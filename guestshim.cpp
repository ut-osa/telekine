#include <hsa/hsa.h>

#include <libhsakmt/hsakmttypes.h>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <iostream>
#include <cstdio>

#include "hip_cpp_bridge.h"
#include "lgm_kernels.hpp"

#include "check_env.h"
#include "./command_scheduler.h"
#include "lgm_memcpy.hpp"
#include "hip_function_info.hpp"

#include <chrono>

using namespace std;

static bool check_batch_complete() {
    static bool ret = CHECK_ENV("HIP_SYNC_CHECK_BATCH_COMPLETE");
    return ret;
}

static bool fixed_rate_command_scheduler_enabled() {
    static bool ret = CHECK_ENV("HIP_ENABLE_COMMAND_SCHEDULER");
    return ret;
}

static bool fixed_rate_sep_memcpy_command_scheduler_enabled() {
    static bool ret = CHECK_ENV("HIP_ENABLE_SEP_MEMCPY_COMMAND_SCHEDULER");
    return ret;
}

std::shared_ptr<CommandScheduler> CommandScheduler::GetForStream(hipStream_t stream)
{
    std::lock_guard<std::mutex> lk(command_scheduler_map_mu_);
    if (!command_scheduler_map_.count(stream)) {
        int batch_size = DEFAULT_BATCH_SIZE;
        char* s = getenv("HIP_COMMAND_SCHEDULER_BATCH_SIZE");
        if (s) batch_size = atoi(s);
        int fixed_rate_interval_us = DEFAULT_FIXED_RATE_INTERVAL_US;
        s = getenv("HIP_COMMAND_SCHEDULER_FR_INTERVAL_US");
        if (s) fixed_rate_interval_us = atoi(s);
        int memcpy_fixed_rate_interval_us = DEFAULT_FIXED_RATE_INTERVAL_US;
        s = getenv("HIP_COMMAND_SCHEDULER_MEMCPY_FR_INTERVAL_US");
        if (s) memcpy_fixed_rate_interval_us = atoi(s);
        int memcpy_n_staging_buffers = DEFAULT_N_STAGING_BUFFERS;
        s = getenv("HIP_COMMAND_SCHEDULER_MEMCPY_N_STAGING_BUFFERS");
        if (s) memcpy_n_staging_buffers = atoi(s);

        if (fixed_rate_sep_memcpy_command_scheduler_enabled()) {
            fprintf(stderr, "Create new SepMemcpyCommandScheduler with stream = %p, "
                    "batch_size = %d, interval = %d, n_staging_buffers = %d\n",
                    (void*)stream, batch_size, fixed_rate_interval_us, memcpy_n_staging_buffers);
            command_scheduler_map_.emplace(stream, std::make_shared<SepMemcpyCommandScheduler>(
                    stream, batch_size, fixed_rate_interval_us,
                    memcpy_fixed_rate_interval_us, memcpy_n_staging_buffers));
        }
        else if (fixed_rate_command_scheduler_enabled()) {
            fprintf(stderr, "Create new BatchCommandScheduler with stream = %p, "
                    "batch_size = %d, interval = %d\n",
                    (void*)stream, batch_size, fixed_rate_interval_us);
            command_scheduler_map_.emplace(stream, std::make_shared<BatchCommandScheduler>(
                    stream, batch_size, fixed_rate_interval_us));
        } else {
            // use BaselineCommandScheduler instead of Batch...
            command_scheduler_map_.emplace(stream, std::make_shared<BaselineCommandScheduler>(
                    stream));
        }
    }
    return command_scheduler_map_.at(stream);
}

std::map<hipStream_t, std::shared_ptr<CommandScheduler>> CommandScheduler::command_scheduler_map_{};
std::mutex CommandScheduler::command_scheduler_map_mu_{};


hipError_t BaselineCommandScheduler::AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql,
        uint8_t *extra, size_t extra_size, hipEvent_t start, hipEvent_t stop) {
    return __do_c_hipHccModuleLaunchKernel(aql, this->stream_, nullptr, (char *)extra,
            extra_size, start, stop);
}

hipError_t BaselineCommandScheduler::AddMemcpyAsync(void* dst, const void* src, size_t size,
        hipMemcpyKind kind) {
    return nw_hipMemcpySync(dst, src, size, kind, this->stream_);
};

hipError_t BaselineCommandScheduler::Wait(void) {
    return nw_hipStreamSynchronize(this->stream_);
}

BatchCommandScheduler::BatchCommandScheduler(hipStream_t stream, int batch_size,
        int fixed_rate_interval_us) : CommandScheduler(stream), batch_size_(batch_size),
    quantum_waiter_((fixed_rate_interval_us == DEFAULT_FIXED_RATE_INTERVAL_US) ? nullptr :
    std::unique_ptr<QuantumWaiter>(new QuantumWaiter(fixed_rate_interval_us))),
    running(true),
    process_thread_(new std::thread(&BatchCommandScheduler::ProcessThread, this))
{
}

BatchCommandScheduler::~BatchCommandScheduler(void) {
    running = false;
    process_thread_->join();
}

SepMemcpyCommandScheduler::SepMemcpyCommandScheduler(hipStream_t stream, int batch_size,
                                                     int fixed_rate_interval_us,
                                                     int memcpy_fixed_rate_interval_us,
                                                     size_t n_staging_buffers)
   : status_buf(nullptr), last_real_batch(0), stg_in_idx(0), stg_out_idx(0),
     cur_batch_id(0), batches_finished(0), n_staging_buffers(n_staging_buffers),
     memcpy_waiter_((memcpy_fixed_rate_interval_us == DEFAULT_FIXED_RATE_INTERVAL_US) ? nullptr :
           std::unique_ptr<QuantumWaiter>(new QuantumWaiter(memcpy_fixed_rate_interval_us))),
     BatchCommandScheduler(stream, batch_size, fixed_rate_interval_us)
{
   hipError_t ret;
   ret = hipStreamCreate(&xfer_stream_);
   assert(ret == hipSuccess);
   in_bufs = static_cast<void**>(malloc(n_staging_buffers * sizeof(void*)));
   out_bufs = static_cast<void**>(malloc(n_staging_buffers * sizeof(void*)));
   for (int i = 0; i < n_staging_buffers; i++) {
      /* allocate space for the buffer plus an 8 byte tag plus the MAC */
      ret = hipMalloc(in_bufs + i, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES);
      assert(ret == hipSuccess);
      ret = hipMalloc(out_bufs + i, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES);
   }
   ret = hipMalloc(&encrypt_out_buf, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES);
   assert(ret == hipSuccess);
   ret = hipMalloc(&status_buf, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES);
   assert(ret == hipSuccess);
   ret = hipMalloc(&out_stg_buf, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES);
   assert(ret == hipSuccess);
   ret = hipMalloc(&in_stg_buf, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES);
   assert(ret == hipSuccess);
   ret = hipMalloc(&nop_buffer, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES);
   assert(ret == hipSuccess);
   memcpy_thread_.reset(new std::thread(&SepMemcpyCommandScheduler::MemcpyThread, this));
}

SepMemcpyCommandScheduler::~SepMemcpyCommandScheduler(void)
{
   running = false;
   memcpy_thread_->join();
   hipStreamDestroy(xfer_stream_);
   for (int i = 0; i < n_staging_buffers; i++) {
      hipFree(in_bufs[i]);
      hipFree(out_bufs[i]);
   }
   hipFree(encrypt_out_buf);
}

hipError_t BatchCommandScheduler::AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql,
        uint8_t *extra, size_t extra_size, hipEvent_t start, hipEvent_t stop)
{
   std::promise<void> done;
   std::future<void> done_future = done.get_future();
    {
        std::lock_guard<std::mutex> lk2(wait_mutex_);
        std::lock_guard<std::mutex> lk1(pending_commands_mutex_);
        pending_commands_.emplace_back(new CommandEntry(aql, extra, extra_size,
                                                        start, stop, std::move(done)));
    }
    pending_commands_cv_.notify_all();
    /* need to wait so that these events are valid if the caller wants to use
     * them in ops since we don't handle events
     */
    if (start || stop)
       done_future.wait();
    return hipSuccess; // TODO more accurate return value
}

hipError_t BatchCommandScheduler::AddMemcpyAsync(void* dst, const void* src, size_t size,
        hipMemcpyKind kind) {
    {
        std::lock_guard<std::mutex> lk2(wait_mutex_);
        pending_commands_.emplace_back(new CommandEntry(dst, src, size, kind));
    }
    pending_commands_cv_.notify_all();
    return hipSuccess; // TODO more accurate return value
}


hipError_t SepMemcpyCommandScheduler::AddMemcpyAsync(void* dst, const void* src, size_t size,
        hipMemcpyKind kind)
{
   hipEvent_t event;
   void *sb1;
   hipError_t err;
   tag_t tag;
   static uint8_t *plaintext;


   std::lock_guard<std::mutex> lk2(wait_mutex_);
   switch (kind) {
      case hipMemcpyHostToDevice: {
         assert(size <= FIXED_SIZE_FULL && "enable hip-memcpy-fixed-size");
         tag = gen_tag();
         sb1 = next_in_buf();
         plaintext = (uint8_t *)malloc(FIXED_SIZE_FULL);
         /* if buffer is too small, copy it to the plaintext buffer so that the
         * transfer is fixed size
         */
         memcpy(plaintext, src, size);

         enqueue_device_copy(dst, sb1, size, tag, true);

         std::unique_lock<std::mutex> lk1(pending_copy_mutex_);
         pending_h2d_commands.emplace_back(
               new CommandEntry(sb1, plaintext, FIXED_SIZE_FULL, kind, tag));
         pending_h2d_cv_.notify_all();
         break;
      }
      case hipMemcpyDeviceToHost: {
         assert(size <= FIXED_SIZE_FULL && "enable hip-memcpy-fixed-size");
         tag = gen_tag();
         sb1 = next_out_buf();
         enqueue_device_copy(sb1, src, size, tag, false);

         std::unique_lock<std::mutex> lk1(pending_copy_mutex_);
         pending_d2h_commands.emplace_back(
               new CommandEntry(dst, sb1, size, kind, tag));
         assert(pending_d2h_commands.size() <= n_staging_buffers);
         pending_h2d_cv_.notify_all();
         break;
      }
      case hipMemcpyDeviceToDevice: {
         std::lock_guard<std::mutex> lk1(pending_commands_mutex_);
         pending_commands_.emplace_back(
               new CommandEntry(vector_copy, dim3(512), dim3(256), 0, stream_,
                                dst, src, size));
         pending_commands_cv_.notify_all();
         break;
      }
      default: {
         assert(false);
         break;
      }
   }
   pending_commands_cv_.notify_all();
   return hipSuccess; // TODO more accurate return value
}

hipError_t BatchCommandScheduler::Wait(void) {
    std::lock_guard<std::mutex> lk2(wait_mutex_);
    {
        std::unique_lock<std::mutex> lk1(pending_commands_mutex_);
        pending_commands_cv_.wait(lk1, [this] () {
                return pending_commands_.size() == 0;
                });
    }
    return hipSuccess; // TODO more accurate return value
}

hipError_t SepMemcpyCommandScheduler::Wait(void)
{
    std::lock_guard<std::mutex> lk2(wait_mutex_);
    {
        std::unique_lock<std::mutex> lk1(pending_commands_mutex_);
        pending_commands_cv_.wait(lk1, [this] () {
                if (check_batch_complete() && last_real_batch > batches_finished)
                  return false;
                return pending_commands_.size() == 0 &&
                       pending_d2h_commands.size() == 0 &&
                       pending_h2d_commands.size() == 0;
                });
    }
    return hipSuccess; // TODO more accurate return value
}

void SepMemcpyCommandScheduler::enqueue_device_copy(void *dst, const void *src,
                                                    size_t size, tag_t tag,
                                                    bool in)
{
   assert(size <= FIXED_SIZE_B);
   std::lock_guard<std::mutex> lk1(pending_commands_mutex_);
   if (in) {
      pending_commands_.emplace_back(new CommandEntry(
          check_tag, dim3(1), dim3(1), 0, stream_, BUF_TAG(src), tag));
   }

   pending_commands_.emplace_back(new CommandEntry(
       vector_copy, dim3(512), dim3(256), 0, stream_, dst, src, size));

   if (!in) {
      pending_commands_.emplace_back(new CommandEntry(
          set_tag, dim3(1), dim3(1), 0, stream_, BUF_TAG(dst), tag));
   }
   pending_commands_cv_.notify_all();
}

void SepMemcpyCommandScheduler::add_extra_kernels(
                             vector<KernelLaunchParam> &extrakerns,
                             const vector<KernelLaunchParam *> &realkerns)
{
   if (realkerns.size() > 0)
         last_real_batch = cur_batch_id;
   for (int i = realkerns.size(); i <= batch_size_; i++) {
      extrakerns.emplace_back(nullKern, dim3(0, 0, 0), dim3(0, 0, 0),
                              0, stream_);
   }
   extrakerns.emplace_back(note_batch_id, dim3(1), dim3(1), 0, stream_,
                           status_buf, cur_batch_id++);
}


template<typename T, typename V>
static inline void
gpu_snapshot_tagged_buf(T *dst, V *src, hipStream_t s)
{
   /* copy the tag first since the producer could complete in the middle and
    * depending on the interleaving we may have incorrect data in the buffer
    */
   hipLaunchNOW(tag_copy, dim3(1), dim3(1), 0, s, BUF_TAG(dst), BUF_TAG(src), sizeof(tag_t));
   hipLaunchNOW(vector_copy, dim3(512), dim3(256), 0, s, dst, src, FIXED_SIZE_B);
}

void SepMemcpyCommandScheduler::do_next_d2h(void)
{
   // This implementation assumes we use FIXED_SIZE_B buffers
   // make scratch buffer big enough to store encrypted (padded + MACed) data
   static uint8_t ciphertext[FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES];
   static uint8_t plaintext[FIXED_SIZE_FULL];
   bool real_copy = false;

   hipError_t err;
   MemcpyParam param(nullptr, status_buf, 0, hipMemcpyDeviceToHost, 0);

   pending_copy_mutex_.lock();
   if (pending_d2h_commands.size() > 0) {
      assert(pending_d2h_commands[0]->kind == MEMCPY);
      param = pending_d2h_commands[0]->memcpy_param;
      real_copy = true;
   }
   pending_copy_mutex_.unlock();

   assert(param.size <= FIXED_SIZE_B);
   if (memcpy_encryption_enabled()) {
      gpu_snapshot_tagged_buf(out_stg_buf, param.src, xfer_stream_);
      lgmEncryptAsync(encrypt_out_buf, out_stg_buf, FIXED_SIZE_FULL, xfer_stream_);
      err = nw_hipMemcpySync(ciphertext, encrypt_out_buf, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES,
            hipMemcpyDeviceToHost, xfer_stream_);
      assert(err == hipSuccess);
      lgmCPUDecrypt(plaintext, ciphertext, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES,
            xfer_stream_);
      lgmNextNonceAsync(xfer_stream_);
   } else {
      gpu_snapshot_tagged_buf(encrypt_out_buf, param.src, xfer_stream_);
      err = nw_hipMemcpySync(plaintext, encrypt_out_buf, FIXED_SIZE_FULL, hipMemcpyDeviceToHost,
            xfer_stream_);
      assert(err == hipSuccess);
   }

   if (real_copy) {
      /* if the tag matches the buffer was ready and we can stop looking for it */
      if (*BUF_TAG(plaintext) == param.tag) {
         memcpy(param.dst, plaintext, param.size);
         pending_copy_mutex_.lock();
         pending_d2h_commands.pop_front();
         pending_copy_mutex_.unlock();

         std::unique_lock<std::mutex> lk1(pending_commands_mutex_);
         pending_commands_cv_.notify_all();
      } else {
         /* if not... then we have to look for it next time */
      }
   } else {
      std::unique_lock<std::mutex> lk1(pending_commands_mutex_);
      memcpy(&batches_finished, plaintext, sizeof(batches_finished));
      pending_commands_cv_.notify_all();
   }
}

void SepMemcpyCommandScheduler::do_next_h2d()
{
    hipError_t err;
    static uint8_t ciphertext[FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES];
    static uint8_t filler_buf[FIXED_SIZE_FULL];
    MemcpyParam param((void *)nop_buffer, (void *)filler_buf, FIXED_SIZE_FULL,
                      hipMemcpyHostToDevice, 0);
    bool real_copy = false;

    pending_copy_mutex_.lock();
    if (pending_h2d_commands.size() > 0) {
       assert(pending_h2d_commands[0]->kind == MEMCPY);
       param = pending_h2d_commands[0]->memcpy_param;
       real_copy = true;
    }
    pending_copy_mutex_.unlock();

    if (memcpy_encryption_enabled()) {
      lgmCPUEncrypt(ciphertext, param.src, FIXED_SIZE_FULL, xfer_stream_);
      err = nw_hipMemcpySync(in_stg_buf, ciphertext, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES,
                             param.kind, xfer_stream_);
      assert(err == hipSuccess);
      lgmDecryptAsync(param.dst, in_stg_buf, FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES, xfer_stream_);
      lgmNextNonceAsync(xfer_stream_);
    } else {
       err = nw_hipMemcpySync(in_stg_buf, param.src, FIXED_SIZE_FULL, param.kind, xfer_stream_);
       assert(err == hipSuccess);
	    hipLaunchNOW(vector_copy, dim3(512), dim3(256), 0, xfer_stream_, param.dst, in_stg_buf, FIXED_SIZE_B);
    }
    hipLaunchNOW(set_tag, dim3(1), dim3(1), 0, xfer_stream_,
                 BUF_TAG(param.dst), param.tag);
    if (real_copy) {
       free((void *)param.src);

       pending_copy_mutex_.lock();
       pending_h2d_commands.pop_front();
       pending_copy_mutex_.unlock();

       std::unique_lock<std::mutex> lk1(pending_commands_mutex_);
       pending_commands_cv_.notify_all();
    }
}

extern __thread int chan_no;
void SepMemcpyCommandScheduler::MemcpyThread()
{
    chan_no = 1;
    static size_t last_d2h_sz;

    while (this->running) {
       if (!memcpy_waiter_) {
          std::unique_lock<std::mutex> lk1(pending_copy_mutex_);
          pending_h2d_cv_.wait(lk1, [this] () {
              return pending_h2d_commands.size() > 0 || pending_d2h_commands.size() > 0;
          });
       } else {
          memcpy_waiter_->WaitNextQuantum();
       }
       do_next_h2d();
       do_next_d2h();
    }
}

void BatchCommandScheduler::ProcessThread() {
    const char* nice_str = getenv("HIP_SCHEDULER_THREAD_NICE");
    if (nice_str != NULL) {
        int value = atoi(nice_str);
        int ret = nice(value);
        if (ret == -1 && errno != 0) {
            fprintf(stderr, "Failed to set nice value\n");
        } else {
            fprintf(stderr, "Set nice value to %d\n", value);
        }
    }
    while (this->running) {
        std::vector<KernelLaunchParam *> params;
        {
            std::unique_lock<std::mutex> lk1(pending_commands_mutex_);
            if (!quantum_waiter_) {
                // if we're not fixed-rating the communication, wait 10 ms
                // or until we have at least half a batch of commands, then continue
                pending_commands_cv_.wait_for(lk1, std::chrono::milliseconds(10), [this] () {
                    return pending_commands_.size() > batch_size_ * 0.5;
                });
            } else {
                lk1.unlock();
                quantum_waiter_->WaitNextQuantum();
                lk1.lock();
            }
            if (pending_commands_.size() == 0) continue;
            if (pending_commands_[0]->kind == MEMCPY) {
                MemcpyParam param = pending_commands_[0]->memcpy_param;
                pending_commands_.pop_front();
                auto ret = nw_hipMemcpySync(param.dst, param.src, param.size, param.kind, stream_);
                assert(ret == hipSuccess);
                pending_commands_cv_.notify_all();
                continue;
            }
            for (int i = 0; i < batch_size_; i++) {
                if (i >= pending_commands_.size()) break;
                if (pending_commands_[i]->kind == MEMCPY) break;
                params.push_back(&pending_commands_[i]->kernel_launch_param);
            }
        }

        std::vector<KernelLaunchParam> extra_kerns;
        add_extra_kernels(extra_kerns, params);

        std::vector<hsa_kernel_dispatch_packet_t> aql;
        std::vector<size_t> extra_size;
        std::vector<hipEvent_t> starts, stops;

        size_t total_extra_size = 0;
        for (int i = 0; i < params.size(); i++) {
            aql.emplace_back(params[i]->aql);
            starts.push_back(params[i]->start);
            stops.push_back(params[i]->stop);
            extra_size.push_back(params[i]->kernArgSize);
            total_extra_size += params[i]->kernArgSize;
        }
        for (int i = 0; i < extra_kerns.size(); i++) {
            aql.emplace_back(extra_kerns[i].aql);
            starts.push_back(extra_kerns[i].start);
            stops.push_back(extra_kerns[i].stop);
            extra_size.push_back(extra_kerns[i].kernArgSize);
            total_extra_size += extra_kerns[i].kernArgSize;
        }
        char* all_extra = (char*)malloc(total_extra_size);
        size_t cursor = 0;
        for (int i = 0; i < params.size(); i++) {
            memcpy(all_extra + cursor, params[i]->kernArg, params[i]->kernArgSize);
            cursor += params[i]->kernArgSize;
        }
        for (int i = 0; i < extra_kerns.size(); i++) {
            memcpy(all_extra + cursor, extra_kerns[i].kernArg, extra_kerns[i].kernArgSize);
            cursor += extra_kerns[i].kernArgSize;
        }

        __do_c_hipHccModuleLaunchMultiKernel(
            params.size() + extra_kerns.size(),
            aql.data(), stream_,
            all_extra, total_extra_size, extra_size.data(),
            starts.data(), stops.data());

        free(all_extra);

        {
            std::lock_guard<std::mutex> lk1(pending_commands_mutex_);
            for (int i = 0; i < params.size(); i++) {
                pending_commands_.pop_front();
            }
        }
        pending_commands_cv_.notify_all();
    }
}

hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent,
                                    hipEvent_t stopEvent)
{
    assert(kernelParams == nullptr);
    assert(extra[0] == HIP_LAUNCH_PARAM_BUFFER_POINTER);
    assert(extra[2] == HIP_LAUNCH_PARAM_BUFFER_SIZE);
    assert(extra[4] == HIP_LAUNCH_PARAM_END);

    hsa_kernel_dispatch_packet_t aql = {0};
    uint8_t *extra_buf = (uint8_t *)extra[1];
    size_t extra_size = *(size_t *)extra[3];

    hip_function_to_aql(&aql, f, globalWorkSizeX, globalWorkSizeY,
                        globalWorkSizeZ, localWorkSizeX, localWorkSizeY,
                        localWorkSizeZ, sharedMemBytes);

    return CommandScheduler::GetForStream(hStream)->AddKernelLaunch(&aql,
            extra_buf, extra_size, startEvent, stopEvent);
}

extern "C" hipError_t
hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
               hipStream_t stream) {
    return lgmMemcpyAsync(dst, src, sizeBytes, kind, stream);
}

extern "C" hipError_t
hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
{
    return lgmMemcpy(dst, src, sizeBytes, kind);
}

extern "C" hipError_t
hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int deviceId)
{
    static std::map<std::pair<hipDeviceAttribute_t, int>, int> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk{mu};
    if (cache.count(std::make_pair(attr, deviceId)) == 0) {
        int value;
        hipError_t status = nw_hipDeviceGetAttribute(&value, attr, deviceId);
        if (status != hipSuccess) {
            return status;
        }
        cache[std::make_pair(attr, deviceId)] = value;
    }
    *pi = cache[std::make_pair(attr, deviceId)];
    return hipSuccess;
}

extern "C" hipError_t
hipStreamSynchronize(hipStream_t stream)
{
    // fprintf(stderr, "hipStreamSynchronize\n");
    return CommandScheduler::GetForStream(stream)->Wait();
}

hipError_t
hipHostMalloc(void** ptr, size_t size, unsigned int flags)
{
   void *res = malloc(size);
   if (res) {
      *ptr = res;
      return hipSuccess;
   }
   return hipErrorMemoryAllocation;
}

hipError_t
hipHostFree(void* ptr)
{
   free(ptr);
   return hipSuccess;
}

hipError_t
hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
					  size_t width, size_t height, hipMemcpyKind kind,
					  hipStream_t stream)
{
	 hipError_t e;
    if((width == dpitch) && (width == spitch)) {
            e = hipMemcpyAsync(dst, src, width*height, kind, stream);
    } else {
			if(kind != hipMemcpyDeviceToDevice){
				 for (int i = 0; i < height && e; ++i)
					  e = hipMemcpyAsync((unsigned char*)dst + i * dpitch,
											   (unsigned char*)src + i * spitch, width,
												kind, stream);
			} else {
				assert("DeviceToDevice hipMemcpy2DAsync not implemented!" && 0);
			}
    }

    return e;
}

extern "C" hipError_t
hipGetDeviceProperties(hipDeviceProp_t *prop, int deviceId)
{
    static std::map<int, hipDeviceProp_t*> cache;
    static std::mutex mu;
    std::lock_guard<std::mutex> lk{mu};
    if (cache.count(deviceId) == 0) {
        hipDeviceProp_t* _prop = new hipDeviceProp_t;
        hipError_t status = __do_c_hipGetDeviceProperties((char *)_prop, deviceId);
        if (status != hipSuccess) {
            return status;
        }
        cache[deviceId] = _prop;
    }
    *prop = *cache[deviceId];
    return hipSuccess;
}

extern "C" hipError_t
hipModuleLaunchKernel(hipFunction_t f, uint32_t gridDimX, uint32_t gridDimY,
                      uint32_t gridDimZ, uint32_t blockDimX, uint32_t blockDimY,
                      uint32_t blockDimZ, uint32_t sharedMemBytes, hipStream_t hStream,
                      void** kernelParams, void** extra)
{
   return hipHccModuleLaunchKernel(f,
               blockDimX * gridDimX, blockDimY * gridDimY, blockDimZ * gridDimZ,
               blockDimX, blockDimY, blockDimZ,
               sharedMemBytes, hStream, kernelParams, extra,
               nullptr, nullptr);
}

extern "C" hsa_status_t HSA_API
nw_hsa_iterate_agents(
      hsa_status_t (*callback)(hsa_agent_t agent, void* data),
      void* data)
{
   hsa_agent_t agents[MAX_AGENTS];
   size_t n_agents = __do_c_get_agents(agents, MAX_AGENTS);
   for (auto agent = agents; agent < agents + n_agents; ++agent) {
      if (callback(*agent, data) != HSA_STATUS_SUCCESS)
         break;
   }
   return HSA_STATUS_SUCCESS;
}

template <uint32_t block_dim, typename RandomAccessIterator, typename N, typename T>
__global__ void hip_fill_n(RandomAccessIterator f, N n, T value) {
    const uint32_t grid_dim = gridDim.x * blockDim.x;

    size_t idx = blockIdx.x * block_dim + threadIdx.x;
    while (idx < n) {
        f[idx] = value;
        idx += grid_dim;
    }
}

template <typename T, typename std::enable_if<std::is_integral<T>{}>::type* = nullptr>
inline const T& clamp_integer(const T& x, const T& lower, const T& upper) {
    assert(!(upper < lower));

    return std::min(upper, std::max(x, lower));
}

template <typename T>
void ihipMemsetKernel(hipStream_t stream, T* ptr, T val, size_t sizeBytes) {
    static constexpr uint32_t block_dim = 256;

    const uint32_t grid_dim = clamp_integer<size_t>(sizeBytes / block_dim, 1, UINT32_MAX);

    hipLaunchKernelGGL(hip_fill_n<block_dim>, dim3(grid_dim), dim3{block_dim}, 0u, stream, ptr,
                       sizeBytes, std::move(val));
}

hipError_t ihipMemset(void* dst, int  value, size_t sizeBytes,
                      hipStream_t stream)
{
    hipError_t e = hipSuccess;

    if ((sizeBytes & 0x3) == 0) {
       // use a faster dword-per-workitem copy:
       try {
           value = value & 0xff;
           uint32_t value32 = (value << 24) | (value << 16) | (value << 8) | (value) ;
           ihipMemsetKernel<uint32_t> (stream, static_cast<uint32_t*> (dst), value32, sizeBytes/sizeof(uint32_t));
       }
       catch (std::exception &ex) {
           e = hipErrorInvalidValue;
       }
    } else {
       // use a slow byte-per-workitem copy:
       try {
           ihipMemsetKernel<char> (stream, static_cast<char*> (dst), value, sizeBytes);
       }
       catch (std::exception &ex) {
           e = hipErrorInvalidValue;
       }
    }
    return e;
};

hipError_t hipMemsetAsync(void* dst, int value, size_t sizeBytes, hipStream_t stream)
{
    return ihipMemset(dst, value, sizeBytes, stream);
}

hipError_t hipMemset(void* dst, int value, size_t sizeBytes)
{
    return hipMemsetAsync(dst, value, sizeBytes,
                          CommandScheduler::GetDefStream());
}

namespace hip_impl
{
    void hipLaunchKernelGGLImpl(
        uintptr_t function_address,
        const dim3& numBlocks,
        const dim3& dimBlocks,
        uint32_t sharedMemBytes,
        hipStream_t stream,
        void** kernarg)
    {
          hipModuleLaunchKernel(
              hip_function_lookup(function_address, stream),
              numBlocks.x,
              numBlocks.y,
              numBlocks.z,
              dimBlocks.x,
              dimBlocks.y,
              dimBlocks.z,
              sharedMemBytes,
              stream,
              nullptr,
              kernarg);
    }
} // namespace hip_impl
