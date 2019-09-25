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
#include "hip_function_info.hpp"
#include "nw/include/n_ava_channels.h"

#include "command_scheduler.h"
#include "config.h"
#include <chrono>

using namespace std;

std::map<hipStream_t, std::shared_ptr<CommandScheduler>> CommandScheduler::command_scheduler_map_{};
std::mutex CommandScheduler::command_scheduler_map_mu_{};

static std::shared_ptr<CommandScheduler> make_scheduler(hipStream_t stream)
{
     switch (tlkine::config.sched_type) {
     case ENCRYPTED:
         return std::make_shared<EncryptedSepMemcpyCommandScheduler>(stream);
     case MANAGED:
         return std::make_shared<SepMemcpyCommandScheduler>(stream);
     case BATCHED:
         return std::make_shared<BatchCommandScheduler>(stream);
     case BASELINE:
         return std::make_shared<BaselineCommandScheduler>(stream);
     default:
         fprintf(stderr, "Impossible!\n");
         abort();
     }
     return std::shared_ptr<CommandScheduler>(nullptr);
}

std::shared_ptr<CommandScheduler> CommandScheduler::GetForStream(hipStream_t stream)
{
    std::lock_guard<std::mutex> lk(command_scheduler_map_mu_);
    if (!command_scheduler_map_.count(stream)) {
        command_scheduler_map_.emplace(stream, make_scheduler(stream));
        std::cerr << "Create new scheduler: " << std::endl
                  << *command_scheduler_map_.at(stream) << std::endl;
    }
    return command_scheduler_map_.at(stream);
}

/**** BaselineCommandScheduler ****/
hipError_t BaselineCommandScheduler::AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql,
        uint8_t *extra, size_t extra_size, hipEvent_t start, hipEvent_t stop)
{
    return __do_c_hipHccModuleLaunchKernel(aql, this->stream_, nullptr, (char *)extra,
            extra_size, start, stop);
}

hipError_t BaselineCommandScheduler::AddMemcpyAsync(void* dst, const void* src, size_t size,
        hipMemcpyKind kind)
{
    return nw_hipMemcpySync(dst, src, size, kind, this->stream_);
};

hipError_t BaselineCommandScheduler::Wait(void)
{
    return nw_hipStreamSynchronize(this->stream_);
}

/**** BatchCommandScheduler ****/
BatchCommandScheduler::BatchCommandScheduler(hipStream_t stream) :
    CommandScheduler(stream),
    batch_size_(tlkine::config.batch_size),
    quantum_waiter_((tlkine::config.fixed_rate_interval_us < 0) ? nullptr :
       std::unique_ptr<QuantumWaiter>(new QuantumWaiter(tlkine::config.fixed_rate_interval_us))),
    running(true),
    process_thread_(new std::thread(&BatchCommandScheduler::ProcessThread, this))
{
}

BatchCommandScheduler::~BatchCommandScheduler(void)
{
    running = false;
    pending_commands_cv_.notify_all();
    process_thread_->join();
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

void BatchCommandScheduler::ProcessThread() {
    SetThreadPriority();
    hipSetDevice(device_index);
    while (this->running) {
        std::vector<KernelLaunchParam *> params;
        {
            std::unique_lock<std::mutex> lk1(pending_commands_mutex_);
            if (!quantum_waiter_) {
                // if we're not fixed-rating the communication, wait 10 ms
                // or until we have at least half a batch of commands, then continue
                pending_commands_cv_.wait_for(lk1, std::chrono::milliseconds(10), [this] () {
                    return pending_commands_.size() > batch_size_ * 0.5 || !this->running;
                });
                if (!this->running) break;
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

        // if (params.size() > 0) fprintf(stderr, "%d real kernels\n", (int) params.size());

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

/**** SepMemcpyCommandSchduler ****/
SepMemcpyCommandScheduler::SepMemcpyCommandScheduler(hipStream_t stream)
   : BatchCommandScheduler(stream),
     cur_batch_id(0), last_real_batch(0), batches_finished(0),
     n_staging_buffers(tlkine::config.memcpy_n_staging_buffers), status_buf(nullptr),
     stg_in_idx(0), stg_out_idx(0),
     h2d_memcpy_waiter_((tlkine::config.memcpy_fixed_rate_interval_us < 0) ? nullptr :
           std::unique_ptr<QuantumWaiter>(new QuantumWaiter(tlkine::config.memcpy_fixed_rate_interval_us))),
     d2h_memcpy_waiter_((tlkine::config.memcpy_fixed_rate_interval_us < 0) ? nullptr :
           std::unique_ptr<QuantumWaiter>(new QuantumWaiter(tlkine::config.memcpy_fixed_rate_interval_us)))
{
   hipError_t ret;
   ret = hipStreamCreate(&h2d_xfer_stream_);
   assert(ret == hipSuccess);
   ret = hipStreamCreate(&d2h_xfer_stream_);
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
   h2d_memcpy_thread_ = std::make_unique<std::thread>(&SepMemcpyCommandScheduler::H2DMemcpyThread, this);
   d2h_memcpy_thread_ = std::make_unique<std::thread>(&SepMemcpyCommandScheduler::D2HMemcpyThread, this);
}

SepMemcpyCommandScheduler::~SepMemcpyCommandScheduler(void)
{
   running = false;
   pending_h2d_cv_.notify_all();
   h2d_memcpy_thread_->join();
   pending_d2h_cv_.notify_all();
   d2h_memcpy_thread_->join();
   hipStreamDestroy(h2d_xfer_stream_);
   hipStreamDestroy(d2h_xfer_stream_);
   for (int i = 0; i < n_staging_buffers; i++) {
      hipFree(in_bufs[i]);
      hipFree(out_bufs[i]);
   }
   free(in_bufs);
   free(out_bufs);
   hipFree(encrypt_out_buf);
}

hipError_t SepMemcpyCommandScheduler::AddMemcpyAsync(void* dst, const void* src, size_t size,
        hipMemcpyKind kind)
{
   void *sb1;
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

         std::unique_lock<std::mutex> lk1(pending_h2d_mutex_);
         pending_h2d_commands_.emplace_back(
               new CommandEntry(sb1, plaintext, FIXED_SIZE_FULL, kind, tag));
         pending_h2d_cv_.notify_all();
         break;
      }
      case hipMemcpyDeviceToHost: {
         assert(size <= FIXED_SIZE_FULL && "enable hip-memcpy-fixed-size");
         tag = gen_tag();
         sb1 = next_out_buf();
         enqueue_device_copy(sb1, src, size, tag, false);

         std::unique_lock<std::mutex> lk1(pending_d2h_mutex_);
         pending_d2h_commands_.emplace_back(
               new CommandEntry(dst, sb1, size, kind, tag));
         assert(pending_d2h_commands_.size() <= n_staging_buffers);
         pending_d2h_cv_.notify_all();
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

hipError_t SepMemcpyCommandScheduler::Wait(void)
{
    std::lock_guard<std::mutex> lk2(wait_mutex_);
    {
        std::unique_lock<std::mutex> lk1(pending_commands_mutex_);
        pending_commands_cv_.wait(lk1, [this] () {
                if (tlkine::config.check_batch_complete && last_real_batch > batches_finished)
                  return false;
                return pending_commands_.size() == 0 &&
                       pending_d2h_commands_.size() == 0 &&
                       pending_h2d_commands_.size() == 0;
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
gpu_snapshot_tagged_buf_on_batch(hip_launch_batch_t* batch, T *dst, V *src, hipStream_t s)
{
   /* copy the tag first since the producer could complete in the middle and
    * depending on the interleaving we may have incorrect data in the buffer
    */
   hipLaunchAddToBatch(batch, tag_copy, dim3(1), dim3(1), 0, s, BUF_TAG(dst), BUF_TAG(src), sizeof(tag_t));
   hipLaunchAddToBatch(batch, vector_copy, dim3(512), dim3(256), 0, s, dst, src, FIXED_SIZE_B);
}

void SepMemcpyCommandScheduler::do_next_d2h(void)
{
   // This implementation assumes we use FIXED_SIZE_B buffers
   // make scratch buffer big enough to store encrypted (padded + MACed) data
   static uint8_t plaintext[FIXED_SIZE_FULL];
   bool real_copy = false;

   MemcpyParam param(nullptr, status_buf, 0, hipMemcpyDeviceToHost, 0);

   pending_d2h_mutex_.lock();
   if (pending_d2h_commands_.size() > 0) {
      assert(pending_d2h_commands_[0]->kind == MEMCPY);
      param = pending_d2h_commands_[0]->memcpy_param;
      real_copy = true;
   }
   pending_d2h_mutex_.unlock();

   assert(param.size <= FIXED_SIZE_B);
   d2h(plaintext, param.src, FIXED_SIZE_FULL, hipMemcpyDeviceToHost, d2h_xfer_stream_);

   if (real_copy) {
      /* if the tag matches the buffer was ready and we can stop looking for it */
      if (*BUF_TAG(plaintext) == param.tag) {
         memcpy(param.dst, plaintext, param.size);
         pending_d2h_mutex_.lock();
         pending_d2h_commands_.pop_front();
         pending_d2h_mutex_.unlock();

         pending_commands_cv_.notify_all();
      } else {
         /* if not... then we have to look for it next time */
      }
   } else {
      memcpy(&batches_finished, plaintext, sizeof(batches_finished));
      pending_commands_cv_.notify_all();
   }
}

void SepMemcpyCommandScheduler::d2h(void* dst, const void* src, size_t sizeBytes,
        hipMemcpyKind kind, hipStream_t stream) {
   hip_launch_memcpy_batch_t batch;
   gpu_snapshot_tagged_buf_on_batch(&batch, encrypt_out_buf, src, stream);

   batch.dst = dst;
   batch.src = encrypt_out_buf;
   batch.sizeBytes = sizeBytes;
   batch.kind = hipMemcpyDeviceToHost;

   hipLaunchMemcpyBatchNOW(&batch, stream);
}

void SepMemcpyCommandScheduler::do_next_h2d()
{
    static uint8_t filler_buf[FIXED_SIZE_FULL];
    MemcpyParam param((void *)nop_buffer, (void *)filler_buf, FIXED_SIZE_FULL,
                      hipMemcpyHostToDevice, 0);
    bool real_copy = false;

    pending_h2d_mutex_.lock();
    if (pending_h2d_commands_.size() > 0) {
       assert(pending_h2d_commands_[0]->kind == MEMCPY);
       param = pending_h2d_commands_[0]->memcpy_param;
       real_copy = true;
    }
    pending_h2d_mutex_.unlock();

    // XXX overloaded by subclasses to add encryption, etc.
    h2d(param.dst, param.src, FIXED_SIZE_FULL, param.kind, h2d_xfer_stream_, param.tag);

    if (real_copy) {
       free((void *)param.src);

       pending_h2d_mutex_.lock();
       pending_h2d_commands_.pop_front();
       pending_h2d_mutex_.unlock();

       pending_commands_cv_.notify_all();
    }
}

void SepMemcpyCommandScheduler::h2d(void* dst, const void* src, size_t sizeBytes,
        hipMemcpyKind kind, hipStream_t stream, tag_t tag) {
   // XXX in_stg_buf is global for scheduler
   hip_launch_memcpy_batch_t batch;
   batch.dst = in_stg_buf;
   batch.src = (void*) src;
   batch.sizeBytes = sizeBytes;
   batch.kind = hipMemcpyHostToDevice;

   hipLaunchAddToBatch(&batch, vector_copy, dim3(512), dim3(256), 0, stream, dst, in_stg_buf, sizeBytes);
   hipLaunchAddToBatch(&batch, set_tag, dim3(1), dim3(1), 0, stream, BUF_TAG(dst), tag);
   hipLaunchMemcpyBatchNOW(&batch, stream);
   // XXX check launch dimensions, for optimality 
}

void SepMemcpyCommandScheduler::H2DMemcpyThread()
{
    SetThreadPriority();
    set_ava_chan_no(1);
    hipSetDevice(device_index);

    while (this->running) {
       if (!h2d_memcpy_waiter_) {
          std::unique_lock<std::mutex> lk1(pending_h2d_mutex_);
          pending_h2d_cv_.wait(lk1, [this] () {
              return pending_h2d_commands_.size() > 0 || !this->running;
          });
          if (!this->running) break;
       } else {
          h2d_memcpy_waiter_->WaitNextQuantum();
       }
       do_next_h2d();
    }
}

void SepMemcpyCommandScheduler::D2HMemcpyThread()
{
    SetThreadPriority();
    set_ava_chan_no(2);
    hipSetDevice(device_index);

    while (this->running) {
       if (!d2h_memcpy_waiter_) {
          std::unique_lock<std::mutex> lk1(pending_d2h_mutex_);
          pending_d2h_cv_.wait(lk1, [this] () {
              return pending_d2h_commands_.size() > 0 || !this->running;
          });
          if (!this->running) break;
       } else {
          d2h_memcpy_waiter_->WaitNextQuantum();
       }
       do_next_d2h();
    }
}

/**** EncryptedSepMemcpyCommandScheduler ****/
EncryptedSepMemcpyCommandScheduler::EncryptedSepMemcpyCommandScheduler(hipStream_t stream)
      : SepMemcpyCommandScheduler(stream) {}

EncryptedSepMemcpyCommandScheduler::~EncryptedSepMemcpyCommandScheduler(void) {}

void EncryptedSepMemcpyCommandScheduler::h2d(void* dst, const void* src, size_t sizeBytes,
        hipMemcpyKind kind, hipStream_t stream, tag_t tag) {
   if (h2d_encryption_state == nullptr) {
       h2d_encryption_state.reset(new lgm::EncryptionState(stream));
   }
   static uint8_t ciphertext[FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES];
   assert(sizeBytes == FIXED_SIZE_FULL && "encryption needs fixed size buffers");
   // we need fixed size buffers because we have this statically allocated ciphertext buffer
   lgm::CPUEncrypt(ciphertext, src, sizeBytes, *h2d_encryption_state.get());
   h2d_encryption_state->nextNonceCPU();

   hip_launch_memcpy_batch_t batch;

   batch.dst = in_stg_buf;
   batch.src = ciphertext;
   batch.sizeBytes = sizeBytes + crypto_aead_aes256gcm_ABYTES;
   batch.kind = hipMemcpyHostToDevice;

   lgm::DecryptAsync(&batch, dst, in_stg_buf, sizeBytes + crypto_aead_aes256gcm_ABYTES, stream,
           *h2d_encryption_state.get());
   h2d_encryption_state->nextNonceGPU(&batch, stream);
   hipLaunchAddToBatch(&batch, set_tag, dim3(1), dim3(1), 0, stream, BUF_TAG(dst), tag);
   hipLaunchMemcpyBatchNOW(&batch, stream);
}

void EncryptedSepMemcpyCommandScheduler::d2h(void* dst, const void* src, size_t sizeBytes,
        hipMemcpyKind kind, hipStream_t stream) {
   if (d2h_encryption_state == nullptr) {
       d2h_encryption_state.reset(new lgm::EncryptionState(stream));
   }
   static uint8_t ciphertext[FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES];
   assert(sizeBytes == FIXED_SIZE_FULL && "encryption needs fixed size buffers");
   hip_launch_memcpy_batch_t batch;
   // we need fixed size buffers because we have this statically allocated ciphertext buffer
   gpu_snapshot_tagged_buf_on_batch(&batch, out_stg_buf, src, stream);
   lgm::EncryptAsync(&batch, encrypt_out_buf, out_stg_buf, sizeBytes, stream, *d2h_encryption_state.get());
   d2h_encryption_state->nextNonceGPU(&batch, stream);

   batch.dst = ciphertext;
   batch.src = encrypt_out_buf;
   batch.sizeBytes = sizeBytes + crypto_aead_aes256gcm_ABYTES;
   batch.kind = hipMemcpyDeviceToHost;

   hipLaunchMemcpyBatchNOW(&batch, stream);

   lgm::CPUDecrypt(dst, ciphertext, sizeBytes + crypto_aead_aes256gcm_ABYTES, *d2h_encryption_state.get());
   d2h_encryption_state->nextNonceCPU();
}
