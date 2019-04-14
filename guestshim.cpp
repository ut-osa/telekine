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

#include "check_env.h"
#include "./command_scheduler.h"
#include "lgm_memcpy.hpp"
#include "hip_function_info.hpp"

#include <chrono>

using namespace std;

#define FIXED_EXTRA_SIZE 256

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

        if (fixed_rate_sep_memcpy_command_scheduler_enabled()) {
            fprintf(stderr, "Create new SepMemcpyCommandScheduler with stream = %p, "
                    "batch_size = %d, interval = %d\n",
                    (void*)stream, batch_size, fixed_rate_interval_us);
            command_scheduler_map_.emplace(stream, std::make_shared<SepMemcpyCommandScheduler>(
                    stream, batch_size, fixed_rate_interval_us));
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
    return nw_hipMemcpyAsync(dst, src, size, kind, this->stream_);
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
                                                     int fixed_rate_interval_us)
   : BatchCommandScheduler(stream, batch_size, fixed_rate_interval_us)
{
   hipError_t ret;
   ret = hipStreamCreate(&xfer_stream_);
   assert(ret == hipSuccess);
   for (int i = 0; i < N_STG_BUFS; i++) {
      /* allocate space for the buffer plus an 8 byte tag */
      ret = hipMalloc(stg_bufs + i, FIXED_SIZE_FULL);
      assert(ret == hipSuccess);
   }
}

SepMemcpyCommandScheduler::~SepMemcpyCommandScheduler(void)
{
   hipStreamDestroy(xfer_stream_);
   for (int i = 0; i < N_STG_BUFS; i++)
      hipFree(stg_bufs[i]);
}

hipError_t BatchCommandScheduler::AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql,
        uint8_t *extra, size_t extra_size, hipEvent_t start, hipEvent_t stop)
{
    assert(!start && !stop);
    {
        std::lock_guard<std::mutex> lk2(wait_mutex_);
        assert(extra_size < FIXED_EXTRA_SIZE);

        CommandEntry command;
        command.kind = KERNEL_LAUNCH;
        command.kernel_launch_param.aql = *aql;
        command.kernel_launch_param.kernArgSize = FIXED_EXTRA_SIZE;
        command.kernel_launch_param.kernArg = malloc(FIXED_EXTRA_SIZE);
        memcpy(command.kernel_launch_param.kernArg, extra, extra_size);
        std::lock_guard<std::mutex> lk1(pending_commands_mutex_);
        pending_commands_.push_back(command);
    }
    pending_commands_cv_.notify_all();
    return hipSuccess; // TODO more accurate return value
}

hipError_t BatchCommandScheduler::AddMemcpyAsync(void* dst, const void* src, size_t size,
        hipMemcpyKind kind) {
    {
        std::lock_guard<std::mutex> lk2(wait_mutex_);
        CommandEntry command;
        command.kind = MEMCPY;
        command.memcpy_param.dst = dst;
        command.memcpy_param.src = src;
        command.memcpy_param.kind = kind;
        command.memcpy_param.size = size;
        pending_commands_.push_back(command);
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
                return pending_commands_.size() == 0 &&
                       pending_d2h_.size() == 0;
                });
    }
    return hipSuccess; // TODO more accurate return value
}

void BatchCommandScheduler::do_memcpy(void *dst, const void *src, size_t size,
                                      hipMemcpyKind kind)
{
   auto ret = nw_hipMemcpyAsync(dst, src, size, kind, stream_);
   assert(ret == hipSuccess);
}

__global__ void
vector_copy_in(uint8_t *C_d, uint8_t *A_d, size_t N, tag_t tag)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x ;

    while (BUF_TAG(A_d) != tag)
       /* spin */;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i];
    }
}


__global__ void
vector_copy_out(uint8_t *C_d, uint8_t *A_d, size_t N, tag_t tag)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i];
    }
    __syncthreads();
    /* set this tag to show that copy was done */
    BUF_TAG(C_d) = tag;
}

#define BLOCKS_THREADS_TO_AQL(blocks, threads) \
   (blocks * threads), 1, 1, threads, 1, 1

void SepMemcpyCommandScheduler::enqueue_device_copy(void *dst, const void *src,
                                                    size_t size, tag_t tag,
                                                    bool in)
{
   static std::once_flag f;
   static hsa_kernel_dispatch_packet_t copy_in_aql = {0};
   static hsa_kernel_dispatch_packet_t copy_out_aql = {0};
   hsa_kernel_dispatch_packet_t *aql;
   hipEvent_t event;

   call_once(f, [&]() {
      auto fun = hip_function_lookup((uintptr_t)vector_copy_out, stream_);
      hip_function_to_aql(&copy_out_aql, fun, BLOCKS_THREADS_TO_AQL(512, 256), 0);
      auto fun_tag = hip_function_lookup((uintptr_t)vector_copy_in, stream_);
      hip_function_to_aql(&copy_in_aql, fun_tag, BLOCKS_THREADS_TO_AQL(512, 256), 0);
   });

   auto cp_args = hip_impl::make_kernarg(dst, src, size, tag);
   aql = in ? &copy_in_aql : &copy_out_aql;

	assert(cp_args.size() < FIXED_EXTRA_SIZE);

	CommandEntry command;
	command.kind = KERNEL_LAUNCH;
	command.kernel_launch_param.aql = *aql;
	command.kernel_launch_param.kernArgSize = FIXED_EXTRA_SIZE;
	command.kernel_launch_param.kernArg = malloc(FIXED_EXTRA_SIZE);
	memcpy(command.kernel_launch_param.kernArg, cp_args.data(), cp_args.size());

   /* push_front because we want to copy kernel to run in the order the memcpy
    * would hav
    */
	pending_commands_.push_front(command);
}

void SepMemcpyCommandScheduler::pre_notify(void)
{
   static uint8_t scratch[FIXED_SIZE_FULL];
   int err;

	/* todo, do memcpy even if there aren't outstanding copies */
	if (pending_d2h_.size() == 0)
      return;

   auto &op = pending_d2h_.at(0);

   /* @Vance: add invocation of encrypt kernel here should take op.src_ and put
    * it in some sb, then change the argument below to copy instead from sb to
    * scratch
    */

   err = nw_hipMemcpyAsync(scratch, op.src_, FIXED_SIZE_FULL,
                           hipMemcpyDeviceToHost, xfer_stream_);
   assert(err == hipSuccess);

   /* then decrypt on the cpu here*/

   /* if the tag matches the buffer was ready and we can stop looking for it */
   if (BUF_TAG(scratch) == op.tag_) {
      memcpy(op.dst_, scratch, op.size_);
      pending_d2h_.pop_front();
      pending_commands_cv_.notify_all();
   } else {
      /* if not... then we have to look for it next time */
   }
}

void SepMemcpyCommandScheduler::do_memcpy(void *dst, const void *src, size_t size,
                                          hipMemcpyKind kind)
{
   hipEvent_t event;
   void *sb;
   hipError_t err;
   tag_t tag;
   static uint8_t scratch[FIXED_SIZE_FULL];

   assert(size <= FIXED_SIZE_B);

   switch (kind) {
   case hipMemcpyHostToDevice:
      /* if buffer is too small, copy it to the scratch so that the
       * transfer is fixed size
       */
      memcpy(scratch, src, size);
      sb = next_stg_buf();
      tag = gen_tag();
      BUF_TAG(scratch) = tag;
      err = nw_hipMemcpyAsync(sb, scratch, FIXED_SIZE_FULL, kind, xfer_stream_);
      assert(err == hipSuccess);

      /* @Vance: add invocation of decrypt kernel here should take
       * sb and put it in sb1, then change the argument below to copy instead
       * from sb1 to dst
       */

      enqueue_device_copy(dst, sb, size, tag, true);
      break;
   case hipMemcpyDeviceToHost:
      tag = gen_tag();
      sb = next_stg_buf();
      enqueue_device_copy(sb, src, size, tag, false);
      pending_d2h_.emplace_back((void *)dst, (void *)sb, size, tag);
      break;
   default:
      err = nw_hipMemcpyAsync(dst, src, size, kind, stream_);
      assert(err == hipSuccess);
      break;
   }
}

void BatchCommandScheduler::ProcessThread() {
    while (this->running) {
        std::vector<KernelLaunchParam> params;
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
            pre_notify();
            if (pending_commands_.size() == 0) continue;
            if (pending_commands_[0].kind == MEMCPY) {
                MemcpyParam param = pending_commands_[0].memcpy_param;
                pending_commands_.pop_front();
                do_memcpy(param.dst, param.src, param.size, param.kind);
                pending_commands_cv_.notify_all();
                continue;
            }
            for (int i = 0; i < batch_size_; i++) {
                if (i >= pending_commands_.size()) break;
                if (pending_commands_[i].kind == MEMCPY) break;
                params.push_back(pending_commands_[i].kernel_launch_param);
            }
        }

        std::vector<hsa_kernel_dispatch_packet_t> aql;
        std::vector<size_t> extra_size;

        size_t total_extra_size = 0;
        for (int i = 0; i < params.size(); i++) {
            aql.emplace_back(params[i].aql);
            extra_size.push_back(params[i].kernArgSize);
            total_extra_size += params[i].kernArgSize;
        }
        char* all_extra = (char*)malloc(total_extra_size);
        size_t cursor = 0;
        for (int i = 0; i < params.size(); i++) {
            memcpy(all_extra + cursor, params[i].kernArg, params[i].kernArgSize);
            cursor += params[i].kernArgSize;
            free(params[i].kernArg);
        }

        __do_c_hipHccModuleLaunchMultiKernel(
            params.size(),
            aql.data(), stream_,
            all_extra, total_extra_size, extra_size.data());

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

    if (startEvent || stopEvent) {
       return __do_c_hipHccModuleLaunchKernel(&aql, hStream, nullptr,
               (char *)extra_buf, extra_size, startEvent, stopEvent);
    } else {
       return CommandScheduler::GetForStream(hStream)->AddKernelLaunch(&aql,
               extra_buf, extra_size, nullptr, nullptr);
   }
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
