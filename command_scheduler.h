#ifndef _COMMAND_SCHEDULER_H_
#define _COMMAND_SCHEDULER_H_

#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <map>
#include <memory>

#include "hip/hip_runtime.h"
#include "hip_function_info.hpp"

#include "quantum_waiter.h"

#define DEFAULT_BATCH_SIZE 64
#define DEFAULT_FIXED_RATE_INTERVAL_US -1

typedef uint64_t tag_t;

#define FIXED_SIZE_FULL (0x1UL << 20) // 1 MB
#define FIXED_SIZE_B (FIXED_SIZE_FULL - sizeof(tag_t))
#define BUF_TAG(buf) (*((uint64_t *)(&((buf)[FIXED_SIZE_B]))))
#define FIXED_EXTRA_SIZE 256


class CommandScheduler {
public:
    CommandScheduler(hipStream_t stream) : stream_(stream) {}
    virtual ~CommandScheduler(void) {}
    virtual hipError_t AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql,
            uint8_t* extra, size_t extra_size, hipEvent_t start, hipEvent_t stop) = 0;
    virtual hipError_t AddMemcpyAsync(void* dst, const void* src, size_t size, hipMemcpyKind kind) = 0;
    virtual hipError_t Wait(void) = 0;

    static std::shared_ptr<CommandScheduler> GetForStream(hipStream_t stream);
protected:
    hipStream_t stream_;
    static std::map<hipStream_t, std::shared_ptr<CommandScheduler>> command_scheduler_map_;
    static std::mutex command_scheduler_map_mu_;
};

class BatchCommandScheduler : public CommandScheduler {
public:
    BatchCommandScheduler(hipStream_t stream, int batch_size, int fixed_rate_interval_us);
    ~BatchCommandScheduler(void);
    hipError_t AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql, uint8_t *extra,
            size_t extra_size, hipEvent_t start, hipEvent_t stop) override;
    hipError_t AddMemcpyAsync(void* dst, const void* src, size_t size, hipMemcpyKind kind) override;
    virtual hipError_t Wait(void) override;
protected:
    void ProcessThread();
    virtual void do_memcpy(void *dst, const void *src, size_t size, hipMemcpyKind kind);
    virtual void pre_notify(void){};

    struct KernelLaunchParam {
        hsa_kernel_dispatch_packet_t aql;
        size_t kernArgSize;
        uint8_t kernArg[FIXED_EXTRA_SIZE];
        KernelLaunchParam(const hsa_kernel_dispatch_packet_t *_aql,
                          uint8_t *kern_arg, size_t kern_arg_size) :
           aql(*_aql), kernArgSize(FIXED_EXTRA_SIZE)
        {
           assert(kern_arg_size < FIXED_EXTRA_SIZE);
           memcpy(kernArg, kern_arg, kern_arg_size);
        }
        template <typename... Args, typename F = void (*)(Args...)>
        KernelLaunchParam(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                          std::uint32_t sharedMemBytes, hipStream_t stream,
                          Args... args) : aql{0}, kernArgSize(FIXED_EXTRA_SIZE)
        {
           auto kern_args = hip_impl::make_kernarg(std::move(args)...);
           auto fun = hip_function_lookup((uintptr_t)kernel, stream);

           hip_function_to_aql(&aql, fun, DIM3_TO_AQL(numBlocks, dimBlocks),
                               sharedMemBytes);
           assert(kern_args.size() < FIXED_EXTRA_SIZE);
           memcpy(kernArg, kern_args.data(), kern_args.size());
        }
    };

    struct MemcpyParam {
        void* dst;
        const void* src;
        size_t size;
        hipMemcpyKind kind;
        MemcpyParam(void *_dst, const void *_src, size_t _size, hipMemcpyKind _kind) :
           dst(_dst), src(_src), size(_size), kind(_kind) {}
    };

    enum CommandKind {
        KERNEL_LAUNCH,
        MEMCPY
    };

    struct CommandEntry {
        CommandKind kind;
        union {
           KernelLaunchParam kernel_launch_param;
           MemcpyParam memcpy_param;
        };
        CommandEntry(const hsa_kernel_dispatch_packet_t *aql, uint8_t *kern_arg,
                     size_t kern_arg_size) :
           kind(KERNEL_LAUNCH),
           kernel_launch_param(aql, kern_arg, kern_arg_size) {}
        CommandEntry(void *dst, const void *src, size_t size, hipMemcpyKind mkind) :
           kind(MEMCPY),
           memcpy_param(dst, src, size, mkind) {}
        template <typename... Args, typename F = void (*)(Args...)>
        CommandEntry(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                          std::uint32_t sharedMemBytes, hipStream_t stream,
                          Args... args) :
           kind(KERNEL_LAUNCH),
           kernel_launch_param(std::move(kernel), numBlocks, dimBlocks, sharedMemBytes,
                               stream, std::move(args)...) {}
    };

    virtual void add_extra_kernels(std::vector<KernelLaunchParam> &extrakerns,
                             const std::vector<KernelLaunchParam *> &params) {};

    std::deque<std::unique_ptr<CommandEntry>> pending_commands_;
    std::mutex pending_commands_mutex_;
    std::condition_variable pending_commands_cv_;
    std::mutex wait_mutex_;
    int batch_size_;
    std::unique_ptr<QuantumWaiter> quantum_waiter_;
    bool running;
    std::unique_ptr<std::thread> process_thread_;
};

#define N_STG_BUFS 256
class SepMemcpyCommandScheduler : public BatchCommandScheduler {
public:
    SepMemcpyCommandScheduler(hipStream_t stream, int batch_size, int fixed_rate_interval_us);
    ~SepMemcpyCommandScheduler(void);
    virtual hipError_t Wait(void) override;
protected:
    void do_memcpy(void *dst, const void *src, size_t size, hipMemcpyKind kind) override;
    void pre_notify(void) override;
    void add_extra_kernels(std::vector<KernelLaunchParam> &extrakerns,
                             const std::vector<KernelLaunchParam *> &params) override;
    void enqueue_device_copy(void *dst, const void *src, size_t size, tag_t tag, bool in);

    inline void *next_in_buf(void) {
      if (stg_in_idx >= N_STG_BUFS)
         stg_in_idx = 0;
      return in_bufs[stg_in_idx++];
    }
    inline void *next_out_buf(void) {
      if (stg_out_idx >= N_STG_BUFS)
         stg_out_idx = 0;
      return out_bufs[stg_out_idx++];
    }
    struct d2h_cpy_op {
       void *dst_;
       void *src_;
       size_t size_;
       uint64_t tag_;
       d2h_cpy_op(void *dst, void *src, size_t size, uint64_t tag) :
          dst_(dst), src_(src), size_(size), tag_(tag) {} ;
    };

    uint64_t cur_batch_id;
    uint64_t last_real_batch;
    uint64_t batches_finished;
    std::deque<d2h_cpy_op> pending_d2h_;
    hipStream_t xfer_stream_;
    void *in_bufs[N_STG_BUFS];
    void *out_bufs[N_STG_BUFS];
    void *encrypt_out_buf;
    void *status_buf;
    unsigned stg_in_idx;
    unsigned stg_out_idx;

	 /* fast way to get tags that won't likely be repeated */
	 tag_t gen_tag(void) {          //period 2^96-1
	 	 static unsigned long x=123456789, y=362436069, z=521288629;
	 	 unsigned long t;
		 x ^= x << 16;
		 x ^= x >> 5;
		 x ^= x << 1;

		 t = x;
		 x = y;
		 y = z;
		 z = t ^ x ^ y;

	    return z;
	 }

};

class BaselineCommandScheduler : public CommandScheduler {
public:
    BaselineCommandScheduler(hipStream_t stream) : CommandScheduler(stream) {}
    hipError_t AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql, uint8_t *extra,
            size_t extra_size, hipEvent_t start, hipEvent_t stop) override;
    hipError_t AddMemcpyAsync(void* dst, const void* src, size_t size, hipMemcpyKind kind) override;
    hipError_t Wait(void) override;
};
#endif
