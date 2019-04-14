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

#include "quantum_waiter.h"

#define DEFAULT_BATCH_SIZE 64
#define DEFAULT_FIXED_RATE_INTERVAL_US -1

#define FIXED_SIZE_B (0x1UL << 20) // 1 MB
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
        void* kernArg;
    };

    struct MemcpyParam {
        void* dst;
        const void* src;
        size_t size;
        hipMemcpyKind kind;
    };

    enum CommandKind {
        KERNEL_LAUNCH,
        MEMCPY
    };

    struct CommandEntry {
        CommandKind kind;
        KernelLaunchParam kernel_launch_param;
        MemcpyParam memcpy_param;
    };

    std::deque<CommandEntry> pending_commands_;
    std::mutex pending_commands_mutex_;
    std::condition_variable pending_commands_cv_;
    std::mutex wait_mutex_;
    int batch_size_;
    std::unique_ptr<QuantumWaiter> quantum_waiter_;
    bool running;
    std::unique_ptr<std::thread> process_thread_;
};

#define N_STG_BUFS 256
#define N_DBELLS 256
class SepMemcpyCommandScheduler : public BatchCommandScheduler {
public:
    SepMemcpyCommandScheduler(hipStream_t stream, int batch_size, int fixed_rate_interval_us);
    ~SepMemcpyCommandScheduler(void);
    virtual hipError_t Wait(void) override;
protected:
    void do_memcpy(void *dst, const void *src, size_t size, hipMemcpyKind kind) override;
    void pre_notify(void) override;
    void enqueue_device_copy(void *dst, const void *src, size_t size, uint64_t *tag, bool now);

    inline void *next_stg_buf(void) {
      if (stg_buf_idx >= N_STG_BUFS)
         stg_buf_idx = 0;
      return stg_bufs[stg_buf_idx++];
    }
    struct d2h_cpy_op {
       void *dst_;
       void *src_;
       size_t size_;
       uint64_t tag_;
       d2h_cpy_op(void *dst, void *src, size_t size, uint64_t tag) :
          dst_(dst), src_(src), size_(size), tag_(tag) {} ;
    };

    std::deque<d2h_cpy_op> pending_d2h_;
    std::mutex pending_d2h_mutex_;
    hipStream_t xfer_stream_;
    void *stg_bufs[N_STG_BUFS];
    int stg_buf_idx;

	 /* fast way to get tags that won't likely be repeated */
	 uint64_t gen_tag(void) {          //period 2^96-1
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
