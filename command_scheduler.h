#ifndef _COMMAND_SCHEDULER_H_
#define _COMMAND_SCHEDULER_H_

#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <future>
#include <ostream>
#include <sstream>

#include "hip/hip_runtime.h"
#include "hip_function_info.hpp"
#include "lgm_memcpy.hpp"
#include "lgm_types.h"

#include "quantum_waiter.h"

#define DEFAULT_BATCH_SIZE 64
#define DEFAULT_FIXED_RATE_INTERVAL_US -1
#define DEFAULT_N_STAGING_BUFFERS 256

#define FIXED_SIZE_FULL (0x1UL << 20) // 1 MB
#define FIXED_SIZE_B (FIXED_SIZE_FULL - sizeof(tag_t))
#define BUF_TAG(buf) ((uint64_t *)(&(((uint8_t *)buf)[FIXED_SIZE_B])))
#define FIXED_EXTRA_SIZE 256

enum SchedulerType {
   BASELINE,
   BATCHED,
   MANAGED,
   ENCRYPTED,
};

class CommandScheduler {
public:
    CommandScheduler(hipStream_t stream) : destroy_stream(false), stream_(stream),
      functions_(hip_impl::functions())
    { 
       hipGetDevice(&device_index);
       if (!stream_) {
          hipStreamCreate(&stream_);
          destroy_stream = true;
       }
    }
    virtual ~CommandScheduler(void) {
       if (destroy_stream) {
          nw_hipStreamDestroy(stream_);
       }
    }
    virtual hipError_t AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql,
            uint8_t* extra, size_t extra_size, hipEvent_t start, hipEvent_t stop) = 0;
    virtual hipError_t AddMemcpyAsync(void* dst, const void* src, size_t size, hipMemcpyKind kind) = 0;
    virtual hipError_t Wait(void) = 0;

    void toStringStream(std::ostringstream &oss) const {
       nameStringStream(oss);
       oss << " with stream = " << stream_;
       parametersStringStream(oss);
    }
    static std::shared_ptr<CommandScheduler> GetForStream(hipStream_t stream);
    static std::map<hipStream_t, std::shared_ptr<CommandScheduler>> command_scheduler_map_;
    static hipStream_t GetDefStream() {
       std::lock_guard<std::mutex> lk(command_scheduler_map_mu_);
       auto it = command_scheduler_map_.begin();
       assert(it != command_scheduler_map_.end());
       return it->first;
    };
protected:
    int device_index;
    bool destroy_stream;
    hipStream_t stream_;
    static std::mutex command_scheduler_map_mu_;
    const std::shared_ptr<
       std::unordered_map<uintptr_t,
                          std::vector<std::pair<hsa_agent_t, hipFunction_t>>>> functions_;
    virtual void nameStringStream(std::ostringstream &) const = 0;
    virtual void parametersStringStream(std::ostringstream &) const = 0;

};

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& out,
                                         const CommandScheduler& t)
{
   std::ostringstream oss;
   t.toStringStream(oss);
   return out << oss.str();
}


class BatchCommandScheduler : public CommandScheduler {
public:
    BatchCommandScheduler(hipStream_t stream);
    ~BatchCommandScheduler(void);
    hipError_t AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql, uint8_t *extra,
            size_t extra_size, hipEvent_t start, hipEvent_t stop) override;
    virtual hipError_t AddMemcpyAsync(void* dst, const void* src, size_t size, hipMemcpyKind kind) override;
    virtual hipError_t Wait(void) override;
protected:
    virtual void nameStringStream(std::ostringstream &oss) const override {
       oss << "BatchCommandScheduler";
    }
    virtual void parametersStringStream(std::ostringstream &oss) const override {
       oss << ", batch_size = " << batch_size_ << ", interval = "
           << (quantum_waiter_ ? quantum_waiter_->interval_us_ : -1);
    }

    void ProcessThread();
    void SetThreadPriority() {
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
    }

    struct KernelLaunchParam {
        hsa_kernel_dispatch_packet_t aql;
        size_t kernArgSize;
        uint8_t kernArg[FIXED_EXTRA_SIZE];
        hipEvent_t start, stop;
        KernelLaunchParam(const hsa_kernel_dispatch_packet_t *_aql,
                          uint8_t *kern_arg, size_t kern_arg_size,
                          hipEvent_t _start, hipEvent_t _stop) :
           aql(*_aql), kernArgSize(FIXED_EXTRA_SIZE), kernArg{0}, start(_start), stop(_stop)
        {
           assert(kern_arg_size < FIXED_EXTRA_SIZE);
           memcpy(kernArg, kern_arg, kern_arg_size);
        }
        template <typename... Args, typename F = void (*)(Args...)>
        KernelLaunchParam(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                          std::uint32_t sharedMemBytes, hipStream_t stream,
                          Args... args) : aql{0}, kernArgSize(FIXED_EXTRA_SIZE), kernArg{0}, 
            start(nullptr), stop(nullptr)
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
        tag_t tag;
        hipMemcpyKind kind;
        MemcpyParam(void *_dst, const void *_src, size_t _size, hipMemcpyKind _kind,
                    tag_t _tag) :
           dst(_dst), src(_src), size(_size), tag(_tag), kind(_kind) {}
        MemcpyParam() : dst(nullptr), src(nullptr), size(0), tag(0) {}
    };

    enum CommandKind {
        KERNEL_LAUNCH,
        MEMCPY
    };

    struct CommandEntry {
        CommandKind kind;
        std::promise<void> done;
        union {
           KernelLaunchParam kernel_launch_param;
           MemcpyParam memcpy_param;
        };
        ~CommandEntry()
        {
           done.set_value();
        }
        CommandEntry(const hsa_kernel_dispatch_packet_t *aql, uint8_t *kern_arg,
                     size_t kern_arg_size) :
           kind(KERNEL_LAUNCH), done(),
           kernel_launch_param(aql, kern_arg, kern_arg_size, nullptr, nullptr) {};

        CommandEntry(const hsa_kernel_dispatch_packet_t *aql, uint8_t *kern_arg,
                     size_t kern_arg_size, hipEvent_t start, hipEvent_t stop) :
           kind(KERNEL_LAUNCH), done(),
           kernel_launch_param(aql, kern_arg, kern_arg_size, start, stop) {};

        CommandEntry(const hsa_kernel_dispatch_packet_t *aql, uint8_t *kern_arg,
                     size_t kern_arg_size, hipEvent_t start, hipEvent_t stop,
                     std::promise<void> _done) :
           kind(KERNEL_LAUNCH), done(std::move(_done)),
           kernel_launch_param(aql, kern_arg, kern_arg_size, start, stop) {};

        CommandEntry(void *dst, const void *src, size_t size, hipMemcpyKind mkind) :
           kind(MEMCPY), done(),
           memcpy_param(dst, src, size, mkind, 0) {};

        CommandEntry(void *dst, const void *src, size_t size, hipMemcpyKind mkind,
                     tag_t tag) :
           kind(MEMCPY), done(),
           memcpy_param(dst, src, size, mkind, tag) {};

        template <typename... Args, typename F = void (*)(Args...)>
        CommandEntry(F kernel, const dim3& numBlocks, const dim3& dimBlocks,
                          std::uint32_t sharedMemBytes, hipStream_t stream,
                          Args... args) :
           kind(KERNEL_LAUNCH), done(),
           kernel_launch_param(std::move(kernel), numBlocks, dimBlocks, sharedMemBytes,
                               stream, std::move(args)...) {};
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

class SepMemcpyCommandScheduler : public BatchCommandScheduler {
public:
    SepMemcpyCommandScheduler(hipStream_t stream);
    ~SepMemcpyCommandScheduler(void);
    virtual hipError_t Wait(void) override;
    virtual hipError_t AddMemcpyAsync(void* dst, const void* src, size_t size, hipMemcpyKind kind) override;
protected:
    virtual void nameStringStream(std::ostringstream &oss) const override {
       oss << "SepMemcpyCommandScheduler";
    }
    virtual void parametersStringStream(std::ostringstream &oss) const override {
       BatchCommandScheduler::parametersStringStream(oss);
       oss << ", n_staging_buffers = " << n_staging_buffers;
    }
    void add_extra_kernels(std::vector<KernelLaunchParam> &extrakerns,
                             const std::vector<KernelLaunchParam *> &params) override;
    void enqueue_device_copy(void *dst, const void *src, size_t size, tag_t tag, bool in);
    void H2DMemcpyThread();
    void D2HMemcpyThread();
    void do_next_h2d();
    virtual void h2d(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream, tag_t tag);
    void do_next_d2h();
    virtual void d2h(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream);

    inline void *next_in_buf(void) {
      if (stg_in_idx >= n_staging_buffers)
         stg_in_idx = 0;
      return in_bufs[stg_in_idx++];
    }
    inline void *next_out_buf(void) {
      if (stg_out_idx >= n_staging_buffers)
         stg_out_idx = 0;
      return out_bufs[stg_out_idx++];
    }

    uint64_t cur_batch_id;
    uint64_t last_real_batch;
    uint64_t batches_finished;
    size_t n_staging_buffers;
    void **in_bufs;
    void **out_bufs;
    void *encrypt_out_buf;
    void *status_buf;
    void *out_stg_buf;
    void *in_stg_buf;
    void *nop_buffer;
    unsigned stg_in_idx;
    unsigned stg_out_idx;

    hipStream_t h2d_xfer_stream_;
    std::mutex pending_h2d_mutex_;
    std::condition_variable pending_h2d_cv_;
    std::deque<std::unique_ptr<CommandEntry>> pending_h2d_commands_;
    std::unique_ptr<std::thread> h2d_memcpy_thread_;
    std::unique_ptr<QuantumWaiter> h2d_memcpy_waiter_;

    hipStream_t d2h_xfer_stream_;
    std::mutex pending_d2h_mutex_;
    std::condition_variable pending_d2h_cv_;
    std::deque<std::unique_ptr<CommandEntry>> pending_d2h_commands_;
    std::unique_ptr<std::thread> d2h_memcpy_thread_;
    std::unique_ptr<QuantumWaiter> d2h_memcpy_waiter_;

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

class EncryptedSepMemcpyCommandScheduler : public SepMemcpyCommandScheduler {
public:
    EncryptedSepMemcpyCommandScheduler(hipStream_t stream);
    ~EncryptedSepMemcpyCommandScheduler(void);
protected:
    virtual void nameStringStream(std::ostringstream &oss) const override {
       oss << "EncryptedSepMemcpyCommandScheduler";
    }
    void h2d(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream, tag_t tag) override;
    void d2h(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind, hipStream_t stream) override;
private:
    std::unique_ptr<lgm::EncryptionState> h2d_encryption_state;
    std::unique_ptr<lgm::EncryptionState> d2h_encryption_state;
};

class BaselineCommandScheduler : public CommandScheduler {
public:
    BaselineCommandScheduler(hipStream_t stream) : CommandScheduler(stream) {}
    hipError_t AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql, uint8_t *extra,
            size_t extra_size, hipEvent_t start, hipEvent_t stop) override;
    hipError_t AddMemcpyAsync(void* dst, const void* src, size_t size, hipMemcpyKind kind) override;
    hipError_t Wait(void) override;
protected:
    virtual void nameStringStream(std::ostringstream &oss) const override {
       oss << "BaselineCommandScheduler";
    }
    virtual void parametersStringStream(std::ostringstream &oss) const override {
    }
};
#endif
