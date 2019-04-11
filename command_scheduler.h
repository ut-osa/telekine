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

class CommandScheduler {
public:
    CommandScheduler(hipStream_t stream, int batch_size, int fixed_rate_interval_us);

    void AddKernelLaunch(hsa_kernel_dispatch_packet_t *aql, void** extra);

    void AddMemcpy(void* dst, const void* src, size_t size, hipMemcpyKind kind);

    void Wait();

    static std::shared_ptr<CommandScheduler> GetForStream(hipStream_t stream);

private:
    void ProcessThread();

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
    hipStream_t stream_;
    std::unique_ptr<QuantumWaiter> quantum_waiter_;
    std::unique_ptr<std::thread> process_thread_;

    static std::map<hipStream_t, std::shared_ptr<CommandScheduler>> command_scheduler_map_;
    static std::mutex command_scheduler_map_mu_;
};

#endif
