#ifndef _COMMAND_SCHEDULER_H_

#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <deque>
#include <map>
#include <memory>

#include "hip/hip_runtime.h"

class CommandScheduler {
public:
    CommandScheduler(hipStream_t stream, int batch_size);

    void AddKernelLaunch(
        hipFunction_t f, hsa_kernel_dispatch_packet_t *aql, void** extra);

    void AddMemcpy(
        void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

    void Wait();

    static CommandScheduler* GetForStream(hipStream_t stream);

private:
    void ProcessThread();

    struct KernelLaunchParam {
        hipFunction_t f;
        hsa_kernel_dispatch_packet_t aql;
        size_t kernArgSize;
        void* kernArg;
    };

    std::deque<KernelLaunchParam> pending_kernel_launches_;
    std::atomic<bool> waiting_;
    std::mutex mu1_;
    std::mutex mu2_;
    std::condition_variable cv_;
    int batch_size_;
    hipStream_t stream_;
    std::unique_ptr<std::thread> process_thread_;

    static std::map<hipStream_t, CommandScheduler*> command_scheduler_map_;
    static std::mutex command_scheduler_map_mu_;
};

#endif
