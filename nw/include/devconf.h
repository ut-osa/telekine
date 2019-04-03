#ifndef __VGPU_DEVCONF_H__
#define __VGPU_DEVCONF_H__

#include "ctype_util.h"
#include "kvm.h"
#include "debug.h"

#define ENABLE_MEASURE

#define ENABLE_KVM_MEDIATION 1
#define ENABLE_RATE_LIMIT    1
#define ENABLE_SWAP          1
#define ENABLE_ASYNC         1

#ifndef __KERNEL__

#define CUDA_SUPPORTED    0
#define TF_C_SUPPORTED    0
#define OPENCL_SUPPORTED  1
#define TF_PY_SUPPORTED   1
#define TF_C_SUPPORTED_FOR_PY 1 // complement for TF_C_SUPPORTED

#else

// must define all for guest kernel compilation
#define CUDA_SUPPORTED    0
#define TF_C_SUPPORTED    0
#define OPENCL_SUPPORTED  1
#define TF_PY_SUPPORTED   0

#endif

#define MAX_VM_NUM        4

#define VGPU_DEV_NAME "scea-vgpu"

#define VGPU_DRIVER_MAJOR 1
#define VGPU_DRIVER_MINOR 0
#define VGPU_DRIVER_PATCHLEVEL 0

#define DEFAULT_PARAM_BLOCK_SIZE  MB(128)

#define VGPU_DEVS_NUM    1
#define VGPU_REG_SIZE    0x10000
#define VGPU_IO_SIZE     0x80
#define VGPU_FIFO_SIZE   MB(128)
#define VGPU_DSTORE_SIZE MB(512)
#define VGPU_VRAM_SIZE   128

#define EXECUTOR_FIFO_SIZE   ((size_t)VGPU_FIFO_SIZE * MAX_VM_NUM)
#define EXECUTOR_DSTORE_SIZE ((size_t)VGPU_DSTORE_SIZE * MAX_VM_NUM)

#define KVM_TASK_QUEUE_SIZE    0x8000
#define EXEC_TASK_QUEUE_SIZE   0x8000
#define KVM_WAIT_QUEUE_SIZE    0x8000
#define GUEST_WAIT_QUEUE_SIZE  0x8000


//
// Transport
//
#define WORKER_MANAGER_PORT 3333
#define WORKER_PORT_BASE    4000
#define WORKER_MANAGER_SOCKET_PATH "/tmp/worker_manager"

//
// Hardware
//
#define DEVICE_MEMORY_TOTAL_SIZE GB(2)

/* fair scheduling */
#define GPU_SCHEDULE_PERIOD         5     /* millisecond */
#define DEVICE_TIME_MEASURE_PERIOD  500   /* millisecond */
#define DEVICE_TIME_DELAY_ADD       1     /* millisecond */
#define DEVICE_TIME_DELAY_MUL_DEC   2

/* swapping */
#define SWAP_SELECTION_DELAY        50    /* millisecond */

/* rate throttling */
#define COMMAND_RATE_LIMIT_BASE     100   /* per second */
#define COMMAND_RATE_PERIOD_INIT    20    /* millisecond */
#define COMMAND_RATE_BUDGET_BASE    (COMMAND_RATE_LIMIT_BASE * COMMAND_RATE_PERIOD_INIT / 1000)  /* per period */
#define COMMAND_RATE_MEASURE_PERIOD 1000  /* millisecond */

/* shares */
static const int PREDEFINED_RATE_SHARES[MAX_VM_NUM+1]  = {0, 1, 2};
static const int PREDEFINED_PRIORITIES[MAX_VM_NUM+1]   = {0, 1, 1, 1, 1};
static const uint64_t DEV_MEM_PARTITIONS[MAX_VM_NUM+1] = {0, GB(2UL), GB(2UL)};

/* auxiliary */
#define vgpu_max(a, b) \
    ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
       _a > _b ? _a : _b; })

#ifdef __KERNEL__
    #include <linux/kernel.h>
#else
    #if !TF_C_SUPPORTED_FOR_PY
        #define max(a, b) vgpu_max(a, b)
    #endif
#endif

#endif
