#include "apifwd.h"
#include "dispatcher.h"
#include "common/devconf.h"
#include "common/task_queue.h"

int spawn_apifwd(void)
{
    int err;
    pthread_t poll_opencl_t;
    pthread_t poll_tensorflow_t;
    pthread_t poll_cuda_t;
    pthread_t poll_mvnc_t;

#if OPENCL_SUPPORTED
    load_ocl_lib();
#endif

    init_task_queue_sem(&ex_st.cl_queue.sem);
    init_task_queue(&ex_st.cl_queue, EXEC_TASK_QUEUE_SIZE);

    //
    // Serialize the invocations if swapping is supported.
    //
    if (ValConfig.SwappingSupported)
    {
        init_task_queue_sem(&ex_st.cl_queue.Serial);
    }

    init_task_queue_sem(&ex_st.tf_queue.sem);
    init_task_queue(&ex_st.tf_queue, EXEC_TASK_QUEUE_SIZE);

    init_task_queue_sem(&ex_st.cuda_queue.sem);
    init_task_queue(&ex_st.cuda_queue, EXEC_TASK_QUEUE_SIZE);

    init_task_queue_sem(&ex_st.mvnc_queue.sem);
    init_task_queue(&ex_st.mvnc_queue, EXEC_TASK_QUEUE_SIZE);
    printf("libraries loaded and task queues initialized\n");

#if OPENCL_SUPPORTED
spawn_poll_opencl:
    err = pthread_create(&poll_opencl_t, NULL, &ocl_apifwd_handler, NULL);
    if (err != 0) {
        printf("create opencl handler thread failed\n");
        return -1;
    }
    DEBUG_PRINT("cl_handler pid = %lx\n", (long)poll_opencl_t);
#endif

#if TF_C_SUPPORTED
spawn_poll_tensorflow:
    err = pthread_create(&poll_tensorflow_t, NULL, &tf_apifwd_handler, NULL);
    if (err != 0) {
        printf("create tensorflow handler thread failed\n");
        return -1;
    }
    DEBUG_PRINT("tf_handler pid = %lx\n", (long)poll_tensorflow_t);
#endif

#if CUDA_SUPPORTED
spawn_poll_cuda:
    err = pthread_create(&poll_cuda_t, NULL, &cuda_apifwd_handler, NULL);
    if (err != 0) {
        printf("create cuda handler thread failed\n");
        return -1;
    }
    DEBUG_PRINT("cuda_handler pid = %lx\n", (long)poll_cuda_t);
#endif

#if MVNC_SUPPORTED
spawn_poll_mvnc:
    err = pthread_create(&poll_mvnc_t, NULL, &mvnc_apifwd_handler, NULL);
    if (err != 0) {
        printf("create mvnc handler thread failed\n");
        return -1;
    }
    DEBUG_PRINT("mvnc_handler pid = %lx\n", (long)poll_mvnc_t);
#endif

    // TODO: fix me

/*
#if OPENCL_SUPPORTED
    pthread_join(poll_opencl_t, NULL);
    printf("respawn opencl handler\n");
    goto spawn_poll_opencl;
#endif
#if TF_C_SUPPORTED
    pthread_join(poll_tensorflow_t, NULL);
#endif
#if CUDA_SUPPORTED
    pthread_join(poll_cuda_t, NULL);
#endif
#if MVNC_SUPPORTED
    pthread_join(poll_mvnc_t, NULL);
#endif
*/

    return 0;
}
