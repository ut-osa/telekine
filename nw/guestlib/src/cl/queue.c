#include "library.h"

CL_API_ENTRY cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context context,
                     cl_device_id device,
                     cl_command_queue_properties properties,
                     cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_command_queue ret;
    struct clCreateCommandQueue_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clCreateCommandQueue_args));
    args = (struct clCreateCommandQueue_args *)slab->args;

    desc->base.cmd_id = CL_CREATE_COMMAND_QUEUE;
    desc->context = context;
    desc->device = device;
    desc->command_queue_properties = properties;
    args->errcode_ret = errcode_ret;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(errcode_ret, cl_int, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    reserve_space_for_type(seeker, desc->errcode_ret, errcode_ret, cl_int, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_CREATE_COMMAND_QUEUE;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseCommandQueue(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clReleaseCommandQueue_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clReleaseCommandQueue_args));
    args = (struct clReleaseCommandQueue_args *)slab->args;

    desc->base.cmd_id = CL_RELEASE_COMMAND_QUEUE;
    desc->command_queue = command_queue;
    desc->base.async = 1; // TODO: support async

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_RELEASE_COMMAND_QUEUE;
    msg.mode = MODE_ASYNC;
    send_message(&msg);

    /* wait for return */
    if (msg.mode | MODE_ASYNC) {
        ret = 0;
    }
    else {
        wait_for_results(slab);
        ret = args->ret_val;
        free(args);
    }

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clFlush(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clFlush_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clFlush_args));
    args = (struct clFlush_args *)slab->args;

    desc->base.cmd_id = CL_FLUSH;
    desc->command_queue = command_queue;

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_FLUSH;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clFinish(cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clFinish_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clFinish_args));
    args = (struct clFinish_args *)slab->args;

    desc->base.cmd_id = CL_FINISH;
    desc->base.barrier = 1;
    desc->base.rate_limit = 0x2;
    desc->command_queue = command_queue;

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_FINISH;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueTask(cl_command_queue command_queue,
              cl_kernel kernel,
              cl_uint num_events_in_wait_list,
              const cl_event *event_wait_list,
              cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clEnqueueTask_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clEnqueueTask_args));
    args = (struct clEnqueueTask_args *)slab->args;

    desc->base.cmd_id = CL_ENQUEUE_TASK;
    desc->base.async = 1;
    desc->command_queue = command_queue;
    desc->kernel = kernel;
    desc->num_events_in_wait_list = num_events_in_wait_list;
    args->event = event;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(event_wait_list, cl_event, num_events_in_wait_list);
    desc->base.dstore_size += compute_ptr_size(event, cl_event, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->event_wait_list, event_wait_list, cl_event, num_events_in_wait_list);
    copy_type(seeker, desc->event, event, cl_event, 1);

    // put into wait queue
    // invocation is already in the slab list
    // TODO: handle event async
    // if (event && desc->base.async) (*event)->wq_id = ret;

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_ENQUEUE_TASK;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel kernel,
                       cl_uint work_dim,
                       const size_t *global_work_offset,
                       const size_t *global_work_size,
                       const size_t *local_work_size,
                       cl_uint num_events_in_wait_list,
                       const cl_event *event_wait_list,
                       cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clEnqueueNDRangeKernel_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clEnqueueNDRangeKernel_args));
    args = (struct clEnqueueNDRangeKernel_args *)slab->args;

    desc->base.cmd_id = CL_ENQUEUE_ND_RANGE_KERNEL;
    desc->base.async = 1;
    desc->base.rate_limit = 0x1;
    desc->command_queue = command_queue;
    desc->kernel = kernel;
    desc->work_dim = work_dim;
    // TODO: support event list
    desc->num_events_in_wait_list = num_events_in_wait_list;
    args->event = event;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(global_work_offset, size_t, work_dim);
    desc->base.dstore_size += compute_ptr_size(global_work_size, size_t, work_dim);
    desc->base.dstore_size += compute_ptr_size(local_work_size, size_t, work_dim);
    desc->base.dstore_size += compute_ptr_size(event_wait_list, cl_event, num_events_in_wait_list);
    desc->base.dstore_size += compute_ptr_size(event, cl_event, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->global_work_offset, global_work_offset, size_t, work_dim);
    copy_type(seeker, desc->global_work_size, global_work_size, size_t, work_dim);
    copy_type(seeker, desc->local_work_size, local_work_size, size_t, work_dim);
    copy_type(seeker, desc->event_wait_list, event_wait_list, cl_event, num_events_in_wait_list);
    copy_type(seeker, desc->event, event, cl_event, 1);

    // put into wait queue
    // invocation is already in the slab list
    // TODO: handle event async
    // if (event && desc->base.async) (*event)->wq_id = ret;

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_ENQUEUE_ND_RANGE_KERNEL;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}
