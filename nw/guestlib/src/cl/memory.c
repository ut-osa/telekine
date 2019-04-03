#include "library.h"

CL_API_ENTRY cl_mem CL_API_CALL
clCreateBuffer(cl_context context,
               cl_mem_flags flags,
               size_t size,
               void *host_ptr,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_mem ret;
    struct clCreateBuffer_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clCreateBuffer_args));
    args = (struct clCreateBuffer_args *)slab->args;

    desc->base.cmd_id = CL_CREATE_BUFFER;
    desc->base.mem_alloc = 1;
    desc->base.FreeCmdId = CL_RELEASE_MEM_OBJECT;
    desc->base.AllocatedMemorySize += compute_space_size(&context, size);

    DEBUG_PRINT("[lib] allocate memory size=%lx\n", desc->base.AllocatedMemorySize);

    desc->context = context;
    desc->flags.mem = flags;
    desc->size = size;
    desc->user_data = host_ptr;
    args->errcode_ret = errcode_ret;

    /* compute block size */
    desc->base.dstore_size += compute_space_size(host_ptr, size);
    desc->base.dstore_size += compute_ptr_size(errcode_ret, cl_int, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_space(seeker, desc->user_data, host_ptr, size);
    reserve_space_for_type(seeker, desc->errcode_ret, errcode_ret, cl_int, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_CREATE_BUFFER;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseMemObject(cl_mem memobj) CL_API_SUFFIX__VERSION_1_0
{

    cl_int ret;
    struct clReleaseMemObject_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clReleaseMemObject_args));
    args = (struct clReleaseMemObject_args *)slab->args;

    desc->base.cmd_id = CL_RELEASE_MEM_OBJECT;
    desc->base.async = 1; // TODO: support async
    desc->base.mem_free = 1;
    desc->mem = memobj;

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_RELEASE_MEM_OBJECT;
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
clEnqueueReadBuffer(cl_command_queue command_queue,
                    cl_mem buffer,
                    cl_bool blocking_read,
                    size_t offset,
                    size_t size,
                    void *ptr,
                    cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list,
                    cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clEnqueueReadBuffer_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clEnqueueReadBuffer_args));
    args = (struct clEnqueueReadBuffer_args *)slab->args;

    desc->base.cmd_id = CL_ENQUEUE_READ_BUFFER;
    desc->command_queue = command_queue;
    desc->mem = buffer;
    desc->blocking = blocking_read;
    desc->offset = offset;
    desc->size = size;
    desc->num_events_in_wait_list = num_events_in_wait_list;
    args->ptr = ptr;
    args->event = event;

    /* compute block size */
    desc->base.dstore_size += compute_space_size(ptr, size);
    desc->base.dstore_size += compute_ptr_size(event_wait_list, cl_event, num_events_in_wait_list);
    desc->base.dstore_size += compute_ptr_size(event, cl_event, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    reserve_space(seeker, desc->user_data, ptr, size);
    copy_type(seeker, desc->event_wait_list, event_wait_list, cl_event, num_events_in_wait_list);
    copy_type(seeker, desc->event, event, cl_event, 1);

    // put into wait queue
    // invocation is already in the slab list
    // TODO: handle event async
    // if (event && desc->base.async) (*event)->wq_id = ret;

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_ENQUEUE_READ_BUFFER;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue command_queue,
                     cl_mem buffer,
                     cl_bool blocking_write,
                     size_t offset,
                     size_t size,
                     const void *ptr,
                     cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list,
                     cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clEnqueueWriteBuffer_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clEnqueueWriteBuffer_args));
    args = (struct clEnqueueWriteBuffer_args *)slab->args;

    desc->base.cmd_id = CL_ENQUEUE_WRITE_BUFFER;
    desc->command_queue = command_queue;
    desc->mem = buffer;
    desc->blocking = blocking_write;
    desc->offset = offset;
    desc->size = size;
    desc->num_events_in_wait_list = num_events_in_wait_list;
    args->event = event;

    /* compute block size */
    desc->base.dstore_size += compute_space_size(ptr, size);
    desc->base.dstore_size += compute_ptr_size(event_wait_list, cl_event, num_events_in_wait_list);
    desc->base.dstore_size += compute_ptr_size(event, cl_event, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_space(seeker, desc->user_data, ptr, size);
    copy_type(seeker, desc->event_wait_list, event_wait_list, cl_event, num_events_in_wait_list);
    copy_type(seeker, desc->event, event, cl_event, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_ENQUEUE_WRITE_BUFFER;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

/* NOT TESTED!
 * Map memory object to shared memory. Need to return more information
 * for future unmap. */
CL_API_ENTRY void * CL_API_CALL
clEnqueueMapBuffer(cl_command_queue command_queue,
                   cl_mem buffer,
                   cl_bool blocking_map,
                   cl_map_flags map_flags,
                   size_t offset,
                   size_t size,
                   cl_uint num_events_in_wait_list,
                   const cl_event *event_wait_list,
                   cl_event *event,
                   cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    void *ret;
    struct clEnqueueMapBuffer_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clEnqueueMapBuffer_args));
    args = (struct clEnqueueMapBuffer_args *)slab->args;

    desc->base.cmd_id = CL_ENQUEUE_MAP_BUFFER;
    desc->command_queue = command_queue;
    desc->mem = buffer;
    desc->blocking = blocking_map;
    desc->flags.map = map_flags;
    desc->offset = offset;
    desc->size = size;
    desc->num_events_in_wait_list = num_events_in_wait_list;
    args->event = event;
    args->errcode_ret = errcode_ret;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(event_wait_list, cl_event, num_events_in_wait_list);
    desc->base.dstore_size += compute_ptr_size(event, cl_event, 1);
    desc->base.dstore_size += compute_ptr_size(errcode_ret, cl_int, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    reserve_space(seeker, desc->user_data, &buffer, size);
    copy_type(seeker, desc->event_wait_list, event_wait_list, cl_event, num_events_in_wait_list);
    copy_type(seeker, desc->event, event, cl_event, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_ENQUEUE_MAP_BUFFER;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue command_queue,
                    cl_mem src_buffer,
                    cl_mem dst_buffer,
                    size_t src_offset,
                    size_t dst_offset,
                    size_t size,
                    cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list,
                    cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clEnqueueCopyBuffer_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clEnqueueCopyBuffer_args));
    args = (struct clEnqueueCopyBuffer_args *)slab->args;

    desc->base.cmd_id = CL_ENQUEUE_COPY_BUFFER;
    desc->command_queue = command_queue;
    desc->mem = src_buffer;
    desc->dst_mem = dst_buffer;
    desc->offset = src_offset;
    desc->dst_offset = dst_offset;
    desc->size = size;
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
    msg.cmd_id = CL_ENQUEUE_COPY_BUFFER;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBufferToImage(cl_command_queue command_queue,
                           cl_mem src_buffer,
                           cl_mem dst_image,
                           size_t src_offset,
                           const size_t *dst_origin, /* [3] */
                           const size_t *region, /* [3] */
                           cl_uint num_events_in_wait_list,
                           const cl_event *event_wait_list,
                           cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clEnqueueCopyBufferToImage_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clEnqueueCopyBufferToImage_args));
    args = (struct clEnqueueCopyBufferToImage_args *)slab->args;

    desc->base.cmd_id = CL_ENQUEUE_COPY_BUFFER_TO_IMAGE;
    desc->command_queue = command_queue;
    desc->mem = src_buffer;
    desc->offset = src_offset;
    desc->dst_mem = dst_image;
    desc->num_events_in_wait_list = num_events_in_wait_list;
    args->event = event;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(dst_origin, size_t, 3);
    desc->base.dstore_size += compute_ptr_size(region, size_t, 3);
    desc->base.dstore_size += compute_ptr_size(event_wait_list, cl_event, num_events_in_wait_list);
    desc->base.dstore_size += compute_ptr_size(event, cl_event, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->origin, dst_origin, size_t, 3);
    copy_type(seeker, desc->region, region, size_t, 3);
    copy_type(seeker, desc->event_wait_list, event_wait_list, cl_event, num_events_in_wait_list);
    copy_type(seeker, desc->event, event, cl_event, 1);

    // put into wait queue
    // invocation is already in the slab list
    // TODO: handle event async
    // if (event && desc->base.async) (*event)->wq_id = ret;

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_ENQUEUE_COPY_BUFFER_TO_IMAGE;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueUnmapMemObject(cl_command_queue command_queue,
                        cl_mem memobj,
                        void *mapped_ptr,
                        cl_uint num_events_in_wait_list,
                        const cl_event *event_wait_list,
                        cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clEnqueueUnmapMemObject_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clEnqueueUnmapMemObject_args));
    args = (struct clEnqueueUnmapMemObject_args *)slab->args;

    desc->base.cmd_id = CL_ENQUEUE_UNMAP_MEM_OBJECT;
    desc->command_queue = command_queue;
    desc->mem = memobj;
    desc->num_events_in_wait_list = num_events_in_wait_list;
    args->event = event;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(event_wait_list, cl_event, num_events_in_wait_list);
    desc->base.dstore_size += compute_ptr_size(event, cl_event, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->event_wait_list, event_wait_list, cl_event, num_events_in_wait_list);
    copy_type(seeker, desc->event, event, cl_event, 1);
    // TODO: mapped_ptr is a negative offset to the current seeker's base
    desc->user_data = (void *)((uintptr_t)mapped_ptr - (uintptr_t)param_block.base - (uintptr_t)seeker.global_offset);

    // put into wait queue
    // invocation is already in the slab list
    // TODO: handle event async
    // if (event && desc->base.async) (*event)->wq_id = ret;

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_ENQUEUE_UNMAP_MEM_OBJECT;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}
