#include "library.h"

/* Deprecated OpenCL 1.1 APIs */
CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage2D(cl_context context,
                cl_mem_flags flags,
                const cl_image_format *image_format,
                size_t image_width,
                size_t image_height,
                size_t image_row_pitch,
                void *host_ptr,
                cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_1
{
    cl_mem ret;
    struct clCreateImage2D_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clCreateImage2D_args));
    args = (struct clCreateImage2D_args *)slab->args;

    desc->base.cmd_id = CL_CREATE_IMAGE_2D;
    desc->base.mem_alloc = 1;
    desc->base.FreeCmdId = CL_RELEASE_MEM_OBJECT;
    desc->context = context;
    desc->flags.mem = flags;
    desc->image_width = image_width;
    desc->image_height = image_height;
    desc->image_row_pitch = image_row_pitch;
    args->errcode_ret = errcode_ret;
    desc->size = image_row_pitch * image_height;
    desc->base.AllocatedMemorySize += desc->size;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(image_format, cl_image_format, 1);
    desc->base.dstore_size += compute_space_size(host_ptr, desc->size);
    desc->base.dstore_size += compute_ptr_size(errcode_ret, cl_int, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->image_format, image_format, cl_image_format, 1);
    copy_space(seeker, desc->user_data, host_ptr, desc->size);
    reserve_space_for_type(seeker, desc->errcode_ret, errcode_ret, cl_int, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_CREATE_IMAGE_2D;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage(cl_context context,
              cl_mem_flags flags,
              const cl_image_format *image_format,
              const cl_image_desc *image_desc,
              void *host_ptr,
              cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_2
{
    cl_mem ret;
    struct clCreateImage_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clCreateImage_args));
    args = (struct clCreateImage_args *)slab->args;

    desc->base.cmd_id = CL_CREATE_IMAGE;
    desc->base.mem_alloc = 1;
    desc->base.FreeCmdId = CL_RELEASE_MEM_OBJECT;
    desc->context = context;
    desc->flags.mem = flags;
    args->errcode_ret = errcode_ret;

    int image_size;
    if (image_desc != NULL)
        switch (image_desc->image_type)
        {
        case CL_MEM_OBJECT_IMAGE1D:
        case CL_MEM_OBJECT_IMAGE1D_BUFFER:
            image_size = image_desc->image_row_pitch;
            break;
        case CL_MEM_OBJECT_IMAGE2D:
            image_size = image_desc->image_row_pitch * image_desc->image_height;
            break;
        case CL_MEM_OBJECT_IMAGE3D:
            image_size = image_desc->image_slice_pitch * image_desc->image_depth;
            break;
        case CL_MEM_OBJECT_IMAGE1D_ARRAY:
        case CL_MEM_OBJECT_IMAGE2D_ARRAY:
            image_size = image_desc->image_slice_pitch * image_desc->image_array_size;
            break;
        default:
            image_size = 0;
        }
    else
        image_size = 0;
    desc->size = image_size;
    desc->base.AllocatedMemorySize += desc->size;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(image_format, cl_image_format, 1);
    desc->base.dstore_size += compute_ptr_size(image_desc, cl_image_desc, 1);
    desc->base.dstore_size += compute_space_size(host_ptr, desc->size);
    desc->base.dstore_size += compute_ptr_size(errcode_ret, cl_int, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->image_format, image_format, cl_image_format, 1);
    copy_type(seeker, desc->image_desc, image_desc, cl_image_desc, 1);
    copy_space(seeker, desc->user_data, host_ptr, desc->size);
    reserve_space_for_type(seeker, desc->errcode_ret, errcode_ret, cl_int, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_CREATE_IMAGE;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadImage(cl_command_queue command_queue,
                   cl_mem image,
                   cl_bool blocking_read,
                   const size_t *origin, /* [3] */
                   const size_t *region, /* [3] */
                   size_t row_pitch,
                   size_t slice_pitch,
                   void *ptr,
                   cl_uint num_events_in_wait_list,
                   const cl_event *event_wait_list,
                   cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clEnqueueReadImage_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clEnqueueReadImage_args));
    args = (struct clEnqueueReadImage_args *)slab->args;

    desc->base.cmd_id = CL_ENQUEUE_READ_IMAGE;
    desc->command_queue = command_queue;
    desc->mem = image;
    desc->blocking = blocking_read;
    desc->image_row_pitch = row_pitch;
    desc->image_slice_pitch = slice_pitch;
    desc->num_events_in_wait_list = num_events_in_wait_list;
    args->ptr = ptr;
    args->event = event;

    // TODO: get correct buffer size
    desc->size = 0;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(origin, size_t, 3);
    desc->base.dstore_size += compute_ptr_size(region, size_t, 3);
    desc->base.dstore_size += compute_space_size(ptr, desc->size);
    desc->base.dstore_size += compute_ptr_size(event_wait_list, cl_event, num_events_in_wait_list);
    desc->base.dstore_size += compute_ptr_size(event, cl_event, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->origin, origin, size_t, 3);
    copy_type(seeker, desc->region, region, size_t, 3);
    reserve_space(seeker, desc->user_data, ptr, desc->size);
    copy_type(seeker, desc->event_wait_list, event_wait_list, cl_event, num_events_in_wait_list);
    copy_type(seeker, desc->event, event, cl_event, 1);

    // put into wait queue
    // invocation is already in the slab list
    // TODO: handle event async
    // if (event && desc->base.async) (*event)->wq_id = ret;

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_ENQUEUE_READ_IMAGE;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteImage(cl_command_queue command_queue,
                    cl_mem image,
                    cl_bool blocking_write,
                    const size_t *origin, /* [3] */
                    const size_t *region, /* [3] */
                    size_t input_row_pitch,
                    size_t input_slice_pitch,
                    const void *ptr,
                    cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list,
                    cl_event *event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clEnqueueWriteImage_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clEnqueueWriteImage_args));
    args = (struct clEnqueueWriteImage_args *)slab->args;

    desc->base.cmd_id = CL_ENQUEUE_WRITE_IMAGE;
    desc->command_queue = command_queue;
    desc->mem = image;
    desc->blocking = blocking_write;
    desc->image_row_pitch = input_row_pitch;
    desc->image_slice_pitch = input_slice_pitch;
    desc->num_events_in_wait_list = num_events_in_wait_list;
    args->event = event;

    // TODO: get correct buffer size
    desc->size = 0;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(origin, size_t, 3);
    desc->base.dstore_size += compute_ptr_size(region, size_t, 3);
    desc->base.dstore_size += compute_space_size(ptr, desc->size);
    desc->base.dstore_size += compute_ptr_size(event_wait_list, cl_event, num_events_in_wait_list);
    desc->base.dstore_size += compute_ptr_size(event, cl_event, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->origin, origin, size_t, 3);
    copy_type(seeker, desc->region, region, size_t, 3);
    reserve_space(seeker, desc->user_data, ptr, desc->size);
    copy_type(seeker, desc->event_wait_list, event_wait_list, cl_event, num_events_in_wait_list);
    copy_type(seeker, desc->event, event, cl_event, 1);

    // put into wait queue
    // invocation is already in the slab list
    // TODO: handle event async
    // if (event && desc->base.async) (*event)->wq_id = ret;

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_ENQUEUE_WRITE_IMAGE;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetSupportedImageFormats(cl_context context,
                           cl_mem_flags flags,
                           cl_mem_object_type image_type,
                           cl_uint num_entries,
                           cl_image_format *image_formats,
                           cl_uint *num_image_formats) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clGetSupportedImageFormats_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clGetSupportedImageFormats_args));
    args = (struct clGetSupportedImageFormats_args *)slab->args;

    desc->base.cmd_id = CL_GET_SUPPORTED_IMAGE_FORMATS;
    desc->context = context;
    desc->flags.mem = flags;
    desc->mem_type = image_type;
    desc->num_entries = num_entries;
    args->image_formats = image_formats;
    args->num_image_formats = num_image_formats;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(image_formats, cl_image_format, num_entries);
    desc->base.dstore_size += compute_ptr_size(num_image_formats, cl_uint, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    reserve_space_for_type(seeker, desc->image_format, image_formats, cl_image_format, num_entries);
    reserve_space_for_type(seeker, desc->num_image_formats, num_image_formats, cl_uint, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_GET_SUPPORTED_IMAGE_FORMATS;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}
