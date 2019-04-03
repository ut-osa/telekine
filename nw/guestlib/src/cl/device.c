#include "library.h"

/**
 * clGetDeviceIDs
 */
CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceIDs(cl_platform_id platform,
               cl_device_type device_type,
               cl_uint num_entries,
               cl_device_id *devices,
               cl_uint *num_devices) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clGetDeviceIDs_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clGetDeviceIDs_args));
    args = (struct clGetDeviceIDs_args *)slab->args;

    desc->base.cmd_id = CL_GET_DEVICE_IDS;
    desc->platform = platform;
    desc->device_type = device_type;
    desc->num_entries = num_entries;
    args->devices = devices;
    args->num_devices = num_devices;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(devices, cl_device_id, num_entries);
    desc->base.dstore_size += compute_ptr_size(num_devices, cl_uint, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    reserve_space_for_type(seeker, desc->devices, devices, cl_device_id, num_entries);
    reserve_space_for_type(seeker, desc->num_devices, num_devices, cl_uint, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_GET_DEVICE_IDS;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

/**
 * clGetDeviceInfo
 */
CL_API_ENTRY cl_int CL_API_CALL
clGetDeviceInfo(cl_device_id device,
                cl_device_info param_name,
                size_t param_value_size,
                void *param_value,
                size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clGetDeviceInfo_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clGetDeviceInfo_args));
    args = (struct clGetDeviceInfo_args *)slab->args;

    desc->base.cmd_id = CL_GET_DEVICE_INFO;
    desc->device = device;
    desc->info.device = param_name;
    desc->param_value_size = param_value_size;
    args->param_value = param_value;
    args->param_value_size_ret = param_value_size_ret;

    /* compute block size */
    desc->base.dstore_size += compute_space_size(param_value, param_value_size);
    desc->base.dstore_size += compute_ptr_size(param_value_size_ret, size_t, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    reserve_space(seeker, desc->param_value, param_value, param_value_size);
    reserve_space_for_type(seeker, desc->param_value_size_ret, param_value_size_ret, size_t, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_GET_DEVICE_INFO;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}
