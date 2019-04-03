#include "library.h"

#include <stdio.h>
#include <string.h>

CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties *properties,
                cl_uint num_devices,
                const cl_device_id *devices,
                void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                void *user_data,
                cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_context ret_context;
    struct clCreateContext_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clCreateContext_args));
    args = (struct clCreateContext_args *)slab->args;

    desc->base.cmd_id = CL_CREATE_CONTEXT;
    desc->count = num_devices;
    args->errcode_ret = errcode_ret;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(devices, cl_device_id, num_devices);
    const cl_context_properties *properties_ptr = properties;
    desc->num_context_properties = 1;
    while (properties_ptr != NULL && *properties_ptr != 0) {
        properties_ptr++;
        desc->num_context_properties++;
    }
    desc->base.dstore_size += compute_ptr_size(properties,
                                               cl_context_properties,
                                               desc->num_context_properties);
    desc->base.dstore_size += compute_ptr_size(errcode_ret, cl_int, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->context_properties, properties, cl_context_properties,
              desc->num_context_properties);
    copy_type(seeker, desc->devices, devices, cl_device_id, num_devices);
    reserve_space_for_type(seeker, desc->errcode_ret, errcode_ret, cl_int, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_CREATE_CONTEXT;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret_context = args->ret_val;
    free(args);

    return ret_context;
}

CL_API_ENTRY cl_context CL_API_CALL
clCreateContextFromType(const cl_context_properties *properties,
                        cl_device_type device_type,
                        void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                        void *user_data,
                        cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_context ret_context;
    struct clCreateContextFromType_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clCreateContextFromType_args));
    args = (struct clCreateContextFromType_args *)slab->args;

    desc->base.cmd_id = CL_CREATE_CONTEXT_FROM_TYPE;
    desc->device_type = device_type;
    args->errcode_ret = errcode_ret;

    /* compute block size */
    const cl_context_properties *properties_ptr = properties;
    desc->num_context_properties = 1;
    while (properties_ptr != NULL && *properties_ptr != 0) {
        properties_ptr++;
        desc->num_context_properties++;
    }
    desc->base.dstore_size += compute_ptr_size(properties,
                                               cl_context_properties,
                                               desc->num_context_properties);
    desc->base.dstore_size += compute_ptr_size(errcode_ret, cl_int, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->context_properties, properties, cl_context_properties,
              desc->num_context_properties);
    reserve_space_for_type(seeker, desc->errcode_ret, errcode_ret, cl_int, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_CREATE_CONTEXT_FROM_TYPE;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret_context = args->ret_val;
    free(args);

    return ret_context;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseContext(cl_context context) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clReleaseContext_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clReleaseContext_args));
    args = (struct clReleaseContext_args *)slab->args;

    desc->base.cmd_id = CL_RELEASE_CONTEXT;
    desc->context= context;
    desc->base.async = 1; // TODO: support async

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_RELEASE_CONTEXT;
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
clGetContextInfo(cl_context context,
                 cl_context_info param_name,
                 size_t param_value_size,
                 void *param_value,
                 size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clGetContextInfo_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clGetContextInfo_args));
    args = (struct clGetContextInfo_args *)slab->args;

    desc->base.cmd_id = CL_GET_CONTEXT_INFO;
    desc->context= context;
    desc->info.context = param_name;
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
    msg.cmd_id = CL_GET_CONTEXT_INFO;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}
