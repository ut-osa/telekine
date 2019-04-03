#include <string.h>

#include "library.h"

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context context,
                          cl_uint count,
                          const char **strings,
                          const size_t *lengths,
                          cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_program ret;
    int i;
    const char **sources;
    size_t *source_lengths = NULL;
    struct clCreateProgramWithSource_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clCreateProgramWithSource_args));
    args = (struct clCreateProgramWithSource_args *)slab->args;

    desc->base.cmd_id = CL_CREATE_PROGRAM_WITH_SOURCE;
    desc->context = context;
    desc->count = count;
    args->errcode_ret = errcode_ret;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(lengths, size_t, count);
    desc->base.dstore_size += compute_ptr_size(strings, const char *, count);
    for (i = 0; i < count; ++i)
        if (lengths)
            desc->base.dstore_size += compute_ptr_size(strings[i], char, lengths[i]);
        else
            desc->base.dstore_size += compute_ptr_size(strings[i], char, strlen(strings[i]) + 1);
    desc->base.dstore_size += compute_ptr_size(errcode_ret, cl_int, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->sources, strings, const char *, count);
    sources = (const char **)DSTORE_ADDR(seeker.local_offset, desc->sources);
    copy_type(seeker, desc->lengths, lengths, size_t, count);
    for (i = 0; i < count; ++i)
    {
        if (lengths) {
            sources[i] = (char *)create_space(seeker, lengths[i]);
            memcpy(DSTORE_ADDR(seeker.local_offset, sources[i]),
                   (void *)strings[i],
                   sizeof(char) * lengths[i]);
        }
        else {
            sources[i] = (char *)create_space(seeker, strlen(strings[i]) + 1);
            memcpy(DSTORE_ADDR(seeker.local_offset, sources[i]),
                   (void *)strings[i],
                   sizeof(char) * (strlen(strings[i]) + 1));
        }
    }
    reserve_space_for_type(seeker, desc->errcode_ret, errcode_ret, cl_int, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_CREATE_PROGRAM_WITH_SOURCE;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseProgram(cl_program program) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clReleaseProgram_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clReleaseProgram_args));
    args = (struct clReleaseProgram_args *)slab->args;

    desc->base.cmd_id = CL_RELEASE_PROGRAM;
    desc->program = program;
    desc->base.async = 1; // TODO: support async

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_RELEASE_PROGRAM;
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
clBuildProgram(cl_program program,
               cl_uint num_devices,
               const cl_device_id *device_list,
               const char *options,
               void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
               void *user_data) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clBuildProgram_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clBuildProgram_args));
    args = (struct clBuildProgram_args *)slab->args;

    desc->base.cmd_id = CL_BUILD_PROGRAM;
    desc->base.async = 1;
    desc->program = program;
    desc->count = num_devices;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(device_list, cl_device_id, num_devices);
    desc->base.dstore_size += compute_ptr_size(options, char, desc->size);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->devices, device_list, cl_device_id, num_devices);
    copy_string(seeker, desc->options, options, strlen(options));

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_BUILD_PROGRAM;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program program,
                      cl_device_id device,
                      cl_program_build_info param_name,
                      size_t param_value_size,
                      void *param_value,
                      size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clGetProgramBuildInfo_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clGetProgramBuildInfo_args));
    args = (struct clGetProgramBuildInfo_args *)slab->args;

    desc->base.cmd_id = CL_GET_PROGRAM_BUILD_INFO;
    desc->program = program;
    desc->device = device;
    desc->info.program_build = param_name;
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
    msg.cmd_id = CL_GET_PROGRAM_BUILD_INFO;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_kernel CL_API_CALL
clCreateKernel(cl_program program,
               const char *kernel_name,
               cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_kernel ret;
    struct clCreateKernel_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clCreateKernel_args));
    args = (struct clCreateKernel_args *)slab->args;

    desc->base.cmd_id = CL_CREATE_KERNEL;
    desc->program = program;
    args->errcode_ret = errcode_ret;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(kernel_name, char, strlen(kernel_name) + 1);
    desc->base.dstore_size += compute_ptr_size(errcode_ret, cl_int, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_string(seeker, desc->kernel_name, kernel_name, strlen(kernel_name));
    reserve_space_for_type(seeker, desc->errcode_ret, errcode_ret, cl_int, 1);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_CREATE_KERNEL;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clReleaseKernel(cl_kernel kernel) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clReleaseKernel_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clReleaseKernel_args));
    args = (struct clReleaseKernel_args *)slab->args;

    desc->base.cmd_id = CL_RELEASE_KERNEL;
    desc->kernel = kernel;
    desc->base.async = 1; // TODO: support async

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_RELEASE_KERNEL;
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
clSetKernelArg(cl_kernel kernel,
               cl_uint arg_index,
               size_t arg_size,
               const void *arg_value) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clSetKernelArg_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clSetKernelArg_args));
    args = (struct clSetKernelArg_args *)slab->args;

    desc->base.cmd_id = CL_SET_KERNEL_ARG;
    desc->base.async = 1;
    desc->kernel = kernel;
    desc->count = arg_index;
    desc->param_value_size = arg_size;

    /* compute block size */
    desc->base.dstore_size += compute_space_size(arg_value, arg_size);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_space(seeker, desc->param_value, arg_value, arg_size);

    // put into wait queue
    // invocation is already in the slab list

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_SET_KERNEL_ARG;
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
clGetKernelWorkGroupInfo(cl_kernel kernel,
                         cl_device_id device,
                         cl_kernel_work_group_info param_name,
                         size_t param_value_size,
                         void *param_value,
                         size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clGetKernelWorkGroupInfo_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clGetKernelWorkGroupInfo_args));
    args = (struct clGetKernelWorkGroupInfo_args *)slab->args;

    desc->base.cmd_id = CL_GET_KERNEL_WORK_GROUP_INFO;
    desc->kernel = kernel;
    desc->device = device;
    desc->info.kernel_work_group = param_name;
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
    msg.cmd_id = CL_GET_KERNEL_WORK_GROUP_INFO;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}
