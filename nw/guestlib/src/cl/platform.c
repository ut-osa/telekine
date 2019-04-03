#include "library.h"
#include <unistd.h>

/**
 * clGetPlatformIDs
 */
CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformIDs(cl_uint num_entries,
                 cl_platform_id *platforms,
                 cl_uint *num_platforms) CL_API_SUFFIX__VERSION_1_0
{
    /* compute block size */
    size_t total_buffer_size = 0;
    if (platforms)
        total_buffer_size += command_channel_buffer_size(chan, num_entries * sizeof(cl_platform_id));
    if (num_platforms)
        total_buffer_size += command_channel_buffer_size(chan, sizeof(cl_uint));
    DEBUG_PRINT("block size=%lx\n", total_buffer_size);

    /* create command */
    struct command_base *cmd = command_channel_new_command(chan,
            sizeof(struct command_base) + sizeof(struct clGetPlatformIDs_args), total_buffer_size);
    struct clGetPlatformIDs_args *args = (struct clGetPlatformIDs_args *)(cmd + sizeof(struct command_base));

    /* assign attributes */
    cmd->command_id = CL_GET_PLATFORM_IDS;

    /* attach buffers */
    args->num_entries = num_entries;
    if (platforms)
        args->platforms = command_channel_attach_buffer(chan, cmd, platforms, num_entries * sizeof(cl_platform_id));
    if (num_platforms)
        args->num_platforms = command_channel_attach_buffer(chan, cmd, num_platforms, sizeof(cl_uint));

    /* send command */
    command_channel_send_command(chan, cmd);
    command_channel_free_command(chan, cmd);

    /* receive response command */
    struct command_base *ret_cmd = command_channel_receive_command(chan);
    assert(ret_cmd->command_id == CL_GET_PLATFORM_IDS && ret_cmd->command_type == MSG_RESPONSE);

    /* save results */
    struct clGetPlatformIDs_ret_args *ret_args = (struct clGetPlatformIDs_ret_args *)(ret_cmd + sizeof(struct command_base));
    if (platforms)
        memcpy(platforms,
               command_channel_get_buffer(chan, ret_cmd, ret_args->platforms),
               num_entries * sizeof(cl_platform_id));
    if (num_platforms)
        memcpy(num_platforms,
               command_channel_get_buffer(chan, ret_cmd, ret_args->num_platforms),
               sizeof(cl_uint));
    cl_int ret = ret_args->ret_val;
    command_channel_free_command(chan, ret_cmd);

    return ret;
}

/**
 * clGetPlatformInfo
 */
CL_API_ENTRY cl_int CL_API_CALL
clGetPlatformInfo(cl_platform_id platform,
                  cl_platform_info param_name,
                  size_t param_value_size,
                  void *param_value,
                  size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clGetPlatformInfo_args *args;

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clGetPlatformInfo_args));
    args = (struct clGetPlatformInfo_args *)slab->args;

    desc->base.cmd_id = CL_GET_PLATFORM_INFO;
    desc->platform = platform;
    desc->info.platform = param_name;
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
    msg.cmd_id = CL_GET_PLATFORM_INFO;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}
