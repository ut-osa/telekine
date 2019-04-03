#include "library.h"

CL_API_ENTRY cl_int CL_API_CALL
clReleaseEvent(cl_event event) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clReleaseEvent_args *args;

    /* simplified dependency check: ensure all previous invocations have
     * returned and work as a pre-request barrier. Must be put before
     * slab initialization.
     * A full version would be to record the desc_slab in a local stub of
     * cl_event, and spin until that slab is freed (back to the slab
     * list).
    */
    wait_until_clear();

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clReleaseEvent_args));
    args = (struct clReleaseEvent_args *)slab->args;

    desc->base.cmd_id = CL_RELEASE_EVENT;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(&event, cl_event, 1);

   /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->event, &event, cl_event, 1);

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_RELEASE_EVENT;
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
clGetEventProfilingInfo(cl_event event,
                        cl_profiling_info param_name,
                        size_t param_value_size,
                        void *param_value,
                        size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clGetEventProfilingInfo_args *args;

    /* simplified dependency check */
    wait_until_clear();

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clGetEventProfilingInfo_args));
    args = (struct clGetEventProfilingInfo_args *)slab->args;

    desc->base.cmd_id = CL_GET_EVENT_PROFILING_INFO;
    desc->info.profiling = param_name;
    desc->param_value_size = param_value_size;
    args->param_value = param_value;
    args->param_value_size_ret = param_value_size_ret;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(&event, cl_event, 1);
    desc->base.dstore_size += compute_space_size(param_value, param_value_size);
    desc->base.dstore_size += compute_ptr_size(param_value_size_ret, size_t, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->event, &event, cl_event, 1);
    reserve_space(seeker, desc->param_value, param_value, param_value_size);
    reserve_space_for_type(seeker, desc->param_value_size_ret, param_value_size_ret, size_t, 1);

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_GET_EVENT_PROFILING_INFO;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clWaitForEvents(cl_uint num_events,
                const cl_event *event_list) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    cl_event *desc_event_list;
    int i;
    struct clWaitForEvents_args *args;

    /* simplified dependency check */
    wait_until_clear();

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clWaitForEvents_args));
    args = (struct clWaitForEvents_args *)slab->args;

    desc->base.cmd_id = CL_WAIT_FOR_EVENTS;
    desc->count = num_events;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(event_list, cl_event*, num_events);
    desc->base.dstore_size += compute_ptr_size(event_list, cl_event, num_events);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    reserve_space_for_type(seeker, desc->event_list, event_list, cl_event, num_events);
    desc_event_list = (cl_event *)DSTORE_ADDR(seeker.local_offset, desc->event_list);
    for (i = 0; i < num_events; i++)
        memcpy(&desc_event_list[i], &event_list[i], sizeof(cl_event));

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_WAIT_FOR_EVENTS;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}

CL_API_ENTRY cl_int CL_API_CALL
clGetEventInfo(cl_event event,
               cl_event_info param_name,
               size_t param_value_size,
               void *param_value,
               size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0
{
    cl_int ret;
    struct clGetEventInfo_args *args;

    /* simplified dependency check */
    wait_until_clear();

    INIT_OCL_DESC(slab, desc);
    slab->args = malloc(sizeof(struct clGetEventInfo_args));
    args = (struct clGetEventInfo_args *)slab->args;

    desc->base.cmd_id = CL_GET_EVENT_INFO;
    desc->info.event = param_name;
    desc->param_value_size = param_value_size;
    args->param_value = param_value;
    args->param_value_size_ret = param_value_size_ret;

    /* compute block size */
    desc->base.dstore_size += compute_ptr_size(&event, cl_event, 1);
    desc->base.dstore_size += compute_space_size(param_value, param_value_size);
    desc->base.dstore_size += compute_ptr_size(param_value_size_ret, size_t, 1);

    /* marshal parameters */
    INIT_BLOCK_SEEKER(slab, desc, param_block, seeker);
    copy_type(seeker, desc->event, event, cl_event, 1);
    reserve_space(seeker, desc->param_value, param_value, param_value_size);
    reserve_space_for_type(seeker, desc->param_value_size_ret, param_value_size_ret, size_t, 1);

    /* notify dispatcher */
    INIT_MESSAGE(msg, slab, MSG_NEW_INVOCATION);
    msg.cmd_id = CL_GET_EVENT_INFO;
    send_message(&msg);

    /* wait for return */
    wait_for_results(slab);
    ret = args->ret_val;
    free(args);

    return ret;
}
