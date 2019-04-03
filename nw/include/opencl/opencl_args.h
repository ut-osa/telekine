#ifndef __VGPU_OPENCL_ARGS_CL_H__
#define __VGPU_OPENCL_ARGS_CL_H__

#include "opencl.h"
#include "common/socket.h"

void copy_cl_results(struct message *msg);

struct clGetPlatformIDs_args
{
    cl_uint num_entries;
    cl_platform_id *platforms;
    cl_uint *num_platforms;
};

struct clGetPlatformInfo_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clGetDeviceIDs_args
{
    cl_device_id *devices;
    cl_uint *num_devices;
    cl_int ret_val;
};

struct clGetDeviceInfo_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clCreateContext_args
{
    cl_int *errcode_ret;
    cl_context ret_val;
};

struct clCreateContextFromType_args
{
    cl_int *errcode_ret;
    cl_context ret_val;
};

struct clReleaseContext_args
{
    cl_int ret_val;
};

struct clGetContextInfo_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clCreateProgramWithSource_args
{
    cl_int *errcode_ret;
    cl_program ret_val;
};

struct clReleaseProgram_args
{
    cl_int ret_val;
};

struct clBuildProgram_args
{
    cl_int ret_val;
};

struct clGetProgramBuildInfo_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clCreateKernel_args
{
    cl_int *errcode_ret;
    cl_kernel ret_val;
};

struct clReleaseKernel_args
{
    cl_int ret_val;
};

struct clSetKernelArg_args
{
    cl_int ret_val;
};

struct clGetKernelWorkGroupInfo_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clCreateCommandQueue_args
{
    cl_int *errcode_ret;
    cl_command_queue ret_val;
};

struct clReleaseCommandQueue_args
{
    cl_int ret_val;
};

struct clFlush_args
{
    cl_int ret_val;
};

struct clFinish_args
{
    cl_int ret_val;
};

struct clEnqueueTask_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clEnqueueNDRangeKernel_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clCreateBuffer_args
{
    cl_int *errcode_ret;
    cl_mem ret_val;
};

struct clReleaseMemObject_args
{
    cl_int ret_val;
};

struct clEnqueueReadBuffer_args
{
    cl_int ret_val;
    void *ptr;
    cl_event *event;
};

struct clEnqueueWriteBuffer_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clEnqueueMapBuffer_args
{
    void *ret_val;
    cl_event *event;
    cl_int *errcode_ret;
};

struct clEnqueueCopyBuffer_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clEnqueueCopyBufferToImage_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clEnqueueUnmapMemObject_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clReleaseEvent_args
{
    cl_int ret_val;
};

struct clGetEventProfilingInfo_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clWaitForEvents_args
{
    cl_int ret_val;
};

struct clGetEventInfo_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clCreateImage2D_args
{
    cl_int *errcode_ret;
    cl_mem ret_val;
};

struct clCreateImage_args
{
    cl_int *errcode_ret;
    cl_mem ret_val;
};

struct clEnqueueReadImage_args
{
    cl_int ret_val;
    void *ptr;
    cl_event *event;
};

struct clEnqueueWriteImage_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clGetSupportedImageFormats_args
{
    cl_int ret_val;
    cl_image_format *image_formats;
    cl_uint *num_image_formats;
};

#endif
