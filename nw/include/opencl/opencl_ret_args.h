#ifndef __VGPU_OPENCL_RET_ARGS_CL_H__
#define __VGPU_OPENCL_RET_ARGS_CL_H__

#include "opencl.h"
#include "common/socket.h"

struct clGetPlatformIDs_ret_args
{
    cl_platform_id *platforms;
    cl_uint *num_platforms;
    cl_int ret_val;
};

struct clGetPlatformInfo_ret_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clGetDeviceIDs_ret_args
{
    cl_device_id *devices;
    cl_uint *num_devices;
    cl_int ret_val;
};

struct clGetDeviceInfo_ret_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clCreateContext_ret_args
{
    cl_int *errcode_ret;
    cl_context ret_val;
};

struct clCreateContextFromType_ret_args
{
    cl_int *errcode_ret;
    cl_context ret_val;
};

struct clReleaseContext_ret_args
{
    cl_int ret_val;
};

struct clGetContextInfo_ret_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clCreateProgramWithSource_ret_args
{
    cl_int *errcode_ret;
    cl_program ret_val;
};

struct clReleaseProgram_ret_args
{
    cl_int ret_val;
};

struct clBuildProgram_ret_args
{
    cl_int ret_val;
};

struct clGetProgramBuildInfo_ret_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clCreateKernel_ret_args
{
    cl_int *errcode_ret;
    cl_kernel ret_val;
};

struct clReleaseKernel_ret_args
{
    cl_int ret_val;
};

struct clSetKernelArg_ret_args
{
    cl_int ret_val;
};

struct clGetKernelWorkGroupInfo_ret_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clCreateCommandQueue_ret_args
{
    cl_int *errcode_ret;
    cl_command_queue ret_val;
};

struct clReleaseCommandQueue_ret_args
{
    cl_int ret_val;
};

struct clFlush_ret_args
{
    cl_int ret_val;
};

struct clFinish_ret_args
{
    cl_int ret_val;
};

struct clEnqueueTask_ret_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clEnqueueNDRangeKernel_ret_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clCreateBuffer_ret_args
{
    cl_int *errcode_ret;
    cl_mem ret_val;
};

struct clReleaseMemObject_ret_args
{
    cl_int ret_val;
};

struct clEnqueueReadBuffer_ret_args
{
    cl_int ret_val;
    void *ptr;
    cl_event *event;
};

struct clEnqueueWriteBuffer_ret_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clEnqueueMapBuffer_ret_args
{
    void *ret_val;
    cl_event *event;
    cl_int *errcode_ret;
};

struct clEnqueueCopyBuffer_ret_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clEnqueueCopyBufferToImage_ret_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clEnqueueUnmapMemObject_ret_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clReleaseEvent_ret_args
{
    cl_int ret_val;
};

struct clGetEventProfilingInfo_ret_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clWaitForEvents_ret_args
{
    cl_int ret_val;
};

struct clGetEventInfo_ret_args
{
    void *param_value;
    size_t *param_value_size_ret;
    cl_int ret_val;
};

struct clCreateImage2D_ret_args
{
    cl_int *errcode_ret;
    cl_mem ret_val;
};

struct clCreateImage_ret_args
{
    cl_int *errcode_ret;
    cl_mem ret_val;
};

struct clEnqueueReadImage_ret_args
{
    cl_int ret_val;
    void *ptr;
    cl_event *event;
};

struct clEnqueueWriteImage_ret_args
{
    cl_int ret_val;
    cl_event *event;
};

struct clGetSupportedImageFormats_ret_args
{
    cl_int ret_val;
    cl_image_format *image_formats;
    cl_uint *num_image_formats;
};

#endif
