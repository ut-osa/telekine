#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE        (0x1000000) // 16 MB
#define MEM_COUNT       (64)
#define STRING_SIZE     (128)
#define MAX_SOURCE_SIZE (0x100000)

int platform_id_inuse = 0;  // platform id in use (default: 0)
int device_id_inuse   = 0;  //device id in use (default : 0)
int device_type       = CL_DEVICE_TYPE_GPU;  // device type, 0:GPU, 1:CPU

clock_t start, end;
double cpu_time_used;

int main(int argc, char* argv[])
{
    cl_device_id     device_id = NULL;
    cl_context       context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem           memobj[MEM_COUNT];
    cl_program       program = NULL;
    cl_kernel        kernel = NULL;
    cl_platform_id   platform_id = NULL;
    cl_uint          ret_num_devices;
    cl_uint          ret_num_platforms;
    cl_int           ret;

    char string[STRING_SIZE];

    FILE *fp;
    char fileName[] = "./hello.cl";
    char *source_str;
    size_t source_size;
    int i;

    /*  Load the source code containing the kernel*/
    fp = fopen(fileName, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *)malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    /*  Get Platform Info */
    ret = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    printf("Platform number = %d\n", ret_num_platforms);

    cl_platform_id* all_platforms = (cl_platform_id *) malloc(ret_num_platforms
            * sizeof(cl_platform_id));
    ret = clGetPlatformIDs(ret_num_platforms, all_platforms, NULL);
    for (int i = 0; i < ret_num_platforms; i++) {
        char pbuff[128];
        ret = clGetPlatformInfo(all_platforms[i], CL_PLATFORM_VENDOR,
                sizeof(pbuff), pbuff, NULL);
        printf("Platform #%d vendor is %s\n", i, pbuff);
    }
    if (platform_id_inuse >= ret_num_platforms) {
        printf("Error: platform_id\n");
        return 1;
    }
    platform_id = all_platforms[platform_id_inuse];
    free(all_platforms);

    /*  Get Device Info */
    ret = clGetDeviceIDs(platform_id, device_type, 0, NULL, &ret_num_devices);
    printf("Devices: %d\n", ret_num_devices);
    cl_device_id *all_devices = (cl_device_id *)malloc(ret_num_devices
            * sizeof(cl_device_id));
    ret = clGetDeviceIDs(platform_id, device_type, ret_num_devices, all_devices, NULL);
    if (device_id_inuse >= ret_num_devices) {
        printf("Error: device_id\n");
        return 1;
    }
    for (int i = 0; i < ret_num_devices; i++) {
        char pbuff[128];
        ret = clGetDeviceInfo(all_devices[i], CL_DEVICE_NAME,
                sizeof(pbuff), pbuff, NULL);
        printf("Device #%d name is %s\n", i, pbuff);
    }
    device_id = all_devices[device_id_inuse];
    free(all_devices);

    /*  Create OpenCL context */
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    /*  Create Command Queue */
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    /*  Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
            (const size_t *)&source_size, &ret);

    /*  Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /*  Create OpenCL Kernel */
    kernel = clCreateKernel(program, "hello", &ret);

    /*  Create Memory Buffer */
    for (i = 0; i < MEM_COUNT; ++i)
    {
        memobj[i] = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   MEM_SIZE * sizeof(char), NULL, &ret);
    }

    /*  Set OpenCL Kernel Parameters */
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj[0]);

    /*  Execute OpenCL Kernel */
    ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
    ret = clFinish(command_queue);

    /*  Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, memobj[0], CL_TRUE, 0,
            STRING_SIZE * sizeof(char),string, 0, NULL, NULL);

    /*  Wait Swap-Main */
    puts("Press any key to continue...");
    getchar();

    /*  Display Result */
    puts(string);

    /*  Finalization */

    /*  Start Timing */
    start = clock();

    ret = clFlush(command_queue);

    /*  Stop Timing */
    end = clock();
    cpu_time_used = ((double) (end - start)) * 1e3 / CLOCKS_PER_SEC;
    printf("swap-in() took %f ms to execute \n", cpu_time_used);

    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    for (i = 0; i < MEM_COUNT; ++i)
    {
        ret = clReleaseMemObject(memobj[i]);
    }
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    /*  Finalization */
    free(source_str);

    return 0;
}
