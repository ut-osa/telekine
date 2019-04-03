#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0x100000)

int parse_cmd_params(int argc, char* argv[]);

int platform_id_inuse = 0;  // platform id in use (default: 0)
int device_id_inuse   = 0;  //device id in use (default : 0)
int device_type       = CL_DEVICE_TYPE_GPU;  // device type, 0:GPU, 1:CPU

clock_t start, end;
double cpu_time_used;

int main(int argc, char* argv[])
{
    if (parse_cmd_params(argc, argv) != 0)
        return 1;
    else {
        printf("Use platform #%d, device #%d, type %s\n",
                platform_id_inuse,
                device_id_inuse,
                device_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU");
    }

    cl_device_id     device_id = NULL;
    cl_context       context = NULL;
    cl_command_queue command_queue = NULL;
    cl_mem           memobj = NULL;
    cl_program       program = NULL;
    cl_kernel        kernel = NULL;
    cl_platform_id   platform_id = NULL;
    cl_uint          ret_num_devices;
    cl_uint          ret_num_platforms;
    cl_int           ret;

    char string[MEM_SIZE];

    FILE *fp;
    char fileName[] = "./hello.cl";
    char *source_str;
    size_t source_size;

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

    /* Get Device Info */
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

    /*  Create Memory Buffer */
    memobj = clCreateBuffer(context, CL_MEM_READ_WRITE,MEM_SIZE * sizeof(char), NULL, &ret);

    /*  Create Kernel Program from the source */
    program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
            (const size_t *)&source_size, &ret);

    /*  Build Kernel Program */
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    /*  Create OpenCL Kernel */
    kernel = clCreateKernel(program, "hello", &ret);

    /*  Set OpenCL Kernel Parameters */
    start = clock();
    for (int i = 0; i < 10; i++)
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memobj);
    end = clock();
    cpu_time_used = ((double) (end - start)) * 1e3 / CLOCKS_PER_SEC;
    printf("fun() took %f ms to execute \n", cpu_time_used);

    /*  Execute OpenCL Kernel */
    ret = clEnqueueTask(command_queue, kernel, 0, NULL,NULL);
    ret = clFinish(command_queue);

    /*  Copy results from the memory buffer */
    ret = clEnqueueReadBuffer(command_queue, memobj, CL_TRUE, 0,
            MEM_SIZE * sizeof(char),string, 0, NULL, NULL);

    /*  Display Result */
    puts(string);

    /*  Finalization */
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(memobj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    /*  Finalization */
    free(source_str);

    return 0;
}

int parse_cmd_params(int argc, char* argv[])
{
    for (int i =0; i < argc; ++i) {
        switch (argv[i][1]) {
        case 'd':	 //--d stands for device id used in computaion
            if (++i < argc) {
                sscanf(argv[i], "%u", &device_id_inuse);
            } else {
                printf("Could not read argument after option %s\n", argv[i-1]);
                return 1;
            }
            break;
        case 'p':   // --p stands for platform id used in computation
            if (++i < argc) {
                sscanf(argv[i], "%u", &platform_id_inuse);
            } else {
                printf("Could not read argument after option %s\n", argv[i-1]);
                return 1;
            }
            break;
        case 't':   // --t stands for device type, 0:GPU, 1:CPU
            if (++i < argc) {
                sscanf(argv[i], "%u", &device_type);
                device_type = (device_type == 0) ? CL_DEVICE_TYPE_GPU
                    : CL_DEVICE_TYPE_CPU;
            } else {
                printf("Could not read argument after option %s\n", argv[i-1]);
                return 1;
            }
            break;
        case 'h':
            printf("Usage: %s [-p platform_id] [-d device_id] [-t device_type (0:gpu, 1:cpu)]\n", argv[0]);
            return 1;
        default:
            ;
        }
    }

    return 0;
}
