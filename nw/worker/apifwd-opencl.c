#include <CL/cl.h>
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

#include "apifwd.h"
#include "common/cmd_channel.h"
#include "common/cmd_channel_impl.h"

extern struct command_channel *chan;

typedef char OCL_FuncName[32];
typedef void (*OCL_Func)(void);

static OCL_FuncName ocl_func_names[CL_FUNCTION_NUM] = {
    { "clGetPlatformIDs"           },
    { "clGetPlatformInfo"          },
    { "clGetDeviceIDs"             },
    { "clCreateContext"            },
    { "clReleaseContext"           },
    { "clCreateCommandQueue"       },
    { "clReleaseCommandQueue"      },
    { "clCreateBuffer"             },
    { "clReleaseMemObject"         },
    { "clFlush"                    },
    { "clFinish"                   },
    { "clEnqueueReadBuffer"        },
    { "clEnqueueTask"              },
    { "clCreateProgramWithSource"  },
    { "clReleaseProgram"           },
    { "clBuildProgram"             },
    { "clCreateKernel"             },
    { "clReleaseKernel"            },
    { "clSetKernelArg"             },
    { "clGetDeviceInfo"            },
    { "clCreateContextFromType"    },
    { "clEnqueueWriteBuffer"       },
    { "clReleaseEvent"             },
    { "clGetEventProfilingInfo"    },
    { "clEnqueueNDRangeKernel"     },
    { "clGetProgramBuildInfo"      },
    { "clEnqueueMapBuffer"         },
    { "clGetContextInfo"           },
    { "clEnqueueCopyBuffer"        },
    { "clEnqueueCopyBufferToImage" },
    { "clCreateImage2D"            },
    { "clCreateImage"              },
    { "clEnqueueReadImage"         },
    { "clEnqueueWriteImage"        },
    { "clEnqueueUnmapMemObject"    },
    { "clGetSupportedImageFormats" },
    { "clWaitForEvents"            },
    { "clGetEventInfo"             },
    { "clGetKernelWorkGroupInfo"   }
};
static OCL_Func ocl_funcs[CL_FUNCTION_NUM];

#define load_ocl_func(name) *(void **)(&name) = ocl_funcs[param->base.cmd_id - 1];


//
// Saved context and command queue for swapping
//
cl_context GlobalContext = 0;
cl_command_queue GlobalCommandQueue = 0;


/**
 * load_ocl_lib: Load OpenCL library
 *
 * Avoid linking OpenCL library to qemu.
 */
void load_ocl_lib(void)
{
    void *libocl;
    libocl = dlopen(OCL_LIB_FILE, RTLD_NODELETE | RTLD_NOW | RTLD_GLOBAL);
    if (libocl == NULL) {
        fprintf(stderr, "OCL library not exist\n");
        return;
    }

    int i;
    for (i = 0; i < CL_FUNCTION_NUM; ++i) {
        ocl_funcs[i] = dlsym(libocl, (char *)ocl_func_names[i]);
        if (ocl_funcs[i] == NULL) {
            printf("Function[%d] %s failed to load\n",
                    i, (char *)ocl_func_names[i]);
        }
    }
    dlclose(libocl);
}

/**
 * ocl_apifwd_handler: Execute OpenCL commands with decoded data
 *
 * Data is compacted in an ocl_param structure. It is defined in both
 * this file (hypervisor-level) and vgpu_ioctl.h (driver-level). All
 * pointers in ocl_param are offset in dstore ram. The ocl_param
 * structure is stored in FIFO ram at state->fifo_ptr + reg[REG_DATA_PTR].
 *
 * The thread is spawned when vgpu is realized.
 */
void *ocl_apifwd_handler(void *opaque)
{
    PTASK_QUEUE q = &ex_st.cl_queue;
    PTASK_QUEUE_SEM sem = &q->sem;
    PTASK_QUEUE_SEM serial = &q->Serial;

    OCL_Param *param;
    PMINI_TASK_NODE task;
    struct command_base *cmd_in;
    struct command_base *cmd_out;
    void *args_base;
    size_t total_buffer_size;
    cl_int *ret;
    int i;
    GHashTable *objectTable;
    PDEVICE_OBJECT_LIST newObject;
    PDEVICE_OBJECT_LIST oldObject;

    task = (PMINI_TASK_NODE)malloc(sizeof(MINI_TASK_NODE));
    objectTable = InitObjectTable();

    while (1) {
        sem_wait(sem->start);
        memcpy(task, &(q->tasks[q->head]), sizeof(MINI_TASK_NODE));

        DEBUG_PRINT("process new cl command\n");

        param = (OCL_Param *)((uintptr_t)ex_st.fifo.addr +
                              VGPU_FIFO_SIZE * (task->vm_id - 1) +
                              task->data_ptr);

        cmd_in = q->tasks[q->head].cmd_in;
        args_base = (void *)(cmd_in + sizeof(struct command_base));

        DEBUG_PRINT("retrieve [vm#%d] task %ld cmd 0x%x, head=%ld\n",
                task->vm_id, task->node_id, param->base.cmd_id, q->head);
        q->head = (q->head + 1) & (q->size - 1);
        sem_post(sem->done);

        DEBUG_PRINT("start to execute [vm#%d] task %ld cmd 0x%x\n"
                "param addr = 0x%lx, dstore size = 0x%lx\n",
                task->vm_id, task->node_id, param->base.cmd_id,
                (unsigned long)param, param->base.dstore_size);

        ret = &param->ret_val;
        switch(cmd_in->command_id) {
            case CL_GET_PLATFORM_IDS:
            {
                struct clGetPlatformIDs_args *args = (struct clGetPlatformIDs_args *)args_base;

                cl_platform_id* platforms = (cl_platform_id *)command_channel_get_buffer(chan, cmd_in, args->platforms);
                cl_uint* num_platforms = (cl_uint *)command_channel_get_buffer(chan, cmd_in, args->num_platforms);
                DEBUG_PRINT("debug ent=%u, plat=%p, num=%p\n",
                        args->num_entries, args->platforms, args->num_platforms);
                DEBUG_PRINT("debug ent=%u, pid=%p, pnum=%p\n",
                        args->num_entries, platforms, num_platforms);

                cl_int (*_clGetPlatformIDs)(cl_uint, cl_platform_id *, cl_uint *);
                load_ocl_func(_clGetPlatformIDs);
                *ret = _clGetPlatformIDs(param->num_entries, platforms, num_platforms);

                if (platforms) { DEBUG_PRINT("platform[0] = %p\n", *platforms); }
                if (num_platforms) { DEBUG_PRINT("platform_num = %u\n", *num_platforms); }

                /* prepare response */
                total_buffer_size = 0;
                if (platforms)
                    total_buffer_size += command_channel_buffer_size(chan, args->num_entries * sizeof(cl_platform_id));
                if (num_platforms)
                    total_buffer_size += command_channel_buffer_size(chan, sizeof(cl_uint));

                cmd_out = command_channel_new_command(chan,
                        sizeof(struct command_base) + sizeof(struct clGetPlatformIDs_ret_args), total_buffer_size);
                struct clGetPlatformIDs_ret_args *ret_args = (struct clGetPlatformIDs_ret_args *)(cmd_out + sizeof(struct command_base));

                ret_args->ret_val = *ret;
                if (platforms)
                    ret_args->platforms = command_channel_attach_buffer(chan, cmd_out, platforms, args->num_entries * sizeof(cl_platform_id));
                if (num_platforms)
                    ret_args->num_platforms = command_channel_attach_buffer(chan, cmd_out, num_platforms, sizeof(cl_uint));

                break;
            }

            case CL_GET_PLATFORM_INFO:
            {
                void *pval = get_ptr_from_dstore(task->vm_id,
                        param->param_value, void *);
                size_t *pval_size = get_ptr_from_dstore(task->vm_id,
                        param->param_value_size_ret, size_t *);
                cl_int (*_clGetPlatformInfo)(cl_platform_id, cl_platform_info,
                        size_t, void *, size_t *);
                load_ocl_func(_clGetPlatformInfo);
                *ret = _clGetPlatformInfo(param->platform, param->info.platform,
                        param->param_value_size, pval, pval_size);
                break;
            }

            case CL_GET_DEVICE_IDS:
            {
                cl_device_id *dlist = get_ptr_from_dstore(task->vm_id,
                        param->devices, cl_device_id *);
                cl_uint *dnum = get_ptr_from_dstore(task->vm_id,
                        param->num_devices, cl_uint *);
                DEBUG_PRINT("debug ent=%u, device=%p, num=%p\n",
                        param->num_entries, param->devices, param->num_devices);

                cl_int (*_clGetDeviceIDs)(cl_platform_id, cl_device_type,
                        cl_uint, cl_device_id *, cl_uint *);
                load_ocl_func(_clGetDeviceIDs);
                *ret = _clGetDeviceIDs(param->platform, param->device_type,
                        param->num_entries, dlist, dnum);

                if (dnum) { DEBUG_PRINT("device_num = %u\n", *dnum); }
                break;
            }

            case CL_CREATE_CONTEXT:
            {
                const cl_context_properties *prop = get_ptr_from_dstore(task->vm_id,
                        param->context_properties, const cl_context_properties *);
                cl_device_id *dlist = get_ptr_from_dstore(task->vm_id,
                        param->devices, cl_device_id *);
                // TODO: callback
                cl_int *errcode = get_ptr_from_dstore(task->vm_id,
                        param->errcode_ret, cl_int *);
                DEBUG_PRINT("debug count=%u, device=%p, prop=%p\n",
                        param->count, *dlist, prop);

                cl_context (*_clCreateContext)(const cl_context_properties *,
                        cl_uint, const cl_device_id *,
                        void(CL_CALLBACK *)(const char *, const void *, size_t, void *), void *,
                        cl_int *);
                load_ocl_func(_clCreateContext);
                param->context = _clCreateContext(prop, param->count, dlist,
                        NULL, NULL, errcode);
                param->context = (cl_context)ex_st.encrypt(task->vm_id, (uintptr_t)param->context);
                break;
            }

            case CL_RELEASE_CONTEXT:
            {
                int (*_clReleaseContext)(cl_context);
                load_ocl_func(_clReleaseContext);
                *ret = _clReleaseContext((cl_context)ex_st.decrypt(task->vm_id, (uintptr_t)param->context));
                ex_st.map_remove(task->vm_id, (uintptr_t)param->context);
                break;
            }

            case CL_CREATE_COMMAND_QUEUE:
            {
                cl_int *errcode = get_ptr_from_dstore(task->vm_id,
                        param->errcode_ret, cl_int *);
                cl_command_queue (*_clCreateCommandQueue)(cl_context,
                        cl_device_id, cl_command_queue_properties, cl_int *);

                load_ocl_func(_clCreateCommandQueue);
                param->command_queue = _clCreateCommandQueue(
                        (cl_context)ex_st.decrypt(task->vm_id, (uintptr_t)param->context),
                        param->device, param->command_queue_properties, errcode);

                if (ValConfig.SwappingSupported)
                {
                    param->base.Context = (HANDLE)param->context;
                    param->base.CommandQueue = (HANDLE)param->command_queue;

                    //
                    // Save context and command queue
                    //
                    if (GlobalCommandQueue == 0)
                    {
                        GlobalCommandQueue = param->command_queue;
                        GlobalContext = param->context;
                    }
                }

                break;
            }

            case CL_RELEASE_COMMAND_QUEUE:
            {
                int (*_clReleaseCommandQueue)(cl_command_queue);
                load_ocl_func(_clReleaseCommandQueue);
                *ret = _clReleaseCommandQueue(param->command_queue);
                break;
            }

            case CL_CREATE_BUFFER:
            {
                void *udata;
                cl_int *errcode;
                cl_mem (*_clCreateBuffer)(cl_context, cl_mem_flags, size_t,
                        void *, cl_int *);
                int (*_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool,
                        size_t, size_t, const void *, cl_uint, const cl_event *,
                        cl_event *);
                size_t size;
                cl_mem_flags memoryFlags;
                uintptr_t encryptedContext;

                load_ocl_func(_clCreateBuffer);

                if (!task->IsSwap || !ValConfig.SwappingSupported)
                {
                    //
                    // To allocate a new object, access data from D-Buff.
                    //
                    DEBUG_PRINT(KRED "[vm#%d] allocate new object: size=%lx, param_size=%lx\n" KRESET,
                                task->vm_id, param->base.AllocatedMemorySize, param->size);

                    encryptedContext = (uintptr_t)param->context;
                    udata = get_ptr_from_dstore(task->vm_id, param->user_data,
                            void *);
                    size = param->size;
                    memoryFlags = param->flags.mem;
                    errcode = get_ptr_from_dstore(task->vm_id,
                            param->errcode_ret, cl_int *);

                    //
                    // Update object-related attributes.
                    //
                    param->base.MemoryFlag = (HANDLE)param->flags.mem;
                    param->base.AllocatedMemorySize = size;
                }
                else
                {
                    //
                    // To swap-in an object, access data from invoker's
                    // memory.
                    //
                    // TODO: for simplicity of swapping, we do not support HOST_PTR
                    // when swapping is turned on.
                    //
                    DEBUG_PRINT(KRED "[vm#%d] swap-in old object: size=%lx, origin_handle=%lx\n" KRESET,
                                task->vm_id,
                                param->base.AllocatedMemorySize,
                                param->base.OriginalObjectHandle);

                    encryptedContext = (uintptr_t)GlobalContext;
                    udata = NULL;
                    size = param->base.AllocatedMemorySize;
                    memoryFlags = param->base.MemoryFlag;
                    errcode = (cl_int *)malloc(sizeof(cl_int));
                }

                param->mem = _clCreateBuffer(
                        (cl_context)ex_st.decrypt(task->vm_id, encryptedContext),
                        memoryFlags, size, udata, errcode);

                //
                // Return new handle of the allocated object.
                //
                param->base.ObjectHandle = (HANDLE)param->mem;
                param->base.Context = encryptedContext;

                if (ValConfig.SwappingSupported && task->IsSwap)
                {
                    //
                    // Swap in data.
                    //
                    *(void **)(&_clEnqueueWriteBuffer) = ocl_funcs[CL_ENQUEUE_WRITE_BUFFER - 1];

                    _clEnqueueWriteBuffer(GlobalCommandQueue,
                                          param->mem,
                                          CL_TRUE,
                                          0, 0,
                                          param->base.SwappedOutAddress,
                                          0, NULL, NULL);

                    free((PVOID)param->base.SwappedOutAddress);
                    free(errcode);
                    param->base.SwappedOutAddress = NULL;

                    //
                    // Update the object int the Object Table.
                    //
                    oldObject = GetObjectByOriginalHandle(objectTable,
                                                          param->base.OriginalObjectHandle);
                    oldObject->ObjectHandle = param->base.ObjectHandle;
                }
                else
                {
                    //
                    // Add the new object to the Object Table.
                    //
                    InsertNewObject(objectTable, param->base.ObjectHandle,
                                    param->base.AllocatedMemorySize);
                }

                break;
            }

            case CL_RELEASE_MEM_OBJECT:
            {
                int (*_clReleaseMemObject)(cl_mem);
                int (*_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool,
                        size_t, size_t, void *, cl_uint, const cl_event *,
                        cl_event *);

                cl_mem memoryObject;

                load_ocl_func(_clReleaseMemObject);

                if (ValConfig.SwappingSupported && task->IsSwap)
                {
                    DEBUG_PRINT("search victim object %lx for vm#%d\n",
                                param->base.OriginalObjectHandle, task->vm_id);

                    oldObject = GetObjectByOriginalHandle(objectTable, param->base.OriginalObjectHandle);
                }
                else
                {
                    DEBUG_PRINT("search vm#%d object %lx\n",
                                task->vm_id, (HANDLE)param->mem);

                    oldObject = GetObjectByOriginalHandle(objectTable, (HANDLE)param->mem);
                }

                if (oldObject == NULL)
                {
                    fprintf(stderr, "object not found!\n");
                    break;
                }

                memoryObject = (cl_mem)oldObject->ObjectHandle;

                if (ValConfig.SwappingSupported && task->IsSwap)
                {
                    //
                    // Swap out the device memory.
                    //
                    *(void **)(&_clEnqueueReadBuffer) = ocl_funcs[CL_ENQUEUE_READ_BUFFER - 1];

                    param->base.SwappedOutAddress = malloc(param->base.AllocatedMemorySize);

                    DEBUG_PRINT(KGRN "swap-out [vm#%d] addr=%lx, size=%lx, cmdqueue=%lx\n" KRESET,
                                task->vm_id,
                                (uintptr_t)param->base.SwappedOutAddress,
                                param->base.AllocatedMemorySize,
                                (HANDLE)GlobalCommandQueue);

                    _clEnqueueReadBuffer(GlobalCommandQueue,
                                         memoryObject,
                                         CL_TRUE, 0, 0,
                                         param->base.SwappedOutAddress,
                                         0, NULL, NULL);

                    oldObject->ObjectHandle = 0;
                    oldObject->SwappedOutAddress = param->base.SwappedOutAddress;

                    DEBUG_PRINT(KGRN "swap-out [vm#%d] old object to 0x%lx: size=%lx, origin_handle=%lx\n" KRESET,
                                task->vm_id,
                                (uintptr_t)oldObject->SwappedOutAddress,
                                oldObject->ObjectSize,
                                oldObject->OriginalObjectHandle);
                }
                else
                {
                    //
                    // Delete the released object from Object Table.
                    //
                    DEBUG_PRINT(KGRN "[vm#%d] release old object: handle=%lx, origin_handle=%lx\n" KRESET,
                                task->vm_id,
                                oldObject->ObjectHandle,
                                oldObject->OriginalObjectHandle);

                    RemoveObject(objectTable, oldObject->OriginalObjectHandle);
                }

                param->base.ObjectHandle = 0;
                param->base.OriginalObjectHandle = oldObject->OriginalObjectHandle;

                *ret = _clReleaseMemObject(memoryObject);

                break;
            }

            case CL_FLUSH:
            case CL_FINISH:
            {
                int (*_clFlushOrFinish)(cl_command_queue);
                load_ocl_func(_clFlushOrFinish);
                *ret = _clFlushOrFinish(param->command_queue);
                break;
            }

            case CL_ENQUEUE_READ_BUFFER:
            {
                void *udata = get_ptr_from_dstore(task->vm_id, param->user_data,
                        void *);
                cl_event *ewaitlist = get_ptr_from_dstore(task->vm_id,
                        param->event_wait_list, cl_event *);
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                int (*_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool,
                        size_t, size_t, void *, cl_uint, const cl_event *,
                        cl_event *);
                load_ocl_func(_clEnqueueReadBuffer);
                *ret = _clEnqueueReadBuffer(param->command_queue, param->mem,
                        param->blocking, param->offset, param->size, udata,
                        param->num_events_in_wait_list, ewaitlist, event);
                DEBUG_PRINT("size=%lu, str=%s\n", param->size, (char *)udata);
                break;
            }

            case CL_ENQUEUE_TASK:
            {
                cl_event *ewaitlist = get_ptr_from_dstore(task->vm_id,
                        param->event_wait_list, cl_event *);
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                int (*_clEnqueueTask)(cl_command_queue, cl_kernel, cl_uint,
                        const cl_event *, cl_event *);
                load_ocl_func(_clEnqueueTask);
                *ret = _clEnqueueTask(param->command_queue, param->kernel,
                        param->num_events_in_wait_list, ewaitlist, event);
                break;
            }

            case CL_CREATE_PROGRAM_WITH_SOURCE:
            {
                const char **sources= get_ptr_from_dstore(task->vm_id,
                        param->sources, const char **);
                const size_t *lengths = get_ptr_from_dstore(task->vm_id,
                        param->lengths, const size_t *);
                for (i = 0; i < param->count; ++i)
                    sources[i] = get_ptr_from_dstore(task->vm_id, sources[i],
                            char *);
                cl_int *errcode = get_ptr_from_dstore(task->vm_id,
                        param->errcode_ret, cl_int *);
                cl_program (*_clCreateProgramWithSource)(cl_context, cl_uint,
                        const char **, const size_t*, cl_int *);
                load_ocl_func(_clCreateProgramWithSource);
                param->program = _clCreateProgramWithSource(
                        (cl_context)ex_st.decrypt(task->vm_id, (uintptr_t)param->context),
                        param->count, sources, lengths, errcode);
                DEBUG_PRINT("num=%u, prog=%p, ctx=%p, err=%d\n", param->count, param->program, param->context, *errcode);
                break;
            }

            case CL_RELEASE_PROGRAM:
            {
                int (*_clReleaseProgram)(cl_program);
                load_ocl_func(_clReleaseProgram);
                *ret = _clReleaseProgram(param->program);
                break;
            }

            case CL_BUILD_PROGRAM:
            {
                cl_device_id *dlist = get_ptr_from_dstore(task->vm_id,
                        param->devices, cl_device_id *);
                const char *opts = get_ptr_from_dstore(task->vm_id,
                        param->options, const char *);
                int (*_clBuildProgram)(cl_program, cl_uint, const cl_device_id *,
                        const char *, void (CL_CALLBACK *)(cl_program, void *), void *);
                load_ocl_func(_clBuildProgram);
                *ret = _clBuildProgram(param->program, param->count, dlist, opts,
                        NULL, NULL);
                break;
            }

            case CL_CREATE_KERNEL:
            {
                const char *kname = get_ptr_from_dstore(task->vm_id,
                        param->kernel_name, const char *);
                cl_int *errcode = get_ptr_from_dstore(task->vm_id,
                        param->errcode_ret, cl_int *);
                cl_kernel (*_clCreateKernel)(cl_program, const char *, cl_int *);
                load_ocl_func(_clCreateKernel);
                param->kernel = _clCreateKernel(param->program, kname, errcode);
                if (errcode)
                    DEBUG_PRINT("prog=%p, name=%s, kernel=%p, err=%d\n", param->program, kname, param->kernel, *errcode);
                break;
            }

            case CL_RELEASE_KERNEL:
            {
                int (*_clReleaseKernel)(cl_kernel);
                load_ocl_func(_clReleaseKernel);
                *ret = _clReleaseKernel(param->kernel);
                break;
            }

            case CL_SET_KERNEL_ARG:
            {
                const void *argv = get_ptr_from_dstore(task->vm_id,
                        param->param_value, const void *);
                int (*_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void *);
                DEBUG_PRINT("kernel=%p, num=%u, size=%lu, argv=%p\n",
                        param->kernel, param->count, param->param_value_size, argv);
                load_ocl_func(_clSetKernelArg);
                *ret = _clSetKernelArg(param->kernel, param->count,
                        param->param_value_size, argv);
                break;
            }

            case CL_GET_DEVICE_INFO:
            {
                void *pval = get_ptr_from_dstore(task->vm_id,
                        param->param_value, void *);
                size_t *pval_size = get_ptr_from_dstore(task->vm_id,
                        param->param_value_size_ret, size_t *);
                cl_int (*_clGetDeviceInfo)(cl_device_id, cl_device_info,
                        size_t, void *, size_t *);
                load_ocl_func(_clGetDeviceInfo);
                *ret = _clGetDeviceInfo(param->device, param->info.device,
                        param->param_value_size, pval, pval_size);
                break;
            }

            case CL_CREATE_CONTEXT_FROM_TYPE:
            {
                cl_context_properties *prop = get_ptr_from_dstore(task->vm_id,
                        param->context_properties, cl_context_properties *);
                // TODO: callback
                cl_int *errcode = get_ptr_from_dstore(task->vm_id,
                        param->errcode_ret, cl_int *);
                if (prop != NULL) {
                    DEBUG_PRINT("device_type=%lu, prop[0]=%lu, prop[1]=0x%lx, prop[2]=%lu\n",
                        param->device_type, prop[0], prop[1], prop[2]);
                }

                cl_context (*_clCreateContextFromType)(const cl_context_properties *,
                        cl_device_type,
                        void(CL_CALLBACK *)(const char *, const void *, size_t, void *),
                        void *, cl_int *);
                load_ocl_func(_clCreateContextFromType);
                param->context = _clCreateContextFromType(prop,
                        param->device_type, NULL, NULL, errcode);
                if (errcode) {
                    DEBUG_PRINT("ctx=%p, err=%d\n", param->context, *errcode);
                }
                param->context = (cl_context)ex_st.encrypt(task->vm_id, (uintptr_t)param->context);
                break;
            }

            case CL_ENQUEUE_WRITE_BUFFER:
            {
                void *udata = get_ptr_from_dstore(task->vm_id, param->user_data,
                        void *);
                cl_event *ewaitlist = get_ptr_from_dstore(task->vm_id,
                        param->event_wait_list, cl_event *);
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                int (*_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool,
                        size_t, size_t, const void *, cl_uint, const cl_event *,
                        cl_event *);
                load_ocl_func(_clEnqueueWriteBuffer);
                *ret = _clEnqueueWriteBuffer(param->command_queue, param->mem,
                        param->blocking, param->offset, param->size, udata,
                        param->num_events_in_wait_list, ewaitlist, event);
                break;
            }

            case CL_RELEASE_EVENT:
            {
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                DEBUG_PRINT("release event real addr = 0x%lx\n", (uintptr_t)*event);
                int (*_clReleaseEvent)(cl_event);
                load_ocl_func(_clReleaseEvent);
                if (event != NULL)
                    *ret = _clReleaseEvent(*event);
                else
                    *ret = CL_INVALID_EVENT;
                break;
            }

            case CL_GET_EVENT_PROFILING_INFO:
            {
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                void *pval = get_ptr_from_dstore(task->vm_id,
                        param->param_value, void *);
                size_t *pval_size = get_ptr_from_dstore(task->vm_id,
                        param->param_value_size_ret, size_t *);
                int (*_clGetEventProfilingInfo)(cl_event, cl_profiling_info,
                        size_t, void *, size_t *);
                DEBUG_PRINT("profile event real addr = 0x%lx\n", (uintptr_t)*event);
                load_ocl_func(_clGetEventProfilingInfo);
                *ret = _clGetEventProfilingInfo(*event, param->info.profiling,
                        param->param_value_size, pval, pval_size);
                break;
            }

            case CL_ENQUEUE_ND_RANGE_KERNEL:
            {
                const size_t *goffset = get_ptr_from_dstore(task->vm_id,
                        param->global_work_offset, const size_t *);
                const size_t *gsize = get_ptr_from_dstore(task->vm_id,
                        param->global_work_size, const size_t *);
                const size_t *lsize = get_ptr_from_dstore(task->vm_id,
                        param->local_work_size, const size_t *);
                cl_event *ewaitlist = get_ptr_from_dstore(task->vm_id,
                        param->event_wait_list, cl_event *);
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);

                int (*_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel,
                        cl_uint, const size_t *, const size_t *, const size_t *,
                        cl_uint, const cl_event *, cl_event *);
                load_ocl_func(_clEnqueueNDRangeKernel);
                *ret = _clEnqueueNDRangeKernel(param->command_queue,
                        param->kernel, param->work_dim,
                        goffset, gsize, lsize, param->num_events_in_wait_list,
                        ewaitlist, event);
                if (event) {
                    DEBUG_PRINT("kernel event real addr = 0x%lx\n", (uintptr_t)*event);
                }
                break;
            }

            case CL_GET_PROGRAM_BUILD_INFO:
            {
                void *pval = get_ptr_from_dstore(task->vm_id,
                        param->param_value, void *);
                size_t *pval_size = get_ptr_from_dstore(task->vm_id,
                        param->param_value_size_ret, size_t *);
                int (*_clGetProgramBuildInfo)(cl_program, cl_device_id,
                        cl_program_build_info, size_t, void *, size_t *);
                load_ocl_func(_clGetProgramBuildInfo);
                *ret = _clGetProgramBuildInfo(param->program, param->device,
                        param->info.program_build, param->param_value_size,
                        pval, pval_size);
                break;
            }

            case CL_ENQUEUE_MAP_BUFFER:
            {
                const cl_event *ewaitlist = get_ptr_from_dstore(task->vm_id,
                        param->event_wait_list, const cl_event *);
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                cl_int *errcode = get_ptr_from_dstore(task->vm_id,
                        param->errcode_ret, cl_int *);
                void *(*_clEnqueueMapBuffer)(cl_command_queue, cl_mem, cl_bool,
                        cl_map_flags, size_t, size_t, cl_uint, const cl_event *,
                        cl_event *, cl_int *);
                load_ocl_func(_clEnqueueMapBuffer);
                // TODO: map it to guest
                param->user_data = _clEnqueueMapBuffer(param->command_queue,
                        param->mem, param->blocking, param->flags.map,
                        param->offset, param->size,
                        param->num_events_in_wait_list, ewaitlist, event,
                        errcode);
                break;
            }

            case CL_GET_CONTEXT_INFO:
            {
                void *pval = get_ptr_from_dstore(task->vm_id,
                        param->param_value, void *);
                size_t *pval_size = get_ptr_from_dstore(task->vm_id,
                        param->param_value_size_ret, size_t *);
                int (*_clGetContextInfo)(cl_context, cl_context_info, size_t,
                        void *, size_t *);
                load_ocl_func(_clGetContextInfo);
                *ret = _clGetContextInfo(
                        (cl_context)ex_st.decrypt(task->vm_id, (uintptr_t)param->context),
                        param->info.context,
                        param->param_value_size, pval, pval_size);
                break;
            }

            case CL_ENQUEUE_COPY_BUFFER:
            {
                cl_event *ewaitlist = get_ptr_from_dstore(task->vm_id,
                        param->event_wait_list, cl_event *);
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                int (*_clEnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem,
                        size_t, size_t, size_t, cl_uint, const cl_event *,
                        cl_event *);
                load_ocl_func(_clEnqueueCopyBuffer);
                *ret = _clEnqueueCopyBuffer(param->command_queue, param->mem,
                        param->dst_mem, param->offset, param->dst_offset,
                        param->size, param->num_events_in_wait_list, ewaitlist,
                        event);
                break;
            }

            case CL_ENQUEUE_COPY_BUFFER_TO_IMAGE:
            {
                const size_t *orig = get_ptr_from_dstore(task->vm_id,
                        param->origin, const size_t *);
                const size_t *reg = get_ptr_from_dstore(task->vm_id,
                        param->region, const size_t *);
                cl_event *ewaitlist = get_ptr_from_dstore(task->vm_id,
                        param->event_wait_list, cl_event *);
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                int (*_clEnqueueCopyBufferToImage)(cl_command_queue, cl_mem,
                        cl_mem, size_t, const size_t *, const size_t *, cl_uint,
                        const cl_event *, cl_event *);
                load_ocl_func(_clEnqueueCopyBufferToImage);
                *ret = _clEnqueueCopyBufferToImage(param->command_queue,
                        param->mem, param->dst_mem, param->offset, orig, reg,
                        param->num_events_in_wait_list, ewaitlist, event);
                break;
            }

            case CL_CREATE_IMAGE_2D:
            {
                const cl_image_format *imgfmt = get_ptr_from_dstore(task->vm_id,
                        param->image_format, const cl_image_format *);
                void *udata = get_ptr_from_dstore(task->vm_id, param->user_data,
                        void *);
                cl_int *errcode = get_ptr_from_dstore(task->vm_id,
                        param->errcode_ret, cl_int *);
                cl_mem (*_clCreateImage2D)(cl_context, cl_mem_flags,
                        const cl_image_format *, size_t, size_t, size_t, void *,
                        cl_int *);
                load_ocl_func(_clCreateImage2D);
                param->mem = _clCreateImage2D(
                        (cl_context)ex_st.decrypt(task->vm_id, (uintptr_t)param->context),
                        param->flags.mem,
                        imgfmt, param->image_width, param->image_height,
                        param->image_row_pitch, udata, errcode);
                break;
            }

            case CL_CREATE_IMAGE:
            {
                const cl_image_format *imgfmt = get_ptr_from_dstore(task->vm_id,
                        param->image_format, const cl_image_format *);
                const cl_image_desc *imgdesc = get_ptr_from_dstore(task->vm_id,
                        param->image_desc, const cl_image_desc *);
                void *udata = get_ptr_from_dstore(task->vm_id, param->user_data,
                        void *);
                cl_int *errcode = get_ptr_from_dstore(task->vm_id,
                        param->errcode_ret, cl_int *);
                cl_mem (*_clCreateImage)(cl_context, cl_mem_flags,
                        const cl_image_format *, const cl_image_desc *, void *,
                        cl_int *);
                load_ocl_func(_clCreateImage);
                param->mem = _clCreateImage(
                        (cl_context)ex_st.decrypt(task->vm_id, (uintptr_t)param->context),
                        param->flags.mem,
                        imgfmt, imgdesc, udata, errcode);
                break;
            }

            case CL_ENQUEUE_READ_IMAGE:
            {
                const size_t *orig = get_ptr_from_dstore(task->vm_id,
                        param->origin, const size_t *);
                const size_t *reg = get_ptr_from_dstore(task->vm_id,
                        param->region, const size_t *);
                void *udata = get_ptr_from_dstore(task->vm_id, param->user_data,
                        void *);
                cl_event *ewaitlist = get_ptr_from_dstore(task->vm_id,
                        param->event_wait_list, cl_event *);
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                int (*_clEnqueueReadImage)(cl_command_queue, cl_mem, cl_bool,
                        const size_t *, const size_t *, size_t, size_t, void *,
                        cl_uint, const cl_event *, cl_event *);
                load_ocl_func(_clEnqueueReadImage);
                *ret = _clEnqueueReadImage(param->command_queue, param->mem,
                        param->blocking, orig, reg, param->image_row_pitch,
                        param->image_slice_pitch, udata,
                        param->num_events_in_wait_list, ewaitlist, event);
                break;
            }

            case CL_ENQUEUE_WRITE_IMAGE:
            {
                const size_t *orig = get_ptr_from_dstore(task->vm_id,
                        param->origin, const size_t *);
                const size_t *reg = get_ptr_from_dstore(task->vm_id,
                        param->region, const size_t *);
                const void *udata = get_ptr_from_dstore(task->vm_id,
                        param->user_data, const void *);
                cl_event *ewaitlist = get_ptr_from_dstore(task->vm_id,
                        param->event_wait_list, cl_event *);
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                int (*_clEnqueueWriteImage)(cl_command_queue, cl_mem, cl_bool,
                        const size_t *, const size_t *, size_t, size_t,
                        const void *, cl_uint, const cl_event *, cl_event *);
                load_ocl_func(_clEnqueueWriteImage);
                *ret = _clEnqueueWriteImage(param->command_queue, param->mem,
                        param->blocking, orig, reg, param->image_row_pitch,
                        param->image_slice_pitch, udata,
                        param->num_events_in_wait_list, ewaitlist, event);
                break;
            }

            case CL_ENQUEUE_UNMAP_MEM_OBJECT:
            {
                cl_event *ewaitlist = get_ptr_from_dstore(task->vm_id,
                        param->event_wait_list, cl_event *);
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);

                // TODO: finx ENQUEUE_MAP_BUFFER first
                int (*_clEnqueueUnmapMemObject)(cl_command_queue, cl_mem,
                        void *, cl_uint, const cl_event *, cl_event *);
                load_ocl_func(_clEnqueueUnmapMemObject);
                *ret = _clEnqueueUnmapMemObject(param->command_queue,
                        param->mem, param->user_data,
                        param->num_events_in_wait_list, ewaitlist, event);
                break;
            }

            case CL_GET_SUPPORTED_IMAGE_FORMATS:
            {
                cl_image_format *imgfmt = get_ptr_from_dstore(task->vm_id,
                        param->image_format, cl_image_format *);
                cl_uint *num_imgfmt = get_ptr_from_dstore(task->vm_id,
                        param->num_image_formats, cl_uint *);
                cl_int (*_clGetSupportedImageFormats)(cl_context, cl_mem_flags,
                        cl_mem_object_type, cl_uint, cl_image_format *,
                        cl_uint *);
                load_ocl_func(_clGetSupportedImageFormats);
                *ret = _clGetSupportedImageFormats(
                        (cl_context)ex_st.decrypt(task->vm_id, (uintptr_t)param->context),
                        param->flags.mem, param->mem_type, param->num_entries,
                        imgfmt, num_imgfmt);
                break;
            }

            case CL_WAIT_FOR_EVENTS:
            {
                const cl_event *elist = get_ptr_from_dstore(task->vm_id,
                        param->event_list, const cl_event *);
                int (*_clWaitForEvent)(cl_uint, const cl_event *);
                load_ocl_func(_clWaitForEvent);
                *ret = _clWaitForEvent(param->count, elist);
                break;
            }

            case CL_GET_EVENT_INFO:
            {
                cl_event *event = get_ptr_from_dstore(task->vm_id, param->event,
                        cl_event *);
                void *pval = get_ptr_from_dstore(task->vm_id,
                        param->param_value, void *);
                size_t *pval_size = get_ptr_from_dstore(task->vm_id,
                        param->param_value_size_ret, size_t *);
                int (*_clGetEventInfo)(cl_event, cl_event_info, size_t, void *,
                        size_t *);
                load_ocl_func(_clGetEventInfo);
                *ret = _clGetEventInfo(*event, param->info.event,
                        param->param_value_size, pval, pval_size);
                break;
            }

            case CL_GET_KERNEL_WORK_GROUP_INFO:
            {
                void *pval = get_ptr_from_dstore(task->vm_id,
                        param->param_value, void *);
                size_t *pval_size = get_ptr_from_dstore(task->vm_id,
                        param->param_value_size_ret, size_t *);
                int (*_clGetKernelWorkGroupInfo)(cl_kernel, cl_device_id,
                        cl_kernel_work_group_info, size_t, void *, size_t *);
                load_ocl_func(_clGetKernelWorkGroupInfo);
                *ret = _clGetKernelWorkGroupInfo(param->kernel, param->device,
                        param->info.kernel_work_group, param->param_value_size,
                        pval, pval_size);
                break;
            }

            default:
                fprintf(stderr, "unsupported OpenCL API\n");
        }

        DEBUG_PRINT("finished [vm#%d] task %ld, cmd 0x%x, high-priority=%d\n",
                task->vm_id, task->node_id, param->base.cmd_id, task->IsHighPriority);

        // TODO: async commands do not need to set "done"
        param->base.done = 1;
        cmd_out->command_id = cmd_in->command_id;
        command_channel_free_command(chan, cmd_in);

        DEBUG_PRINT("notify guestlib of completion\n");

        cmd_out->command_type = MSG_RESPONSE;
        cmd_out->status = STATUS_TASK_DONE;

        command_channel_send_command(chan, cmd_out);
        command_channel_free_command(chan, cmd_out);

        //
        // Read next invocation.
        //
        if (ValConfig.SwappingSupported)
        {
            sem_post(serial->start);
        }
    }

    // never reached
    return NULL;
}
