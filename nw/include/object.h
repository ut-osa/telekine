#ifndef __VGPU_OBJECT_H__
#define __VGPU_OBJECT_H__


#include "devconf.h"
#include "ctype_util.h"


typedef struct obj_info
{
#ifdef __KERNEL__
    struct list_head list;
#endif

    //
    // Runtime type of the object. Only OpenCL is supported for now.
    //
    uint8_t RuntimeType;

    //
    // Command used to allocate and release the object.
    //
    uint32_t AllocateCmdId;
    uint32_t FreeCmdId;

    //
    // Size of the object.
    //
    size_t ObjectSize;

    //
    // Virtual address in invoker of the swapped-out object.
    //
    PVOID SwappedOutAddress;

    //
    // Object handle
    //
    HANDLE OriginalObjectHandle;
    HANDLE ObjectHandle;

    //
    // Other runtime-dependent attributes for swapping.
    //
    union
    {
        //
        // OpenCL
        //
        struct
        {
            HANDLE Context;
            HANDLE MemoryFlag;
            HANDLE CommandQueue;
        };
    };

} DEVICE_OBJECT_LIST, *PDEVICE_OBJECT_LIST;


#endif

