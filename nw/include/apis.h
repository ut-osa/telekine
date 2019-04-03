#ifndef __VGPU_APIS_H__
#define __VGPU_APIS_H__


#include "devconf.h"

#if CUDA_SUPPORTED
#include "cuda.h"
#endif

#if TF_C_SUPPORTED
#include "tensorflow.h"
#endif

#if TF_PY_SUPPORTED
#include "tensorflow_py.h"
#endif


typedef union _DESCRIPTOR
{
#if CUDA_SUPPORTED
    CUDA_PARAM cuda_param;
#endif

#if TF_C_SUPPORTED
    TF_PARAM tf_param;
#endif

#if TF_PY_SUPPORTED
    TF_PY_PARAM tf_py_param;
#endif

} DESCRIPTOR, *PDESCRIPTOR;

#ifndef __KERNEL__
#define PAGE_SIZE 0x1000
#endif
#define PAGE_ROUND_UP(x) ( (((uintptr_t)(x)) + PAGE_SIZE-1)  & (~(PAGE_SIZE-1)) )
#define DESC_SLAB_SIZE PAGE_ROUND_UP(sizeof(DESCRIPTOR))


#endif
