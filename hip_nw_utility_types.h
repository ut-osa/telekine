#ifndef __HIP_NW_UTILITY_TYPES_H__
#define __HIP_NW_UTILITY_TYPES_H__

struct hipFuncAttributes;
typedef struct hipFuncAttributes hipFuncAttributes;
typedef struct {
    /* argument types */
    int func_argc;
    char func_arg_is_handle[64];
} Metadata;

#endif                                           // ndef __HIP_NW_UTILITY_TYPES_H__
