#ifndef __VGPU_TENSORFLOW_PY_H__
#define __VGPU_TENSORFLOW_PY_H__

#include "devconf.h"

#if TF_PY_SUPPORTED

#ifdef __KERNEL__
#include "tensorflow_types.h"
#else
#include <string.h>
#endif

#include "attribute.h"

#define TF_PY_NW_OBJECT                15
#define TF_PY_NW_METHOD                16
#define TF_PY_NW_CALLBACK              17
#define TF_PY_NW_CALLBACK_INIT         18
#define TF_PY_NW_CALLBACK_DONE         40
#define TF_PY_NW_CALLBACK_TEST         19

#define TF_PY_SESSION_INIT              1
#define TF_PY_SESSION_ENTER             6
#define TF_PY_SESSION_EXIT              7
#define TF_PY_SESSION_DEL               4
#define TF_PY_SESSION_RUN               5
#define TF_PY_TPU_CLUSTER_RESOLVER_INIT 2
#define TF_PY_TPU_CLUSTER_RESOLVER_MASTER 3

#define TF_PY_TPU_INITIALIZE_SYSTEM     8
#define TF_PY_TPU_SHUTDOWN_SYSTEM       9
#define TF_PY_TPU_REWRITE              11
#define TF_PY_TPU_RUN_CONFIG           13
#define TF_PY_TPU_TPU_ESTIMATOR        14

#define TF_PY_GLOBAL_VARIABLES_INITIALIZER 10
#define TF_PY_ONES                         12
#define TF_PY_RANDOM_UNIFORM               20
#define TF_PY_TRANSPOSE                    21
#define TF_PY_CAST                         22
#define TF_PY_RESHAPE                      23
#define TF_PY_EXPAND_DIMS                  24
#define TF_PY_CONCAT                       25
#define TF_PY_EQUAL                        26
#define TF_PY_SLICE                        30
#define TF_PY_SHAPE                        32
#define TF_PY_FIXED_LEN_FEATURE            36
#define TF_PY_VAR_LEN_FEATURE              37
#define TF_PY_PARSE_SINGLE_EXAMPLE         38

#define TF_PY_CONTROL_FLOW_OPS_SWITCH      27
#define TF_PY_CONTROL_FLOW_OPS_MERGE       28

#define TF_PY_IMAGE_RESIZE_IMAGES          29
#define TF_PY_IMAGE_SAMPLE_DISTORTED_BOUNDING_BOX 31
#define TF_PY_IMAGE_DRAW_BOUNDING_BOXES    33
#define TF_PY_IMAGE_DECODE_JPEG            34
#define TF_PY_IMAGE_CONVERT_IMAGE_DTYPE    35

#define TF_PY_DATA_DATASET_LIST_FILES      39


#define TF_PY_PARAM_PAYLOAD_NUM 12
#define TF_PY_PARAM_RET_VAL_NUM 1

typedef struct _TF_PY_PARAM_PAYLOAD {
    char      *addr;
    size_t    size;
    uintptr_t offset;

} TF_PY_PARAM_PAYLOAD, *PTF_PY_PARAM_PAYLOAD;

typedef struct _TF_PY_PARAM {
    PARAM_BASE base;

    TF_PY_PARAM_PAYLOAD param[TF_PY_PARAM_PAYLOAD_NUM];
    TF_PY_PARAM_PAYLOAD ret_val[TF_PY_PARAM_RET_VAL_NUM];

} TF_PY_PARAM, *PTF_PY_PARAM;

#define TF_PY_PARAM_SIZE sizeof(TF_PY_PARAM)

#else

#define TF_PY_PARAM_SIZE 0

#endif // #if TF_PY_SUPPORTED

#endif
