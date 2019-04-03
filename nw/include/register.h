#ifndef __VGPU_REGISTER_H__
#define __VGPU_REGISTER_H__

/* BAR2 registers */

/* 1 byte */
#define REG_VT_MODE   0x1
#define REG_RT_TYPE   0x2
#define REG_CMD_ID    0x3
#define REG_FLUSH_CMD 0x4

#define REG_MOD_INIT  0x5 /* used for VM realize/unrealize */
#define REG_MOD_EXIT  0x6

#define REG_APP_INIT  0x10
#define REG_APP_EXIT  0x11

/* 8 bytes */
#define REG_DATA_PTR  0x8
#define REG_VM_ID     0x18

/* VT_MODE: virtualization mode */

#define API_REMOTE      0

/* RT_TYPE: runtime type */

#define OPENCL_API        0
#define CUDA_API          1
#define CUDA_DRIVER_API   2
#define OPENGL_API        3
#define TENSORFLOW_API    4
#define MVNC_API          5
#define TENSORFLOW_PY_API 6

#endif
