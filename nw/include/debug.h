#ifndef __VGPU_DEBUG_H__
#define __VGPU_DEBUG_H__

#ifndef __KERNEL__
#include <stdio.h>
#endif

#define DEBUG
#undef DEBUG

/* debug print */
#ifdef DEBUG
    #ifdef __KERNEL__
    #define DEBUG_PRINT(fmt, args...) printk(KERN_INFO fmt, ## args)
    #else
    #define DEBUG_PRINT(fmt, args...) fprintf(stderr, fmt, ## args)
    #endif
#else
    #define DEBUG_PRINT(fmt, args...)
#endif

#endif
