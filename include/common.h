#ifndef _COMMON_H_
#define _COMMON_H_

#include <hip/hip_runtime.h>

#define AES_KEYLEN 32u
#define AES_ROUNDKEYLEN 240u
#define AES_BLOCKLEN 16u
#define AES_MACLEN 12u
#define AES_GCM_STEP 64u

constexpr int kBaseThreadBits = 8;
constexpr int kBaseThreadNum  = 1 << kBaseThreadBits;

#define HIP_CHECK(cmd)                                                                             \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            abort();                                                                    \
        }                                                                                          \
    }

#endif
