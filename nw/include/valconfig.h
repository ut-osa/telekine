#ifndef __VGPU_VAL_CONFIG_H__
#define __VGPU_VAL_CONFIG_H__


#include "devconf.h"
#include "ctype_util.h"


typedef struct _VAL_CONFIG
{
    //
    // Indicates whether to use KVM mode.
    //
    BOOLEAN KvmSupported;

    //
    // Indicates whether rate limiting is enabled.
    //
    BOOLEAN RateLimitSupported;

    //
    // Indicates whether device memory swapping is enabled.
    //
    BOOLEAN SwappingSupported;

    //
    // Indicates whether debug print is enabled.
    //
    BOOLEAN DebugSupported;

} VAL_CONFIG, *PVAL_CONFIG;

#ifdef DEBUG

static const VAL_CONFIG ValConfig = {
    .RateLimitSupported = ENABLE_RATE_LIMIT,
    .SwappingSupported = ENABLE_SWAP,
    .DebugSupported = TRUE
};

#else

static const VAL_CONFIG ValConfig = {
    .KvmSupported = ENABLE_KVM_MEDIATION,
    .RateLimitSupported = ENABLE_RATE_LIMIT,
    .SwappingSupported = ENABLE_SWAP,
    .DebugSupported = FALSE
};

#endif

#endif
