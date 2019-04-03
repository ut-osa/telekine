#ifndef __HIP_NW_UTILITIES_H__
#define __HIP_NW_UTILITIES_H__

#define ava_utility static
ava_utility size_t
hipLaunchKernel_extra_size(void **extra)
{
    size_t size = 1;
    while (extra[size - 1] != HIP_LAUNCH_PARAM_END)
        size++;
    return size;
}

#undef ava_utility

#endif                                           // ndef __HIP_NW_UTILITIES_H__
