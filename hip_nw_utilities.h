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

ava_utility size_t
calc_image_size(const void *image)
{
    const Elf64_Ehdr *h = (Elf64_Ehdr *) image;

    return sizeof(Elf64_Ehdr) + h->e_shoff + h->e_shentsize * h->e_shnum;
}

#undef ava_utility

#endif                                           // ndef __HIP_NW_UTILITIES_H__
