#include "lgm_kernels.hpp"

#include <hip/hip_runtime.h>

__global__ void nullKern(void)
{
}

__global__ void
vector_copy(uint8_t *C_d, uint8_t *A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i];
    }
}

__global__ void
tag_copy(tag_t *dst, tag_t *src)
{
	atomicExch((unsigned long long *)dst, (unsigned long long)*src);
}

__global__ void
check_tag(tag_t *tag_p, tag_t tag)
{
    /* TODO: one  thread  only */
    while (atomicExch((unsigned long long *)tag_p, 1) != tag)
      /* spin */;

}

__global__ void
set_tag(unsigned long long *ptr, unsigned long long tag)
{
    atomicExch((unsigned long long *)ptr, tag);
}

__global__ void
note_batch_id(uint8_t *buf, uint64_t batchid)
{
    *((uint64_t *)buf) = batchid;
}

