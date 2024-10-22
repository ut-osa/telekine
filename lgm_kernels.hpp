#ifndef _LGM_KERNELS_HPP_
#define _LGM_KERNELS_HPP_ 1

#include <hip/hip_runtime_api.h>
#include <lgm_types.h>

/*
 * do nothing,used to pad out kernel batches
 */
void nullKern(void);

/*
 * copy from GPU memory to GPU memory
 */
void vector_copy(uint8_t *C_d, uint8_t *A_d, size_t N);

/*
 * perform an atomic tag update when the tag resides on the GPU
 */
void tag_copy(tag_t *dst, tag_t *src);

/*
 * spin until the memory at tag_p contains tag
 */
void check_tag(tag_t *tag_p, tag_t tag);

/*
 * perform an atomic tag update when we know the tag
 */
void set_tag(unsigned long long *ptr, unsigned long long tag);

/*
 * write batchid into buf, used for tracking completed batches
 */
void note_batch_id(uint8_t *buf, uint64_t batchid);

#endif