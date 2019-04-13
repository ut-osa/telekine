#ifndef __LGM_HIP_FUNCTION_INFO_H_
#define __LGM_HIP_FUNCTION_INFO_H_ 1

#define MAX_AGENTS 16

#include <hip/hip_runtime.h>
#include "hip_cpp_bridge.h"

hipFunction_t hip_function_lookup(uintptr_t function_address,
                                  hipStream_t stream);

const struct nw_kern_info *hip_function_kernel_info(hipFunction_t f);

static inline
void hip_function_to_aql(hsa_kernel_dispatch_packet_t *aql, hipFunction_t f,
               uint32_t globalWorkSizeX, uint32_t globalWorkSizeY,
               uint32_t globalWorkSizeZ, uint32_t localWorkSizeX,
               uint32_t localWorkSizeY, uint32_t localWorkSizeZ,
               size_t sharedMemBytes)
{
    const struct nw_kern_info *kern_info = hip_function_kernel_info(f);

    aql->workgroup_size_x = localWorkSizeX;
    aql->workgroup_size_y = localWorkSizeY;
    aql->workgroup_size_z = localWorkSizeZ;
    aql->grid_size_x = globalWorkSizeX;
    aql->grid_size_y = globalWorkSizeY;
    aql->grid_size_z = globalWorkSizeZ;
    aql->group_segment_size = sharedMemBytes + kern_info->workgroup_group_segment_byte_size;
    aql->private_segment_size = kern_info->workitem_private_segment_byte_size;
    aql->kernel_object = kern_info->_object;
    aql->setup = 3 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    aql->header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);
    aql->header |= (1 << HSA_PACKET_HEADER_BARRIER);
    aql->header |= (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
            (HSA_FENCE_SCOPE_AGENT << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
}

#endif
