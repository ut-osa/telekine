#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__ 1
#endif

#include "./hip_cpp_bridge.h"

#include <hip/hcc_detail/program_state.hpp>

#include <string>
#include <stdio.h>
#include <stdlib.h>

using std::string;


extern "C" int
__do_c_load_executable(
    const char *file_buf,
    size_t file_len,
    hsa_executable_t * executable,
    hsa_agent_t * agent)
{
   printf("OOOOF\n");
   abort();
   *executable = hip_impl::load_executable(string(file_buf, file_len),
            *executable, *agent);
   return 0;
}
