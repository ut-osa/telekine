#include <string>
#include <hsa.h>

#include <stdio.h>
#include <stdlib.h>

#include "./hip_cpp_bridge.h"
using std::string;


namespace hip_impl {
    hsa_executable_t load_executable(
        const string& file, hsa_executable_t executable, hsa_agent_t agent)
    {
       printf("XXXX you failed!\n");
       abort();
       /*
       __do_c_load_executable(file.data(), file.length(), &executable, &agent);
       */
       return executable;

    }
} // namespace hip_impl
