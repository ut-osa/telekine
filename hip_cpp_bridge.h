#include <stddef.h>
#include <hsa/hsa.h>

#ifdef __cplusplus
extern "C" {
#endif

struct hipFuncAttributes;
typedef struct hipFuncAttributes hipFuncAttributes;

int __do_c_load_executable(
    const char *file_buf,
    size_t file_len,
    hsa_executable_t * executable,
    hsa_agent_t * agent);

#ifdef __cplusplus
}
#endif
