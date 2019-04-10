#ifndef _CHECK_ENV_H_
#define _CHECK_ENV_H_

#include <stdlib.h>
#include <string.h>

static inline bool CHECK_ENV(const char *key) {
  const char *val = getenv(key);
  /* true if val exists and is not 0 */
  return val && strcmp("0", val);
}

#endif
