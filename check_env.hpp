#ifndef _CHECK_ENV_H_
#define _CHECK_ENV_H_

#include <cstdlib>
#include <climits>

static inline bool _env_parse(const char *s, bool def)
{
   switch (s[0]) {
   case '0':
   case 'n':
   case 'N':
   case 'f':
   case 'F':
      return false;
   case '1':
   case 'y':
   case 'Y':
   case 't':
   case 'T':
      return true;
   default:
      return def;
   }
}

static inline int _env_parse(const char *s, int def)
{
   auto full = strtol(s, NULL, 0);

   if (full > INT_MAX)
      return INT_MAX;
   if (full < INT_MIN)
      return INT_MIN;
   return static_cast<int>(full);
}

template<typename T>
static inline T CHECK_ENV(const char *key, T def) {
  const char *s = getenv(key);
  return s ? _env_parse(s, def) : def;
}
#endif
