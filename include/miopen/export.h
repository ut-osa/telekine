
#ifndef MIOPEN_EXPORT_H
#define MIOPEN_EXPORT_H

#ifdef MIOPEN_STATIC_DEFINE
#  define MIOPEN_EXPORT
#  define MIOPEN_NO_EXPORT
#else
#  ifndef MIOPEN_EXPORT
#    ifdef MIOpen_EXPORTS
        /* We are building this library */
#      define MIOPEN_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define MIOPEN_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef MIOPEN_NO_EXPORT
#    define MIOPEN_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef MIOPEN_DEPRECATED
#  define MIOPEN_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef MIOPEN_DEPRECATED_EXPORT
#  define MIOPEN_DEPRECATED_EXPORT MIOPEN_EXPORT MIOPEN_DEPRECATED
#endif

#ifndef MIOPEN_DEPRECATED_NO_EXPORT
#  define MIOPEN_DEPRECATED_NO_EXPORT MIOPEN_NO_EXPORT MIOPEN_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define MIOPEN_NO_DEPRECATED
#endif

#endif
