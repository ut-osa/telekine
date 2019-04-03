#ifndef __VGPU_KVM_H__
#define __VGPU_KVM_H__


#ifdef __KERNEL__
#include <linux/types.h>
#else
#include <stdint.h>
#include <stddef.h>
#endif


/* Invocation Status/State */

#define STATUS_TASK_UNDEFINED 0
#define STATUS_TASK_DONE      1
#define STATUS_TASK_RUNNING   2
#define STATUS_TASK_ASYNC     3
#define STATUS_TASK_CALLBACK  4
#define STATUS_TASK_ERROR     5
#define STATUS_TASK_CONTINUE  9

#define STATUS_CALLBACK_POLL   6
#define STATUS_CALLBACK_FILLED 7
#define STATUS_CALLBACK_DONE   8
#define STATUS_CALLBACK_TIMEOUT 10
#define STATUS_CALLBACK_ERROR   11

#endif
