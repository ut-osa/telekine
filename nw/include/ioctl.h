#ifndef __VGPU_IOCTL_H__
#define __VGPU_IOCTL_H__

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
#else
#include <sys/ioctl.h>
#include <stdio.h>
#endif

#include <linux/kvm.h>

/* General */

#define IOCTL_SET_MSG     _IOR(VGPU_DRIVER_MAJOR, 0, char *)
#define IOCTL_SET_VT_MODE _IOR(VGPU_DRIVER_MAJOR, REG_VT_MODE, uint8_t)
#define IOCTL_SET_RT_TYPE _IOR(VGPU_DRIVER_MAJOR, REG_RT_TYPE, uint8_t)
#define IOCTL_SET_CMD_ID  _IOR(VGPU_DRIVER_MAJOR, REG_CMD_ID, uint8_t)
#define IOCTL_FLUSH_CMD   _IO(VGPU_DRIVER_MAJOR, REG_FLUSH_CMD)
#define IOCTL_GET_VGPU_ID _IO(VGPU_DRIVER_MAJOR, 0x17)

#define IOCTL_SEND_VMCALL     _IOR(VGPU_DRIVER_MAJOR, 0x10, uintptr_t)
#define IOCTL_WAIT_TASK_ASYNC _IO(VGPU_DRIVER_MAJOR, 0x11)
#define IOCTL_APP_ENTRY       _IO(VGPU_DRIVER_MAJOR, 0x12)
#define IOCTL_APP_EXIT        _IO(VGPU_DRIVER_MAJOR, 0x13)

#define IOCTL_REQUEST_DESC_SLAB    _IO(VGPU_DRIVER_MAJOR, 0x15)
#define IOCTL_REQUEST_PARAM_BLOCK  _IOR(VGPU_DRIVER_MAJOR, 0x16, size_t)

/* KVM */

#define KVM_NOTIFY_EXEC_FIFO   _IOR(KVMIO, 0x100, uintptr_t)
#define KVM_NOTIFY_EXEC_DSTORE _IOR(KVMIO, 0x101, uintptr_t)
#define KVM_NOTIFY_EXEC_SPAWN  _IOR(KVMIO, 0x107, int64_t)
#define IOCTL_KVM_NOTIFY_EXEC_SPAWN 0x107

#define KVM_NOTIFY_VGPU_PARAM  _IOR(KVMIO, 0x102, uintptr_t)
#define KVM_NOTIFY_VGPU_DSTORE _IOR(KVMIO, 0x103, uintptr_t)
#define KVM_WAIT_TASK_ASYNC    _IO(KVMIO, 0x105)
#define KVM_NOTIFY_VGPU_EXIT   _IO(KVMIO, 0x106)
#define KVM_GET_VGPU_ID        _IO(KVMIO, 0x10B)
#define KVM_SET_VGPU_CID       _IOR(KVMIO, 0x10C, int)

#define KVM_NOTIFY_APP_INIT    _IO(KVMIO, 0x108)
#define KVM_NOTIFY_APP_EXIT    _IO(KVMIO, 0x109)

#define KVM_NOTIFY_TASK_FINISHED               _IOR(KVMIO, 0x104, size_t)
#define IOCTL_KVM_NOTIFY_TASK_FINISHED 0x104
#define KVM_NOTIFY_HIGH_PRIORITY_TASK_FINISHED _IOR(KVMIO, 0x10A, size_t)

/* APIs */

#define IOCTL_TF_PY_CMD      0x54
#define IOCTL_TF_PY_CALLBACK 0x55

#endif
