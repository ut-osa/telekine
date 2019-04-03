#!/home/hyu/venv/bin/python

from absl import app
import os, sys, fcntl, select, atexit, mmap
import threading, Queue

from ctypes import *
from structs import *
sys.path.insert(1, os.path.join(sys.path[0], '../include/common'))
from devconf import *
from tensorflow_py import *
import apifwd_tf


print(sys.executable, __file__, sys.argv[1:])


# Global
kvm_fd = None
vm_id = None
mm = None
poller = None
tf_queue = Queue.Queue()


# Register cleanup function
@atexit.register
def dispatcher_exit():
    if mm is not None:
        mm.close()
    if poller is not None:
        poller.unregister(kvm_fd)
    if kvm_fd is not None:
        os.close(kvm_fd)
    print("halt dispatcher")


# Setup SIGINT handler

# Setup SIGSEGV handler


# Open KVM device
if os.geteuid() != 0:
    print("need sudo permission")
    exit(-1)
if len(sys.argv) <= 1:
    print("Usage: ./invoker.py <vm_id>")
    exit(0)
kvm_fd = os.open("/dev/kvm-vgpu", os.O_RDWR | os.O_NONBLOCK)


# Map kernel memory
#mm = mmap.mmap(kvm_fd, INVOKER_FIFO_SIZE + INVOKER_DSTORE_SIZE, mmap.MAP_SHARED,
#               mmap.PROT_READ | mmap.PROT_WRITE)
mm = mmap.mmap(kvm_fd, INVOKER_FIFO_SIZE + INVOKER_DSTORE_SIZE, mmap.MAP_SHARED,
               access=mmap.ACCESS_WRITE)
if mm is None:
    print("mmap KVM memory failed")
    exit(-1)
else:
    print("mmap KVM memory success")


# Setup invoker info
vm_id = int(sys.argv[1])
print("spawn invoker#%d" % vm_id)
ret = fcntl.ioctl(kvm_fd, IOCTL_KVM_NOTIFY_EXEC_SPAWN, vm_id)
if ret < 0:
    print("failed to notify invoker respawn err=%d" % ret);
    exit(-1);
print("[invoker#%d] KVM notified" % vm_id)


# Spawn task-poller
def poll_task():
    poller = select.poll()
    poller.register(kvm_fd, select.POLLIN | select.POLLERR)
    print("invoker#%d starts polling tasks" % vm_id)

    while True:
        events = poller.poll(-1)
        for fd, flag in events:
            if flag & select.POLLERR:
                print("failed to poll")
                exit(-1)

            if flag & select.POLLIN:
                node_bytes = os.read(kvm_fd, sizeof(MINI_TASK_NODE))
                if len(node_bytes) != sizeof(MINI_TASK_NODE):
                    print("read task node failed, bytes read=%d" %
                          len(node_bytes))
                    continue
                node = MINI_TASK_NODE.from_buffer_copy(node_bytes)

                if node.node_id < 0:
                    print("invoker#%d exits" % vm_id)
                    exit(0)

                if node.rt_type == TENSORFLOW_PY_API:
                    tf_queue.put(node)
                    print("[invoker#%ld] new tf_py task global_offset=%lx"
                          % (vm_id, node.data_ptr))
                else:
                    print("[invoker#%d] wrong runtime type" % vm_id)



t = threading.Thread(target = poll_task)
t.start()
print("[invoker#%d] poll task spawned" % vm_id)

# Spawn API handlers
#t_tf = threading.Thread(target = apifwd_tf.handler,
#                        args = (tf_queue, kvm_fd, mm))
#t_tf.start()

# Execute handler in the main thread
apifwd_tf.handler(tf_queue, kvm_fd, mm)
