#!/home/hyu/venv/bin/python

import os, sys, signal, select
import atexit
from structs import *


# Global
kvm_fd = None
poller = None


# Register cleanup function
@atexit.register
def dispatcher_exit():
    if kvm_fd is not None:
        poller.unregister(kvm_fd)
        os.close(kvm_fd)
    print("halt dispatcher")


# Setup SIGINT handler
def signal_handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


# Open KVM device
if os.geteuid() != 0:
    print("need sudo permission")
    exit(-1)
kvm_fd = os.open("/dev/kvm-vgpu", os.O_RDWR | os.O_NONBLOCK)


# Poll VM info
poller = select.poll()
poller.register(kvm_fd, select.POLLIN | select.POLLERR)
print("start polling VMs")

try:
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
                vm_id = node.vm_id

                # node.rt_type = 0: create, 1: shutdown
                if node.rt_type == 0:
                    print("new [vm#%ld] spawned" % vm_id)
                    child = os.fork()
                    if (child == 0):
                        raise StopIteration
                else:
                    print("[vm#%ld] shutdown" % vm_id)
except StopIteration: pass

# Spawn invoker

argv = [sys.executable,
        'invoker.py',
        str(vm_id)] + sys.argv[1:]
os.execv(sys.executable, argv)
