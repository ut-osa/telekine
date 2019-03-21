python
import os
guest_lib = os.environ['PWD'] + '/hip_nw/libguestlib.so'
shim = os.environ['PWD'] + '/guestshim.so'
gdb.execute('set environment AVA_LOCAL=1')
gdb.execute('set environment HCC_LAZYINIT=1')
gdb.execute('set environment LD_PRELOAD=' + shim + ':' + guest_lib)
end
