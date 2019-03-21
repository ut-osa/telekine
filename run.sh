#!/bin/bash


guest_lib=$PWD/hip_nw/libguestlib.so
shim=$PWD/guestshim.so

echo LD_PRELOAD="$shim:$guest_lib" AVA_LOCAL=1 exec "$@"
LD_PRELOAD="$shim:$guest_lib" AVA_LOCAL=1 exec "$@"
