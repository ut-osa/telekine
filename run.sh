#!/bin/bash

dir=$(dirname $0)

guest_lib=$dir/libguestlib.so
shim=$dir/guestshim.so

echo HCC_LAZYINIT=1 LD_PRELOAD="$shim:$guest_lib" AVA_LOCAL=1 "$@"
HCC_LAZYINIT=1 LD_PRELOAD="$shim:$guest_lib" AVA_LOCAL=1 exec "$@"
