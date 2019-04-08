#!/bin/bash

dir=$(dirname $0)

guest_lib=$dir/libguestlib.so
shim=$dir/crypto_guestshim.so
crypt_lib=$dir/crypto_impl.so

echo LD_PRELOAD="$shim:$guest_lib:$crypt_lib" AVA_LOCAL=1 "$@"
HCC_LAZYINIT=1 LD_PRELOAD="$shim:$guest_lib:$crypt_lib" AVA_LOCAL=1 exec "$@"
