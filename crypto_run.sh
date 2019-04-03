#!/bin/bash


guest_lib=$PWD/libguestlib.so
shim=$PWD/crypto_guestshim.so
crypt_lib=$PWD/crypto_impl.so

echo LD_PRELOAD="$shim:$guest_lib:$crypt_lib" AVA_LOCAL=1 "$@"
HCC_LAZYINIT=1 LD_PRELOAD="$shim:$guest_lib:$crypt_lib" AVA_LOCAL=1 exec "$@"
