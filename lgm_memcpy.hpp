#ifndef LGM_MEMCPY_H
#define LGM_MEMCPY_H

#include "check_env.hpp"

#include "crypto/aes_gcm.h"
#include <algorithm>
#include <hip/hip_runtime_api.h>
#include <map>
#include <memory>
#include <stdio.h>
#include <openssl/evp.h>
#include <unistd.h>

#ifndef crypto_aead_aes256gcm_ABYTES
#define crypto_aead_aes256gcm_ABYTES 16U
#endif
#ifndef crypto_aead_aes256gcm_NPUBBYTES
#define crypto_aead_aes256gcm_NPUBBYTES 12U
#endif

namespace lgm {

// AES Encryption State //
struct EncryptionState {
  uint8_t key[AES_KEYLEN];
  uint8_t nonce_host[crypto_aead_aes256gcm_NPUBBYTES];
  uint8_t* nonce_device;
  AES_GCM_engine* engine_device;
  EVP_CIPHER_CTX* encrypt_ctx;
  EVP_CIPHER_CTX* decrypt_ctx;

  EncryptionState(hipStream_t stream);
  ~EncryptionState(void);
  void nextNonceCPU();
  void nextNonceGPU(hip_launch_batch_t* batch, hipStream_t stream);
};

// ciphertext_len includes MAC
void DecryptAsync(hip_launch_batch_t* batch, void* plaintext, const void* ciphertext, size_t ciphertext_len,
    hipStream_t stream, EncryptionState& state);

// sizes are padded
void EncryptAsync(hip_launch_batch_t* batch, void* ciphertext, const void* plaintext, size_t sizeBytes,
    hipStream_t stream, EncryptionState& state);

// provide a ciphertext buffer of size FIXED_SIZE_FULL + crypto_aead_aes256gcm_ABYTES
void CPUEncrypt(void* ciphertext, const void* src, size_t size, EncryptionState& state);

void CPUDecrypt(void* dst, const void* ciphertext, size_t size, EncryptionState& state);

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
    hipStream_t stream);

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind);

} // namespace lgm
#endif // LGM_MEMCPY_H
