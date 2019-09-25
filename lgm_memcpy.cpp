#include "lgm_memcpy.hpp"
#include "command_scheduler.h"
#include <openssl/rand.h>

namespace lgm {

static bool memcpy_size_fixed(void) {
  static bool ret = CHECK_ENV("LGM_MEMCPY_FIX_SIZE", false);
  return ret;
}

EncryptionState::EncryptionState(hipStream_t stream) {
  assert(RAND_bytes(key, AES_KEYLEN) == 1);
  AES_GCM_init(&engine_device, key, stream);
  HIP_CHECK(hipMalloc(&nonce_device, crypto_aead_aes256gcm_NPUBBYTES));
  HIP_CHECK(nw_hipMemcpySync(nonce_device, nonce_host, crypto_aead_aes256gcm_NPUBBYTES,
      hipMemcpyHostToDevice, stream));
  encrypt_ctx = EVP_CIPHER_CTX_new();
  assert(encrypt_ctx != NULL);
  decrypt_ctx = EVP_CIPHER_CTX_new();
  assert(decrypt_ctx != NULL);
}

EncryptionState::~EncryptionState(void) {
  HIP_CHECK(hipFree(nonce_device));
  AES_GCM_destroy(engine_device);
  EVP_CIPHER_CTX_free(encrypt_ctx);
  EVP_CIPHER_CTX_free(decrypt_ctx);
}

void EncryptionState::nextNonceCPU() {
  int i = 0;
  while (i < 12) {
      nonce_host[i]++;
      if (nonce_host[i] > 0) break;
      i++;
  }
}

void EncryptionState::nextNonceGPU(hip_launch_batch_t* batch, hipStream_t stream) {
  AES_GCM_next_nonce(batch, nonce_device, stream);
}

void DecryptAsync(hip_launch_batch_t* batch, void* plaintext, const void* ciphertext, size_t ciphertext_len,
    hipStream_t stream, EncryptionState& state) {
  AES_GCM_decrypt(batch, static_cast<uint8_t*>(plaintext), state.engine_device,
      state.nonce_device, static_cast<const uint8_t*>(ciphertext),
      ciphertext_len - crypto_aead_aes256gcm_ABYTES, stream);
}

void EncryptAsync(hip_launch_batch_t* batch, void* ciphertext, const void* plaintext, size_t sizeBytes,
    hipStream_t stream, EncryptionState& state) {
  AES_GCM_encrypt(batch, static_cast<uint8_t*>(ciphertext), state.engine_device,
      state.nonce_device, static_cast<const uint8_t*>(plaintext), sizeBytes, stream);
}

void CPUEncrypt(void* ciphertext, const void* src, size_t size, EncryptionState& state) {
  assert(EVP_EncryptInit_ex(state.encrypt_ctx, EVP_aes_256_gcm(), NULL, state.key, state.nonce_host) == 1);
  int len;
  assert(EVP_EncryptUpdate(state.encrypt_ctx, (unsigned char*)ciphertext, &len, (const unsigned char*)src, size) == 1);
  assert(EVP_EncryptFinal_ex(state.encrypt_ctx, (unsigned char*)ciphertext + len, &len) == 1);
  assert(EVP_CIPHER_CTX_ctrl(state.encrypt_ctx, EVP_CTRL_GCM_GET_TAG, 16, (unsigned char*)ciphertext + size) == 1);
}

void CPUDecrypt(void* dst, const void* ciphertext, size_t size, EncryptionState& state) {
  assert(EVP_DecryptInit_ex(state.decrypt_ctx, EVP_aes_256_gcm(), NULL, state.key, state.nonce_host) == 1);
  int len;
  assert(EVP_DecryptUpdate(state.decrypt_ctx, (unsigned char*)dst, &len, (const unsigned char*)ciphertext, size - 16) == 1);
  unsigned char mac_copy[16];
  memcpy(mac_copy, (unsigned char*)ciphertext + size - 16, 16);
  assert(EVP_CIPHER_CTX_ctrl(state.decrypt_ctx, EVP_CTRL_GCM_SET_TAG, 16, mac_copy) == 1);
  if (EVP_DecryptFinal_ex(state.decrypt_ctx, (unsigned char*)dst + len, &len) != 1) {
    throw std::runtime_error("failed to decrypt"); 
  }
}

static hipError_t fixedSizeHipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
    hipMemcpyKind kind, hipStream_t stream) {
  hipError_t ret;
  for (size_t i = 0; i < sizeBytes; i+= FIXED_SIZE_B) {
    size_t memcpy_size = std::min(sizeBytes - i, FIXED_SIZE_B);
    ret = CommandScheduler::GetForStream(stream)->AddMemcpyAsync(
        static_cast<void*>(static_cast<char*>(dst) + i),
        static_cast<const void*>(static_cast<const char*>(src) + i), memcpy_size, kind);
    if (ret != hipSuccess) break;
  }
  return ret;
}

hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
    hipStream_t stream) {
  if (memcpy_size_fixed() &&
      (kind == hipMemcpyDeviceToHost || kind == hipMemcpyHostToDevice)) {
    return fixedSizeHipMemcpyAsync(dst, src, sizeBytes, kind, stream);
  } else {
    return CommandScheduler::GetForStream(stream)->AddMemcpyAsync(dst, src, sizeBytes, kind);
  }
}

hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  HIP_CHECK(lgm::hipMemcpyAsync(dst, src, sizeBytes, kind, nullptr));
  // async memcpy on the default stream
  // then synchronize because this API is blocking from the application's perspective
  return hipStreamSynchronize(nullptr);
}
} // namespace lgm
