#include "lgm_memcpy.hpp"
#include "command_scheduler.h"

namespace lgm {

static bool memcpy_size_fixed(void) {
  static bool ret = CHECK_ENV("LGM_MEMCPY_FIX_SIZE");
  return ret;
}

EncryptionState::EncryptionState(hipStream_t stream) {
  randombytes_buf(key, AES_KEYLEN);
  AES_GCM_init(&engine_device, key, stream);
  HIP_CHECK(hipMalloc(&nonce_device, crypto_aead_aes256gcm_NPUBBYTES));
  HIP_CHECK(nw_hipMemcpySync(nonce_device, nonce_host, crypto_aead_aes256gcm_NPUBBYTES,
      hipMemcpyHostToDevice, stream));
}

EncryptionState::~EncryptionState(void) {
  HIP_CHECK(hipFree(nonce_device));
  AES_GCM_destroy(engine_device);
}

void EncryptionState::nextNonceAsync(hipStream_t stream) {
  sodium_increment(nonce_host, crypto_aead_aes256gcm_NPUBBYTES); // TODO this is blocking
  AES_GCM_next_nonce(nonce_device, stream);
}

void DecryptAsync(void* plaintext, const void* ciphertext, size_t ciphertext_len,
    hipStream_t stream, EncryptionState& state) {
  AES_GCM_decrypt(static_cast<uint8_t*>(plaintext), state.engine_device,
      state.nonce_device, static_cast<const uint8_t*>(ciphertext),
      ciphertext_len - crypto_aead_aes256gcm_ABYTES, stream);
}

void EncryptAsync(void* ciphertext, const void* plaintext, size_t sizeBytes,
    hipStream_t stream, EncryptionState& state) {
  AES_GCM_encrypt(static_cast<uint8_t*>(ciphertext), state.engine_device,
      state.nonce_device, static_cast<const uint8_t*>(plaintext), sizeBytes, stream);
}

void CPUEncrypt(void* ciphertext, const void* src, size_t size, EncryptionState& state) {
  unsigned long long ciphertext_len;
  if (crypto_aead_aes256gcm_encrypt(static_cast<uint8_t*>(ciphertext), &ciphertext_len,
      static_cast<const uint8_t*>(src), size, NULL, 0, NULL, state.nonce_host, state.key) < 0) {
    throw std::runtime_error("failed to encrypt");
  }
}

void CPUDecrypt(void* dst, const void* ciphertext, size_t size, EncryptionState& state) {
  unsigned long long plaintext_len;
  if (crypto_aead_aes256gcm_decrypt(static_cast<uint8_t*>(dst), &plaintext_len, NULL,
        static_cast<const uint8_t*>(ciphertext), size, NULL, 0, state.nonce_host, state.key) < 0) {
    throw std::runtime_error("failed to decrypt");
  }
  assert(plaintext_len + crypto_aead_aes256gcm_ABYTES == size);
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
