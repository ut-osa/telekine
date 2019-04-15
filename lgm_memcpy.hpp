#ifndef LGM_MEMCPY_H
#define LGM_MEMCPY_H

#include "command_scheduler.h"
#include "check_env.h"

#include "crypto/aes_gcm.h"
#include <algorithm>
#include <hip/hip_runtime_api.h>
#include <map>
#include <memory>
#include <stdio.h>
#include <sodium.h>
#include <unistd.h>

static bool memcpy_size_fixed(void) {
  static bool ret = CHECK_ENV("LGM_MEMCPY_FIX_SIZE");
  return ret;
}

static bool memcpy_encryption_enabled(void) {
  static bool ret = CHECK_ENV("LGM_MEMCPY_ENABLE_ENCRYPTION");
  return ret;
}

// AES Encryption State //
template <size_t TRANSFER_UNIT_SIZE>
struct EncryptionState {
  struct Nonce {
    uint8_t bytes[crypto_aead_aes256gcm_NPUBBYTES];
  };

  struct DeviceState {
    int device;
    std::map<hipStream_t, Nonce> _nonce_host;
    std::map<hipStream_t, uint8_t*> _nonce_device;
    AES_GCM_engine* engine_device;

    DeviceState(void) {}

    DeviceState(int device, uint8_t key[AES_KEYLEN], hipStream_t stream) : device(device) {
      int current_device;
      HIP_CHECK(hipGetDevice(&current_device));
      HIP_CHECK(hipSetDevice(device));
      AES_GCM_init(&engine_device, key, stream);
      HIP_CHECK(hipSetDevice(current_device));
    }

    ~DeviceState(void) {
      int current_device;
      HIP_CHECK(hipGetDevice(&current_device));
      for (auto& n : _nonce_device) HIP_CHECK(hipFree(n.second));
      AES_GCM_destroy(engine_device);
      HIP_CHECK(hipSetDevice(current_device));
    }

    uint8_t* nonce(hipStream_t stream) {
      if (_nonce_host.find(stream) != _nonce_host.end())
        return _nonce_host.at(stream).bytes;
      else {
        _nonce_host.emplace(stream, Nonce{});
        randombytes_buf(_nonce_host.at(stream).bytes, crypto_aead_aes256gcm_NPUBBYTES);
        return _nonce_host.at(stream).bytes;
      }
    }

    uint8_t* nonce_device(hipStream_t stream) {
      if (_nonce_device.find(stream) != _nonce_device.end())
        return _nonce_device.at(stream);
      else {
        int current_device;
        uint8_t* _nonce;
        HIP_CHECK(hipGetDevice(&current_device));
        HIP_CHECK(hipMalloc(&_nonce, crypto_aead_aes256gcm_NPUBBYTES));
        HIP_CHECK(nw_hipMemcpySync(_nonce, nonce(stream), crypto_aead_aes256gcm_NPUBBYTES,
              hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipSetDevice(current_device));
        _nonce_device.emplace(stream, _nonce);
        return _nonce_device.at(stream);
      }
    }

    void nextNonceAsync(hipStream_t stream) {
      int current_device;
      sodium_increment(nonce(stream), crypto_aead_aes256gcm_NPUBBYTES); // TODO this is blocking
      HIP_CHECK(hipGetDevice(&current_device));
      AES_GCM_next_nonce(_nonce_device.at(stream), stream);
      HIP_CHECK(hipSetDevice(current_device));
    }
  };

  const hipStream_t stream;
  std::map<int, DeviceState> dstate;
  uint8_t key[AES_KEYLEN];

  EncryptionState(hipStream_t stream) : stream(stream) {
    randombytes_buf(key, AES_KEYLEN);
    int device_count = 1;
    //HIP_CHECK(hipGetDeviceCount(&device_count));
    // TODO enable when multiple devices are supported
    for (int i = 0; i < device_count; i++) {
      dstate.emplace(std::piecewise_construct, std::forward_as_tuple(i),
          std::forward_as_tuple(i, key, stream));
    }
  }

  const AES_GCM_engine* engine_device(int device = -1) {
    if (device == -1) HIP_CHECK(hipGetDevice(&device));
    return dstate.at(device).engine_device;
  }

  uint8_t* nonce(int device = -1) {
    if (device == -1) HIP_CHECK(hipGetDevice(&device));
    return dstate.at(device).nonce(stream);
  }

  uint8_t* nonce_device(int device = -1) {
    if (device == -1) HIP_CHECK(hipGetDevice(&device));
    return dstate.at(device).nonce_device(stream);
  }

  void nextNonceAsync(int device = -1) {
    if (device == -1) HIP_CHECK(hipGetDevice(&device));
    dstate.at(device).nextNonceAsync(stream);
  }
};

// One encryption state per thread
static thread_local std::unique_ptr<EncryptionState<FIXED_SIZE_B>> _state;
// Constructs state on demand
static EncryptionState<FIXED_SIZE_B>& state(hipStream_t stream) {
  if (!_state)
    _state = std::unique_ptr<EncryptionState<FIXED_SIZE_B>>(
        new EncryptionState<FIXED_SIZE_B>(stream));
  return *_state;
}

static std::map<const void*, int> ptrDevice; // Global map from memory allocations to devices

#define lgmEncPad(sizeBytes) \
  (((sizeBytes + (AES_BLOCKLEN - 1)) / AES_BLOCKLEN) * AES_BLOCKLEN)

// ciphertext_len includes MAC
void lgmDecryptAsync(void* ciphertext, size_t ciphertext_len, hipStream_t stream) {
  AES_GCM_decrypt(state(stream).engine_device(), state(stream).nonce_device(),
      static_cast<uint8_t*>(ciphertext), ciphertext_len - crypto_aead_aes256gcm_ABYTES,
      &static_cast<uint8_t*>(ciphertext)[ciphertext_len - crypto_aead_aes256gcm_ABYTES],
      stream);
}

// sizes are padded
void lgmEncryptAsync(void* buf, size_t sizeBytes, hipStream_t stream) {
  AES_GCM_encrypt(state(stream).engine_device(), state(stream).nonce_device(),
      static_cast<uint8_t*>(buf), sizeBytes, &static_cast<uint8_t*>(buf)[sizeBytes], stream);
}

// provide a ciphertext buffer of size lgmEncPad(size) + crypto_aead_aes256gcm_ABYTES
void lgmCPUEncrypt(void* ciphertext, const void* src, size_t size, hipStream_t stream) {
  unsigned long long ciphertext_len;
  if (crypto_aead_aes256gcm_encrypt(static_cast<uint8_t*>(ciphertext), &ciphertext_len,
      static_cast<const uint8_t*>(src), size, NULL, 0, NULL, state(stream).nonce(),
      state(stream).key) < 0) {
    throw std::runtime_error("failed to encrypt");
  }
}

void lgmCPUDecrypt(void* dst, const void* ciphertext, size_t size, hipStream_t stream) {
  unsigned long long plaintext_len;
  if (crypto_aead_aes256gcm_decrypt(static_cast<uint8_t*>(dst), &plaintext_len, NULL,
        static_cast<const uint8_t*>(ciphertext), size, NULL, 0, state(stream).nonce(),
        state(stream).key) < 0) {
    throw std::runtime_error("failed to decrypt");
  }
  assert(plaintext_len + crypto_aead_aes256gcm_ABYTES == size);
}

void lgmNextNonceAsync(hipStream_t stream, int device=-1) {
  state(stream).nextNonceAsync();
}

#if 0
hipError_t lgmEncMemcpyAsyncD2D(void* dst, const void* src, size_t sizeBytes, hipStream_t stream) {
  int dst_device(ptrDevice.at(dst));
  int src_device(ptrDevice.at(src));
  int current_device;
  HIP_CHECK(hipGetDevice(&current_device));
  const size_t paddedSize(((sizeBytes + (AES_BLOCKLEN - 1)) / AES_BLOCKLEN) * AES_BLOCKLEN);
  // Move data to staging buffer
  HIP_CHECK(nw_hipMemcpySync(state().ciphertext_device(stream, src_device), src, sizeBytes,
        hipMemcpyDeviceToDevice, stream));
  // Encrypt on GPU
  HIP_CHECK(hipSetDevice(src_device));
  AES_GCM_encrypt(state().engine_device(src_device), state().nonce_device(stream, src_device),
      state().ciphertext_device(stream, src_device), paddedSize,
      &state().ciphertext_device(stream, src_device)[paddedSize], stream);
  // Update nonce
  state().nextNonceAsync(stream, src_device);
  // Copy to other GPU
  HIP_CHECK(nw_hipMemcpySync(state().ciphertext_device(nullptr, dst_device),
        state().ciphertext_device(nullptr, src_device), paddedSize + crypto_aead_aes256gcm_ABYTES,
        hipMemcpyDeviceToDevice, stream));
  // wait until data has been copied to other GPU, we can't move past this point until the data is
  // present.
  hipError_t ret = hipStreamSynchronize(stream);
  if (ret != hipSuccess) return ret;
  // Decrypt on GPU
  HIP_CHECK(hipSetDevice(dst_device));
  // we don't have a valid stream for the other GPU, so we are forced to use the
  // default stream after this point
  AES_GCM_decrypt(state().engine_device(dst_device), state().nonce_device(nullptr, dst_device),
      state().ciphertext_device(nullptr, dst_device), paddedSize,
      &state().ciphertext_device(nullptr, dst_device)[paddedSize], nullptr); // execute on null stream
  // Update nonce
  state().nextNonceAsync(nullptr, dst_device); // execute on null stream
  // Move from staging buffer to real memory
  HIP_CHECK(nw_hipMemcpySync(dst, state().ciphertext_device(nullptr, dst_device), sizeBytes,
        hipMemcpyDeviceToDevice, nullptr)); // execute on null stream
  // Reset to original device
  HIP_CHECK(hipSetDevice(current_device));
  return ret;
}
#endif
template <size_t SIZE>
static hipError_t fixedSizeHipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
    hipMemcpyKind kind, hipStream_t stream) {
  hipError_t ret;
  for (size_t i = 0; i < sizeBytes; i+= SIZE) {
    size_t memcpy_size = std::min(sizeBytes - i, SIZE);
    ret = CommandScheduler::GetForStream(nullptr)->AddMemcpyAsync(
        static_cast<void*>(static_cast<char*>(dst) + i),
        static_cast<const void*>(static_cast<const char*>(src) + i), memcpy_size, kind);
    if (ret != hipSuccess) break;
  }
  return ret;
}

// TODO for multi-gpu support, register location of all hipmallocs with this method
void lgm_register_gpu_ptr(void *ptr) {
  int device;
  HIP_CHECK(hipGetDevice(&device));
  ptrDevice.emplace(ptr, device); // Record map from allocation to device
}

template <size_t SIZE = FIXED_SIZE_B>
hipError_t lgmMemcpyAsync(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
    hipStream_t stream) {
  if (memcpy_size_fixed()) {  // split memcpys into fixed size batches
    hipError_t ret;
    for (size_t i = 0; i < sizeBytes; i+= SIZE) {
      size_t memcpy_size = std::min(sizeBytes - i, SIZE);
      ret = CommandScheduler::GetForStream(stream)->AddMemcpyAsync(
          static_cast<void*>(static_cast<char*>(dst) + i),
          static_cast<const void*>(static_cast<const char*>(src) + i), memcpy_size, kind);
      if (ret != hipSuccess) break;
    }
    return ret;
  } else {
    return CommandScheduler::GetForStream(stream)->AddMemcpyAsync(dst, src, sizeBytes, kind);
  }
}

hipError_t lgmMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind) {
  HIP_CHECK(lgmMemcpyAsync(dst, src, sizeBytes, kind, nullptr));
  // async memcpy on the default stream
  // then synchronize because this API is blocking from the application's perspective
  return hipStreamSynchronize(nullptr);
}

#endif // LGM_MEMCPY_H
