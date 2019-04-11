#ifndef LGM_MEMCPY_H
#define LGM_MEMCPY_H

#include "hip_cpp_bridge.h"
#include "crypto/aes_gcm.h"

#include <algorithm>
#include <hip/hip_runtime_api.h>
#include <map>
#include <stdio.h>
#include <sodium.h>
#include <unistd.h>

#define FIXED_MEMCPY_SIZE_B (1048576ul) // 1 MB
#define SPLIT_MEMCPY 1
#define ENCRYPTION_BLOCK_SIZE 64
#define MAX_DEVICES 4

static bool memcpy_size_fixed(void) {
  return getenv("LGM_MEMCPY_FIX_SIZE") != nullptr;
}

static bool memcpy_encryption_enabled(void) {
  return getenv("LGM_MEMCPY_ENABLE_ENCRYPTION") != nullptr;
}

// AES Encryption State //
struct EncryptionState {
  struct Nonce {
    uint8_t bytes[crypto_aead_aes256gcm_NPUBBYTES];
  };

  struct DeviceState {
    int device;
    std::map<hipStream_t, Nonce> _nonce_host;
    std::map<hipStream_t, uint8_t*> _nonce_device;
    std::map<hipStream_t, uint8_t*> _ciphertext_device;
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
      for (auto& ct : _ciphertext_device) HIP_CHECK(hipFree(ct.second));
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
        HIP_CHECK(nw_hipMemcpyAsync(_nonce, nonce(stream), crypto_aead_aes256gcm_NPUBBYTES,
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

    uint8_t* ciphertext_device(hipStream_t stream) {
      if (_ciphertext_device.find(stream) != _ciphertext_device.end())
        return _ciphertext_device.at(stream);
      else {
        int current_device;
        uint8_t* ciphertext;
        HIP_CHECK(hipGetDevice(&current_device));
        HIP_CHECK(hipMalloc(&ciphertext, FIXED_MEMCPY_SIZE_B + crypto_aead_aes256gcm_ABYTES));
        HIP_CHECK(hipSetDevice(current_device));
        _ciphertext_device.emplace(stream, ciphertext);
        return _ciphertext_device.at(stream);
      }
    }
  };

  std::map<int, DeviceState> dstate;
  bool initialized = false;
  uint8_t key[AES_KEYLEN];
  uint8_t *ciphertext;

  EncryptionState(void) {
    randombytes_buf(key, AES_KEYLEN);
    ciphertext = new uint8_t[FIXED_MEMCPY_SIZE_B + crypto_aead_aes256gcm_ABYTES]; // TODO remove me
    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    assert(device_count <= MAX_DEVICES);
    for (int i = 0; i < device_count; i++) {
      dstate.emplace(std::piecewise_construct, std::forward_as_tuple(i),
          std::forward_as_tuple(i, key, nullptr)); // initialize on default stream
    }
    initialized = true;
  }

  ~EncryptionState(void) {
    delete[] ciphertext;
  }

  const AES_GCM_engine* engine_device(int device = -1) {
    if (device == -1) HIP_CHECK(hipGetDevice(&device));
    return dstate.at(device).engine_device;
  }

  uint8_t* nonce(hipStream_t stream, int device = -1) {
    if (device == -1) HIP_CHECK(hipGetDevice(&device));
    return dstate.at(device).nonce(stream);
  }

  uint8_t* nonce_device(hipStream_t stream, int device = -1) {
    if (device == -1) HIP_CHECK(hipGetDevice(&device));
    return dstate.at(device).nonce_device(stream);
  }

  void nextNonceAsync(hipStream_t stream, int device = -1) {
    if (device == -1) HIP_CHECK(hipGetDevice(&device));
    dstate.at(device).nextNonceAsync(stream);
  }

  uint8_t* ciphertext_device(hipStream_t stream, int device = -1) {
    if (device == -1) HIP_CHECK(hipGetDevice(&device));
    return dstate.at(device).ciphertext_device(stream);
  }
};

static thread_local EncryptionState state; // One encryption state per thread
static std::map<const void*, int> ptrDevice; // Global map from memory allocations to devices

// Helper functions //
static hipError_t sendAsync(void* dst, const void* src, size_t sizeBytes, hipStream_t stream) {
  hipError_t ret = hipSuccess;
  const size_t paddedSize((sizeBytes + (ENCRYPTION_BLOCK_SIZE - 1)) / ENCRYPTION_BLOCK_SIZE);
  uint8_t *tmp(nullptr);
  if (sizeBytes < paddedSize) {
    // Copy src to staging buffer because crypto_aead_aes256gcm will read past end
    tmp = new uint8_t[paddedSize];
    memcpy(tmp, src, sizeBytes);
    src = tmp;
  }
  // Encrypt on CPU
  unsigned long long ciphertext_len;
  if (crypto_aead_aes256gcm_encrypt(state.ciphertext, &ciphertext_len,
        static_cast<const uint8_t*>(src), paddedSize, NULL, 0, NULL, state.nonce(stream),
        state.key) < 0) {
    throw std::runtime_error("failed to encrypt");
  }
  // TODO crypto_aead_aes256gcm_encrypt blocks forward progress
  // Copy to GPU
  ret = nw_hipMemcpyAsync(state.ciphertext_device(stream), state.ciphertext, ciphertext_len,
      hipMemcpyHostToDevice, stream);
  // Decrypt on GPU
  AES_GCM_decrypt(state.engine_device(), state.nonce_device(stream), state.ciphertext_device(stream),
      ciphertext_len - crypto_aead_aes256gcm_ABYTES,
      &state.ciphertext_device(stream)[ciphertext_len - crypto_aead_aes256gcm_ABYTES], stream);
  // Update nonce
  state.nextNonceAsync(stream);
  // Move from staging buffer to real memory
  HIP_CHECK(nw_hipMemcpyAsync(dst, state.ciphertext_device(stream), sizeBytes,
        hipMemcpyDeviceToDevice, stream));
  // Clean up padded input
  if (tmp) delete[] tmp;
  return ret;
}

static hipError_t receiveAsync(void* dst, const void* src, size_t sizeBytes, hipStream_t stream) {
  const size_t paddedSize((sizeBytes + (ENCRYPTION_BLOCK_SIZE - 1)) / ENCRYPTION_BLOCK_SIZE);
  // Copy to staging buffer
  HIP_CHECK(nw_hipMemcpyAsync(state.ciphertext_device(stream), src, sizeBytes,
        hipMemcpyDeviceToDevice, stream));
  // Encrypt on GPU
  // Write mac at end of ciphertext
  AES_GCM_encrypt(state.engine_device(), state.nonce_device(stream), state.ciphertext_device(stream),
      paddedSize, &state.ciphertext_device(stream)[paddedSize], stream);
  // Update nonce
  state.nextNonceAsync(stream);
  // Copy to CPU
  HIP_CHECK(nw_hipMemcpyAsync(state.ciphertext, state.ciphertext_device(stream), paddedSize +
        crypto_aead_aes256gcm_ABYTES, hipMemcpyDeviceToHost, stream));
  // TODO add a future and copy data asynchronously
  hipError_t ret = hipStreamSynchronize(stream);
  if (ret != hipSuccess) return ret;
  // Decrypt on CPU
  uint8_t *plaintext = new uint8_t[paddedSize];
  unsigned long long plaintext_len;
  if (crypto_aead_aes256gcm_decrypt(plaintext, &plaintext_len, NULL, state.ciphertext,
        paddedSize + crypto_aead_aes256gcm_ABYTES, NULL, 0, state.nonce(stream), state.key) < 0) {
    throw std::runtime_error("failed to decrypt");
  }
  memcpy(dst, plaintext, sizeBytes);
  delete[] plaintext;
  return ret;
}

static hipError_t sendReceiveAsync(void* dst, const void* src, size_t sizeBytes, hipStream_t stream) {
  int dst_device(ptrDevice.at(dst));
  int src_device(ptrDevice.at(src));
  int current_device;
  HIP_CHECK(hipGetDevice(&current_device));
  const size_t paddedSize((sizeBytes + (ENCRYPTION_BLOCK_SIZE - 1)) / ENCRYPTION_BLOCK_SIZE);
  // Move data to staging buffer
  HIP_CHECK(nw_hipMemcpyAsync(state.ciphertext_device(stream, src_device), src, sizeBytes,
        hipMemcpyDeviceToDevice, stream));
  // Encrypt on GPU
  HIP_CHECK(hipSetDevice(src_device));
  AES_GCM_encrypt(state.engine_device(src_device), state.nonce_device(stream, src_device),
      state.ciphertext_device(stream, src_device), paddedSize,
      &state.ciphertext_device(stream, src_device)[paddedSize], stream);
  // Update nonce
  state.nextNonceAsync(stream, src_device);
  // Copy to other GPU
  HIP_CHECK(nw_hipMemcpyAsync(state.ciphertext_device(nullptr, dst_device),
        state.ciphertext_device(nullptr, src_device), paddedSize + crypto_aead_aes256gcm_ABYTES,
        hipMemcpyDeviceToDevice, stream));
  // wait until data has been copied to other GPU, we can't move past this point until the data is
  // present.
  hipError_t ret = hipStreamSynchronize(stream);
  if (ret != hipSuccess) return ret;
  // Decrypt on GPU
  HIP_CHECK(hipSetDevice(dst_device));
  // we don't have a valid stream for the other GPU, so we are forced to use the
  // default stream after this point
  AES_GCM_decrypt(state.engine_device(dst_device), state.nonce_device(nullptr, dst_device),
      state.ciphertext_device(nullptr, dst_device), paddedSize,
      &state.ciphertext_device(nullptr, dst_device)[paddedSize], nullptr); // execute on null stream
  // Update nonce
  state.nextNonceAsync(nullptr, dst_device); // execute on null stream
  // Move from staging buffer to real memory
  HIP_CHECK(nw_hipMemcpyAsync(dst, state.ciphertext_device(nullptr, dst_device), sizeBytes,
        hipMemcpyDeviceToDevice, nullptr)); // execute on null stream
  // Reset to original device
  HIP_CHECK(hipSetDevice(current_device));
  return ret;
}

template <typename F>
static hipError_t fixed_size_memcpy(F f, void* dst, const void* src, size_t sizeBytes,
    hipMemcpyKind kind, hipStream_t stream) {
  hipError_t ret;
  for (size_t i = 0; i < sizeBytes; i+= FIXED_MEMCPY_SIZE_B) {
    size_t memcpy_size = std::min(sizeBytes - i, FIXED_MEMCPY_SIZE_B);
    ret = f(static_cast<void*>(static_cast<char*>(dst) + i),
        static_cast<const void*>(static_cast<const char*>(src) + i), memcpy_size, kind, stream);
    if (ret != hipSuccess) break;
  }
  return ret;
}

static hipError_t encryptedMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
    hipMemcpyKind kind, hipStream_t stream) {
  switch (kind) {
    case hipMemcpyHostToDevice:
      return sendAsync(dst, src, sizeBytes, stream);
    case hipMemcpyDeviceToHost:
      return receiveAsync(dst, src, sizeBytes, stream);
    case hipMemcpyDeviceToDevice:
      return sendReceiveAsync(dst, src, sizeBytes, stream);
    default:
      throw std::runtime_error("unsupported memcpy type in encrypted memcpy");
  }
}

void lgm_register_gpu_ptr(void *ptr) {
  int device;
  HIP_CHECK(hipGetDevice(&device));
  ptrDevice.emplace(ptr, device); // Record map from allocation to device
}

hipError_t lgm_memcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
    hipStream_t stream) {
  if (memcpy_size_fixed() && memcpy_encryption_enabled()) {
    return fixed_size_memcpy(encryptedMemcpyAsync, dst, src, sizeBytes, kind, stream);
  } else if (memcpy_size_fixed()) {
    return fixed_size_memcpy(nw_hipMemcpyAsync, dst, src, sizeBytes, kind, stream);
  } else if (memcpy_encryption_enabled()) {
    return encryptedMemcpyAsync(dst, src, sizeBytes, kind, stream);
  } else {
    return nw_hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
  }
}

#endif // LGM_MEMCPY_H
