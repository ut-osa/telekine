/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <nw_kern_info.h>

#include <cstddef>
#include <istream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

struct ihipModuleSymbol_t;
using hipFunction_t = ihipModuleSymbol_t*;

namespace std {
template <>
struct hash<hsa_agent_t> {
    size_t operator()(hsa_agent_t x) const { return hash<decltype(x.handle)>{}(x.handle); }
};
}  // namespace std

inline constexpr bool operator==(hsa_agent_t x, hsa_agent_t y) { return x.handle == y.handle; }

namespace hip_impl {
class Kernel_descriptor {
public:
    std::uint64_t kernel_object_{};
    amd_kernel_code_t const* kernel_header_{nullptr};
    std::string name_{};
    amd_kernel_code_t kernel_header_buffer;
    Kernel_descriptor() = default;
    Kernel_descriptor(std::uint64_t kernel_object, const std::string& name);
    Kernel_descriptor(const Kernel_descriptor&) = default;
    Kernel_descriptor(Kernel_descriptor&&) = default;
    ~Kernel_descriptor() = default;

    Kernel_descriptor& operator=(const Kernel_descriptor&) = default;
    Kernel_descriptor& operator=(Kernel_descriptor&&) = default;

    operator hipFunction_t() const {  // TODO: this is awful and only meant for illustration.
        return reinterpret_cast<hipFunction_t>(const_cast<Kernel_descriptor*>(this));
    }
};

const std::unordered_map<hsa_agent_t, std::vector<hsa_executable_t>>& executables();
const std::unordered_map<std::uintptr_t, std::string>& function_names();
std::unordered_map<std::string, void*>& globals();

hsa_executable_t load_executable(const std::string& file, hsa_executable_t executable,
                                 hsa_agent_t agent);

template <typename K, typename V>
class LockedMap {
private:
    std::mutex lock;
    std::unordered_map<K,V> _map;
public:
    LockedMap() : _map() {}
    void add(const K& k, const V& v) {
        std::lock_guard<std::mutex> lk(lock);
        _map.emplace(k, v);
    }
    void remove(const K& k) {
        std::lock_guard<std::mutex> lk(lock);
        auto it = _map.find(k);
        if (it != _map.end())
            _map.erase(it);
    }
    V get(const K& k) {
        std::lock_guard<std::mutex> lk(lock);
        auto it = _map.find(k);
        if (it == _map.end()) {
            std::printf("%s:%d no key\n", __FILE__, __LINE__);
            std::abort();
        }
        return it->second;
    }
    V* get_ptr(const K& k) {
        std::lock_guard<std::mutex> lk(lock);
        auto it = _map.find(k);
        if (it == _map.end()) {
           return nullptr;
        }
        return &it->second;
    }
};

class program_state {
private:
    static std::shared_ptr<program_state> instance;
    static std::mutex init_mutex;
    std::mutex stream_agent_mutex;
    program_state();

    /* no copies */
    program_state& operator=(const program_state&) = delete;
    program_state(const program_state &) = delete;

public:
    std::unordered_map<std::uintptr_t, std::vector<std::pair<hsa_agent_t, hipFunction_t>>> functions;

    LockedMap<hipStream_t, hsa_agent_t> stream_to_agent;
    LockedMap<hipFunction_t, struct nw_kern_info> kern_info_cache;

    friend std::shared_ptr<program_state> program_state_handle();
};

std::shared_ptr<program_state> program_state_handle();

}  // Namespace hip_impl.
