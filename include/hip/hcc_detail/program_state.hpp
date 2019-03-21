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
    std::uint64_t kernel_object_{};
    amd_kernel_code_t const* kernel_header_{nullptr};
    std::string name_{};
    amd_kernel_code_t kernel_header_buffer;
public:
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
const std::unordered_map<std::uintptr_t, std::vector<std::pair<hsa_agent_t, hipFunction_t>>>&
functions();
const std::unordered_map<std::uintptr_t, std::string>& function_names();
std::unordered_map<std::string, void*>& globals();

hsa_executable_t load_executable(const std::string& file, hsa_executable_t executable,
                                 hsa_agent_t agent);
}  // Namespace hip_impl.
