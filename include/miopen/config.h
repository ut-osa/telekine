/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef GUARD_CONFIG_H_IN
#define GUARD_CONFIG_H_IN

#define MIOPEN_BACKEND_OPENCL 0
#define MIOPEN_BACKEND_HCC 0
#define MIOPEN_BACKEND_HIP 1
#define MIOPEN_USE_MIOPENGEMM 1
#define MIOPEN_BUILD_DEV 0
#define MIOPEN_GPU_SYNC 0

#define MIOPEN_AMDGCN_ASSEMBLER "/opt/rocm/opencl/bin/x86_64/clang"
#define HIP_OC_COMPILER "/opt/rocm/bin/clang-ocl"
#define MIOPEN_CACHE_DIR "~/.cache/miopen/"

#endif
