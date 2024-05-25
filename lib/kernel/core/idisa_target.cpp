/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/core/idisa_target.h>

#include <toolchain/toolchain.h>
#include <idisa/idisa_i64_builder.h>
#ifdef PARABIX_ARM_TARGET
#include <idisa/idisa_arm_builder.h>
#endif
#ifdef PARABIX_X86_TARGET
#include <idisa/idisa_sse_builder.h>
#include <idisa/idisa_avx_builder.h>
#endif
#ifdef PARABIX_NVPTX_TARGET
#include <idisa/idisa_nvptx_builder.h>
#endif
#include <llvm/IR/Module.h>

#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
#include <llvm/TargetParser/Triple.h>
#else
#include <llvm/ADT/Triple.h>
#endif

#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/core/kernel_builder.h>

#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(17, 0, 0)
#include <llvm/TargetParser/Host.h>
#elif LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(11, 0, 0)
#include <llvm/Support/Host.h>
#endif

using namespace kernel;
using namespace llvm;

struct Features {
    bool hasAVX;
    bool hasAVX2;
    bool hasAVX512F;
    Features() : hasAVX(0), hasAVX2(0), hasAVX512F(0) { }
};

Features getHostCPUFeatures() {
    Features hostCPUFeatures;
    StringMap<bool> features;
    if (sys::getHostCPUFeatures(features)) {
        hostCPUFeatures.hasAVX = features.lookup("avx");
        hostCPUFeatures.hasAVX2 = features.lookup("avx2");
        hostCPUFeatures.hasAVX512F = features.lookup("avx512f");
    }
    return hostCPUFeatures;
}

bool ARM_available() {
    StringMap<bool> features;
    if (sys::getHostCPUFeatures(features)) {
        return features.lookup("neon");
    }
    return false;
}

bool SSSE3_available() {
    StringMap<bool> features;
    if (sys::getHostCPUFeatures(features)) {
        return features.lookup("ssse3");
    }
    return false;
}

bool BMI2_available() {
    // FIXME: Workaround to prevent this from returning true on AVX2 machines even when the SSE builder is made
    if (codegen::BlockSize < 256) return false;
    StringMap<bool> features;
    if (sys::getHostCPUFeatures(features)) {
        return features.lookup("bmi2");
    }
    return false;
}

bool AVX2_available() {
    StringMap<bool> features;
    if (sys::getHostCPUFeatures(features)) {
        return features.lookup("avx2");
    }
    return false;
}

bool AVX512BW_available() {
    StringMap<bool> features;
    if (sys::getHostCPUFeatures(features)) {
        return features.lookup("avx512bw");
    }
    return false;
}

namespace IDISA {

KernelBuilder * GetIDISA_Builder(llvm::LLVMContext & C) {
    if (codegen::BlockSize == 64) {
        return new KernelBuilderImpl<IDISA_I64_Builder>(C, codegen::BlockSize, codegen::LaneWidth);
    }
#ifdef PARABIX_ARM_TARGET
    if (LLVM_LIKELY(codegen::BlockSize == 0)) {  // No BlockSize override: use processor SIMD width
        codegen::BlockSize = 128;
    }
    if (ARM_available()) return new KernelBuilderImpl<IDISA_ARM_Builder>(C, codegen::BlockSize, codegen::LaneWidth);
#endif
#ifdef PARABIX_X86_TARGET
    const auto hostCPUFeatures = getHostCPUFeatures();
    if (LLVM_LIKELY(codegen::BlockSize == 0)) {  // No BlockSize override: use processor SIMD width
        if (hostCPUFeatures.hasAVX512F) codegen::BlockSize = 512;
        else
        if (hostCPUFeatures.hasAVX2) codegen::BlockSize = 256;
        else codegen::BlockSize = 128;
    }
    else if (((codegen::BlockSize & (codegen::BlockSize - 1)) != 0) || (codegen::BlockSize < 64)) {
        llvm::report_fatal_error("BlockSize must be a power of 2 and >=64");
    }
    if (codegen::BlockSize >= 512) {
        // AVX512BW builder can only be used for BlockSize multiples of 512
        if (hostCPUFeatures.hasAVX512F) {
            return new KernelBuilderImpl<IDISA_AVX512F_Builder>(C, codegen::BlockSize, codegen::LaneWidth);
        }
    }
    if (codegen::BlockSize >= 256) {
        // AVX2 or AVX builders can only be used for BlockSize multiples of 256
        if (hostCPUFeatures.hasAVX2) {
            return new KernelBuilderImpl<IDISA_AVX2_Builder>(C, codegen::BlockSize, codegen::LaneWidth);
        } else if (hostCPUFeatures.hasAVX) {
            return new KernelBuilderImpl<IDISA_AVX_Builder>(C, codegen::BlockSize, codegen::LaneWidth);
        }
    }
    if (codegen::BlockSize == 128) {
        if (SSSE3_available()) return new KernelBuilderImpl<IDISA_SSSE3_Builder>(C, codegen::BlockSize, codegen::LaneWidth);
        return new KernelBuilderImpl<IDISA_SSE2_Builder>(C, codegen::BlockSize, codegen::LaneWidth);
    }
#else
    llvm::errs() << "No PARABIX_X86_TARGET!\n";
#endif
    llvm::errs() << "BlockSize 64 default!\n";
    codegen::BlockSize = 64;
    return new KernelBuilderImpl<IDISA_I64_Builder>(C, codegen::BlockSize, codegen::LaneWidth);
}
#ifdef PARABIX_NVPTX_TARGET
KernelBuilder * GetIDISA_GPU_Builder(llvm::LLVMContext & C) {
    return new KernelBuilderImpl<IDISA_NVPTX20_Builder>(C, 64 * 64, 64);
}
#endif
}
