/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <llvm/Support/Compiler.h>

namespace llvm { class LLVMContext; }
namespace kernel { class KernelBuilder; }

extern LLVM_READNONE bool BMI2_available();
extern LLVM_READNONE bool AVX2_available();
extern LLVM_READNONE bool AVX512BW_available();

namespace IDISA {
    
kernel::KernelBuilder * GetIDISA_Builder(llvm::LLVMContext & C);

#ifdef CUDA_ENABLED
kernel::KernelBuilder * GetIDISA_GPU_Builder(llvm::LLVMContext & C);
#endif
}

