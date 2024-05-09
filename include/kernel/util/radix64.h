/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>

namespace llvm { class Module; }
namespace llvm { class Value; }

namespace IDISA { class IDISA_Builder; }

namespace kernel {

/*  expand3_4 transforms a byte sequence by duplicating every third byte.
    Each 3 bytes of the input abc produces a 4 byte output abcc.
    This is a useful preparatory transformation in various radix-64 encodings. */

class expand3_4Kernel final : public MultiBlockKernel {
public:
    expand3_4Kernel(KernelBuilder & b, StreamSet * input, StreamSet * expandedOutput);
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

class radix64Kernel final : public BlockOrientedKernel {
public:
    radix64Kernel(KernelBuilder &, StreamSet * input, StreamSet * output);
private:
    virtual void generateDoBlockMethod(KernelBuilder & b) override;
    virtual void generateFinalBlockMethod(KernelBuilder & b, llvm::Value * remainingBytes) override;
    llvm::Value * processPackData(KernelBuilder & b, llvm::Value* packData) const;
};

class base64Kernel final : public BlockOrientedKernel {
public:
    base64Kernel(KernelBuilder &, StreamSet * input, StreamSet * output);
private:
    virtual void generateDoBlockMethod(KernelBuilder & b) override;
    virtual void generateFinalBlockMethod(KernelBuilder & b, llvm::Value * remainingBytes) override;
    llvm::Value* processPackData(KernelBuilder & b, llvm::Value* packData) const;
};

}
