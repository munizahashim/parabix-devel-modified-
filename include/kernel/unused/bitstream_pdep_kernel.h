/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>
#include <llvm/IR/Value.h>
#include <string>

namespace kernel {

class BitStreamPDEPKernel final : public MultiBlockKernel {
public:
    BitStreamPDEPKernel(BuilderRef b, const unsigned numberOfStream = 8, std::string name = "BitStreamPDEPKernel");
protected:
    void generateMultiBlockLogic(BuilderRef b, llvm::Value * const numOfStrides) final;
private:
    const unsigned mNumberOfStream;
};

}

