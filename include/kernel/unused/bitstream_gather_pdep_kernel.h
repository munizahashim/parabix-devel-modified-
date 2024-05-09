
#pragma once

#include <kernel/core/kernel.h>
#include <llvm/IR/Value.h>
#include <string>

namespace kernel {

class BitStreamGatherPDEPKernel final : public MultiBlockKernel {
public:
    BitStreamGatherPDEPKernel(KernelBuilder & b, const unsigned numberOfStream = 8, std::string name = "BitStreamGatherPDEPKernel");
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) final;
private:
    const unsigned mNumberOfStream;

    llvm::Value* fill_address(KernelBuilder & b, unsigned fw, unsigned field_count, llvm::Value* a);
};

}

