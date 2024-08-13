#pragma once

#include <kernel/core/kernel.h>

namespace kernel {

class ZeroExtend final : public MultiBlockKernel {
public:
    ZeroExtend(LLVMTypeSystemInterface & ts,
               StreamSet * const input, StreamSet * const output);
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
};

}

