#pragma once

#include <kernel/core/kernel.h>

namespace kernel {

class PopCountKernel final : public MultiBlockKernel {
public:

    static bool classof(const Kernel * const k) {
        return k->getTypeId() == TypeId::PopCountKernel;
    }

    enum PopCountType { POSITIVE, NEGATIVE, BOTH };

    explicit PopCountKernel(KernelBuilder & b, const PopCountType type, const unsigned stepFactor, StreamSet * input, StreamSet * const output);

    explicit PopCountKernel(KernelBuilder & b, const PopCountType type, const unsigned stepFactor, StreamSet * input, StreamSet * const positive, StreamSet * negative);

    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) final;

private:

    const PopCountType mType;
};

}

