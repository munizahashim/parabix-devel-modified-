#pragma once

#include <kernel/core/kernel_compiler.h>

namespace kernel {

class BlockKernelCompiler : public KernelCompiler {
public:

    BlockKernelCompiler(BlockOrientedKernel * const kernel) noexcept;

    void generateMultiBlockLogic(BuilderRef b, llvm::Value * const numOfBlocks);

    void generateDefaultFinalBlockMethod(BuilderRef b);

protected:

    llvm::Value * getRemainingItems(BuilderRef b);

    void incrementCountableItemCounts(BuilderRef b);

    llvm::Value * getPopCountRateItemCount(BuilderRef b, const ProcessingRate & rate);

    void writeDoBlockMethod(BuilderRef b);

    void writeFinalBlockMethod(BuilderRef b, llvm::Value * remainingItems);

private:

    llvm::Function *            mDoBlockMethod;
    llvm::BasicBlock *          mStrideLoopBody;
    llvm::IndirectBrInst *      mStrideLoopBranch;
    llvm::PHINode *             mStrideLoopTarget;
    llvm::PHINode *             mStrideBlockIndex;
};

}

