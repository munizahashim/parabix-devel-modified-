/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <kernel/core/kernel.h>

namespace kernel {

/**
 * Abstract scanning kernel class. Implements a basic scanning framework,
 * delegating procssing logic on scan bits to subclasses.
 * 
 * Accepts only a single scanning stream (i.e., <i1>[1]).
 */
class SingleStreamScanKernelTemplate : public MultiBlockKernel {
public:

    SingleStreamScanKernelTemplate(KernelBuilder & b, std::string && name, StreamSet * scan);

    virtual ~SingleStreamScanKernelTemplate() {}

    static const uint32_t MaxStrideWidth;

protected:

    struct ScanWordContext {
        const unsigned strideMaskWidth = 64;
        const unsigned minScanWordWidth = 8;

        const unsigned width;
        const unsigned wordsPerBlock;
        const unsigned wordsPerStride;
        const unsigned fieldWidth;

        llvm::Type * const Ty;
        llvm::Type * const PointerTy;
        llvm::Type * const StrideMaskTy;

        llvm::Constant * const WIDTH;
        llvm::Constant * const WORDS_PER_BLOCK;
        llvm::Constant * const WORDS_PER_STRIDE;
        llvm::Constant * const NUM_BLOCKS_PER_STRIDE;

        ScanWordContext(KernelBuilder & b, unsigned strideWidth);
    };

    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) final override;

    virtual void initialize(KernelBuilder & b) { }

    virtual void willProcessStride(KernelBuilder & b, llvm::Value * const strideNo) { }

    virtual void maskBuildingIterationHead(KernelBuilder & b) { }

    virtual void maskBuildingIterationBody(KernelBuilder & b, llvm::Value * const blockIndex) { }

    virtual void didBuildMask(KernelBuilder & b, llvm::Value * const mask) { }

    virtual void willProcessWord(KernelBuilder & b, llvm::Value * const word) { }

    virtual void generateProcessingLogic(KernelBuilder & b, llvm::Value * const absoluteIndex) { }

    virtual void generateProcessingLogic(KernelBuilder & b, llvm::Value * const absoluteIndex, llvm::Value * const blockIndex, llvm::Value * const bitOffset) { 
        generateProcessingLogic(b, absoluteIndex);
    }

    virtual void didProcessWord(KernelBuilder & b) { }

    virtual void didProcessStride(KernelBuilder & b, llvm::Value * strideNo) { }

    virtual void finalize(KernelBuilder & b) { }

    ScanWordContext mSW;
    llvm::BasicBlock * mEntryBlock = nullptr;
    llvm::BasicBlock * mStrideStart = nullptr;
    llvm::BasicBlock * mBuildStrideMask = nullptr;
    llvm::BasicBlock * mMaskReady = nullptr;
    llvm::BasicBlock * mProcessMask = nullptr;
    llvm::BasicBlock * mProcessWord = nullptr;
    llvm::BasicBlock * mWordDone = nullptr;
    llvm::BasicBlock * mStrideDone = nullptr;
    llvm::BasicBlock * mExitBlock = nullptr;
    llvm::Value * mStrideNo = nullptr;
    llvm::Value * mBlockNo = nullptr;
    llvm::Value * mWordOffset = nullptr;
    llvm::Value * mProcessingMask = nullptr;
    llvm::Value * mProcessingWord = nullptr;
};

}
