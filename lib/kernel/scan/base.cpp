/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/scan/base.h>

#include <kernel/core/kernel_builder.h>
#include <llvm/IR/Module.h>

#define IS_POW_2(i) ((i > 0) && ((i & (i - 1)) == 0))

using namespace llvm;

namespace kernel {

SingleStreamScanKernelTemplate::ScanWordContext::ScanWordContext(LLVMTypeSystemInterface & ts, unsigned strideWidth)
: width(std::max(minScanWordWidth, strideWidth / strideMaskWidth))
, wordsPerBlock(ts.getBitBlockWidth() / width)
, wordsPerStride(strideMaskWidth)
, fieldWidth(width)
, Ty(ts.getIntNTy(width))
, PointerTy(Ty->getPointerTo())
, StrideMaskTy(ts.getIntNTy(strideMaskWidth))
, WIDTH(ts.getSize(width))
, WORDS_PER_BLOCK(ts.getSize(wordsPerBlock))
, WORDS_PER_STRIDE(ts.getSize(wordsPerStride))
, NUM_BLOCKS_PER_STRIDE(ts.getSize(strideWidth / ts.getBitBlockWidth()))
{
    assert (IS_POW_2(strideWidth) && strideWidth >= ts.getBitBlockWidth() && strideWidth <= MaxStrideWidth);
}

void SingleStreamScanKernelTemplate::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    Type * const sizeTy = b.getSizeTy();
    Value * const sz_ZERO = b.getSize(0);
    Value * const sz_ONE = b.getSize(1);

    mEntryBlock = b.GetInsertBlock();
    mStrideStart = b.CreateBasicBlock("strideStart");
    mBuildStrideMask = b.CreateBasicBlock("buildStrideMask");
    mMaskReady = b.CreateBasicBlock("maskReady");
    mProcessMask = b.CreateBasicBlock("processMask");
    mProcessWord = b.CreateBasicBlock("processWord");
    mWordDone = b.CreateBasicBlock("wordDone");
    mStrideDone = b.CreateBasicBlock("strideDone");
    mExitBlock = b.CreateBasicBlock("exitBlock");
    initialize(b);
    Value * const scanStreamProcessedItemCount = b.getProcessedItemCount("scan");
    Value * const MASK_ZERO = Constant::getNullValue(mSW.StrideMaskTy);
    b.CreateBr(mStrideStart);

    b.SetInsertPoint(mStrideStart);
    PHINode * const strideNo = b.CreatePHI(sizeTy, 2, "strideNo");
    strideNo->addIncoming(sz_ZERO, mEntryBlock);
    mStrideNo = strideNo;
    Value * const nextStrideNo = b.CreateAdd(strideNo, sz_ONE, "nextStrideNo");
    willProcessStride(b, strideNo);
    b.CreateBr(mBuildStrideMask);

    b.SetInsertPoint(mBuildStrideMask);
    PHINode * maskAccumPhi = b.CreatePHI(mSW.StrideMaskTy, 2);
    maskAccumPhi->addIncoming(MASK_ZERO, mStrideStart);
    PHINode * const blockNo = b.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, mStrideStart);
    mBlockNo = blockNo;
    maskBuildingIterationHead(b);
    Value * const nextBlockNo = b.CreateAdd(blockNo, sz_ONE);
    blockNo->addIncoming(nextBlockNo, mBuildStrideMask);
    Value * const blockIndex = b.CreateAdd(blockNo, b.CreateMul(strideNo, mSW.NUM_BLOCKS_PER_STRIDE));
    maskBuildingIterationBody(b, blockIndex);
    Value * const block = b.loadInputStreamBlock("scan", b.getInt32(0), blockIndex);
    Value * const any = b.simd_any(mSW.fieldWidth, block);
    Value * const signMask = b.CreateZExt(b.hsimd_signmask(mSW.fieldWidth, any), mSW.StrideMaskTy);
    Value * const shiftedSignMask = b.CreateShl(signMask, b.CreateZExtOrTrunc(b.CreateMul(blockNo, mSW.WORDS_PER_BLOCK), mSW.StrideMaskTy));
    Value * const strideMask = b.CreateOr(maskAccumPhi, shiftedSignMask);
    maskAccumPhi->addIncoming(strideMask, mBuildStrideMask);
    b.CreateCondBr(b.CreateICmpNE(nextBlockNo, mSW.NUM_BLOCKS_PER_STRIDE), mBuildStrideMask, mMaskReady);

    b.SetInsertPoint(mMaskReady);
    didBuildMask(b, strideMask);
    b.CreateLikelyCondBr(b.CreateICmpNE(strideMask, MASK_ZERO), mProcessMask, mStrideDone);

    b.SetInsertPoint(mProcessMask);
    PHINode * const processingMask = b.CreatePHI(mSW.StrideMaskTy, 2, "processingMask");
    mProcessingMask = processingMask;
    processingMask->addIncoming(strideMask, mMaskReady);
    Value * const wordOffset = b.CreateCountForwardZeroes(processingMask, "wordOffset", true);
    mWordOffset = wordOffset;
    Value * const strideOffset = b.CreateMul(strideNo, mSW.NUM_BLOCKS_PER_STRIDE);
    Value * const blockNumOfWord = b.CreateUDiv(wordOffset, mSW.WORDS_PER_BLOCK);
    Value * const blockOffset = b.CreateURem(wordOffset, mSW.WORDS_PER_BLOCK);
    Value * const processingBlockIndex = b.CreateAdd(strideOffset, blockNumOfWord);
    Value * const stridePtr = b.CreateBitCast(b.getInputStreamBlockPtr("scan", b.getInt32(0), processingBlockIndex), mSW.PointerTy);
    Value * const wordPtr = b.CreateGEP(mSW.Ty, stridePtr, blockOffset);
    Value * const word = b.CreateLoad(mSW.Ty, wordPtr);
    willProcessWord(b, word);
    b.CreateBr(mProcessWord);

    b.SetInsertPoint(mProcessWord);
    PHINode * const processingWord = b.CreatePHI(mSW.Ty, 2, "processingWord");
    processingWord->addIncoming(word, mProcessMask);
    mProcessingWord = processingWord;
    Value * const bitIndex_InWord = b.CreateZExt(b.CreateCountForwardZeroes(processingWord, "inWord", true), sizeTy);
    Value * const wordIndex_InStride = b.CreateMul(wordOffset, mSW.WIDTH);
    Value * const strideIndex = b.CreateAdd(scanStreamProcessedItemCount, b.CreateMul(strideNo, b.getSize(mStride)));
    Value * const absoluteWordIndex = b.CreateAdd(strideIndex, wordIndex_InStride);
    Value * const absoluteIndex = b.CreateAdd(absoluteWordIndex, bitIndex_InWord);

    generateProcessingLogic(b, absoluteIndex, processingBlockIndex, bitIndex_InWord);

    Value * const processedWord = b.CreateResetLowestBit(processingWord);
    processingWord->addIncoming(processedWord, mProcessWord);
    b.CreateCondBr(b.CreateICmpNE(processedWord, Constant::getNullValue(mSW.Ty)), mProcessWord, mWordDone);

    b.SetInsertPoint(mWordDone);
    didProcessWord(b);
    Value * const processedMask = b.CreateResetLowestBit(processingMask, "processedMask");
    processingMask->addIncoming(processedMask, mWordDone);
    b.CreateCondBr(b.CreateICmpNE(processedMask, MASK_ZERO), mProcessMask, mStrideDone);

    b.SetInsertPoint(mStrideDone);
    didProcessStride(b, strideNo);
    strideNo->addIncoming(nextStrideNo, mStrideDone);
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), mStrideStart, mExitBlock);

    b.SetInsertPoint(mExitBlock);
    finalize(b);
}

const uint32_t SingleStreamScanKernelTemplate::MaxStrideWidth = 4096;

SingleStreamScanKernelTemplate::SingleStreamScanKernelTemplate(LLVMTypeSystemInterface & ts, std::string && name, StreamSet * scan)
: MultiBlockKernel(ts, name + "_sb" + std::to_string(codegen::ScanBlocks), {{"scan", scan}}, {}, {}, {}, {})
, mSW(ts, std::min(codegen::ScanBlocks * ts.getBitBlockWidth(), MaxStrideWidth))
{
    assert (scan->getNumElements() == 1 && scan->getFieldWidth() == 1);
    uint32_t strideWidth = std::min(codegen::ScanBlocks * ts.getBitBlockWidth(), MaxStrideWidth);
    if (!IS_POW_2(codegen::ScanBlocks)) {
        report_fatal_error("scan-blocks must be a power of 2");
    }
    if ((codegen::ScanBlocks * ts.getBitBlockWidth()) > MaxStrideWidth) {
        report_fatal_error(StringRef("scan-blocks exceeds maximum allowed size of ") + std::to_string(MaxStrideWidth / ts.getBitBlockWidth()));
    }
    setStride(strideWidth);
}

} // namespace kernel
