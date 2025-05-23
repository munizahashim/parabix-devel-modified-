/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/scan/scanmatchgen.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/core/kernel_builder.h>

using namespace llvm;

namespace kernel {

const unsigned BITS_PER_BYTE = 8;
const unsigned SIZE_T_BITS = sizeof(size_t) * BITS_PER_BYTE;

struct ScanWordParameters {
    unsigned width;
    unsigned indexWidth;
    Type * const Ty;
    Type * const pointerTy;
    Constant * const WIDTH;
    Constant * const ix_MAXBIT;
    Constant * const WORDS_PER_BLOCK;
    Constant * const WORDS_PER_STRIDE;

    ScanWordParameters(KernelBuilder & b, unsigned stride) :
#ifdef PREFER_NARROW_SCANWIDTH
    width(std::max(BITS_PER_BYTE, stride/SIZE_T_BITS)),
#else
    width(std::min(SIZE_T_BITS, stride/BITS_PER_BYTE)),
#endif
    indexWidth(stride/width),
    Ty(b.getIntNTy(width)),
    pointerTy(Ty->getPointerTo()),
    WIDTH(b.getSize(width)),
    ix_MAXBIT(b.getSize(indexWidth - 1)),
    WORDS_PER_BLOCK(b.getSize(b.getBitBlockWidth()/width)),
    WORDS_PER_STRIDE(b.getSize(indexWidth))
    {   //  The stride must be a power of 2 and a multiple of the BitBlock width.
        assert((((stride & (stride - 1)) == 0) && (stride >= b.getBitBlockWidth()) && (stride <= SIZE_T_BITS * SIZE_T_BITS)));
    }
};


void ScanMatchKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    const bool mLineNumbering = true;
    // Determine the parameters for two-level scanning.
    ScanWordParameters sw(b, mStride);

    Module * const m = b.getModule();
    Constant * const sz_STRIDE = b.getSize(mStride);
    assert ((mStride % b.getBitBlockWidth()) == 0);
    Constant * const sz_BLOCKS_PER_STRIDE = b.getSize(mStride/b.getBitBlockWidth());
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);
    Constant * const sz_BITS = b.getSize(SIZE_T_BITS);
    Constant * const sz_MAXBIT = b.getSize(SIZE_T_BITS - 1);
    Type * const sizeTy = b.getSizeTy();
    Type * const blockTy = b.getBitBlockType();

    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const stridePrologue = b.CreateBasicBlock("stridePrologue");
    BasicBlock * const stridePrecomputation = b.CreateBasicBlock("stridePrecomputation");
    BasicBlock * const strideMasksReady = b.CreateBasicBlock("strideMasksReady");
    BasicBlock * const updateLineInfo = b.CreateBasicBlock("updateLineInfo");
    BasicBlock * const strideMatchLoop = b.CreateBasicBlock("strideMatchLoop");
    BasicBlock * const dispatch = b.CreateBasicBlock("dispatch");
    BasicBlock * const matchesDone = b.CreateBasicBlock("matchesDone");
    BasicBlock * const stridesDone = b.CreateBasicBlock("stridesDone");
    BasicBlock * const callFinalizeScan = b.CreateBasicBlock("callFinalizeScan");
    BasicBlock * const scanReturn = b.CreateBasicBlock("scanReturn");

    Value * const initialPos = b.getProcessedItemCount("matchResult");
    Value * const accumulator = b.getScalarField("accumulator_address");
    Value * const avail = b.getAvailableItemCount("InputStream");
    Value * const initialLineStart = b.getProcessedItemCount("InputStream");
    Value * initialLineNum = nullptr;
    Value * lineCountArrayBlockPtr = nullptr;
    Value * lineCountArrayWordPtr = nullptr;
    if (mLineNumbering) {
        initialLineNum = b.getScalarField("LineNum");
        lineCountArrayBlockPtr = b.CreateAlignedAllocaAtEntryPoint(blockTy,
                                                        b.getBitBlockWidth()/BITS_PER_BYTE,
                                                        sz_BLOCKS_PER_STRIDE);
        // Bitcast the lineNumberArrayptr to access by scanWord number
        lineCountArrayWordPtr = b.CreateBitCast(lineCountArrayBlockPtr, sw.pointerTy);
    }
    b.CreateBr(stridePrologue);

    b.SetInsertPoint(stridePrologue);
    // Set up the loop variables as PHI nodes at the beginning of each stride.
    PHINode * const strideNo = b.CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode * const pendingLineStart = b.CreatePHI(sizeTy, 2);
    pendingLineStart->addIncoming(initialLineStart, entryBlock);
    PHINode * pendingLineNum = nullptr;
    if (mLineNumbering) {
        pendingLineNum = b.CreatePHI(sizeTy, 2);
        pendingLineNum->addIncoming(initialLineNum, entryBlock);
    }
    Value * stridePos = b.CreateAdd(initialPos, b.CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b.CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * nextStrideNo = b.CreateAdd(strideNo, sz_ONE);
    b.CreateBr(stridePrecomputation);


    // Precompute index masks for one stride of the match result and line break streams,
    // as well as a partial sum popcount of line numbers if line numbering is on.
    b.SetInsertPoint(stridePrecomputation);
    PHINode * const matchMaskAccum = b.CreatePHI(sizeTy, 2);
    matchMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const breakMaskAccum = b.CreatePHI(sizeTy, 2);
    breakMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const blockNo = b.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, stridePrologue);
    PHINode * baseCounts = nullptr;
    if (mLineNumbering) {
        baseCounts = b.CreatePHI(blockTy, 2);
        baseCounts->addIncoming(b.allZeroes(), stridePrologue);
    }

    Value * strideBlockIndex = b.CreateAdd(strideBlockOffset, blockNo);
    Value * matchBitBlock = b.loadInputStreamBlock("matchResult", sz_ZERO, strideBlockIndex);
    Value * breakBitBlock = b.loadInputStreamBlock("lineBreak", sz_ZERO, strideBlockIndex);
    Value * const anyMatch = b.simd_any(sw.width, matchBitBlock);
    Value * const anyBreak = b.simd_any(sw.width, breakBitBlock);
    if (mLineNumbering) {
        Value * breakCounts = b.hsimd_partial_sum(sw.width, b.simd_popcount(sw.width, breakBitBlock));
        breakCounts = b.simd_add(sw.width, breakCounts, baseCounts);
        b.CreateBlockAlignedStore(b.bitCast(breakCounts), b.CreateGEP(blockTy, lineCountArrayBlockPtr, blockNo));
        Value * baseCountsNext = b.bitCast(b.simd_fill(sw.width, b.mvmd_extract(sw.width, breakCounts, b.getBitBlockWidth()/sw.width - 1)));
        baseCounts->addIncoming(baseCountsNext, stridePrecomputation);
    }
    Value * matchWordMask = b.CreateZExt(b.hsimd_signmask(sw.width, anyMatch), sizeTy);
    Value * breakWordMask = b.CreateZExt(b.hsimd_signmask(sw.width, anyBreak), sizeTy);
    Value * matchMask = b.CreateOr(matchMaskAccum, b.CreateShl(matchWordMask, b.CreateMul(blockNo, sw.WORDS_PER_BLOCK)), "matchMask");
    Value * breakMask = b.CreateOr(breakMaskAccum, b.CreateShl(breakWordMask, b.CreateMul(blockNo, sw.WORDS_PER_BLOCK)), "breakMask");
    Value * const nextBlockNo = b.CreateAdd(blockNo, sz_ONE);

    matchMaskAccum->addIncoming(matchMask, stridePrecomputation);
    breakMaskAccum->addIncoming(breakMask, stridePrecomputation);
    blockNo->addIncoming(nextBlockNo, stridePrecomputation);
    b.CreateCondBr(b.CreateICmpNE(nextBlockNo, sz_BLOCKS_PER_STRIDE), stridePrecomputation, strideMasksReady);

    b.SetInsertPoint(strideMasksReady);
    // If there are no breaks in the stride, there are no matches.   We can move on to
    // the next stride immediately.
    b.CreateUnlikelyCondBr(b.CreateICmpEQ(breakMask, sz_ZERO), matchesDone, updateLineInfo);

    b.SetInsertPoint(updateLineInfo);
    // We have at least one line break.   Determine the end-of-stride line start position
    // and line number, if needed.
    Value * matchWordBasePtr = b.getInputStreamBlockPtr("matchResult", sz_ZERO, strideBlockOffset);
    matchWordBasePtr = b.CreatePointerCast(matchWordBasePtr, sw.pointerTy);
    Value * breakWordBasePtr = b.getInputStreamBlockPtr("lineBreak", sz_ZERO, strideBlockOffset);
    breakWordBasePtr = b.CreatePointerCast(breakWordBasePtr, sw.pointerTy);

    Value * finalBreakIdx = b.CreateSub(sz_MAXBIT, b.CreateCountReverseZeroes(breakMask), "finalBreakIdx");
    Value * finalBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, finalBreakIdx)), sizeTy);
    Value * finalLineStartInWord = b.CreateSub(sz_BITS, b.CreateCountReverseZeroes(finalBreakWord));
    Value * finalLineStartPos = b.CreateAdd(stridePos, b.CreateMul(finalBreakIdx, sw.WIDTH));
    finalLineStartPos = b.CreateAdd(finalLineStartInWord, finalLineStartPos);
    Value * strideFinalLineNum = nullptr;
    if (mLineNumbering) {
        // compute the final line number.
        Value * strideLineCount = b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, sw.ix_MAXBIT));
        strideFinalLineNum = b.CreateAdd(pendingLineNum, b.CreateZExtOrTrunc(strideLineCount, sizeTy));
    }
    // Now check whether there are any matches at all in the stride.   If not, we
    // can immediately move on to the next stride.
    // We optimize for the case of no matches; the cost of the branch penalty
    // is expected to be small relative to the processing of each match.
    b.CreateLikelyCondBr(b.CreateICmpEQ(matchMask, sz_ZERO), matchesDone, strideMatchLoop);

    // Precondition: we have at least one more match to process.
    b.SetInsertPoint(strideMatchLoop);
    PHINode * const matchMaskPhi = b.CreatePHI(sizeTy, 2);
    matchMaskPhi->addIncoming(matchMask, updateLineInfo);
    PHINode * const matchWordPhi = b.CreatePHI(sizeTy, 2);
    matchWordPhi->addIncoming(sz_ZERO, updateLineInfo);

    // If we have any bits in the current matchWordPhi, continue with those, otherwise load
    // the next match word.
    Value * matchWordIdx = b.CreateCountForwardZeroes(matchMaskPhi, "matchWordIdx");
    Value * nextMatchWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, matchWordBasePtr, matchWordIdx)), sizeTy);
    Value * matchBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, matchWordIdx)), sizeTy);
    Value * theMatchWord = b.CreateSelect(b.CreateICmpEQ(matchWordPhi, sz_ZERO), nextMatchWord, matchWordPhi);
    Value * matchWordPos = b.CreateAdd(stridePos, b.CreateMul(matchWordIdx, sw.WIDTH));
    Value * matchEndPosInWord = b.CreateCountForwardZeroes(theMatchWord);
    Value * matchEndPos = b.CreateAdd(matchWordPos, matchEndPosInWord, "matchEndPos");
    // Find the prior line break.  There are three possibilities.
    // (a) a prior break in the break word corresponding to the current match word.
    // (b) the last break in a prior word within the current stride.
    // (c) the pending line start from previous iterations.
    // Case (b) is most likely and requires a load of the prior break word.
    // We avoid branching by safely loading a prior word in any case and then
    // using selects to handle cases (a) and (c).
    Value * priorBreaksThisWord = b.CreateZeroHiBitsFrom(matchBreakWord, matchEndPosInWord);
    Value * priorBreaksInStride = b.CreateZeroHiBitsFrom(breakMask, matchWordIdx);
    Value * inWordCond = b.CreateICmpNE(priorBreaksThisWord, sz_ZERO);
    Value * inStrideCond = b.CreateICmpNE(priorBreaksInStride, sz_ZERO);
    Value * breakWordIdx = b.CreateSub(sz_MAXBIT, b.CreateCountReverseZeroes(priorBreaksInStride), "breakWordIdx_");
    // Create a safe index to load; the loaded value will be ignored for cases (a), (c).
    breakWordIdx = b.CreateSelect(inStrideCond, breakWordIdx, sz_ZERO);
    breakWordIdx = b.CreateSelect(inWordCond, matchWordIdx, breakWordIdx, "breakWordIdx");
    Value * breakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, breakWordIdx)), sizeTy);
    // For case (a), we use the previously masked value of the break word.
    breakWord = b.CreateSelect(inWordCond, priorBreaksThisWord, breakWord);   // cases (a) and (b)
    Value * lineStartInWord = b.CreateSub(sz_BITS, b.CreateCountReverseZeroes(breakWord));
    Value * lineStartBase = b.CreateAdd(stridePos, b.CreateMul(breakWordIdx, sw.WIDTH));
    Value * lineStartPos = b.CreateAdd(lineStartBase, lineStartInWord);
    // The break position is the line start for cases (a), (b); otherwise use the pending value.
    Value * const matchStart = b.CreateSelect(b.CreateOr(inWordCond, inStrideCond), lineStartPos, pendingLineStart, "matchStart");
    Value * matchRecordNum = nullptr;
    if (mLineNumbering) {
        Value * lineCountInStride = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, matchWordIdx)), sizeTy);
        // Subtract the number of remaining breaks in the match word to get the relative line number.
        Value * extraBreaks = b.CreateXor(matchBreakWord, priorBreaksThisWord);
        lineCountInStride = b.CreateSub(lineCountInStride, b.CreatePopcount(extraBreaks));
        matchRecordNum = b.CreateAdd(pendingLineNum, lineCountInStride);
    }

    // It is possible that the matchRecordEnd position is one past EOF.  Make sure not
    // to access past EOF.
    Value * const bufLimit = b.CreateSub(avail, sz_ONE);
    matchEndPos = b.CreateUMin(matchEndPos, bufLimit);
    // matchStart should never be past EOF, but in case it is....
    //b.CreateAssert(b.CreateICmpULT(matchStart, avail), "match position past EOF");
    b.CreateCondBr(b.CreateICmpULT(matchStart, avail), dispatch, callFinalizeScan);

    b.SetInsertPoint(dispatch);
    Function * const dispatcher = m->getFunction("accumulate_match_wrapper"); assert (dispatcher);

    Value * const startPtr = b.getRawInputPointer("InputStream", matchStart);
    Value * const endPtr = b.getRawInputPointer("InputStream", matchEndPos);

    Type * expectedMatchRecNumTy = dispatcher->getFunctionType()->getParamType(1);
    assert (expectedMatchRecNumTy->isIntegerTy());
    Value * const matchRecNum = b.CreateZExtOrTrunc(matchRecordNum, expectedMatchRecNumTy);
    b.CreateCall(dispatcher->getFunctionType(), dispatcher, {accumulator, matchRecNum, startPtr, endPtr});

    //  We've dealt with the match, now prepare for the next one, if any.
    // There may be more matches in the current word.
    Value * dropMatch = b.CreateResetLowestBit(theMatchWord, "dropMatch");
    Value * thisWordDone = b.CreateICmpEQ(dropMatch, sz_ZERO);
    // There may be more matches in the match mask.
    Value * resetMatchMask = b.CreateResetLowestBit(matchMaskPhi, "nextMatchMask");
    Value * nextMatchMask = b.CreateSelect(thisWordDone, resetMatchMask, matchMaskPhi);
    BasicBlock * currentBB = b.GetInsertBlock();
    matchMaskPhi->addIncoming(nextMatchMask, currentBB);
    matchWordPhi->addIncoming(dropMatch, currentBB);
    b.CreateCondBr(b.CreateICmpNE(nextMatchMask, sz_ZERO), strideMatchLoop, matchesDone);

    b.SetInsertPoint(matchesDone);
    PHINode * strideFinalLineStart = b.CreatePHI(sizeTy, 3);
    strideFinalLineStart->addIncoming(pendingLineStart, strideMasksReady);
    strideFinalLineStart->addIncoming(finalLineStartPos, updateLineInfo);
    strideFinalLineStart->addIncoming(finalLineStartPos, currentBB);
    PHINode * strideFinalLineNumPhi = nullptr;
    if (mLineNumbering) {
        strideFinalLineNumPhi = b.CreatePHI(sizeTy, 3);
        strideFinalLineNumPhi->addIncoming(pendingLineNum, strideMasksReady);
        strideFinalLineNumPhi->addIncoming(strideFinalLineNum, updateLineInfo);
        strideFinalLineNumPhi->addIncoming(strideFinalLineNum, currentBB);
    }
    strideNo->addIncoming(nextStrideNo, matchesDone);
    pendingLineStart->addIncoming(strideFinalLineStart, matchesDone);
    if (mLineNumbering) {
        pendingLineNum->addIncoming(strideFinalLineNumPhi, matchesDone);
    }
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);

    b.SetInsertPoint(stridesDone);
    if (mLineNumbering) {
        b.setScalarField("LineNum", strideFinalLineNumPhi);
    }
    Value * processed = b.CreateSelect(b.isFinal(), avail, strideFinalLineStart);
    b.setProcessedItemCount("InputStream", processed);
    b.CreateCondBr(b.isFinal(), callFinalizeScan, scanReturn);

    b.SetInsertPoint(callFinalizeScan);
    Function * finalizer = m->getFunction("finalize_match_wrapper"); assert (finalizer);
    FunctionType * fTy = finalizer->getFunctionType();
    Value * const bufferEnd = b.getRawInputPointer("InputStream", avail);
    b.CreateCall(fTy, finalizer, {accumulator, bufferEnd});
    b.CreateBr(scanReturn);

    b.SetInsertPoint(scanReturn);
}

ScanMatchKernel::ScanMatchKernel(LLVMTypeSystemInterface & ts, StreamSet * const Matches, StreamSet * const LineBreakStream, StreamSet * const ByteStream, Scalar * const callbackObject, unsigned strideBlocks)
    : MultiBlockKernel(ts, "scanMatch" + std::to_string(strideBlocks),
// inputs
{Binding{"matchResult", Matches}
,Binding{"lineBreak", LineBreakStream}
,Binding{"InputStream", ByteStream, FixedRate(), { Deferred() }}},
// outputs
{},
// input scalars
{Binding{"accumulator_address", callbackObject}},
// output scalars
{},
// kernel state
{InternalScalar{ts.getSizeTy(), "LineNum"}}) {
    addAttribute(SideEffecting());
    setStride(std::min(ts.getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
}

void ScanBatchKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    const bool mLineNumbering = true;
    // Determine the parameters for two-level scanning.
    ScanWordParameters sw(b, mStride);

    Module * const m = b.getModule();
    Constant * const sz_STRIDE = b.getSize(mStride);
    Constant * const sz_BLOCKS_PER_STRIDE = b.getSize(mStride/b.getBitBlockWidth());
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);
    Constant * const sz_BITS = b.getSize(SIZE_T_BITS);
    Constant * const sz_MAXBIT = b.getSize(SIZE_T_BITS - 1);
    Type * const sizeTy = b.getSizeTy();
    Type * const blockTy = b.getBitBlockType();

    Function * const getFileCount = m->getFunction("get_file_count_wrapper"); assert (getFileCount);
    Function * const getFileStartPos = m->getFunction("get_file_start_pos_wrapper"); assert (getFileStartPos);
    Function * const setBatchLineNumber = m->getFunction("set_batch_line_number_wrapper"); assert (setBatchLineNumber);
    Function * const dispatcher = m->getFunction("accumulate_match_wrapper"); assert (dispatcher);
    Function * const finalizer = m->getFunction("finalize_match_wrapper"); assert (finalizer);

    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const stridePrologue = b.CreateBasicBlock("stridePrologue");
    BasicBlock * const stridePrecomputation = b.CreateBasicBlock("stridePrecomputation");
    BasicBlock * const strideMasksReady = b.CreateBasicBlock("strideMasksReady");
    BasicBlock * const updateLineInfo = b.CreateBasicBlock("updateLineInfo");
    BasicBlock * const nextInBatch = b.CreateBasicBlock("nextInBatch");
    BasicBlock * const nextInBatch2 = b.CreateBasicBlock("nextInBatch2");
    BasicBlock * const strideMatchLoop = b.CreateBasicBlock("strideMatchLoop");
    BasicBlock * const dispatch = b.CreateBasicBlock("dispatch");
    BasicBlock * const matchesDone = b.CreateBasicBlock("matchesDone");
    BasicBlock * const strideFinal = b.CreateBasicBlock("strideFinal");
    BasicBlock * const stridesDone = b.CreateBasicBlock("stridesDone");
    BasicBlock * const callFinalizeScan = b.CreateBasicBlock("callFinalizeScan");
    BasicBlock * const scanReturn = b.CreateBasicBlock("scanReturn");

    Value * const initialPos = b.getProcessedItemCount("matchResult");
    Value * const accumulator = b.getScalarField("accumulator_address");
    Value * const avail = b.getAvailableItemCount("InputStream");
    Value * fileCount = b.CreateCall(getFileCount->getFunctionType(), getFileCount, {accumulator});
    Value * maxFileNum = b.CreateSub(fileCount, b.getSize(1));
    Value * const initialLineStart = b.getProcessedItemCount("InputStream");
    Value * initialLineNum = nullptr;
    Value * lineCountArrayBlockPtr = nullptr;
    Value * lineCountArrayWordPtr = nullptr;
    if (mLineNumbering) {
        initialLineNum = b.getScalarField("LineNum");
        lineCountArrayBlockPtr = b.CreateAlignedAllocaAtEntryPoint(blockTy,
                                                        b.getBitBlockWidth()/BITS_PER_BYTE,
                                                        sz_BLOCKS_PER_STRIDE);
        // Bitcast the lineNumberArrayptr to access by scanWord number
        lineCountArrayWordPtr = b.CreateBitCast(lineCountArrayBlockPtr, sw.pointerTy);
    }
    b.CreateBr(stridePrologue);

    b.SetInsertPoint(stridePrologue);
    // Set up the loop variables as PHI nodes at the beginning of each stride.
    PHINode * const strideNo = b.CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode * const pendingLineStart = b.CreatePHI(sizeTy, 2);
    pendingLineStart->addIncoming(initialLineStart, entryBlock);
    PHINode * pendingLineNum = nullptr;
    if (mLineNumbering) {
        pendingLineNum = b.CreatePHI(sizeTy, 2);
        pendingLineNum->addIncoming(initialLineNum, entryBlock);
    }
    Value * stridePos = b.CreateAdd(initialPos, b.CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b.CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * matchWordBasePtr = b.getInputStreamBlockPtr("matchResult", sz_ZERO, strideBlockOffset);
    matchWordBasePtr = b.CreatePointerCast(matchWordBasePtr, sw.pointerTy);
    Value * breakWordBasePtr = b.getInputStreamBlockPtr("lineBreak", sz_ZERO, strideBlockOffset);
    breakWordBasePtr = b.CreatePointerCast(breakWordBasePtr, sw.pointerTy);
    Value * nextStrideNo = b.CreateAdd(strideNo, sz_ONE);
    Value * batchFileNum = b.getScalarField("batchFileNum");
    Value * inFinalFile = b.CreateICmpEQ(batchFileNum, maxFileNum);
    Value * availableLimit = b.getAvailableItemCount("matchResult");
    Value * nextFileNum = b.CreateAdd(batchFileNum, b.getSize(1));
    Value * fileLimit = b.CreateCall(getFileStartPos->getFunctionType(), getFileStartPos, {accumulator, b.CreateSelect(inFinalFile, maxFileNum, nextFileNum)});
    Value * pendingLimit = b.CreateSelect(inFinalFile, availableLimit, fileLimit);

    b.setScalarField("pendingFileLimit", pendingLimit);
    b.CreateBr(stridePrecomputation);

    // Precompute index masks for one stride of the match result and line break streams,
    // as well as a partial sum popcount of line numbers if line numbering is on.
    b.SetInsertPoint(stridePrecomputation);
    PHINode * const matchMaskAccum = b.CreatePHI(sizeTy, 2);
    matchMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const breakMaskAccum = b.CreatePHI(sizeTy, 2);
    breakMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const blockNo = b.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, stridePrologue);
    PHINode * baseCounts = nullptr;
    if (mLineNumbering) {
        baseCounts = b.CreatePHI(blockTy, 2);
        baseCounts->addIncoming(b.allZeroes(), stridePrologue);
    }
    Value * strideBlockIndex = b.CreateAdd(strideBlockOffset, blockNo);
    Value * matchBitBlock = b.loadInputStreamBlock("matchResult", sz_ZERO, strideBlockIndex);
    Value * breakBitBlock = b.loadInputStreamBlock("lineBreak", sz_ZERO, strideBlockIndex);

    Value * const anyMatch = b.simd_any(sw.width, matchBitBlock);
    Value * const anyBreak = b.simd_any(sw.width, breakBitBlock);
    if (mLineNumbering) {
        Value * breakCounts = b.hsimd_partial_sum(sw.width, b.simd_popcount(sw.width, breakBitBlock));
        breakCounts = b.simd_add(sw.width, breakCounts, baseCounts);
        b.CreateBlockAlignedStore(b.bitCast(breakCounts), b.CreateGEP(blockTy, lineCountArrayBlockPtr, blockNo));
        Value * baseCountsNext = b.bitCast(b.simd_fill(sw.width, b.mvmd_extract(sw.width, breakCounts, b.getBitBlockWidth()/sw.width - 1)));
        baseCounts->addIncoming(baseCountsNext, stridePrecomputation);
    }
    Value * matchWordMask = b.CreateZExt(b.hsimd_signmask(sw.width, anyMatch), sizeTy);
    Value * breakWordMask = b.CreateZExt(b.hsimd_signmask(sw.width, anyBreak), sizeTy);
    Value * const shift = b.CreateMul(blockNo, sw.WORDS_PER_BLOCK);
    Value * matchMask = b.CreateOr(matchMaskAccum, b.CreateShl(matchWordMask, shift), "matchMask");
    Value * breakMask = b.CreateOr(breakMaskAccum, b.CreateShl(breakWordMask, shift), "breakMask");
    Value * const nextBlockNo = b.CreateAdd(blockNo, sz_ONE);

    matchMaskAccum->addIncoming(matchMask, stridePrecomputation);
    breakMaskAccum->addIncoming(breakMask, stridePrecomputation);
    blockNo->addIncoming(nextBlockNo, stridePrecomputation);
    b.CreateCondBr(b.CreateICmpNE(nextBlockNo, sz_BLOCKS_PER_STRIDE), stridePrecomputation, strideMasksReady);

    b.SetInsertPoint(strideMasksReady);
    // If there are no breaks in the stride, there are no matches.   We can move on to
    // the next stride immediately.
    b.CreateUnlikelyCondBr(b.CreateICmpEQ(breakMask, sz_ZERO), matchesDone, updateLineInfo);

    b.SetInsertPoint(updateLineInfo);
    // We have at least one line break.   Determine the end-of-stride line start position
    // and line number, if needed.

    Value * finalBreakIdx = b.CreateSub(sz_MAXBIT, b.CreateCountReverseZeroes(breakMask), "finalBreakIdx");
    Value * finalBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, finalBreakIdx)), sizeTy);
    Value * finalLineStartInWord = b.CreateSub(sz_BITS, b.CreateCountReverseZeroes(finalBreakWord));
    Value * finalLineStartPos = b.CreateAdd(stridePos, b.CreateMul(finalBreakIdx, sw.WIDTH));
    finalLineStartPos = b.CreateAdd(finalLineStartInWord, finalLineStartPos);
    Value * strideFinalLineNum = nullptr;
    if (mLineNumbering) {
        // compute the final line number.
        Value * strideLineCount = b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, sw.ix_MAXBIT));
        strideFinalLineNum = b.CreateAdd(pendingLineNum, b.CreateZExtOrTrunc(strideLineCount, sizeTy));
    }
    // Now check whether there are any matches at all in the stride.   If not, we
    // can immediately move on to the next stride.
    // We optimize for the case of no matches; the cost of the branch penalty
    // is expected to be small relative to the processing of each match.
    b.CreateLikelyCondBr(b.CreateICmpEQ(matchMask, sz_ZERO), matchesDone, strideMatchLoop);

    // Precondition: we have at least one more match to process.
    b.SetInsertPoint(strideMatchLoop);
    PHINode * const matchMaskPhi = b.CreatePHI(sizeTy, 2);
    matchMaskPhi->addIncoming(matchMask, updateLineInfo);
    PHINode * const matchWordPhi = b.CreatePHI(sizeTy, 2);
    matchWordPhi->addIncoming(sz_ZERO, updateLineInfo);

    // If we have any bits in the current matchWordPhi, continue with those, otherwise load
    // the next match word.
    Value * matchWordIdx = b.CreateCountForwardZeroes(matchMaskPhi, "matchWordIdx");
    Value * nextMatchWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, matchWordBasePtr, matchWordIdx)), sizeTy);
    Value * matchBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, matchWordIdx)), sizeTy);
    Value * theMatchWord = b.CreateSelect(b.CreateICmpEQ(matchWordPhi, sz_ZERO), nextMatchWord, matchWordPhi);
    Value * matchWordPos = b.CreateAdd(stridePos, b.CreateMul(matchWordIdx, sw.WIDTH));
    Value * matchEndPosInWord = b.CreateCountForwardZeroes(theMatchWord);
    Value * matchEndPos = b.CreateAdd(matchWordPos, matchEndPosInWord, "matchEndPos");
    // Find the prior line break.  There are three possibilities.
    // (a) a prior break in the break word corresponding to the current match word.
    // (b) the last break in a prior word within the current stride.
    // (c) the pending line start from previous iterations.
    // Case (b) is most likely and requires a load of the prior break word.
    // We avoid branching by safely loading a prior word in any case and then
    // using selects to handle cases (a) and (c).
    Value * priorBreaksThisWord = b.CreateZeroHiBitsFrom(matchBreakWord, matchEndPosInWord);
    Value * priorBreaksInStride = b.CreateZeroHiBitsFrom(breakMask, matchWordIdx);
    Value * inWordCond = b.CreateICmpNE(priorBreaksThisWord, sz_ZERO);
    Value * inStrideCond = b.CreateICmpNE(priorBreaksInStride, sz_ZERO);
    Value * breakWordIdx = b.CreateSub(sz_MAXBIT, b.CreateCountReverseZeroes(priorBreaksInStride), "breakWordIdx_");
    // Create a safe index to load; the loaded value will be ignored for cases (a), (c).
    breakWordIdx = b.CreateSelect(inStrideCond, breakWordIdx, sz_ZERO);
    breakWordIdx = b.CreateSelect(inWordCond, matchWordIdx, breakWordIdx, "breakWordIdx");
    Value * breakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, breakWordIdx)), sizeTy);
    // For case (a), we use the previously masked value of the break word.
    breakWord = b.CreateSelect(inWordCond, priorBreaksThisWord, breakWord);   // cases (a) and (b)
    Value * lineStartInWord = b.CreateSub(sz_BITS, b.CreateCountReverseZeroes(breakWord));
    Value * lineStartBase = b.CreateAdd(stridePos, b.CreateMul(breakWordIdx, sw.WIDTH));
    Value * lineStartPos = b.CreateAdd(lineStartBase, lineStartInWord);
    // The break position is the line start for cases (a), (b); otherwise use the pending value.
    Value * const matchStart = b.CreateSelect(b.CreateOr(inWordCond, inStrideCond), lineStartPos, pendingLineStart, "matchStart");
    Value * matchRecordNum = nullptr;
    if (mLineNumbering) {
        Value * lineCountInStride = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, matchWordIdx)), sizeTy);
        // Subtract the number of remaining breaks in the match word to get the relative line number.
        Value * extraBreaks = b.CreateXor(matchBreakWord, priorBreaksThisWord);
        lineCountInStride = b.CreateSub(lineCountInStride, b.CreatePopcount(extraBreaks));
        matchRecordNum = b.CreateAdd(pendingLineNum, lineCountInStride);
    }

    // It is possible that the matchRecordEnd position is one past EOF.  Make sure not
    // to access past EOF.
    Value * const bufLimit = b.CreateSub(avail, sz_ONE);
    matchEndPos = b.CreateUMin(matchEndPos, bufLimit);

    pendingLimit = b.getScalarField("pendingFileLimit");
    Value * beyondFileEnd = b.CreateICmpUGE(matchStart, pendingLimit);

    b.CreateUnlikelyCondBr(beyondFileEnd, nextInBatch, dispatch);

    b.SetInsertPoint(nextInBatch);
    batchFileNum = b.getScalarField("batchFileNum");
    pendingLimit = b.getScalarField("pendingFileLimit");
    if (mLineNumbering) {
        Value * strideOffsetPos = b.CreateSub(pendingLimit, stridePos);
        Value * offsetIdx = b.CreateUDiv(strideOffsetPos, sw.WIDTH);
        Value * offsetPosInWord = b.CreateURem(strideOffsetPos, sw.WIDTH);
        // Get the count of all line breaks including the offset word.
        Value * offsetLineCount = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, offsetIdx)), sizeTy);
        // Subtract the breaaks that are past the start position.
        Value * fileBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, offsetIdx)), sizeTy);
        Value * excess = b.CreatePopcount(b.CreateLShr(fileBreakWord, offsetPosInWord));
        Value * priorLineCount = b.CreateAdd(b.CreateSub(offsetLineCount, excess), pendingLineNum);
        //b.CreateCall(finalizer->getFunctionType(), finalizer, {accumulator, batchFileNum, priorLineCount});
        b.CreateCall(setBatchLineNumber->getFunctionType(), setBatchLineNumber, {accumulator, batchFileNum, priorLineCount});
    } else {
        b.CreateCall(setBatchLineNumber->getFunctionType(), setBatchLineNumber, {accumulator, batchFileNum, sz_ZERO});
    }
    
    nextFileNum = b.CreateAdd(batchFileNum, b.getSize(1));
    b.setScalarField("batchFileNum", nextFileNum);
    inFinalFile = b.CreateICmpEQ(nextFileNum, maxFileNum);
    Value * nextFileLimit = b.CreateCall(getFileStartPos->getFunctionType(), getFileStartPos, {accumulator, b.CreateSelect(inFinalFile, maxFileNum, b.CreateAdd(nextFileNum, b.getSize(1)))});
    Value * limit = b.CreateSelect(inFinalFile, availableLimit, nextFileLimit);
    b.setScalarField("pendingFileLimit", limit);
    beyondFileEnd = b.CreateICmpUGE(matchStart, limit);

    b.CreateUnlikelyCondBr(beyondFileEnd, nextInBatch, dispatch);

    // matchStart should never be past EOF, but in case it is....
    //b.CreateAssert(b.CreateICmpULT(matchStart, avail), "match position past EOF");
    //b.CreateCondBr(b.CreateICmpULT(matchStart, avail), dispatch, callFinalizeScan);

    b.SetInsertPoint(dispatch);

    Value * const startPtr = b.getRawInputPointer("InputStream", matchStart);
    Value * const endPtr = b.getRawInputPointer("InputStream", matchEndPos);

//    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
//        Value * const A = b.CreateSub(matchEndPos, matchStart);
//        Value * const B = b.CreatePtrDiff(endPtr, startPtr);
//        b.CreateAssert(b.CreateICmpEQ(A, B), "InputStream is not contiguous");
//    }

    Type * expectedMatchRecNumTy = dispatcher->getFunctionType()->getParamType(1);
    assert (expectedMatchRecNumTy->isIntegerTy());
    Value * const matchRecNum = b.CreateZExtOrTrunc(matchRecordNum, expectedMatchRecNumTy);
    b.CreateCall(dispatcher->getFunctionType(), dispatcher, {accumulator, matchRecNum, startPtr, endPtr});

    //  We've dealt with the match, now prepare for the next one, if any.
    // There may be more matches in the current word.
    Value * dropMatch = b.CreateResetLowestBit(theMatchWord, "dropMatch");
    Value * thisWordDone = b.CreateICmpEQ(dropMatch, sz_ZERO);
    // There may be more matches in the match mask.
    Value * resetMatchMask = b.CreateResetLowestBit(matchMaskPhi, "nextMatchMask");
    Value * nextMatchMask = b.CreateSelect(thisWordDone, resetMatchMask, matchMaskPhi);
    BasicBlock * currentBB = b.GetInsertBlock();
    matchMaskPhi->addIncoming(nextMatchMask, currentBB);
    matchWordPhi->addIncoming(dropMatch, currentBB);
    b.CreateCondBr(b.CreateICmpNE(nextMatchMask, sz_ZERO), strideMatchLoop, matchesDone);

    b.SetInsertPoint(matchesDone);
    PHINode * strideFinalLineStart = b.CreatePHI(sizeTy, 3);
    strideFinalLineStart->addIncoming(pendingLineStart, strideMasksReady);
    strideFinalLineStart->addIncoming(finalLineStartPos, updateLineInfo);
    strideFinalLineStart->addIncoming(finalLineStartPos, currentBB);
    PHINode * strideFinalLineNumPhi = nullptr;
    if (mLineNumbering) {
        strideFinalLineNumPhi = b.CreatePHI(sizeTy, 3);
        strideFinalLineNumPhi->addIncoming(pendingLineNum, strideMasksReady);
        strideFinalLineNumPhi->addIncoming(strideFinalLineNum, updateLineInfo);
        strideFinalLineNumPhi->addIncoming(strideFinalLineNum, currentBB);
    }
    //  We've processed available stride data looking for matches, now check
    //  for any files that are terminated within the stride and finalize them.
    Value * strideLimit = b.CreateUMin(b.CreateAdd(stridePos, sz_STRIDE), availableLimit);
    pendingLimit = b.getScalarField("pendingFileLimit");
    //  We use a strictly greater than test here; if the pendingLimit is the availableLimit,
    //  this means that we are at the end of the available data, not necessarily a file end.
    Value * notFinalFile = b.CreateICmpNE(b.getScalarField("batchFileNum"), maxFileNum);
    b.CreateUnlikelyCondBr(b.CreateAnd(notFinalFile, b.CreateICmpUGT(strideLimit, pendingLimit)), nextInBatch2, strideFinal);

    b.SetInsertPoint(nextInBatch2);
    batchFileNum = b.getScalarField("batchFileNum");
    // pendingLimit = b.getScalarField("pendingFileLimit");
    if (mLineNumbering) {
        Value * strideOffsetPos = b.CreateSub(pendingLimit, stridePos);
        Value * offsetIdx = b.CreateUDiv(strideOffsetPos, sw.WIDTH);
        Value * offsetPosInWord = b.CreateURem(strideOffsetPos, sw.WIDTH);
        // Get the count of all line breaks including the offset word.
        Value * offsetLineCount = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, offsetIdx)), sizeTy);
        // Subtract the breaaks that are past the start position.
        Value * fileBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, offsetIdx)), sizeTy);
        Value * excess = b.CreatePopcount(b.CreateLShr(fileBreakWord, offsetPosInWord));
        Value * priorLineCount = b.CreateAdd(b.CreateSub(offsetLineCount, excess), pendingLineNum);
        b.CreateCall(setBatchLineNumber->getFunctionType(), setBatchLineNumber, {accumulator, batchFileNum, priorLineCount});
    } else {
        b.CreateCall(setBatchLineNumber->getFunctionType(), setBatchLineNumber, {accumulator, batchFileNum, sz_ZERO});
    }
    nextFileNum = b.CreateAdd(batchFileNum, b.getSize(1));
    b.setScalarField("batchFileNum", nextFileNum);
    inFinalFile = b.CreateICmpEQ(nextFileNum, maxFileNum);
    nextFileLimit = b.CreateCall(getFileStartPos->getFunctionType(), getFileStartPos, {accumulator, b.CreateSelect(inFinalFile, maxFileNum, b.CreateAdd(nextFileNum, b.getSize(1)))});
    limit = b.CreateSelect(inFinalFile, strideLimit, nextFileLimit);
    b.setScalarField("pendingFileLimit", limit);
    beyondFileEnd = b.CreateICmpUGT(strideLimit, limit);
    b.CreateUnlikelyCondBr(beyondFileEnd, nextInBatch2, strideFinal);

    b.SetInsertPoint(strideFinal);
    strideNo->addIncoming(nextStrideNo, strideFinal);
    pendingLineStart->addIncoming(strideFinalLineStart, strideFinal);
    if (mLineNumbering) {
        pendingLineNum->addIncoming(strideFinalLineNumPhi, strideFinal);
    }
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);

    b.SetInsertPoint(stridesDone);
    if (mLineNumbering) {
        b.setScalarField("LineNum", strideFinalLineNumPhi);
    }
    Value * processed = b.CreateSelect(b.isFinal(), avail, strideFinalLineStart);
    b.setProcessedItemCount("InputStream", processed);
    b.CreateCondBr(b.isFinal(), callFinalizeScan, scanReturn);

    b.SetInsertPoint(callFinalizeScan);
    Value * const bufferEnd = b.getRawInputPointer("InputStream", avail);
    b.CreateCall(finalizer->getFunctionType(), finalizer, {accumulator, bufferEnd});
    /*if (mLineNumbering) {
        b.CreateCall(setBatchLineNumber->getFunctionType(), setBatchLineNumber, {accumulator, maxFileNum, strideFinalLineNumPhi});
    } else {
        b.CreateCall(setBatchLineNumber->getFunctionType(), setBatchLineNumber, {accumulator, maxFileNum, sz_ZERO});
    }
     */
    b.CreateBr(scanReturn);

    b.SetInsertPoint(scanReturn);
}

ScanBatchKernel::ScanBatchKernel(LLVMTypeSystemInterface & ts, StreamSet * const Matches, StreamSet * const LineBreakStream, StreamSet * const ByteStream, Scalar * const callbackObject, unsigned strideBlocks)
    : MultiBlockKernel(ts, "scanBatch" + std::to_string(strideBlocks),
// inputs
{Binding{"matchResult", Matches}
,Binding{"lineBreak", LineBreakStream, FixedRate(), ZeroExtended()}
,Binding{"InputStream", ByteStream, FixedRate(), { ZeroExtended(), Deferred() }}},
// outputs
{},
// input scalars
{Binding{"accumulator_address", callbackObject}},
// output scalars
{},
// kernel state
{InternalScalar{ts.getSizeTy(), "LineNum"}, InternalScalar{ts.getSizeTy(), "batchFileNum"}, InternalScalar{ts.getSizeTy(), "pendingFileLimit"}}) {
    addAttribute(SideEffecting());
    setStride(std::min(ts.getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
}



enum MatchCoordinatesEnum {LINE_STARTS = 0, LINE_ENDS = 1, LINE_NUMBERS = 2};

MatchCoordinatesKernel::MatchCoordinatesKernel(LLVMTypeSystemInterface & ts,
                                               StreamSet * const Matches, StreamSet * const LineBreakStream,
                                               StreamSet * const Coordinates, unsigned strideBlocks)
: MultiBlockKernel(ts, "matchCoordinates" + std::to_string(strideBlocks),
// inputs
{Binding{"matchResult", Matches}, Binding{"lineBreak", LineBreakStream, FixedRate(1), ZeroExtended()}},
// outputs
{Bind("Coordinates", Coordinates, PopcountOf("matchResult"), MaximumDistribution())},
// input scalars
{},
// output scalars
{},
// kernel state
{InternalScalar{ts.getSizeTy(), "LineNum"},
 InternalScalar{ts.getSizeTy(), "LineStart"}}) {
     // The stride size must be limited so that the scanword mask is a single size_t value.
     setStride(std::min(ts.getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
     assert (Matches->getNumElements() == 1);
     assert (LineBreakStream->getNumElements() == 1);
     assert (Coordinates->getNumElements() == 3);
}

void MatchCoordinatesKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    const bool mLineNumbering = true;

    // Determine the parameters for two-level scanning.
    ScanWordParameters sw(b, mStride);

    Constant * const sz_STRIDE = b.getSize(mStride);
    Constant * const sz_BLOCKS_PER_STRIDE = b.getSize(mStride/b.getBitBlockWidth());
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);
    Constant * const sz_BITS = b.getSize(SIZE_T_BITS);
    Constant * const sz_MAXBIT = b.getSize(SIZE_T_BITS - 1);
    Type * sizeTy = b.getSizeTy();
    Type * const blockTy = b.getBitBlockType();

    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const stridePrologue = b.CreateBasicBlock("stridePrologue");
    BasicBlock * const stridePrecomputation = b.CreateBasicBlock("stridePrecomputation");
    BasicBlock * const strideMasksReady = b.CreateBasicBlock("strideMasksReady");
    BasicBlock * const updateLineInfo = b.CreateBasicBlock("updateLineInfo");
    BasicBlock * const strideCoordinateLoop = b.CreateBasicBlock("strideCoordinateLoop");
    BasicBlock * const strideCoordinatesDone = b.CreateBasicBlock("strideCoordinatesDone");
    BasicBlock * const stridesDone = b.CreateBasicBlock("stridesDone");

    Value * const initialPos = b.getProcessedItemCount("matchResult");
    Value * const initialLineStart = b.getScalarField("LineStart");
    Value * initialLineNum = nullptr;
    Value * lineCountArrayBlockPtr = nullptr;
    Value * lineCountArrayWordPtr = nullptr;
    if (mLineNumbering) {
        initialLineNum = b.getScalarField("LineNum");
        lineCountArrayBlockPtr = b.CreateAlignedAllocaAtEntryPoint(blockTy,
                                                        b.getBitBlockWidth()/BITS_PER_BYTE,
                                                        sz_BLOCKS_PER_STRIDE);
        // Bitcast the lineNumberArrayptr to access by scanWord number
        lineCountArrayWordPtr = b.CreateBitCast(lineCountArrayBlockPtr, sw.pointerTy);
    }
    Value * const initialMatchCount = b.getProducedItemCount("Coordinates");
    b.CreateBr(stridePrologue);

    b.SetInsertPoint(stridePrologue);
    // Set up the loop variables as PHI nodes at the beginning of each stride.
    PHINode * const strideNo = b.CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode * const currenMatchCount = b.CreatePHI(sizeTy, 2);
    currenMatchCount->addIncoming(initialMatchCount, entryBlock);
    PHINode * const pendingLineStart = b.CreatePHI(sizeTy, 2);
    pendingLineStart->addIncoming(initialLineStart, entryBlock);
    PHINode * pendingLineNum = nullptr;
    if (mLineNumbering) {
        pendingLineNum = b.CreatePHI(sizeTy, 2);
        pendingLineNum->addIncoming(initialLineNum, entryBlock);
    }
    Value * stridePos = b.CreateAdd(initialPos, b.CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b.CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * nextStrideNo = b.CreateAdd(strideNo, sz_ONE);
    b.CreateBr(stridePrecomputation);
    // Precompute index masks for one stride of the match result and line break streams,
    // as well as a partial sum popcount of line numbers if line numbering is on.
    b.SetInsertPoint(stridePrecomputation);
    PHINode * const matchMaskAccum = b.CreatePHI(sizeTy, 2);
    matchMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const breakMaskAccum = b.CreatePHI(sizeTy, 2);
    breakMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const blockNo = b.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, stridePrologue);
    PHINode * baseCounts = nullptr;
    if (mLineNumbering) {
        baseCounts = b.CreatePHI(blockTy, 2);
        baseCounts->addIncoming(b.allZeroes(), stridePrologue);
    }
    Value * strideBlockIndex = b.CreateAdd(strideBlockOffset, blockNo);
    Value * matchBitBlock = b.loadInputStreamBlock("matchResult", sz_ZERO, strideBlockIndex);
    Value * breakBitBlock = b.loadInputStreamBlock("lineBreak", sz_ZERO, strideBlockIndex);
    Value * const anyMatch = b.simd_any(sw.width, matchBitBlock);
    Value * const anyBreak = b.simd_any(sw.width, breakBitBlock);
    if (mLineNumbering) {
        Value * breakCounts = b.hsimd_partial_sum(sw.width, b.simd_popcount(sw.width, breakBitBlock));
        breakCounts = b.simd_add(sw.width, breakCounts, baseCounts);
        b.CreateBlockAlignedStore(b.bitCast(breakCounts), b.CreateGEP(blockTy, lineCountArrayBlockPtr, blockNo));
        Value * baseCountsNext = b.bitCast(b.simd_fill(sw.width, b.mvmd_extract(sw.width, breakCounts, b.getBitBlockWidth()/sw.width - 1)));
        baseCounts->addIncoming(baseCountsNext, stridePrecomputation);
    }
    Value * matchWordMask = b.CreateZExtOrTrunc(b.hsimd_signmask(sw.width, anyMatch), sizeTy);
    Value * breakWordMask = b.CreateZExtOrTrunc(b.hsimd_signmask(sw.width, anyBreak), sizeTy);
    Value * matchMask = b.CreateOr(matchMaskAccum, b.CreateShl(matchWordMask, b.CreateMul(blockNo, sw.WORDS_PER_BLOCK)), "matchMask");
    Value * breakMask = b.CreateOr(breakMaskAccum, b.CreateShl(breakWordMask, b.CreateMul(blockNo, sw.WORDS_PER_BLOCK)), "breakMask");
    Value * const nextBlockNo = b.CreateAdd(blockNo, sz_ONE);
    matchMaskAccum->addIncoming(matchMask, stridePrecomputation);
    breakMaskAccum->addIncoming(breakMask, stridePrecomputation);
    blockNo->addIncoming(nextBlockNo, stridePrecomputation);
    b.CreateCondBr(b.CreateICmpNE(nextBlockNo, sz_BLOCKS_PER_STRIDE), stridePrecomputation, strideMasksReady);

    b.SetInsertPoint(strideMasksReady);
    // If there are no breaks in the stride, there are no matches.   We can move on to
    // the next stride immediately.
    b.CreateUnlikelyCondBr(b.CreateIsNull(breakMask), strideCoordinatesDone, updateLineInfo);

    b.SetInsertPoint(updateLineInfo);
    // We have at least one line break.   Determine the end-of-stride line start position
    // and line number, if needed.
    Value * matchWordBasePtr = b.getInputStreamBlockPtr("matchResult", sz_ZERO, strideBlockOffset);
    matchWordBasePtr = b.CreateBitCast(matchWordBasePtr, sw.pointerTy);
    Value * breakWordBasePtr = b.getInputStreamBlockPtr("lineBreak", sz_ZERO, strideBlockOffset);
    breakWordBasePtr = b.CreateBitCast(breakWordBasePtr, sw.pointerTy);

    Value * finalBreakIdx = b.CreateSub(sz_MAXBIT, b.CreateCountReverseZeroes(breakMask), "finalBreakIdx");
    Value * finalBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, finalBreakIdx)), sizeTy);
    Value * finalLineStartInWord = b.CreateSub(sz_BITS, b.CreateCountReverseZeroes(finalBreakWord));
    Value * finalLineStartPos = b.CreateAdd(stridePos, b.CreateMul(finalBreakIdx, sw.WIDTH));
    finalLineStartPos = b.CreateAdd(finalLineStartInWord, finalLineStartPos);
    Value * strideFinalLineNum = nullptr;
    if (mLineNumbering) {
        // compute the final line number.
        Value * strideLineCount = b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, sw.ix_MAXBIT));
        strideFinalLineNum = b.CreateAdd(pendingLineNum, b.CreateZExtOrTrunc(strideLineCount, sizeTy));
   }
    // Now check whether there are any matches at all in the stride.   If not, we
    // can immediately move on to the next stride.
    // We optimize for the case of no matches; the cost of the branch penalty
    // is expected to be small relative to the processing of each match.
    b.CreateLikelyCondBr(b.CreateIsNull(matchMask), strideCoordinatesDone, strideCoordinateLoop);

    // Precondition: we have at least one more match to process.
    b.SetInsertPoint(strideCoordinateLoop);
    PHINode * const matchMaskPhi = b.CreatePHI(sizeTy, 2);
    matchMaskPhi->addIncoming(matchMask, updateLineInfo);
    PHINode * const matchWordPhi = b.CreatePHI(sizeTy, 2);
    matchWordPhi->addIncoming(sz_ZERO, updateLineInfo);
    PHINode * const matchNumPhi = b.CreatePHI(sizeTy, 2, "matchNumPhi");
    matchNumPhi->addIncoming(currenMatchCount, updateLineInfo);

    // If we have any bits in the current matchWordPhi, continue with those, otherwise load
    // the next match word.
    Value * matchWordIdx = b.CreateCountForwardZeroes(matchMaskPhi, "matchWordIdx");
    Value * nextMatchWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, matchWordBasePtr, matchWordIdx)), sizeTy);
    Value * matchBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, matchWordIdx)), sizeTy);
    Value * theMatchWord = b.CreateSelect(b.CreateIsNull(matchWordPhi), nextMatchWord, matchWordPhi);
    Value * matchWordPos = b.CreateAdd(stridePos, b.CreateMul(matchWordIdx, sw.WIDTH));
    Value * matchEndPosInWord = b.CreateCountForwardZeroes(theMatchWord);
    Value * matchEndPos = b.CreateAdd(matchWordPos, matchEndPosInWord, "matchEndPos");
    // Find the prior line break.  There are three possibilities.
    // (a) a prior break in the break word corresponding to the current match word.
    // (b) the last break in a prior word within the current stride.
    // (c) the pending line start from previous iterations.
    // Case (b) is most likely and requires a load of the prior break word.
    // We avoid branching by safely loading a prior word in any case and then
    // using selects to handle cases (a) and (c).
    Value * priorBreaksThisWord = b.CreateZeroHiBitsFrom(matchBreakWord, matchEndPosInWord);
    Value * priorBreaksInStride = b.CreateZeroHiBitsFrom(breakMask, matchWordIdx);
    Value * inWordCond = b.CreateICmpNE(priorBreaksThisWord, sz_ZERO);
    Value * inStrideCond = b.CreateICmpNE(priorBreaksInStride, sz_ZERO);
    Value * breakWordIdx = b.CreateSub(sz_MAXBIT, b.CreateCountReverseZeroes(priorBreaksInStride), "breakWordIdx_");
    // Create a safe index to load; the loaded value will be ignored for cases (a), (c).
    breakWordIdx = b.CreateSelect(inWordCond, matchWordIdx, b.CreateSelect(inStrideCond, breakWordIdx, sz_ZERO), "breakWordIdx");
    Value * breakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, breakWordIdx)), sizeTy);
    // For case (a), we use the previously masked value of the break word.
    breakWord = b.CreateSelect(inWordCond, priorBreaksThisWord, breakWord);   // cases (a) and (b)
    Value * lineStartInWord = b.CreateSub(sz_BITS, b.CreateCountReverseZeroes(breakWord));
    Value * lineStartBase = b.CreateAdd(stridePos, b.CreateMul(breakWordIdx, sw.WIDTH));
    Value * lineStartPos = b.CreateAdd(lineStartBase, lineStartInWord);
    // The break position is the line start for cases (a), (b); otherwise use the pending value.
    Value * const matchStart = b.CreateSelect(b.CreateOr(inWordCond, inStrideCond), lineStartPos, pendingLineStart, "matchStart");

    Value * const matchStartPtr = b.getRawOutputPointer("Coordinates", b.getInt32(LINE_STARTS), matchNumPhi);
    b.CreateStore(matchStart, matchStartPtr);
    Value * const lineEndsPtr = b.getRawOutputPointer("Coordinates", b.getInt32(LINE_ENDS), matchNumPhi);
    b.CreateStore(matchEndPos, lineEndsPtr);
    if (mLineNumbering) {
        Value * lineCountInStride = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, matchWordIdx)), sizeTy);
        // Subtract the number of remaining breaks in the match word to get the relative line number.
        Value * extraBreaks = b.CreateXor(matchBreakWord, priorBreaksThisWord);
        lineCountInStride = b.CreateSub(lineCountInStride, b.CreatePopcount(extraBreaks));
        Value * lineNum = b.CreateAdd(pendingLineNum, lineCountInStride);
        b.CreateStore(lineNum, b.getRawOutputPointer("Coordinates", b.getInt32(LINE_NUMBERS), matchNumPhi));
    }
    //  We've dealt with the match, now prepare for the next one, if any.
    // There may be more matches in the current word.
    Value * dropMatch = b.CreateResetLowestBit(theMatchWord);
    Value * thisWordDone = b.CreateIsNull(dropMatch);
    // There may be more matches in the match mask.
    Value * nextMatchMask = b.CreateSelect(thisWordDone, b.CreateResetLowestBit(matchMaskPhi), matchMaskPhi);
    BasicBlock * currentBB = b.GetInsertBlock();
    matchMaskPhi->addIncoming(nextMatchMask, currentBB);
    matchWordPhi->addIncoming(dropMatch, currentBB);
    Value * nextMatchNum = b.CreateAdd(matchNumPhi, sz_ONE);
    matchNumPhi->addIncoming(nextMatchNum, currentBB);
    b.CreateCondBr(b.CreateICmpNE(nextMatchMask, sz_ZERO), strideCoordinateLoop, strideCoordinatesDone);

    b.SetInsertPoint(strideCoordinatesDone);
    PHINode * finalStrideMatchCount = b.CreatePHI(sizeTy, 3);
    finalStrideMatchCount->addIncoming(currenMatchCount, strideMasksReady);
    finalStrideMatchCount->addIncoming(currenMatchCount, updateLineInfo);
    finalStrideMatchCount->addIncoming(nextMatchNum, currentBB);
    PHINode * strideFinalLineStart = b.CreatePHI(sizeTy, 3);
    strideFinalLineStart->addIncoming(pendingLineStart, strideMasksReady);
    strideFinalLineStart->addIncoming(finalLineStartPos, updateLineInfo);
    strideFinalLineStart->addIncoming(finalLineStartPos, currentBB);
    PHINode * strideFinalLineNumPhi = nullptr;
    if (mLineNumbering) {
        strideFinalLineNumPhi = b.CreatePHI(sizeTy, 3);
        strideFinalLineNumPhi->addIncoming(pendingLineNum, strideMasksReady);
        strideFinalLineNumPhi->addIncoming(strideFinalLineNum, updateLineInfo);
        strideFinalLineNumPhi->addIncoming(strideFinalLineNum, currentBB);
    }
    strideNo->addIncoming(nextStrideNo, strideCoordinatesDone);
    currenMatchCount->addIncoming(finalStrideMatchCount, strideCoordinatesDone);
    pendingLineStart->addIncoming(strideFinalLineStart, strideCoordinatesDone);
    if (mLineNumbering) {
        pendingLineNum->addIncoming(strideFinalLineNumPhi, strideCoordinatesDone);
    }
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);

    b.SetInsertPoint(stridesDone);
    b.setScalarField("LineStart", strideFinalLineStart);
    if (mLineNumbering) {
        b.setScalarField("LineNum", strideFinalLineNumPhi);
    }
    // b.setProducedItemCount("Coordinates", finalStrideMatchCount);
}

enum BatchCoordinatesEnum {BATCH_LINE_NUMBERS = 0, BATCH_FILE_NUMBERS = 1};

void BatchCoordinatesKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    const bool mLineNumbering = true;
    // Determine the parameters for two-level scanning.
    ScanWordParameters sw(b, mStride);

    Module * const m = b.getModule();
    Constant * const sz_STRIDE = b.getSize(mStride);
    Constant * const sz_BLOCKS_PER_STRIDE = b.getSize(mStride/b.getBitBlockWidth());
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);
    Constant * const sz_MAXBIT = b.getSize(SIZE_T_BITS - 1);
    Type * const sizeTy = b.getSizeTy();
    Type * const blockTy = b.getBitBlockType();

    Function * const getFileCount = m->getFunction("get_file_count_wrapper"); assert (getFileCount);
    Function * const getFileStartPos = m->getFunction("get_file_start_pos_wrapper"); assert (getFileStartPos);
    Function * const setBatchLineNumber = m->getFunction("set_batch_line_number_wrapper"); assert (setBatchLineNumber);

    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const stridePrologue = b.CreateBasicBlock("stridePrologue");
    BasicBlock * const stridePrecomputation = b.CreateBasicBlock("stridePrecomputation");
    BasicBlock * const strideMasksReady = b.CreateBasicBlock("strideMasksReady");
    BasicBlock * const updateLineInfo = b.CreateBasicBlock("updateLineInfo");
    BasicBlock * const nextInBatch = b.CreateBasicBlock("nextInBatch");
    BasicBlock * const nextInBatch2 = b.CreateBasicBlock("nextInBatch2");
    BasicBlock * const strideMatchLoop = b.CreateBasicBlock("strideMatchLoop");
    BasicBlock * const storeCoord = b.CreateBasicBlock("storeCoord");
    BasicBlock * const matchesDone = b.CreateBasicBlock("matchesDone");
    BasicBlock * const strideFinal = b.CreateBasicBlock("strideFinal");
    BasicBlock * const stridesDone = b.CreateBasicBlock("stridesDone");

    Value * const initialPos = b.getProcessedItemCount("matchResult");
    Value * const accumulator = b.getScalarField("accumulator_address");
    Value * maxFileNum = b.CreateSub(b.CreateCall(getFileCount->getFunctionType(), getFileCount, {accumulator}), b.getSize(1));
    Value * initialLineNum = nullptr;
    Value * lineCountArrayBlockPtr = nullptr;
    Value * lineCountArrayWordPtr = nullptr;
    if (mLineNumbering) {
        initialLineNum = b.getScalarField("pendingLineNum");
        lineCountArrayBlockPtr = b.CreateAlignedAllocaAtEntryPoint(blockTy,
                                                        b.getBitBlockWidth()/BITS_PER_BYTE,
                                                        sz_BLOCKS_PER_STRIDE);
        // Bitcast the lineNumberArrayptr to access by scanWord number
        lineCountArrayWordPtr = b.CreateBitCast(lineCountArrayBlockPtr, sw.pointerTy);
    }
    Value * const initialMatchCount = b.getProducedItemCount("Coordinates");
    b.CreateBr(stridePrologue);

    b.SetInsertPoint(stridePrologue);
    // Set up the loop variables as PHI nodes at the beginning of each stride.
    PHINode * const strideNo = b.CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode * const currenMatchCount = b.CreatePHI(sizeTy, 2);
    currenMatchCount->addIncoming(initialMatchCount, entryBlock);
    PHINode * pendingLineNum = nullptr;
    if (mLineNumbering) {
        pendingLineNum = b.CreatePHI(sizeTy, 2);
        pendingLineNum->addIncoming(initialLineNum, entryBlock);
    }
    Value * stridePos = b.CreateAdd(initialPos, b.CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b.CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * matchWordBasePtr = b.getInputStreamBlockPtr("matchResult", sz_ZERO, strideBlockOffset);
    matchWordBasePtr = b.CreatePointerCast(matchWordBasePtr, sw.pointerTy);
    Value * breakWordBasePtr = b.getInputStreamBlockPtr("lineBreak", sz_ZERO, strideBlockOffset);
    breakWordBasePtr = b.CreatePointerCast(breakWordBasePtr, sw.pointerTy);
    Value * nextStrideNo = b.CreateAdd(strideNo, sz_ONE);
    Value * batchFileNum = b.getScalarField("batchFileNum");
    Value * inFinalFile = b.CreateICmpEQ(batchFileNum, maxFileNum);
    Value * availableLimit = b.getAvailableItemCount("matchResult");
    Value * nextFileNum = b.CreateAdd(batchFileNum, b.getSize(1));
    Value * fileLimit = b.CreateCall(getFileStartPos->getFunctionType(), getFileStartPos, {accumulator, b.CreateSelect(inFinalFile, maxFileNum, nextFileNum)});
    Value * pendingLimit = b.CreateSelect(inFinalFile, availableLimit, fileLimit);
    b.setScalarField("pendingFileLimit", pendingLimit);
    b.CreateBr(stridePrecomputation);


    // Precompute index masks for one stride of the match result and line break streams,
    // as well as a partial sum popcount of line numbers if line numbering is on.
    b.SetInsertPoint(stridePrecomputation);
    PHINode * const matchMaskAccum = b.CreatePHI(sizeTy, 2);
    matchMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const breakMaskAccum = b.CreatePHI(sizeTy, 2);
    breakMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const blockNo = b.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, stridePrologue);
    PHINode * baseCounts = nullptr;
    if (mLineNumbering) {
        baseCounts = b.CreatePHI(blockTy, 2);
        baseCounts->addIncoming(b.allZeroes(), stridePrologue);
    }
    Value * strideBlockIndex = b.CreateAdd(strideBlockOffset, blockNo);
    Value * matchBitBlock = b.loadInputStreamBlock("matchResult", sz_ZERO, strideBlockIndex);
    Value * breakBitBlock = b.loadInputStreamBlock("lineBreak", sz_ZERO, strideBlockIndex);

    Value * const anyMatch = b.simd_any(sw.width, matchBitBlock);
    Value * const anyBreak = b.simd_any(sw.width, breakBitBlock);
    if (mLineNumbering) {
        Value * breakCounts = b.hsimd_partial_sum(sw.width, b.simd_popcount(sw.width, breakBitBlock));
        breakCounts = b.simd_add(sw.width, breakCounts, baseCounts);
        b.CreateBlockAlignedStore(b.bitCast(breakCounts), b.CreateGEP(blockTy, lineCountArrayBlockPtr, blockNo));
        Value * baseCountsNext = b.bitCast(b.simd_fill(sw.width, b.mvmd_extract(sw.width, breakCounts, b.getBitBlockWidth()/sw.width - 1)));
        baseCounts->addIncoming(baseCountsNext, stridePrecomputation);
    }
    Value * matchWordMask = b.CreateZExt(b.hsimd_signmask(sw.width, anyMatch), sizeTy);
    Value * breakWordMask = b.CreateZExt(b.hsimd_signmask(sw.width, anyBreak), sizeTy);
    Value * matchMask = b.CreateOr(matchMaskAccum, b.CreateShl(matchWordMask, b.CreateMul(blockNo, sw.WORDS_PER_BLOCK)), "matchMask");
    Value * breakMask = b.CreateOr(breakMaskAccum, b.CreateShl(breakWordMask, b.CreateMul(blockNo, sw.WORDS_PER_BLOCK)), "breakMask");
    Value * const nextBlockNo = b.CreateAdd(blockNo, sz_ONE);

    matchMaskAccum->addIncoming(matchMask, stridePrecomputation);
    breakMaskAccum->addIncoming(breakMask, stridePrecomputation);
    blockNo->addIncoming(nextBlockNo, stridePrecomputation);
    b.CreateCondBr(b.CreateICmpNE(nextBlockNo, sz_BLOCKS_PER_STRIDE), stridePrecomputation, strideMasksReady);

    b.SetInsertPoint(strideMasksReady);
    // If there are no breaks in the stride, there are no matches.   We can move on to
    // the next stride immediately.
    b.CreateUnlikelyCondBr(b.CreateICmpEQ(breakMask, sz_ZERO), matchesDone, updateLineInfo);

    b.SetInsertPoint(updateLineInfo);
    // We have at least one line break.   Determine the end-of-stride line start position
    // and line number, if needed.
    Value * strideFinalLineNum = nullptr;
    if (mLineNumbering) {
        // compute the final line number.
        Value * strideLineCount = b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, sw.ix_MAXBIT));
        strideFinalLineNum = b.CreateAdd(pendingLineNum, b.CreateZExtOrTrunc(strideLineCount, sizeTy));
    }
    // Now check whether there are any matches at all in the stride.   If not, we
    // can immediately move on to the next stride.
    // We optimize for the case of no matches; the cost of the branch penalty
    // is expected to be small relative to the processing of each match.
    b.CreateLikelyCondBr(b.CreateICmpEQ(matchMask, sz_ZERO), matchesDone, strideMatchLoop);

    // Precondition: we have at least one more match to process.
    b.SetInsertPoint(strideMatchLoop);
    PHINode * const matchMaskPhi = b.CreatePHI(sizeTy, 2);
    matchMaskPhi->addIncoming(matchMask, updateLineInfo);
    PHINode * const matchWordPhi = b.CreatePHI(sizeTy, 2);
    matchWordPhi->addIncoming(sz_ZERO, updateLineInfo);
    PHINode * const matchNumPhi = b.CreatePHI(sizeTy, 2, "matchNumPhi");
    matchNumPhi->addIncoming(currenMatchCount, updateLineInfo);

    // If we have any bits in the current matchWordPhi, continue with those, otherwise load
    // the next match word.
    Value * matchWordIdx = b.CreateCountForwardZeroes(matchMaskPhi, "matchWordIdx");
    Value * nextMatchWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, matchWordBasePtr, matchWordIdx)), sizeTy);
    Value * matchBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, matchWordIdx)), sizeTy);
    Value * theMatchWord = b.CreateSelect(b.CreateICmpEQ(matchWordPhi, sz_ZERO), nextMatchWord, matchWordPhi);
    Value * matchWordPos = b.CreateAdd(stridePos, b.CreateMul(matchWordIdx, sw.WIDTH));
    Value * matchEndPosInWord = b.CreateCountForwardZeroes(theMatchWord);
    Value * matchEndPos = b.CreateAdd(matchWordPos, matchEndPosInWord, "matchEndPos");
    // Find the prior line break.  There are three possibilities.
    // (a) a prior break in the break word corresponding to the current match word.
    // (b) the last break in a prior word within the current stride.
    // (c) the pending line start from previous iterations.
    // Case (b) is most likely and requires a load of the prior break word.
    // We avoid branching by safely loading a prior word in any case and then
    // using selects to handle cases (a) and (c).
    Value * priorBreaksThisWord = b.CreateZeroHiBitsFrom(matchBreakWord, matchEndPosInWord);
    Value * priorBreaksInStride = b.CreateZeroHiBitsFrom(breakMask, matchWordIdx);
    Value * inWordCond = b.CreateICmpNE(priorBreaksThisWord, sz_ZERO);
    Value * inStrideCond = b.CreateICmpNE(priorBreaksInStride, sz_ZERO);
    Value * breakWordIdx = b.CreateSub(sz_MAXBIT, b.CreateCountReverseZeroes(priorBreaksInStride), "breakWordIdx_");
    // Create a safe index to load; the loaded value will be ignored for cases (a), (c).
    breakWordIdx = b.CreateSelect(inStrideCond, breakWordIdx, sz_ZERO);
    breakWordIdx = b.CreateSelect(inWordCond, matchWordIdx, breakWordIdx, "breakWordIdx");
    Value * breakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, breakWordIdx)), sizeTy);
    // For case (a), we use the previously masked value of the break word.
    breakWord = b.CreateSelect(inWordCond, priorBreaksThisWord, breakWord);   // cases (a) and (b)
    //Value * matchRecordNum = nullptr;
    if (mLineNumbering) {
        Value * lineCountInStride = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, matchWordIdx)), sizeTy);
        // Subtract the number of remaining breaks in the match word to get the relative line number.
        Value * extraBreaks = b.CreateXor(matchBreakWord, priorBreaksThisWord);
        lineCountInStride = b.CreateSub(lineCountInStride, b.CreatePopcount(extraBreaks));
        //matchRecordNum = b.CreateAdd(pendingLineNum, lineCountInStride);
    }

    pendingLimit = b.getScalarField("pendingFileLimit");
    Value * beyondFileEnd = b.CreateICmpUGT(matchEndPos, pendingLimit);
    b.CreateUnlikelyCondBr(beyondFileEnd, nextInBatch, storeCoord);

    b.SetInsertPoint(nextInBatch);
    batchFileNum = b.getScalarField("batchFileNum");
    pendingLimit = b.getScalarField("pendingFileLimit");
    if (mLineNumbering) {
        Value * strideOffsetPos = b.CreateSub(pendingLimit, stridePos);
        Value * offsetIdx = b.CreateUDiv(strideOffsetPos, sw.WIDTH);
        Value * offsetPosInWord = b.CreateURem(strideOffsetPos, sw.WIDTH);
        // Get the count of all line breaks including the offset word.
        Value * offsetLineCount = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, offsetIdx)), sizeTy);
        // Subtract the breaaks that are past the start position.
        Value * fileBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, offsetIdx)), sizeTy);
        Value * excess = b.CreatePopcount(b.CreateLShr(fileBreakWord, offsetPosInWord));
        Value * priorLineCount = b.CreateAdd(b.CreateSub(offsetLineCount, excess), pendingLineNum);
        b.CreateCall(setBatchLineNumber->getFunctionType(), setBatchLineNumber, {accumulator, batchFileNum, priorLineCount});
    } else {
        b.CreateCall(setBatchLineNumber->getFunctionType(), setBatchLineNumber, {accumulator, batchFileNum, sz_ZERO});
    }

    nextFileNum = b.CreateAdd(batchFileNum, b.getSize(1));
    b.setScalarField("batchFileNum", nextFileNum);
    inFinalFile = b.CreateICmpEQ(nextFileNum, maxFileNum);
    Value * nextFileLimit = b.CreateCall(getFileStartPos->getFunctionType(), getFileStartPos, {accumulator, b.CreateSelect(inFinalFile, maxFileNum, b.CreateAdd(nextFileNum, b.getSize(1)))});
    Value * limit = b.CreateSelect(inFinalFile, availableLimit, nextFileLimit);
    b.setScalarField("pendingFileLimit", limit);
    beyondFileEnd = b.CreateICmpUGT(matchEndPos, limit);

    b.CreateUnlikelyCondBr(beyondFileEnd, nextInBatch, storeCoord);

    b.SetInsertPoint(storeCoord);
#ifdef WRITE_FILE_NUMBERS
    Value * const fileNumPtr = b.getRawOutputPointer("Coordinates", b.getInt32(BATCH_FILE_NUMBERS), matchNumPhi);
    //b.CallPrintInt("storing batchFileNum", b.getScalarField("batchFileNum"));
    b.CreateStore(b.CreateZExt(b.getScalarField("batchFileNum"), sizeTy), fileNumPtr);
#endif
    if (mLineNumbering) {
        Value * lineCountInStride = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, matchWordIdx)), sizeTy);
        // Subtract the number of remaining breaks in the match word to get the relative line number.
        Value * extraBreaks = b.CreateXor(matchBreakWord, priorBreaksThisWord);
        lineCountInStride = b.CreateSub(lineCountInStride, b.CreatePopcount(extraBreaks));
        Value * lineNum = b.CreateAdd(pendingLineNum, lineCountInStride);
        b.CreateStore(lineNum, b.getRawOutputPointer("Coordinates", b.getInt32(BATCH_LINE_NUMBERS), matchNumPhi));
    }

    //  We've dealt with the match, now prepare for the next one, if any.
    // There may be more matches in the current word.
    Value * dropMatch = b.CreateResetLowestBit(theMatchWord, "dropMatch");
    Value * thisWordDone = b.CreateICmpEQ(dropMatch, sz_ZERO);
    // There may be more matches in the match mask.
    Value * resetMatchMask = b.CreateResetLowestBit(matchMaskPhi, "nextMatchMask");
    Value * nextMatchMask = b.CreateSelect(thisWordDone, resetMatchMask, matchMaskPhi);
    BasicBlock * currentBB = b.GetInsertBlock();
    matchMaskPhi->addIncoming(nextMatchMask, currentBB);
    matchWordPhi->addIncoming(dropMatch, currentBB);
    Value * nextMatchNum = b.CreateAdd(matchNumPhi, sz_ONE);
    matchNumPhi->addIncoming(nextMatchNum, currentBB);
    b.CreateCondBr(b.CreateICmpNE(nextMatchMask, sz_ZERO), strideMatchLoop, matchesDone);

    b.SetInsertPoint(matchesDone);
    PHINode * strideFinalLineNumPhi = nullptr;
    if (mLineNumbering) {
        strideFinalLineNumPhi = b.CreatePHI(sizeTy, 3);
        strideFinalLineNumPhi->addIncoming(pendingLineNum, strideMasksReady);
        strideFinalLineNumPhi->addIncoming(strideFinalLineNum, updateLineInfo);
        strideFinalLineNumPhi->addIncoming(strideFinalLineNum, currentBB);
    }
    PHINode * finalStrideMatchCount = b.CreatePHI(sizeTy, 3);
    finalStrideMatchCount->addIncoming(currenMatchCount, strideMasksReady);
    finalStrideMatchCount->addIncoming(currenMatchCount, updateLineInfo);
    finalStrideMatchCount->addIncoming(nextMatchNum, currentBB);
    //  We've processed available stride data looking for matches, now check
    //  for any files that are terminated within the stride and finalize them.
    Value * strideLimit = b.CreateUMin(b.CreateAdd(stridePos, sz_STRIDE), availableLimit);
    pendingLimit = b.getScalarField("pendingFileLimit");
    //  We use a strictly greater than test here; if the pendingLimit is the availableLimit,
    //  this means that we are at the end of the available data, not necessarily a file end.
    Value * notFinalFile = b.CreateICmpNE(b.getScalarField("batchFileNum"), maxFileNum);
    b.CreateUnlikelyCondBr(b.CreateAnd(notFinalFile, b.CreateICmpUGT(strideLimit, pendingLimit)), nextInBatch2, strideFinal);

    b.SetInsertPoint(nextInBatch2);
    batchFileNum = b.getScalarField("batchFileNum");
    pendingLimit = b.getScalarField("pendingFileLimit");
    if (mLineNumbering) {
        Value * strideOffsetPos = b.CreateSub(pendingLimit, stridePos);
        Value * offsetIdx = b.CreateUDiv(strideOffsetPos, sw.WIDTH);
        Value * offsetPosInWord = b.CreateURem(strideOffsetPos, sw.WIDTH);
        // Get the count of all line breaks including the offset word.
        Value * offsetLineCount = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, lineCountArrayWordPtr, offsetIdx)), sizeTy);
        // Subtract the breaaks that are past the start position.
        Value * fileBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, offsetIdx)), sizeTy);
        Value * excess = b.CreatePopcount(b.CreateLShr(fileBreakWord, offsetPosInWord));
        Value * priorLineCount = b.CreateAdd(b.CreateSub(offsetLineCount, excess), pendingLineNum);
        b.CreateCall(setBatchLineNumber->getFunctionType(), setBatchLineNumber, {accumulator, batchFileNum, priorLineCount});
    } else {
        b.CreateCall(setBatchLineNumber->getFunctionType(), setBatchLineNumber, {accumulator, batchFileNum, sz_ZERO});
    }
    nextFileNum = b.CreateAdd(batchFileNum, b.getSize(1));
    b.setScalarField("batchFileNum", nextFileNum);
    inFinalFile = b.CreateICmpEQ(nextFileNum, maxFileNum);
    nextFileLimit = b.CreateCall(getFileStartPos->getFunctionType(), getFileStartPos, {accumulator, b.CreateSelect(inFinalFile, maxFileNum, b.CreateAdd(nextFileNum, b.getSize(1)))});
    limit = b.CreateSelect(inFinalFile, strideLimit, nextFileLimit);
    b.setScalarField("pendingFileLimit", limit);
    beyondFileEnd = b.CreateICmpUGT(strideLimit, limit);
    b.CreateUnlikelyCondBr(beyondFileEnd, nextInBatch2, strideFinal);

    b.SetInsertPoint(strideFinal);
    strideNo->addIncoming(nextStrideNo, strideFinal);
    currenMatchCount->addIncoming(finalStrideMatchCount, strideFinal);
    if (mLineNumbering) {
        pendingLineNum->addIncoming(strideFinalLineNumPhi, strideFinal);
    }
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);

    b.SetInsertPoint(stridesDone);
    if (mLineNumbering) {
        b.setScalarField("pendingLineNum", strideFinalLineNumPhi);
    }
}

BatchCoordinatesKernel::BatchCoordinatesKernel(LLVMTypeSystemInterface & ts,
                                               StreamSet * const Matches, StreamSet * const LineBreakStream,
                                               StreamSet * const Coordinates, Scalar * const callbackObject, unsigned strideBlocks)
: MultiBlockKernel(ts, "batchCoordinates" + std::to_string(strideBlocks),
// inputs
{Binding{"matchResult", Matches}, Binding{"lineBreak", LineBreakStream, FixedRate(1), ZeroExtended()}},
// outputs
{Binding{"Coordinates", Coordinates, PopcountOf("matchResult")}},
// input scalars
{Binding{"accumulator_address", callbackObject}},
// output scalars
{},
// kernel state
{InternalScalar{ts.getSizeTy(), "batchFileNum"},
    InternalScalar{ts.getSizeTy(), "pendingFileLimit"},
    //InternalScalar{b.getSizeTy(), "pendingFileStartLine"},
    InternalScalar{ts.getSizeTy(), "pendingLineNum"}}) {
     // The stride size must be limited so that the scanword mask is a single size_t value.
     setStride(std::min(ts.getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
     assert (Matches->getNumElements() == 1);
     assert (LineBreakStream->getNumElements() == 1);
#ifdef WRITE_FILE_NUMBERS
     assert (Coordinates->getNumElements() == 2);
#else
    assert (Coordinates->getNumElements() == 1);
#endif
}

MatchReporter::MatchReporter(LLVMTypeSystemInterface & ts, StreamSet * ByteStream, StreamSet * const Coordinates, Scalar * const callbackObject)
: SegmentOrientedKernel(ts, "matchReporter" + std::to_string(Coordinates->getNumElements()),
// inputs
{Binding{"InputStream", ByteStream, GreedyRate(), Deferred()},
 Binding{"Coordinates", Coordinates, GreedyRate(1)}},
// outputs
{},
// input scalars
{Binding{"accumulator_address", callbackObject}},
// output scalars
{},
// kernel state
{}) {
    setStride(1);
    addAttribute(SideEffecting());
}

    // TO DO:  investigate add linebreaks as input:  set consumed by the last linebreak?

void MatchReporter::generateDoSegmentMethod(KernelBuilder & b) {
    Module * const m = b.getModule();
    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const processMatchCoordinates = b.CreateBasicBlock("processMatchCoordinates");
    BasicBlock * const dispatch = b.CreateBasicBlock("dispatch");
    BasicBlock * const coordinatesDone = b.CreateBasicBlock("coordinatesDone");
    BasicBlock * const callFinalizeScan = b.CreateBasicBlock("callFinalizeScan");
    BasicBlock * const scanReturn = b.CreateBasicBlock("scanReturn");

    Value * accumulator = b.getScalarField("accumulator_address");
    Value * const avail = b.getAvailableItemCount("InputStream");
    Value * matchesProcessed = b.getProcessedItemCount("Coordinates");
    Value * matchesAvail = b.getAvailableItemCount("Coordinates");

    Constant * const sz_ONE = b.getSize(1);
    Type * const sizeTy = b.getSizeTy();

    b.CreateCondBr(b.CreateICmpNE(matchesProcessed, matchesAvail), processMatchCoordinates, coordinatesDone);

    b.SetInsertPoint(processMatchCoordinates);
    PHINode * phiMatchNum = b.CreatePHI(b.getSizeTy(), 2, "matchNum");
    phiMatchNum->addIncoming(matchesProcessed, entryBlock);

    Value * nextMatchNum = b.CreateAdd(phiMatchNum, sz_ONE);

    Value * matchRecordStart = b.CreateLoad(sizeTy, b.getRawInputPointer("Coordinates", b.getInt32(LINE_STARTS), phiMatchNum), "matchStartLoad");
    Value * matchRecordEnd = b.CreateLoad(sizeTy, b.getRawInputPointer("Coordinates", b.getInt32(LINE_ENDS), phiMatchNum), "matchEndLoad");
    Value * matchRecordNum = b.CreateLoad(sizeTy, b.getRawInputPointer("Coordinates", b.getInt32(LINE_NUMBERS), phiMatchNum), "matchNumLoad");

    // It is possible that the matchRecordEnd position is one past EOF.  Make sure not
    // to access past EOF.
    Value * const bufLimit = b.CreateSub(avail, sz_ONE);
    matchRecordEnd = b.CreateUMin(matchRecordEnd, bufLimit);
    // matchStart should never be past EOF, but in case it is....
    //b.CreateAssert(b.CreateICmpULT(matchRecordStart, avail), "match position past EOF");
    b.CreateCondBr(b.CreateICmpULT(matchRecordStart, avail), dispatch, callFinalizeScan);

    b.SetInsertPoint(dispatch);
    Function * const dispatcher = m->getFunction("accumulate_match_wrapper"); assert (dispatcher);
    Value * const startPtr = b.getRawInputPointer("InputStream", matchRecordStart);
    Value * const endPtr = b.getRawInputPointer("InputStream", matchRecordEnd);
    Type * expectedMatchRecNumTy = dispatcher->getFunctionType()->getParamType(1);
    assert (expectedMatchRecNumTy->isIntegerTy());
    Value * const matchRecNum = b.CreateZExtOrTrunc(matchRecordNum, expectedMatchRecNumTy);
    b.CreateCall(dispatcher->getFunctionType(), dispatcher, {accumulator, matchRecNum, startPtr, endPtr});
    Value * haveMoreMatches = b.CreateICmpNE(nextMatchNum, matchesAvail);
    phiMatchNum->addIncoming(nextMatchNum, b.GetInsertBlock());
    b.CreateCondBr(haveMoreMatches, processMatchCoordinates, coordinatesDone);

    b.SetInsertPoint(coordinatesDone);
    //b.setProcessedItemCount("InputStream", matchRecordEnd);
    b.CreateCondBr(b.isFinal(), callFinalizeScan, scanReturn);

    b.SetInsertPoint(callFinalizeScan);
    b.setProcessedItemCount("InputStream", avail);
    Function * finalizer = m->getFunction("finalize_match_wrapper"); assert (finalizer);
    Value * const bufferEnd = b.getRawInputPointer("InputStream", avail);
    b.CreateCall(finalizer->getFunctionType(), finalizer, {accumulator, bufferEnd});
    b.CreateBr(scanReturn);

    b.SetInsertPoint(scanReturn);

}

MatchFilterKernel::MatchFilterKernel(LLVMTypeSystemInterface & ts,
                                     StreamSet * const MatchStarts, StreamSet * const LineBreakStream,
                                     StreamSet * const InputStream, StreamSet * Output, unsigned strideBlocks)
: MultiBlockKernel(ts, "matchFilter" + std::to_string(strideBlocks),
{Binding{"matchStarts", MatchStarts}, Binding{"lineBreak", LineBreakStream}, Binding{"InputStream", InputStream}},
{Binding{"Output", Output, BoundedRate(0,1)}},
{},
{},
{InternalScalar{ts.getInt1Ty(), "pendingMatch"}}) {
    // The stride size must be limited so that the scanword mask is a single size_t value.
    setStride(std::min(ts.getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
    assert (MatchStarts->getNumElements() == 1);
    assert (LineBreakStream->getNumElements() == 1);
}

void MatchFilterKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    // Determine the parameters for two-level scanning.
    ScanWordParameters sw(b, mStride);

    Constant * const sz_STRIDE = b.getSize(mStride);
    Constant * const sz_BLOCKS_PER_STRIDE = b.getSize(mStride/b.getBitBlockWidth());
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);
    Type * sizeTy = b.getSizeTy();

    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const stridePrologue = b.CreateBasicBlock("stridePrologue");
    BasicBlock * const stridePrecomputation = b.CreateBasicBlock("stridePrecomputation");
    BasicBlock * const strideMasksReady = b.CreateBasicBlock("strideMasksReady");
    BasicBlock * const strideMatchProcessing = b.CreateBasicBlock("strideMatchProcessing");
    BasicBlock * const strideMatchLoop = b.CreateBasicBlock("strideMatchLoop");
    BasicBlock * const pendingMatchProcessing = b.CreateBasicBlock("pendingMatchProcessing");
    BasicBlock * const strideInitialMatch = b.CreateBasicBlock("strideInitialMatch");
    BasicBlock * const writeLF = b.CreateBasicBlock("writeLF");
    BasicBlock * const strideInitialDone = b.CreateBasicBlock("strideInitialDone");
    BasicBlock * const inStrideMatch = b.CreateBasicBlock("inStrideMatch");
    BasicBlock * const strideEndMatch = b.CreateBasicBlock("strideEndMatch");
    BasicBlock * const strideMatchesDone = b.CreateBasicBlock("strideMatchesDone");
    BasicBlock * const stridesDone = b.CreateBasicBlock("stridesDone");

    Value * const initialPos = b.getProcessedItemCount("matchStarts");
    Value * const pendingMatch = b.getScalarField("pendingMatch");
    Value * const initialProduced = b.getProducedItemCount("Output");
    Value * const avail = b.getAvailableItemCount("InputStream");
    b.CreateBr(stridePrologue);

    b.SetInsertPoint(stridePrologue);
    // Set up the loop variables as PHI nodes at the beginning of each stride.
    PHINode * const strideNo = b.CreatePHI(sizeTy, 3);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode * const pendingMatchPhi = b.CreatePHI(pendingMatch->getType(), 3, "pendingMatchPhi");
    pendingMatchPhi->addIncoming(pendingMatch, entryBlock);
    PHINode * const strideProducedPhi = b.CreatePHI(sizeTy, 3, "strideProducedPhi");
    strideProducedPhi->addIncoming(initialProduced, entryBlock);
    Value * stridePos = b.CreateAdd(initialPos, b.CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b.CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * nextStrideNo = b.CreateAdd(strideNo, sz_ONE);
    b.CreateBr(stridePrecomputation);
    // Precompute index masks for one stride of the match result and line break streams,
    // as well as a partial sum popcount of line numbers if line numbering is on.
    b.SetInsertPoint(stridePrecomputation);
    PHINode * const matchMaskAccum = b.CreatePHI(sizeTy, 2);
    matchMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const breakMaskAccum = b.CreatePHI(sizeTy, 2);
    breakMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const blockNo = b.CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, stridePrologue);
    Value * strideBlockIndex = b.CreateAdd(strideBlockOffset, blockNo);
    Value * matchBitBlock = b.loadInputStreamBlock("matchStarts", sz_ZERO, strideBlockIndex);
    Value * breakBitBlock = b.loadInputStreamBlock("lineBreak", sz_ZERO, strideBlockIndex);

    Value * const anyMatch = b.simd_any(sw.width, matchBitBlock);
    Value * const anyBreak = b.simd_any(sw.width, breakBitBlock);
    Value * matchWordMask = b.CreateZExtOrTrunc(b.hsimd_signmask(sw.width, anyMatch), sizeTy);
    Value * breakWordMask = b.CreateZExtOrTrunc(b.hsimd_signmask(sw.width, anyBreak), sizeTy);
    Value * matchMask = b.CreateOr(matchMaskAccum, b.CreateShl(matchWordMask, b.CreateMul(blockNo, sw.WORDS_PER_BLOCK)), "matchMask");
    Value * breakMask = b.CreateOr(breakMaskAccum, b.CreateShl(breakWordMask, b.CreateMul(blockNo, sw.WORDS_PER_BLOCK)), "breakMask");
    Value * const nextBlockNo = b.CreateAdd(blockNo, sz_ONE);

    matchMaskAccum->addIncoming(matchMask, stridePrecomputation);
    breakMaskAccum->addIncoming(breakMask, stridePrecomputation);
    blockNo->addIncoming(nextBlockNo, stridePrecomputation);
    b.CreateCondBr(b.CreateICmpNE(nextBlockNo, sz_BLOCKS_PER_STRIDE), stridePrecomputation, strideMasksReady);

    b.SetInsertPoint(strideMasksReady);
    // First check whether there is any pending match or any match in the stride.
    // If not we can immediately move on to the next stride.
    // We optimize for the case of no matches; the cost of the branch penalty
    // is expected to be small relative to the processing of each match.
    Value * anyMatches = b.CreateOr(b.CreateIsNotNull(pendingMatchPhi), b.CreateIsNotNull(matchMask));
    b.CreateUnlikelyCondBr(anyMatches, strideMatchProcessing, strideMatchesDone);

    b.SetInsertPoint(strideMatchProcessing);
    Value * matchWordBasePtr = b.getInputStreamBlockPtr("matchStarts", sz_ZERO, strideBlockOffset);
    matchWordBasePtr = b.CreateBitCast(matchWordBasePtr, sw.pointerTy);
    Value * breakWordBasePtr = b.getInputStreamBlockPtr("lineBreak", sz_ZERO, strideBlockOffset);
    breakWordBasePtr = b.CreateBitCast(breakWordBasePtr, sw.pointerTy);

    // Do we have a pending matched line continuing from the previous stride?
    b.CreateUnlikelyCondBr(b.CreateIsNotNull(pendingMatchPhi), pendingMatchProcessing, strideMatchLoop);

    // Precondition: we have at least one more match to process.
    b.SetInsertPoint(strideMatchLoop);
    PHINode * const matchMaskPhi = b.CreatePHI(sizeTy, 3);
    matchMaskPhi->addIncoming(matchMask, strideMatchProcessing);
    PHINode * const matchWordPhi = b.CreatePHI(sizeTy, 3);
    matchWordPhi->addIncoming(sz_ZERO, strideMatchProcessing);
    PHINode * const producedPosPhi = b.CreatePHI(sizeTy, 3);
    producedPosPhi->addIncoming(strideProducedPhi, strideMatchProcessing);

    // If we have any bits in the current matchWordPhi, continue with those, otherwise load
    // the next match word.
    Value * matchWordIdx = b.CreateCountForwardZeroes(matchMaskPhi, "matchWordIdx");
    Value * nextMatchWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, matchWordBasePtr, matchWordIdx)), sizeTy);
    Value * theMatchWord = b.CreateSelect(b.CreateICmpEQ(matchWordPhi, sz_ZERO), nextMatchWord, matchWordPhi);
    Value * matchWordPos = b.CreateAdd(stridePos, b.CreateMul(matchWordIdx, sw.WIDTH));
    Value * matchPosInWord = b.CreateCountForwardZeroes(theMatchWord);
    Value * matchPos = b.CreateAdd(matchWordPos, matchPosInWord, "matchPos");
    // We have the match start position, now find the match end position.
    Value * matchBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, matchWordIdx)), sizeTy);
    Value * notBeforeMatch = b.CreateNot(b.CreateMaskToLowestBitExclusive(theMatchWord));
    Value * followingMatchIdx = b.CreateNot(b.CreateMaskToLowestBitInclusive(matchMaskPhi));
    Value * breaksInWord = b.CreateAnd(matchBreakWord, notBeforeMatch);
    Value * breaksAfterWord = b.CreateAnd(breakMask, followingMatchIdx);
    Value * hasBreak = b.CreateOr(breaksInWord, breaksAfterWord);
    b.CreateUnlikelyCondBr(b.CreateIsNull(hasBreak), strideEndMatch, inStrideMatch);

    b.SetInsertPoint(inStrideMatch);
    // We have a matched line completely within the stride.
    Value * breakWordIdx = b.CreateCountForwardZeroes(breaksAfterWord, "breakWordIdx");
    breakWordIdx = b.CreateSelect(b.CreateIsNotNull(breaksInWord), matchWordIdx, breakWordIdx);
    Value * breakWordPos = b.CreateAdd(stridePos, b.CreateMul(breakWordIdx, sw.WIDTH));
    Value * nextBreakWord = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, breakWordIdx)), sizeTy);
    Value * breakWord = b.CreateSelect(b.CreateIsNull(breaksInWord), nextBreakWord, breaksInWord);
    Value * breakPosInWord = b.CreateCountForwardZeroes(breakWord);
    Value * matchEndPos = b.CreateAdd(breakWordPos, breakPosInWord, "breakPos");
    Value * const bufLimit = b.CreateSub(avail, sz_ONE);
    matchEndPos = b.CreateUMin(matchEndPos, bufLimit);
    Value * lineLength = b.CreateAdd(b.CreateSub(matchEndPos, matchPos), sz_ONE);
    Value * const matchStartPtr = b.getRawInputPointer("InputStream", matchPos);
    Value * const outputPtr = b.getRawOutputPointer("Output", producedPosPhi);
    b.CreateMemCpy(outputPtr, matchStartPtr, lineLength, 1);
    Value * nextProducedPos = b.CreateAdd(producedPosPhi, lineLength);

    //  We've dealt with the match, now prepare for the next one, if any.
    // There may be more matches in the current word.
    Value * dropMatch = b.CreateResetLowestBit(theMatchWord, "dropMatch");
    Value * thisWordDone = b.CreateIsNull(dropMatch);
    // There may be more matches in the match mask.
    Value * resetMatchMask = b.CreateResetLowestBit(matchMaskPhi, "nextMatchMask");
    Value * nextMatchMask = b.CreateSelect(thisWordDone, resetMatchMask, matchMaskPhi);
    BasicBlock * currentBB = b.GetInsertBlock();
    matchMaskPhi->addIncoming(nextMatchMask, currentBB);
    matchWordPhi->addIncoming(dropMatch, currentBB);
    producedPosPhi->addIncoming(nextProducedPos, currentBB);
    b.CreateCondBr(b.CreateIsNotNull(nextMatchMask), strideMatchLoop, strideMatchesDone);

    b.SetInsertPoint(pendingMatchProcessing);
    b.CreateLikelyCondBr(b.CreateIsNotNull(breakMask), strideInitialMatch, strideEndMatch);

    b.SetInsertPoint(strideInitialMatch);
    Value * breakWordIdx1 = b.CreateCountForwardZeroes(breakMask);
    Value * breakWord1 = b.CreateZExtOrTrunc(b.CreateLoad(sw.Ty, b.CreateGEP(sw.Ty, breakWordBasePtr, breakWordIdx1)), sizeTy);
    Value * breakWord1Pos = b.CreateAdd(stridePos, b.CreateMul(breakWordIdx1, sw.WIDTH));
    Value * break1Pos = b.CreateAdd(breakWord1Pos, b.CreateCountForwardZeroes(breakWord1));
    Value * initialLineLgth = b.CreateAdd(b.CreateSub(break1Pos, stridePos), sz_ONE);
    Value * const strideStartPtr = b.getRawInputPointer("InputStream", stridePos);
    Value * const outputPtr1 = b.getRawOutputPointer("Output", strideProducedPhi);
    b.CreateMemCpy(outputPtr1, strideStartPtr, initialLineLgth, 1);
    Value * producedPos1 = b.CreateAdd(strideProducedPhi, initialLineLgth);
    b.CreateCondBr(b.CreateICmpUGE(break1Pos, avail), writeLF, strideInitialDone);

    b.SetInsertPoint(writeLF);
    Value * finalBytePtr = b.getRawOutputPointer("Output", b.CreateSub(producedPos1, sz_ONE));
    finalBytePtr = b.CreateBitCast(finalBytePtr, b.getInt8PtrTy());
    b.CreateStore(b.getInt8(0x0A), finalBytePtr);
    b.CreateBr(strideInitialDone);

    b.SetInsertPoint(strideInitialDone);
    matchMaskPhi->addIncoming(matchMask, strideInitialDone);
    matchWordPhi->addIncoming(sz_ZERO, strideInitialDone);
    producedPosPhi->addIncoming(producedPos1, strideInitialDone);
    b.CreateCondBr(b.CreateIsNotNull(matchMask), strideMatchLoop, strideMatchesDone);

    b.SetInsertPoint(strideEndMatch);
    PHINode * const partialLineStart = b.CreatePHI(sizeTy, 2);
    partialLineStart->addIncoming(matchPos, strideMatchLoop);
    partialLineStart->addIncoming(stridePos, pendingMatchProcessing);
    PHINode * const partialProducedPhi = b.CreatePHI(sizeTy, 2);
    partialProducedPhi->addIncoming(producedPosPhi, strideMatchLoop);
    partialProducedPhi->addIncoming(strideProducedPhi, pendingMatchProcessing);
    Value * partialLineBytes = b.CreateSub(sz_STRIDE, b.CreateSub(partialLineStart, stridePos));
    Value * const partialLineStartPtr = b.getRawInputPointer("InputStream", partialLineStart);
    Value * const partialLineOutputPtr = b.getRawOutputPointer("Output", partialProducedPhi);
    b.CreateMemCpy(partialLineOutputPtr, partialLineStartPtr, partialLineBytes, 1);
    Value * partialProducedPos = b.CreateAdd(partialProducedPhi, partialLineBytes, "partialProducedPos");
    strideNo->addIncoming(nextStrideNo, strideEndMatch);
    pendingMatchPhi->addIncoming(ConstantInt::get(pendingMatch->getType(), 1), strideEndMatch);
    strideProducedPhi->addIncoming(partialProducedPos, strideEndMatch);
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);

    b.SetInsertPoint(strideMatchesDone);
    PHINode * const strideFinalProduced = b.CreatePHI(sizeTy, 3);
    strideFinalProduced->addIncoming(nextProducedPos, inStrideMatch);
    strideFinalProduced->addIncoming(producedPos1, strideInitialDone);
    strideFinalProduced->addIncoming(strideProducedPhi, strideMasksReady);
    strideNo->addIncoming(nextStrideNo, strideMatchesDone);
    pendingMatchPhi->addIncoming(Constant::getNullValue(pendingMatch->getType()), strideMatchesDone);
    strideProducedPhi->addIncoming(strideFinalProduced, strideMatchesDone);
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);

    b.SetInsertPoint(stridesDone);
    PHINode * const finalProducedPhi = b.CreatePHI(sizeTy, 2);
    finalProducedPhi->addIncoming(partialProducedPos, strideEndMatch);
    finalProducedPhi->addIncoming(strideFinalProduced, strideMatchesDone);
    PHINode * const finalPendingPhi = b.CreatePHI(pendingMatch->getType(), 2);
    finalPendingPhi->addIncoming(ConstantInt::get(pendingMatch->getType(), 1), strideEndMatch);
    finalPendingPhi->addIncoming(Constant::getNullValue(pendingMatch->getType()), strideMatchesDone);
    b.setScalarField("pendingMatch", finalPendingPhi);
    b.setProducedItemCount("Output", finalProducedPhi);
}

ColorizedReporter::ColorizedReporter(LLVMTypeSystemInterface & ts, StreamSet * ByteStream, StreamSet * const SourceCoords, StreamSet * const ColorizedCoords, Scalar * const callbackObject)
: SegmentOrientedKernel(ts, "colorizedReporter" + std::to_string(SourceCoords->getNumElements()) + std::to_string(ColorizedCoords->getNumElements()),
// inputs
{Binding{"InputStream", ByteStream, GreedyRate(), Deferred()},
    Binding{"SourceCoords", SourceCoords, FixedRate(1)}, Binding{"ColorizedCoords", ColorizedCoords, FixedRate(1)}},
                        // outputs
{},
                        // input scalars
{Binding{"accumulator_address", callbackObject}},
                        // output scalars
{},
                        // kernel state
{}), mColorizedLineNumberIndex(SourceCoords->getNumElements() == 3 ? (unsigned)LINE_NUMBERS : (unsigned)BATCH_LINE_NUMBERS) {
    setStride(1);
    addAttribute(SideEffecting());
}

// TO DO:  investigate add linebreaks as input:  set consumed by the last linebreak?

void ColorizedReporter::generateDoSegmentMethod(KernelBuilder & b) {
    Module * const m = b.getModule();
    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const processMatchCoordinates = b.CreateBasicBlock("processMatchCoordinates");
    BasicBlock * const dispatch = b.CreateBasicBlock("dispatch");
    BasicBlock * const coordinatesDone = b.CreateBasicBlock("coordinatesDone");
    BasicBlock * const checkFinal = b.CreateBasicBlock("checkFinal");
    BasicBlock * const callFinalizeScan = b.CreateBasicBlock("callFinalizeScan");
    BasicBlock * const scanReturn = b.CreateBasicBlock("scanReturn");

    Value * accumulator = b.getScalarField("accumulator_address");
    Value * const avail = b.getAvailableItemCount("InputStream");
    Value * matchesProcessed = b.getProcessedItemCount("SourceCoords");
    Value * matchesAvail = b.getAvailableItemCount("SourceCoords");

    Constant * const sz_ONE = b.getSize(1);
    Type * const sizeTy = b.getSizeTy();

    b.CreateCondBr(b.CreateICmpNE(matchesProcessed, matchesAvail), processMatchCoordinates, checkFinal);

    b.SetInsertPoint(processMatchCoordinates);
    PHINode * phiMatchNum = b.CreatePHI(b.getSizeTy(), 2, "matchNum");
    phiMatchNum->addIncoming(matchesProcessed, entryBlock);

    Value * nextMatchNum = b.CreateAdd(phiMatchNum, sz_ONE);

    Value * matchRecordStart = b.CreateLoad(sizeTy, b.getRawInputPointer("ColorizedCoords", b.getInt32(LINE_STARTS), phiMatchNum), "matchStartLoad");
    Value * matchRecordEnd = b.CreateLoad(sizeTy, b.getRawInputPointer("ColorizedCoords", b.getInt32(LINE_ENDS), phiMatchNum), "matchEndLoad");
    #ifdef WRITE_FILE_NUMBERS
    Value * matchFileNum = b.CreateLoad(sizeTy, b.getRawInputPointer("SourceCoords", b.getInt32(BATCH_FILE_NUMBERS), phiMatchNum), "matchFileNumLoad");
    #endif
    Value * matchRecordNum = b.CreateLoad(sizeTy, b.getRawInputPointer("SourceCoords", b.getInt32(mColorizedLineNumberIndex), phiMatchNum), "matchNumLoad");

    // It is possible that the matchRecordEnd position is one past EOF.  Make sure not
    // to access past EOF.
    Value * const bufLimit = b.CreateSub(avail, sz_ONE);
    matchRecordEnd = b.CreateUMin(matchRecordEnd, bufLimit);
    // matchStart should never be past EOF, but in case it is....
    //b.CreateAssert(b.CreateICmpULT(matchRecordStart, avail), "match position past EOF");
    b.CreateCondBr(b.CreateICmpULT(matchRecordStart, avail), dispatch, callFinalizeScan);

    b.SetInsertPoint(dispatch);
    Function * const dispatcher = m->getFunction("accumulate_match_wrapper"); assert (dispatcher);



    Value * const startPtr = b.getRawInputPointer("InputStream", matchRecordStart);
    Value * const endPtr = b.getRawInputPointer("InputStream", matchRecordEnd);
    Type * expectedMatchRecNumTy = dispatcher->getFunctionType()->getParamType(1);
    assert (expectedMatchRecNumTy->isIntegerTy());
    Value * const matchRecNum = b.CreateZExtOrTrunc(matchRecordNum, expectedMatchRecNumTy);
    b.CreateCall(dispatcher->getFunctionType(), dispatcher, {accumulator, matchRecNum, startPtr, endPtr});
    Value * haveMoreMatches = b.CreateICmpNE(nextMatchNum, matchesAvail);
    BasicBlock * const dispatchEnd = b.GetInsertBlock();
    phiMatchNum->addIncoming(nextMatchNum, dispatchEnd);
    b.CreateCondBr(haveMoreMatches, processMatchCoordinates, coordinatesDone);

    b.SetInsertPoint(coordinatesDone);
    b.setProcessedItemCount("InputStream", matchRecordEnd);
    b.CreateBr(checkFinal);

    b.SetInsertPoint(checkFinal);
    b.CreateCondBr(b.isFinal(), callFinalizeScan, scanReturn);

    b.SetInsertPoint(callFinalizeScan);
    b.setProcessedItemCount("InputStream", avail);
    Function * finalizer = m->getFunction("finalize_match_wrapper"); assert (finalizer);
    Value * const bufferEnd = b.getRawInputPointer("InputStream", avail);
    b.CreateCall(finalizer->getFunctionType(), finalizer, {accumulator, bufferEnd});
    b.CreateBr(scanReturn);

    b.SetInsertPoint(scanReturn);

}

}
