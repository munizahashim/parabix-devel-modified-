/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/streamutils/until_n.h>

#include <llvm/IR/Module.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <toolchain/toolchain.h>
#include <boost/intrusive/detail/math.hpp>

using boost::intrusive::detail::floor_log2;

namespace llvm { class Type; }

using namespace llvm;

namespace kernel {

void UntilNkernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {

/*
   Strategy:  first form an index consisting of one bit per packsize input positions,
   with a 1 bit signifying that the corresponding pack has at least one 1 bit.
   Build the index one pack at a time, i.e, packsize * packsize positions at a time.
   After an index pack is constructed, scan the index pack for 1 bits.  Each 1 bit
   found identifies an input pack with a nonzero popcount.  Take the actual popcount
   of the corresponding input pack and update the total number of bits seen.   If
   the number of bits seen reaches N with any pack, determine the position of the
   Nth bit and signal termination at that point.

   For normal processing, we process whole blocks only, always advanced processed
   and produced item counts by an integral number of blocks.   For final block
   processing, we treat the final partial block as a whole block for the purpose
   of finding the Nth bit.   However, if the located bit position is past the
   EOF position, then this is treated as if the Nth bit does not exist in the
   input stream.
*/
    IntegerType * const sizeTy = b.getSizeTy();
    const unsigned packSize = sizeTy->getBitWidth();
    Constant * const ZERO = b.getSize(0);
    Constant * const ONE = b.getSize(1);
    Constant * const BLOCK_BYTES = b.getSize(b.getBitBlockWidth()/8);
    const auto packsPerBlock = b.getBitBlockWidth() / packSize;
    Constant * const PACK_SIZE = b.getSize(packSize);
    Constant * const PACKS_PER_BLOCK = b.getSize(packsPerBlock);
    const auto blocksPerStride = getStride() / b.getBitBlockWidth();
    Constant * const BLOCKS_PER_STRIDE = b.getSize(blocksPerStride);
    const auto maximumBlocksPerIteration = packSize / packsPerBlock;
    Constant * const MAXIMUM_BLOCKS_PER_ITERATION = b.getSize(maximumBlocksPerIteration);
    VectorType * const packVectorTy = b.fwVectorType(packSize);

    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const segmentDone = b.CreateBasicBlock("segmentDone");

    Value * const numOfBlocks = b.CreateMul(numOfStrides, BLOCKS_PER_STRIDE);
    BasicBlock * const strideLoop = b.CreateBasicBlock("strideLoop");
    Value * const N = b.getScalarField("N");
    Value * const observedSoFar = b.getScalarField("observed");

    if (mMode == UntilNkernel::Mode::TerminateAtN) {
        b.CreateBr(strideLoop);
    } else {
        BasicBlock * const memZeroSegment = b.CreateBasicBlock("memZeroSegment");
        b.CreateCondBr(b.CreateICmpULT(observedSoFar, N), strideLoop, memZeroSegment);
        b.SetInsertPoint(memZeroSegment);
        Value * outputPtr = b.getOutputStreamBlockPtr("uptoN", ZERO, ZERO);
        outputPtr = b.CreatePointerCast(outputPtr, b.getInt8PtrTy());
        Value * bytesToZero = b.CreateMul(numOfBlocks, BLOCK_BYTES);
        b.CreateMemZero(outputPtr, bytesToZero, /* alignment = */ b.getBitBlockWidth()/8);
        b.CreateBr(segmentDone);
    }

    b.SetInsertPoint(strideLoop);
    PHINode * const baseBlockIndex = b.CreatePHI(sizeTy, 2);
    baseBlockIndex->addIncoming(ZERO, entry);
    PHINode * const blocksRemaining = b.CreatePHI(sizeTy, 2);
    blocksRemaining->addIncoming(numOfBlocks, entry);
    Value * const blocksToDo = b.CreateUMin(blocksRemaining, MAXIMUM_BLOCKS_PER_ITERATION);
    BasicBlock * const iteratorLoop = b.CreateBasicBlock("iteratorLoop");
    BasicBlock * const checkForMatches = b.CreateBasicBlock("checkForMatches");
    b.CreateBr(iteratorLoop);

    // Construct the outer iterator mask indicating whether any markers are in the stream.
    b.SetInsertPoint(iteratorLoop);
    PHINode * const groupMaskPhi = b.CreatePHI(sizeTy, 2);
    groupMaskPhi->addIncoming(ZERO, strideLoop);
    PHINode * const localIndex = b.CreatePHI(sizeTy, 2);
    localIndex->addIncoming(ZERO, strideLoop);
    Value * const blockIndex = b.CreateAdd(baseBlockIndex, localIndex);
    Value * inputValue = b.loadInputStreamBlock("bits", ZERO, blockIndex);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::DisableInOutAttributes))) {
        b.storeOutputStreamBlock("uptoN", ZERO, blockIndex, inputValue);
    }
    Value * const inputPackValue = b.simd_any(packSize, inputValue);
    Value * iteratorMask = b.CreateZExtOrTrunc(b.hsimd_signmask(packSize, inputPackValue), sizeTy);
    iteratorMask = b.CreateShl(iteratorMask, b.CreateMul(localIndex, PACKS_PER_BLOCK));
    iteratorMask = b.CreateOr(groupMaskPhi, iteratorMask);
    groupMaskPhi->addIncoming(iteratorMask, iteratorLoop);
    Value * const nextLocalIndex = b.CreateAdd(localIndex, ONE);
    localIndex->addIncoming(nextLocalIndex, iteratorLoop);
    b.CreateCondBr(b.CreateICmpNE(nextLocalIndex, blocksToDo), iteratorLoop, checkForMatches);

    // Now check whether we have any matches
    b.SetInsertPoint(checkForMatches);

    BasicBlock * const processGroups = b.CreateBasicBlock("processGroups");
    BasicBlock * const nextStride = b.CreateBasicBlock("nextStride");
    b.CreateLikelyCondBr(b.CreateIsNull(iteratorMask), nextStride, processGroups);

    b.SetInsertPoint(processGroups);
    Value * const initiallyObserved = b.getScalarField("observed");
    BasicBlock * const processGroup = b.CreateBasicBlock("processGroup", nextStride);
    b.CreateBr(processGroup);

    b.SetInsertPoint(processGroup);
    PHINode * const observed = b.CreatePHI(initiallyObserved->getType(), 2);
    observed->addIncoming(initiallyObserved, processGroups);
    PHINode * const groupMarkers = b.CreatePHI(iteratorMask->getType(), 2);
    groupMarkers->addIncoming(iteratorMask, processGroups);

    Value * const groupIndex = b.CreateZExtOrTrunc(b.CreateCountForwardZeroes(groupMarkers), sizeTy);
    Value * const blockIndex2 = b.CreateAdd(baseBlockIndex, b.CreateUDiv(groupIndex, PACKS_PER_BLOCK));
    Value * const packOffset = b.CreateURem(groupIndex, PACKS_PER_BLOCK);
    Value * const groupValue = b.loadInputStreamBlock("bits", ZERO, blockIndex2);
    Value * const packBits = b.CreateExtractElement(b.CreateBitCast(groupValue, packVectorTy), packOffset);
    Value * const packCount = b.CreateZExtOrTrunc(b.CreatePopcount(packBits), sizeTy);
    Value * const observedUpTo = b.CreateAdd(observed, packCount);
    BasicBlock * const haveNotSeenEnough = b.CreateBasicBlock("haveNotSeenEnough", nextStride);
    BasicBlock * const seenNOrMore = b.CreateBasicBlock("seenNOrMore", nextStride);
    b.CreateLikelyCondBr(b.CreateICmpULT(observedUpTo, N), haveNotSeenEnough, seenNOrMore);

    // update our kernel state and check whether we have any other groups to process
    b.SetInsertPoint(haveNotSeenEnough);
    observed->addIncoming(observedUpTo, haveNotSeenEnough);
    b.setScalarField("observed", observedUpTo);
    Value * const remainingGroupMarkers = b.CreateResetLowestBit(groupMarkers);
    groupMarkers->addIncoming(remainingGroupMarkers, haveNotSeenEnough);
    b.CreateLikelyCondBr(b.CreateIsNull(remainingGroupMarkers), nextStride, processGroup);

    // we've seen N non-zero items; determine the position of our items and clear any subsequent markers
    b.SetInsertPoint(seenNOrMore);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b.CreateAssert(b.CreateICmpUGT(N, observed), "N must be greater than observed count!");
    }
    Value * const bitsToFind = b.CreateSub(N, observed);
    BasicBlock * const findNthBit = b.CreateBasicBlock("findNthBit", nextStride);
    BasicBlock * const foundNthBit = b.CreateBasicBlock("foundNthBit", nextStride);
    b.CreateBr(findNthBit);

    b.SetInsertPoint(findNthBit);
    PHINode * const remainingBitsToFind = b.CreatePHI(bitsToFind->getType(), 2);
    remainingBitsToFind->addIncoming(bitsToFind, seenNOrMore);
    PHINode * const remainingBits = b.CreatePHI(packBits->getType(), 2);
    remainingBits->addIncoming(packBits, seenNOrMore);
    Value * const nextRemainingBits = b.CreateResetLowestBit(remainingBits);
    remainingBits->addIncoming(nextRemainingBits, findNthBit);
    Value * const nextRemainingBitsToFind = b.CreateSub(remainingBitsToFind, ONE);
    remainingBitsToFind->addIncoming(nextRemainingBitsToFind, findNthBit);
    b.CreateLikelyCondBr(b.CreateIsNull(nextRemainingBitsToFind), foundNthBit, findNthBit);

    // If we've found the n-th bit, end the segment after clearing the markers
    b.SetInsertPoint(foundNthBit);
    b.setScalarField("observed", N);
    Value * const packPosition = b.CreateZExtOrTrunc(b.CreateCountForwardZeroes(remainingBits), sizeTy);
    Value * const basePosition = b.CreateMul(packOffset, PACK_SIZE);
    Value * const blockOffset = b.CreateOr(basePosition, packPosition);
    Value * const priorProducedItemCount = b.getProducedItemCount("uptoN");
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::DisableInOutAttributes))) {
        if ((mMode == Mode::ZeroAfterN) || (mMode == Mode::TerminateAtN)) {
            Value * const inputValue2 = b.loadInputStreamBlock("bits", ZERO, blockIndex2);
            Value * const mask = b.bitblock_mask_to(blockOffset, true);
            Value * const maskedInputValue = b.CreateAnd(inputValue2, mask, "untilNmasked");
            b.storeOutputStreamBlock("uptoN", ZERO, blockIndex2, maskedInputValue);
        }
    }
    const auto log2BlockWidth = floor_log2(b.getBitBlockWidth());
    Value * positionOfNthItem = nullptr;
    if ((mMode == Mode::TerminateAtN) || (mMode == Mode::ReportAcceptedLengthAtAndBeforeN)) {
        positionOfNthItem = b.CreateShl(blockIndex2, log2BlockWidth);
        positionOfNthItem = b.CreateAdd(positionOfNthItem, b.CreateAdd(blockOffset, ONE));
        positionOfNthItem = b.CreateAdd(positionOfNthItem, priorProducedItemCount);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            Value * const availableBits = b.getAvailableItemCount("bits");
            Value * const positionLessThanAvail = b.CreateICmpULT(positionOfNthItem, availableBits);
            b.CreateAssert(positionLessThanAvail, "position of n-th item exceeds available items!");
        }
        b.setProducedItemCount("uptoN", positionOfNthItem);
        b.setTerminationSignal();
    } else {
        Value * nextBlk = b.CreateAdd(blockIndex2, ONE);
        BasicBlock * const memZeroRemaining = b.CreateBasicBlock("memZeroRemaining");
        b.CreateCondBr(b.CreateICmpULT(nextBlk, numOfBlocks), memZeroRemaining, segmentDone);

        b.SetInsertPoint(memZeroRemaining);
        Value * outputPtr = b.getOutputStreamBlockPtr("uptoN", ZERO, nextBlk);
        outputPtr = b.CreatePointerCast(outputPtr, b.getInt8PtrTy());
        Value * bytesToZero = b.CreateMul(b.CreateSub(numOfBlocks, nextBlk), BLOCK_BYTES);
        b.CreateMemZero(outputPtr, bytesToZero, /* alignment = */ b.getBitBlockWidth()/8);
    }
    b.CreateBr(segmentDone);

    b.SetInsertPoint(nextStride);
    blocksRemaining->addIncoming(b.CreateSub(blocksRemaining, MAXIMUM_BLOCKS_PER_ITERATION), nextStride);
    baseBlockIndex->addIncoming(b.CreateAdd(baseBlockIndex, MAXIMUM_BLOCKS_PER_ITERATION), nextStride);
    Value * const done = b.CreateICmpULE(blocksRemaining, MAXIMUM_BLOCKS_PER_ITERATION);
    b.CreateLikelyCondBr(done, segmentDone, strideLoop);

    b.SetInsertPoint(segmentDone);
}

UntilNkernel::UntilNkernel(LLVMTypeSystemInterface & ts, Scalar * N, StreamSet * Markers, StreamSet * FirstN, UntilNkernel::Mode m)
: MultiBlockKernel(ts, [&]() -> std::string {
    std::string tmp;
    raw_string_ostream nm(tmp);
    nm << "UntilN_";
    switch (m) {
        case Mode::TerminateAtN:
            nm << "t"; break;
        case Mode::ZeroAfterN:
            nm << "z"; break;
        case Mode::ReportAcceptedLengthAtAndBeforeN:
            nm << "r"; break;
    }
    nm.flush();
    return tmp;
}(),
{Binding{"bits", Markers}},
// outputs
{Binding{"uptoN", FirstN, FixedRate(), InOut("bits")}},
// input scalar
{Binding{"N", N}}, {},
// internal state
{InternalScalar{N->getType(), "observed"}}), mMode(m) {
    switch (m) {
        case Mode::TerminateAtN:
        case Mode::ReportAcceptedLengthAtAndBeforeN:
            addAttribute(CanTerminateEarly());
        default: break;
    }
}

}
