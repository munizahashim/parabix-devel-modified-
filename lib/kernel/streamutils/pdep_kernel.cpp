/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/deletion.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <kernel/core/idisa_target.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/pipeline/driver/driver.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <pablo/pablo_kernel.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/bixnum/bixnum.h>

using namespace llvm;
using namespace pablo;
using namespace kernel;

namespace kernel {

void SpreadByMask(PipelineBuilder & P,
                  StreamSet * mask, StreamSet * toSpread, StreamSet * outputs,
                  unsigned streamOffset,
                  bool zeroExtend,
                  StreamExpandOptimization opt,
                  unsigned expansionFieldWidth,
                  ProcessingRateProbabilityDistribution itemsPerOutputUnit) {
    unsigned streamCount = outputs->getNumElements();
    StreamSet * const expanded = P.CreateStreamSet(streamCount);
    Scalar * base = P.CreateConstant(P.getDriver().getBuilder().getSize(streamOffset));
    P.CreateKernelCall<StreamExpandKernel>(mask, toSpread, expanded, base, zeroExtend, opt, expansionFieldWidth, itemsPerOutputUnit);
    P.CreateKernelCall<FieldDepositKernel>(mask, expanded, outputs, expansionFieldWidth);
}


void MergeByMask(PipelineBuilder & P,
                 StreamSet * mask, StreamSet * a, StreamSet * b, StreamSet * merged) {
    unsigned elems = merged->getNumElements();
    if ((a->getNumElements() != elems) || (b->getNumElements() != elems)) {
        llvm::report_fatal_error("MergeByMask called with incompatible element counts");
    }
    unsigned streamOffset = 0;
    Scalar * base = P.CreateConstant(P.getDriver().getBuilder().getSize(streamOffset));
    int expansionFieldWidth=64;
    P.CreateKernelCall<StreamMergeKernel>(mask,a,b,merged,base,expansionFieldWidth);

}


const unsigned StreamExpandStrideSize = 4;

StreamExpandKernel::StreamExpandKernel(KernelBuilder & b,
                                       StreamSet * mask,
                                       StreamSet * source,
                                       StreamSet * expanded,
                                       Scalar * base,
                                       bool zeroExtend,
                                       const StreamExpandOptimization opt,
                                       const unsigned FieldWidth, ProcessingRateProbabilityDistribution itemsPerOutputUnitProbability)
: MultiBlockKernel(b, [&]() -> std::string {
                        std::string tmp;
                        raw_string_ostream nm(tmp);
                        nm << "streamExpand"  << StreamExpandStrideSize << ':' << FieldWidth;
                        if (opt == StreamExpandOptimization::NullCheck)  {
                            nm << 'N';
                        }
                        nm << '_' << source->getNumElements() << ':' << expanded->getNumElements();
                        if (zeroExtend) nm << "z";
                        nm.flush();
                        return tmp;
                    }(),
{Bind("marker", mask, Principal())},
{Binding{"output", expanded}},
// input scalar
{Binding{"base", base}},
{}, {})
, mFieldWidth(FieldWidth)
, mSelectedStreamCount(expanded->getNumElements())
, mOptimization(opt) {
    setStride(StreamExpandStrideSize * b.getBitBlockWidth());
    if (zeroExtend) {
        mInputStreamSets.push_back(Bind("source", source, PopcountOf("marker"), itemsPerOutputUnitProbability, ZeroExtended(), BlockSize(b.getBitBlockWidth())));
    } else {
        mInputStreamSets.push_back(Bind("source", source, PopcountOf("marker"), itemsPerOutputUnitProbability, BlockSize(b.getBitBlockWidth())));
    }
}

void StreamExpandKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    Type * fieldWidthTy = b.getIntNTy(mFieldWidth);
    Type * sizeTy = b.getSizeTy();
    const unsigned numFields = b.getBitBlockWidth() / mFieldWidth;

    Constant * const ZERO = b.getSize(0);
    Constant * BLOCK_WIDTH = ConstantInt::get(sizeTy, b.getBitBlockWidth());
    Constant * FIELD_WIDTH = ConstantInt::get(sizeTy, mFieldWidth);
    Constant * fwSplat = b.getSplat(numFields, ConstantInt::get(fieldWidthTy, mFieldWidth));
    Constant * fw_sub1Splat = b.getSplat(numFields, ConstantInt::get(fieldWidthTy, mFieldWidth - 1));

    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * expandLoop = b.CreateBasicBlock("expandLoop");
    BasicBlock * expansionDone = b.CreateBasicBlock("expansionDone");
    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        assert ((getStride() % b.getBitBlockWidth()) == 0);
        ConstantInt * const mult = b.getSize(getStride() / b.getBitBlockWidth());
        numOfBlocks = b.CreateMul(numOfStrides, mult);
    }
    Value * processedSourceItems = b.getProcessedItemCount("source");
    Value * initialSourceOffset = b.CreateURem(processedSourceItems, BLOCK_WIDTH);
    Value * const streamBase = b.getScalarField("base");

    SmallVector<Value *, 16> pendingData(mSelectedStreamCount);
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        Constant * const streamIndex = ConstantInt::get(streamBase->getType(), i);
        Value * const streamOffset = b.CreateAdd(streamBase, streamIndex);
        pendingData[i] = b.loadInputStreamBlock("source", streamOffset, ZERO);
    }

    b.CreateBr(expandLoop);
    // Main Loop
    b.SetInsertPoint(expandLoop);
    const unsigned incomingCount = (mOptimization == StreamExpandOptimization::NullCheck) ? 3 : 2;
    PHINode * blockNoPhi = b.CreatePHI(b.getSizeTy(), incomingCount);
    PHINode * pendingOffsetPhi = b.CreatePHI(b.getSizeTy(), incomingCount);
    SmallVector<PHINode *, 16> pendingDataPhi(mSelectedStreamCount);
    blockNoPhi->addIncoming(ZERO, entry);
    pendingOffsetPhi->addIncoming(initialSourceOffset, entry);

    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        pendingDataPhi[i] = b.CreatePHI(b.getBitBlockType(), incomingCount);
        pendingDataPhi[i]->addIncoming(pendingData[i], entry);
    }
    Value * deposit_mask = b.loadInputStreamBlock("marker", ZERO, blockNoPhi);
    Value * nextBlk = b.CreateAdd(blockNoPhi, b.getSize(1));
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
    if (mOptimization == StreamExpandOptimization::NullCheck) {
        BasicBlock * expandLoopContinue = b.CreateBasicBlock("expandLoopContinue");
        BasicBlock * nullMarkers = b.CreateBasicBlock("nullMarkers");
        b.CreateCondBr(b.bitblock_any(deposit_mask), expandLoopContinue, nullMarkers);

        b.SetInsertPoint(nullMarkers);
        Constant * zeroes = b.allZeroes();
        for (unsigned i = 0; i < mSelectedStreamCount; i++) {
            b.storeOutputStreamBlock("output", b.getInt32(i), blockNoPhi, zeroes);
        }
        blockNoPhi->addIncoming(nextBlk, nullMarkers);
        pendingOffsetPhi->addIncoming(pendingOffsetPhi, nullMarkers);
        for (unsigned i = 0; i < mSelectedStreamCount; i++) {
            pendingDataPhi[i]->addIncoming(pendingDataPhi[i], nullMarkers);
        }
        b.CreateCondBr(moreToDo, expandLoop, expansionDone);

        b.SetInsertPoint(expandLoopContinue);
    }
    // Calculate the field values and offsets we need for assembling a
    // a full block of source bits.  Assembly will use the following operations.
    // A = b.simd_srlv(fw, b.mvmd_dsll(fw, source, pending, field_offset_lo), bit_offset);
    // B = b.simd_sllv(fw, b.mvmd_dsll(fw, source, pending, field_offset_hi), shift_fwd);
    // all_source_bits = simd_or(A, B);
    Value * pendingOffset = b.CreateURem(pendingOffsetPhi, BLOCK_WIDTH);

    // Value * pendingItems = b.CreateURem(b.CreateSub(bwConst, pendingOffset), bwConst);
    Value * pendingItems = b.CreateSub(BLOCK_WIDTH, pendingOffset);

    Value * field_offset_lo = b.CreateCeilUDiv(pendingItems, FIELD_WIDTH);
    Value * bit_offset = b.simd_fill(mFieldWidth, b.CreateURem(pendingOffset, FIELD_WIDTH));
    // Carefully avoid a shift by the full fieldwith (which gives a poison value).
    // field_offset_lo + 1 unless the bit_offset is 0, in which case it is just field_offset_lo.
    Value * field_offset_hi =  b.CreateUDiv(pendingItems, FIELD_WIDTH);
    // fw - bit_offset, unless bit_offset is 0, in which case, the shift_fwd is 0.
    Value * shift_fwd = b.CreateURem(b.CreateSub(fwSplat, bit_offset), fwSplat);

    // Once all source bits are assembled, they need to be distributed to the
    // output fields in accord with the popcounts of the deposit mask fields.
    // The bits for each output field will typically come from (at most) two
    // source fields, with offsets.  Calculate the field numbers and offsets.

    Value * fieldPopCounts = b.simd_popcount(mFieldWidth, deposit_mask);
    // For each field determine the (partial) sum popcount of all fields prior to
    // the current field.

    Value * partialSum = b.hsimd_partial_sum(mFieldWidth, fieldPopCounts);
    Value * const blockPopCount = b.CreateZExtOrTrunc(b.CreateExtractElement(partialSum, numFields - 1), sizeTy);
    partialSum = b.mvmd_slli(mFieldWidth, partialSum, 1);

    Value * const source_field_lo = b.CreateUDiv(partialSum, fwSplat);
    Value * const source_field_hi = b.CreateUDiv(b.CreateAdd(partialSum, fw_sub1Splat), fwSplat);
    Value * const source_shift_lo = b.CreateAnd(partialSum, fw_sub1Splat);  // parallel URem
    Value * const source_shift_hi = b.CreateAnd(b.CreateSub(fwSplat, source_shift_lo), fw_sub1Splat);
    // The source stream may not be positioned at a block boundary.  Partial data
    // has been saved in the kernel state, determine the next full block number
    // for loading source streams.
    Value * const newPendingOffset = b.CreateAdd(pendingOffsetPhi, blockPopCount);
    Value * const srcBlockNo = b.CreateUDiv(newPendingOffset, BLOCK_WIDTH);

    // Now load and process source streams.
    SmallVector<Value *, 16> sourceData(mSelectedStreamCount);
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        Constant * const streamIndex = ConstantInt::get(streamBase->getType(), i);
        Value * const streamOffset = b.CreateAdd(streamBase, streamIndex);
        sourceData[i] = b.loadInputStreamBlock("source", streamOffset, srcBlockNo);
        Value * A = b.simd_srlv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData[i], pendingDataPhi[i], field_offset_lo), bit_offset);
        Value * B = b.simd_sllv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData[i], pendingDataPhi[i], field_offset_hi), shift_fwd);
        Value * full_source_block = b.CreateOr(A, B, "toExpand");
        Value * C = b.simd_srlv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block, source_field_lo), source_shift_lo);
        Value * D = b.simd_sllv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block, source_field_hi), source_shift_hi);
        Value * output = b.bitCast(b.CreateOr(C, D, "expanded"));
        b.storeOutputStreamBlock("output", b.getInt32(i), blockNoPhi, output);
    }
    //
    // Update loop control Phis for the next iteration.
    //
    blockNoPhi->addIncoming(nextBlk, b.GetInsertBlock());
    pendingOffsetPhi->addIncoming(newPendingOffset, b.GetInsertBlock());
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        pendingDataPhi[i]->addIncoming(sourceData[i], b.GetInsertBlock());
    }
    //
    // Now continue the loop if there are more blocks to process.
    b.CreateCondBr(moreToDo, expandLoop, expansionDone);

    b.SetInsertPoint(expansionDone);
}



/*************************StreamMergeKernel*********************************/
const unsigned StreamMergeStrideSize = 4;
StreamMergeKernel::StreamMergeKernel(KernelBuilder & b,
                                       StreamSet * mask,
                                       StreamSet * source1,
                                       StreamSet * source2,
                                       StreamSet * merged,
                                       Scalar * base,
                                       const unsigned FieldWidth)
: MultiBlockKernel(b, [&]() -> std::string {
                        std::string tmp;
                        raw_string_ostream nm(tmp);
                        nm << "streamMerge"  << StreamMergeStrideSize << ':' << FieldWidth;
                        nm << '_' << source1->getNumElements() << ':' << merged->getNumElements();
                        nm.flush();
                        return tmp;
                    }(),
{Binding("marker", mask, FixedRate(1), Principal()),Binding("source1", source1, PopcountOf("marker")), Binding("source2", source2, PopcountOfNot("marker"))},
{Binding{"mergedStreamSet", merged}},
// input scalar
{Binding{"base", base}},
{}, {})
, mFieldWidth(FieldWidth)
, mSelectedStreamCount(merged->getNumElements()){
    setStride(StreamMergeStrideSize * b.getBitBlockWidth());
}

void StreamMergeKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    Type * fieldWidthTy = b.getIntNTy(mFieldWidth);
    Type * sizeTy = b.getSizeTy();
    const unsigned numFields = b.getBitBlockWidth() / mFieldWidth;

    Constant * const ZERO = b.getSize(0);
    Constant * BLOCK_WIDTH = ConstantInt::get(sizeTy, b.getBitBlockWidth());
    Constant * FIELD_WIDTH = ConstantInt::get(sizeTy, mFieldWidth);
    Constant * fwSplat = b.getSplat(numFields, ConstantInt::get(fieldWidthTy, mFieldWidth));
    Constant * fw_sub1Splat = b.getSplat(numFields, ConstantInt::get(fieldWidthTy, mFieldWidth - 1));

    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * mergeLoop = b.CreateBasicBlock("mergeLoop");
    BasicBlock * mergenceDone = b.CreateBasicBlock("mergenceDone");
    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        assert ((getStride() % b.getBitBlockWidth()) == 0);
        ConstantInt * const mult = b.getSize(getStride() / b.getBitBlockWidth());
        numOfBlocks = b.CreateMul(numOfStrides, mult);
    }
    Value * processedSourceItems1 = b.getProcessedItemCount("source1");
    Value * processedSourceItems2 = b.getProcessedItemCount("source2");
    Value * initialSourceOffset1 = b.CreateURem(processedSourceItems1, BLOCK_WIDTH);
    Value * initialSourceOffset2 = b.CreateURem(processedSourceItems2, BLOCK_WIDTH);
    Value * const streamBase = b.getScalarField("base");

    SmallVector<Value *, 16> pendingData1(mSelectedStreamCount);
    SmallVector<Value *, 16> pendingData2(mSelectedStreamCount);
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        Constant * const streamIndex = ConstantInt::get(streamBase->getType(), i);
        Value * const streamOffset = b.CreateAdd(streamBase, streamIndex);
        pendingData1[i] = b.loadInputStreamBlock("source1", streamOffset, ZERO);
        pendingData2[i] = b.loadInputStreamBlock("source2", streamOffset, ZERO);
    }

    b.CreateBr(mergeLoop);
    // Main Loop
    b.SetInsertPoint(mergeLoop);
    const unsigned incomingCount = 2;
    PHINode * blockNoPhi = b.CreatePHI(b.getSizeTy(), incomingCount);
    PHINode * pendingOffsetPhi1 = b.CreatePHI(b.getSizeTy(), incomingCount);
    PHINode * pendingOffsetPhi2 = b.CreatePHI(b.getSizeTy(), incomingCount);
    SmallVector<PHINode *, 16> pendingDataPhi1(mSelectedStreamCount);
    SmallVector<PHINode *, 16> pendingDataPhi2(mSelectedStreamCount);
    blockNoPhi->addIncoming(ZERO, entry);
    pendingOffsetPhi1->addIncoming(initialSourceOffset1, entry);
    pendingOffsetPhi2->addIncoming(initialSourceOffset2, entry);

    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        pendingDataPhi1[i] = b.CreatePHI(b.getBitBlockType(), incomingCount);
        pendingDataPhi1[i]->addIncoming(pendingData1[i], entry);
        pendingDataPhi2[i] = b.CreatePHI(b.getBitBlockType(), incomingCount);
        pendingDataPhi2[i]->addIncoming(pendingData2[i], entry);
    }
    Value * mask1 = b.loadInputStreamBlock("marker", ZERO, blockNoPhi);
    Value * mask2 = b.CreateNot(mask1);
    Value * nextBlk = b.CreateAdd(blockNoPhi, b.getSize(1));
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);

    Value * pendingOffset1 = b.CreateURem(pendingOffsetPhi1, BLOCK_WIDTH);
    Value * pendingOffset2 = b.CreateURem(pendingOffsetPhi2, BLOCK_WIDTH);

    Value * pendingItems1 = b.CreateSub(BLOCK_WIDTH, pendingOffset1);
    Value * pendingItems2 = b.CreateSub(BLOCK_WIDTH, pendingOffset2);

    Value * field_offset_lo1 = b.CreateCeilUDiv(pendingItems1, FIELD_WIDTH);
    Value * bit_offset1 = b.simd_fill(mFieldWidth, b.CreateURem(pendingOffset1, FIELD_WIDTH));

    Value * field_offset_lo2 = b.CreateCeilUDiv(pendingItems2, FIELD_WIDTH);
    Value * bit_offset2 = b.simd_fill(mFieldWidth, b.CreateURem(pendingOffset2, FIELD_WIDTH));

    Value * field_offset_hi1 =  b.CreateUDiv(pendingItems1, FIELD_WIDTH);
    Value * field_offset_hi2 =  b.CreateUDiv(pendingItems2, FIELD_WIDTH);

    Value * shift_fwd1 = b.CreateURem(b.CreateSub(fwSplat, bit_offset1), fwSplat);
    Value * shift_fwd2 = b.CreateURem(b.CreateSub(fwSplat, bit_offset2), fwSplat);

    Value * fieldPopCounts1 = b.simd_popcount(mFieldWidth, mask1);
    Value * fieldPopCounts2 = b.simd_popcount(mFieldWidth, mask2);


    Value * partialSum1 = b.hsimd_partial_sum(mFieldWidth, fieldPopCounts1);
    Value * partialSum2 = b.hsimd_partial_sum(mFieldWidth, fieldPopCounts2);
    Value * const blockPopCount1 = b.CreateZExtOrTrunc(b.CreateExtractElement(partialSum1, numFields - 1), sizeTy);
    Value * const blockPopCount2 = b.CreateZExtOrTrunc(b.CreateExtractElement(partialSum2, numFields - 1), sizeTy);
    partialSum1 = b.mvmd_slli(mFieldWidth, partialSum1, 1);
    partialSum2 = b.mvmd_slli(mFieldWidth, partialSum2, 1);

    Value * const source_field_lo1 = b.CreateUDiv(partialSum1, fwSplat);
    Value * const source_field_hi1 = b.CreateUDiv(b.CreateAdd(partialSum1, fw_sub1Splat), fwSplat);
    Value * const source_shift_lo1 = b.CreateAnd(partialSum1, fw_sub1Splat);  // parallel URem
    Value * const source_shift_hi1 = b.CreateAnd(b.CreateSub(fwSplat, source_shift_lo1), fw_sub1Splat);

    Value * const source_field_lo2 = b.CreateUDiv(partialSum2, fwSplat);
    Value * const source_field_hi2 = b.CreateUDiv(b.CreateAdd(partialSum2, fw_sub1Splat), fwSplat);
    Value * const source_shift_lo2 = b.CreateAnd(partialSum2, fw_sub1Splat);
    Value * const source_shift_hi2 = b.CreateAnd(b.CreateSub(fwSplat, source_shift_lo2), fw_sub1Splat);

    Value * const newPendingOffset1 = b.CreateAdd(pendingOffsetPhi1, blockPopCount1);
    Value * const srcBlockNo1 = b.CreateUDiv(newPendingOffset1, BLOCK_WIDTH);

    Value * const newPendingOffset2 = b.CreateAdd(pendingOffsetPhi2, blockPopCount2);
    Value * const srcBlockNo2 = b.CreateUDiv(newPendingOffset2, BLOCK_WIDTH);

    // Now load and process source streams.
    SmallVector<Value *, 16> sourceData1(mSelectedStreamCount);
    SmallVector<Value *, 16> sourceData2(mSelectedStreamCount);
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        Constant * const streamIndex = ConstantInt::get(streamBase->getType(), i);
        Value * const streamOffset = b.CreateAdd(streamBase, streamIndex);
        sourceData1[i] = b.loadInputStreamBlock("source1", streamOffset, srcBlockNo1);
        sourceData2[i] = b.loadInputStreamBlock("source2", streamOffset, srcBlockNo2);
        Value * A1 = b.simd_srlv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData1[i], pendingDataPhi1[i], field_offset_lo1), bit_offset1);
        Value * B1 = b.simd_sllv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData1[i], pendingDataPhi1[i], field_offset_hi1), shift_fwd1);
        Value * full_source_block1 = b.CreateOr(A1, B1, "toExpand1");
        Value * C1 = b.simd_srlv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block1, source_field_lo1), source_shift_lo1);
        Value * D1 = b.simd_sllv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block1, source_field_hi1), source_shift_hi1);
        Value * expanded1 = b.bitCast(b.CreateOr(C1, D1, "expanded1"));

        Value * A2 = b.simd_srlv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData2[i], pendingDataPhi2[i], field_offset_lo2), bit_offset2);
        Value * B2 = b.simd_sllv(mFieldWidth, b.mvmd_dsll(mFieldWidth, sourceData2[i], pendingDataPhi2[i], field_offset_hi2), shift_fwd2);
        Value * full_source_block2 = b.CreateOr(A2, B2, "toExpand1");
        Value * C2 = b.simd_srlv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block2, source_field_lo2), source_shift_lo2);
        Value * D2 = b.simd_sllv(mFieldWidth, b.mvmd_shuffle(mFieldWidth, full_source_block2, source_field_hi2), source_shift_hi2);
        Value * expanded2 = b.bitCast(b.CreateOr(C2, D2, "expanded2"));

        Value * toMerge1 = b.simd_pdep(mFieldWidth, expanded1, mask1);
        Value * toMerge2 = b.simd_pdep(mFieldWidth, expanded2, mask2);
        Value * merged=b.CreateOr(toMerge1,toMerge2);//combine 2 inputs using bitwise OR
        b.storeOutputStreamBlock("mergedStreamSet", b.getInt32(i), blockNoPhi, merged);
    }
    //
    // Update loop control Phis for the next iteration.
    //
    blockNoPhi->addIncoming(nextBlk, b.GetInsertBlock());
    pendingOffsetPhi1->addIncoming(newPendingOffset1, b.GetInsertBlock());
    pendingOffsetPhi2->addIncoming(newPendingOffset2, b.GetInsertBlock());
    for (unsigned i = 0; i < mSelectedStreamCount; i++) {
        pendingDataPhi1[i]->addIncoming(sourceData1[i], b.GetInsertBlock());
        pendingDataPhi2[i]->addIncoming(sourceData2[i], b.GetInsertBlock());
    }
    // Now continue the loop if there are more blocks to process.
    b.CreateCondBr(moreToDo, mergeLoop, mergenceDone);

    b.SetInsertPoint(mergenceDone);
}
/*******************************************************************************************/

FieldDepositKernel::FieldDepositKernel(KernelBuilder & b
                                       , StreamSet * mask, StreamSet * input, StreamSet * output
                                       , const unsigned fieldWidth)
: MultiBlockKernel(b, "FieldDeposit" + std::to_string(fieldWidth) + "_" + std::to_string(input->getNumElements()),
{Bind("depositMask", mask)
, Bind("inputStreamSet", input)},
{Bind("outputStreamSet", output)},
{}, {}, {})
, mFieldWidth(fieldWidth)
, mStreamCount(input->getNumElements()) {

}

void PDEPFieldDepositLogic(KernelBuilder & b, llvm::Value * const numOfStrides, unsigned fieldWidth, unsigned streamCount, unsigned stride);

void FieldDepositKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    if (AVX2_available() && BMI2_available() && ((mFieldWidth == 32) || (mFieldWidth == 64))) {
        PDEPFieldDepositLogic(b, numOfStrides, mFieldWidth, mStreamCount, getStride());
    } else {
        BasicBlock * entry = b.GetInsertBlock();
        BasicBlock * processBlock = b.CreateBasicBlock("processBlock");
        BasicBlock * done = b.CreateBasicBlock("done");
        Constant * const ZERO = b.getSize(0);
        Value * numOfBlocks = numOfStrides;
        if (getStride() != b.getBitBlockWidth()) {
            numOfBlocks = b.CreateShl(numOfStrides, b.getSize(std::log2(getStride()/b.getBitBlockWidth())));
        }
        b.CreateBr(processBlock);

        b.SetInsertPoint(processBlock);
        PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
        blockOffsetPhi->addIncoming(ZERO, entry);
        Value * depositMask = b.loadInputStreamBlock("depositMask", ZERO, blockOffsetPhi);
        for (unsigned j = 0; j < mStreamCount; ++j) {
            Value * input = b.loadInputStreamBlock("inputStreamSet", b.getInt32(j), blockOffsetPhi);
            Value * output = b.simd_pdep(mFieldWidth, input, depositMask);
            b.storeOutputStreamBlock("outputStreamSet", b.getInt32(j), blockOffsetPhi, output);
        }
        Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
        blockOffsetPhi->addIncoming(nextBlk, processBlock);
        Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
        b.CreateCondBr(moreToDo, processBlock, done);

        b.SetInsertPoint(done);
    }
}

//#define PREFER_FIELD_STORES_OVER_INSERT_ELEMENT

void PDEPFieldDepositLogic(KernelBuilder & b, llvm::Value * const numOfStrides, unsigned fieldWidth, unsigned streamCount, unsigned stride) {
#ifdef PREFER_FIELD_LOADS_OVER_EXTRACT_ELEMENT
    Type * fieldTy = b.getIntNTy(fieldWidth);
    Type * fieldPtrTy = PointerType::get(fieldTy, 0);
#elif PREFER_FIELD_STORES_OVER_INSERT_ELEMENT
    Type * fieldTy = b.getIntNTy(fieldWidth);
    Type * fieldPtrTy = PointerType::get(fieldTy, 0);
#endif
    BasicBlock * entry = b.GetInsertBlock();
    BasicBlock * processBlock = b.CreateBasicBlock("processBlock");
    BasicBlock * done = b.CreateBasicBlock("done");
    Constant * const ZERO = b.getSize(0);
    const unsigned fieldsPerBlock = b.getBitBlockWidth()/fieldWidth;
    Value * numOfBlocks = numOfStrides;
    if (stride != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(std::log2(stride/b.getBitBlockWidth())));
    }
    b.CreateBr(processBlock);
    b.SetInsertPoint(processBlock);
    PHINode * blockOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    blockOffsetPhi->addIncoming(ZERO, entry);

    SmallVector<Value *, 16> mask(fieldsPerBlock);
    //  When operating on fields individually, we can use vector load/store with
    //  extract/insert element operations, or we can use individual field load
    //  and stores.   Individual field operations require fewer total operations,
    //  but more memory instructions.   It may be that vector load/extract is better,
    //  while field store is better.   Vector insert then store creates long dependence
    //  chains.
    //
#ifdef PREFER_FIELD_LOADS_OVER_EXTRACT_ELEMENT
    Value * depositMaskPtr = b.getInputStreamBlockPtr("depositMask", ZERO, blockOffsetPhi);
    depositMaskPtr = b.CreatePointerCast(depositMaskPtr, fieldPtrTy);
    for (unsigned i = 0; i < fieldsPerBlock; i++) {
        mask[i] = b.CreateLoad(fieldTy, b.CreateGEP(fiedlTy, depositMaskPtr, b.getInt32(i)));
    }
#else

    Value * depositMask = b.fwCast(fieldWidth, b.loadInputStreamBlock("depositMask", ZERO, blockOffsetPhi));
    for (unsigned i = 0; i < fieldsPerBlock; i++) {
        mask[i] = b.CreateExtractElement(depositMask, b.getInt32(i));
    }
#endif
    for (unsigned j = 0; j < streamCount; ++j) {
#ifdef PREFER_FIELD_LOADS_OVER_EXTRACT_ELEMENT
        Value * inputPtr = b.getInputStreamBlockPtr("inputStreamSet", b.getInt32(j), blockOffsetPhi);
        inputPtr = b.CreatePointerCast(inputPtr, fieldPtrTy);
#else
        Value * const input = b.loadInputStreamBlock("inputStreamSet", b.getInt32(j), blockOffsetPhi);
        Value * inputStrm = b.fwCast(fieldWidth, input);
#endif
#ifdef PREFER_FIELD_STORES_OVER_INSERT_ELEMENT
        Value * outputPtr = b.getOutputStreamBlockPtr("outputStreamSet", b.getInt32(j), blockOffsetPhi);
        outputPtr = b.CreatePointerCast(outputPtr, fieldPtrTy);
#else
        // Value * outputStrm = b.fwCast(mPDEPWidth, b.allZeroes());
        Value * outputStrm = UndefValue::get(b.fwVectorType(fieldWidth));
#endif
        for (unsigned i = 0; i < fieldsPerBlock; i++) {
#ifdef PREFER_FIELD_LOADS_OVER_EXTRACT_ELEMENT
            Value * field = b.CreateLoad(fieldTy, b.CreateGEP(fiedlTy, inputPtr, b.getInt32(i)));
#else
            Value * field = b.CreateExtractElement(inputStrm, b.getInt32(i));
#endif
            Value * compressed = b.CreatePdeposit(field, mask[i]);
#ifdef PREFER_FIELD_STORES_OVER_INSERT_ELEMENT
            b.CreateStore(compressed, b.CreateGEP(fieldTy, outputPtr, b.getInt32(i)));
        }
#else
        outputStrm = b.CreateInsertElement(outputStrm, compressed, b.getInt32(i));
    }
    b.storeOutputStreamBlock("outputStreamSet", b.getInt32(j), blockOffsetPhi, outputStrm);
#endif
    }
    Value * nextBlk = b.CreateAdd(blockOffsetPhi, b.getSize(1));
    blockOffsetPhi->addIncoming(nextBlk, processBlock);
    Value * moreToDo = b.CreateICmpNE(nextBlk, numOfBlocks);
    b.CreateCondBr(moreToDo, processBlock, done);
    b.SetInsertPoint(done);
}

PDEPFieldDepositKernel::PDEPFieldDepositKernel(KernelBuilder & b
                                               , StreamSet * mask, StreamSet * input, StreamSet * output
                                               , const unsigned fieldWidth)
: MultiBlockKernel(b, "PDEPFieldDeposit" + std::to_string(fieldWidth) + "_" + std::to_string(input->getNumElements()) ,
                   {Binding{"depositMask", mask},
                    Binding{"inputStreamSet", input}},
                   {Binding{"outputStreamSet", output}},
                   {}, {}, {})
, mPDEPWidth(fieldWidth)
, mStreamCount(input->getNumElements()) {
    if ((fieldWidth != 32) && (fieldWidth != 64))
        llvm::report_fatal_error("Unsupported PDEP width for PDEPFieldDepositKernel");
}

void PDEPFieldDepositKernel::generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) {
    PDEPFieldDepositLogic(b, numOfStrides, mPDEPWidth, mStreamCount, getStride());
}


PDEPkernel::PDEPkernel(KernelBuilder & b, const unsigned swizzleFactor, std::string name)
: MultiBlockKernel(b, std::move(name),
                   // input stream sets
{Binding{b.getStreamSetTy(), "marker", FixedRate(), Principal()},
    Binding{b.getStreamSetTy(swizzleFactor), "source", PopcountOf("marker"), BlockSize(b.getBitBlockWidth() / swizzleFactor) }},
                   // output stream set
{Binding{b.getStreamSetTy(swizzleFactor), "output", FixedRate(), BlockSize(b.getBitBlockWidth() / swizzleFactor)}},
{}, {}, {})
, mSwizzleFactor(swizzleFactor) {
}

void PDEPkernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const processBlock = b.CreateBasicBlock("processBlock");
    BasicBlock * const finishedStrides = b.CreateBasicBlock("finishedStrides");
    const auto pdepWidth = b.getBitBlockWidth() / mSwizzleFactor;
    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());
    ConstantInt * const PDEP_WIDTH = b.getSize(pdepWidth);

    Constant * const ZERO = b.getSize(0);
    Value * const sourceItemCount = b.getProcessedItemCount("source");

    Value * numOfBlocks = numOfStrides;
    if (getStride() != b.getBitBlockWidth()) {
        numOfBlocks = b.CreateShl(numOfStrides, b.getSize(std::log2(getStride()/b.getBitBlockWidth())));
    }
    Value * const initialSourceOffset = b.CreateURem(sourceItemCount, BLOCK_WIDTH);
    b.CreateBr(processBlock);

    b.SetInsertPoint(processBlock);
    PHINode * const strideIndex = b.CreatePHI(b.getSizeTy(), 2);
    strideIndex->addIncoming(ZERO, entry);
    PHINode * const bufferPhi = b.CreatePHI(b.getBitBlockType(), 2);
    bufferPhi->addIncoming(Constant::getNullValue(b.getBitBlockType()), entry);
    PHINode * const sourceOffsetPhi = b.CreatePHI(b.getSizeTy(), 2);
    sourceOffsetPhi->addIncoming(initialSourceOffset, entry);
    PHINode * const bufferSizePhi = b.CreatePHI(b.getSizeTy(), 2);
    bufferSizePhi->addIncoming(ZERO, entry);

    // Extract the values we will use in the main processing loop
    Value * const markerStream = b.getInputStreamBlockPtr("marker", ZERO, strideIndex);
    Type * const vecType = b.fwVectorType(pdepWidth);
    Value * const selectors = b.CreateBlockAlignedLoad(vecType, markerStream);
    Value * const numOfSelectors = b.simd_popcount(pdepWidth, selectors);

    // For each element of the marker block
    Value * bufferSize = bufferSizePhi;
    Value * sourceOffset = sourceOffsetPhi;
    Value * buffer = bufferPhi;
    for (unsigned i = 0; i < mSwizzleFactor; i++) {

        // How many bits will we deposit?
        Value * const required = b.CreateExtractElement(numOfSelectors, b.getSize(i));

        // Aggressively enqueue any additional bits
        BasicBlock * const entry = b.GetInsertBlock();
        BasicBlock * const enqueueBits = b.CreateBasicBlock();
        b.CreateBr(enqueueBits);

        b.SetInsertPoint(enqueueBits);
        PHINode * const updatedBufferSize = b.CreatePHI(bufferSize->getType(), 2);
        updatedBufferSize->addIncoming(bufferSize, entry);
        PHINode * const updatedSourceOffset = b.CreatePHI(sourceOffset->getType(), 2);
        updatedSourceOffset->addIncoming(sourceOffset, entry);
        PHINode * const updatedBuffer = b.CreatePHI(buffer->getType(), 2);
        updatedBuffer->addIncoming(buffer, entry);

        // Calculate the block and swizzle index of the current swizzle row
        Value * const blockOffset = b.CreateUDiv(updatedSourceOffset, BLOCK_WIDTH);
        Value * const swizzleIndex = b.CreateUDiv(b.CreateURem(updatedSourceOffset, BLOCK_WIDTH), PDEP_WIDTH);
        Value * const swizzle = b.CreateBlockAlignedLoad(b.getBitBlockType(), b.getInputStreamBlockPtr("source", swizzleIndex, blockOffset));
        Value * const swizzleOffset = b.CreateURem(updatedSourceOffset, PDEP_WIDTH);

        // Shift the swizzle to the right to clear off any used bits ...
        Value * const swizzleShift = b.simd_fill(pdepWidth, swizzleOffset);
        Value * const unreadBits = b.CreateLShr(swizzle, swizzleShift);

        // ... then to the left to align the bits with the buffer and combine them.
        Value * const bufferShift = b.simd_fill(pdepWidth, updatedBufferSize);
        Value * const pendingBits = b.CreateShl(unreadBits, bufferShift);

        buffer = b.CreateOr(updatedBuffer, pendingBits);
        updatedBuffer->addIncoming(buffer, enqueueBits);

        // Update the buffer size with the number of bits we have actually enqueued
        Value * const maxBufferSize = b.CreateAdd(b.CreateSub(PDEP_WIDTH, swizzleOffset), updatedBufferSize);
        bufferSize = b.CreateUMin(maxBufferSize, PDEP_WIDTH);
        updatedBufferSize->addIncoming(bufferSize, enqueueBits);

        // ... and increment the source offset by the number we actually inserted
        Value * const inserted = b.CreateSub(bufferSize, updatedBufferSize);
        sourceOffset = b.CreateAdd(updatedSourceOffset, inserted);
        updatedSourceOffset->addIncoming(sourceOffset, enqueueBits);

        // INVESTIGATE: we can branch at most once here. I'm not sure whether the potential
        // branch misprediction is better or worse than always filling from two swizzles to
        // ensure that we have enough bits to deposit.
        BasicBlock * const depositBits = b.CreateBasicBlock();
        b.CreateUnlikelyCondBr(b.CreateICmpULT(bufferSize, required), enqueueBits, depositBits);

        b.SetInsertPoint(depositBits);
        // Apply PDEP to each element of the combined swizzle using the current PDEP mask
        Value * const mask = b.CreateExtractElement(selectors, i);
        Value* result = b.simd_pdep(pdepWidth, buffer, b.simd_fill(pdepWidth, mask));
        // Store the result
        Value * const outputStreamPtr = b.getOutputStreamBlockPtr("output", b.getSize(i), strideIndex);
        b.CreateBlockAlignedStore(result, outputStreamPtr);
        // Shift away any used bits from the buffer and decrement our buffer size by the number we used
        Value * const usedShift = b.simd_fill(pdepWidth, required);
        buffer = b.CreateLShr(buffer, usedShift);
        bufferSize = b.CreateSub(bufferSize, required);
    }

    BasicBlock * const finishedBlock = b.GetInsertBlock();
    sourceOffsetPhi->addIncoming(sourceOffset, finishedBlock);
    bufferSizePhi->addIncoming(bufferSize, finishedBlock);
    bufferPhi->addIncoming(buffer, finishedBlock);
    Value * const nextStrideIndex = b.CreateAdd(strideIndex, b.getSize(1));
    strideIndex->addIncoming(nextStrideIndex, finishedBlock);
    b.CreateLikelyCondBr(b.CreateICmpNE(nextStrideIndex, numOfBlocks), processBlock, finishedStrides);

    b.SetInsertPoint(finishedStrides);
}

std::string InsertString(StreamSet * mask, InsertPosition p) {
    std::string s = std::to_string(mask->getNumElements()) + "x1_";
    return s + (p == InsertPosition::Before ? "Before" : "After");
}

class UnitInsertionExtractionMasks final : public BlockOrientedKernel {
public:
    UnitInsertionExtractionMasks(KernelBuilder & b,
                                 StreamSet * insertion_mask, StreamSet * stream01, StreamSet * valid01, InsertPosition p = InsertPosition::Before)
    : BlockOrientedKernel(b, "unitInsertionExtractionMasks" + InsertString(insertion_mask, p),
        {Binding{"insertion_mask", insertion_mask}},
        {Binding{"stream01", stream01, FixedRate(2)}, Binding{"valid01", valid01, FixedRate(2)}},
        {}, {},
        {InternalScalar{ScalarType::NonPersistent, b.getBitBlockType(), "EOFmask"}}),
    mInsertPos(p) {}
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
    void generateFinalBlockMethod(KernelBuilder & b, llvm::Value * const remainingBytes) override;
private:
    const InsertPosition mInsertPos;
};

void UnitInsertionExtractionMasks::generateDoBlockMethod(KernelBuilder & b) {
    Value * fileExtentMask = b.CreateNot(b.getScalarField("EOFmask"));
    Value * insertion_mask = b.loadInputStreamBlock("insertion_mask", b.getSize(0), b.getSize(0));
    const auto n = b.getInputStreamSet("insertion_mask")->getNumElements();
    for (unsigned i = 1; i < n; i++) {
        insertion_mask = b.CreateOr(insertion_mask, b.loadInputStreamBlock("insertion_mask", b.getSize(i), b.getSize(0)));
    }
    Constant * mask01 = nullptr;
    Value * extract_mask_lo = nullptr;
    Value * extract_mask_hi = nullptr;
    if (mInsertPos == InsertPosition::Before) {
        mask01 = b.simd_himask(2);
        extract_mask_lo = b.esimd_mergel(1, insertion_mask, fileExtentMask);
        extract_mask_hi = b.esimd_mergeh(1, insertion_mask, fileExtentMask);
    } else {
        mask01 = b.simd_lomask(2);
        extract_mask_lo = b.esimd_mergel(1, fileExtentMask, insertion_mask);
        extract_mask_hi = b.esimd_mergeh(1, fileExtentMask, insertion_mask);
    }
    b.storeOutputStreamBlock("stream01", b.getSize(0), b.getSize(0), mask01);
    b.storeOutputStreamBlock("stream01", b.getSize(0), b.getSize(1), mask01);
    b.storeOutputStreamBlock("valid01", b.getSize(0), b.getSize(0), extract_mask_lo);
    b.storeOutputStreamBlock("valid01", b.getSize(0), b.getSize(1), extract_mask_hi);
}

void UnitInsertionExtractionMasks::generateFinalBlockMethod(KernelBuilder & b, Value * const remainingBytes) {
    // Standard Pablo convention for final block processing: set a bit marking
    // the position just past EOF, as well as a mask marking all positions past EOF.
    b.setScalarField("EOFmask", b.bitblock_mask_from(remainingBytes));
    RepeatDoBlockLogic(b);
}
#define USE_FILTER_BY_MASK_KERNEL
StreamSet * UnitInsertionSpreadMask(PipelineBuilder & P, StreamSet * insertion_mask, InsertPosition p,
                                    ProcessingRateProbabilityDistribution insertionProbabilityDistribution) {
    auto stream01 = P.CreateStreamSet(1);
    auto valid01 = P.CreateStreamSet(1);
    P.CreateKernelCall<UnitInsertionExtractionMasks>(insertion_mask, stream01, valid01, p);
    auto spread_mask = P.CreateStreamSet(1);
#ifndef USE_FILTER_BY_MASK_KERNEL
    FilterByMask(P, valid01, stream01, spread_mask, spreadCountDensity);
#else
    P.CreateKernelCall<FilterByMaskKernel>
        (Select(valid01, {0}),
         SelectOperationList{Select(stream01, {0})},
         spread_mask, 64, insertionProbabilityDistribution);
#endif
    return spread_mask;
}

class UGT_Kernel final : public pablo::PabloKernel {
public:
    UGT_Kernel(KernelBuilder & b, StreamSet * bixnum, unsigned immediate, StreamSet * result) :
    pablo::PabloKernel(b, "ugt_" + std::to_string(immediate) + "_" + std::to_string(bixnum->getNumElements()),
                {Binding{"bixnum", bixnum}}, {Binding{"result", result}}), mTestVal(immediate) {}
protected:
    void generatePabloMethod() override;
private:
    const unsigned mTestVal;
};

void UGT_Kernel::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::BixNum bixnum = getInputStreamSet("bixnum");
    pablo::Var * output = getOutputStreamVar("result");
    pb.createAssign(pb.createExtract(output, 0), bnc.UGT(bixnum, mTestVal));
}

class SpreadMaskStep final : public pablo::PabloKernel {
public:
    SpreadMaskStep(KernelBuilder & b,
                   StreamSet * bixnum, StreamSet * result, InsertPosition p = InsertPosition::Before) :
    pablo::PabloKernel(b, "spreadMaskStep_" + InsertString(bixnum, p),
                {Bind("bixnum", bixnum, LookAhead(1))},
                {Bind("result", result)}), mInsertPos(p) {}
protected:
    void generatePabloMethod() override;
private:
    const InsertPosition mInsertPos;
};

void SpreadMaskStep::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::BixNum insert_counts = getInputStreamSet("bixnum");
    pablo::BixNum has_insert(1);
    has_insert[0] = bnc.UGT(insert_counts, 0);
    // Since we have inserted one position, subtract one from the insert count.
    pablo::BixNum remaining_to_insert = bnc.SubModular(insert_counts, has_insert);
    // Split the original insert counts into 2
    unsigned remaining_bits = insert_counts.size() - 1;
    pablo::BixNum divided_counts = bnc.HighBits(insert_counts, remaining_bits);
    pablo::BixNum divided_remaining = bnc.HighBits(remaining_to_insert, remaining_bits);
    for (unsigned i = 0; i < remaining_bits; i++) {
        std::string vname = "divided_counts[" + std::to_string(i) + "]";
        if (mInsertPos == InsertPosition::Before) {
            divided_counts[i] = pb.createOr(divided_remaining[i], pb.createLookahead(divided_counts[i], 1), vname);
        } else {
            divided_counts[i] = pb.createOr(divided_counts[i], pb.createAdvance(divided_remaining[i], 1), vname);
        }
    }
    pablo::Var * result = getOutputStreamVar("result");
    for (unsigned i = 0; i < remaining_bits; i++) {
        pb.createAssign(pb.createExtract(result, i), divided_counts[i]);
    }
}

StreamSet * InsertionSpreadMask(PipelineBuilder & P,
                                StreamSet * bixNumInsertCount, InsertPosition pos,
                                ProcessingRateProbabilityDistribution itemsPerOutputUnit,
                                ProcessingRateProbabilityDistribution expansionRate) {
    unsigned steps = bixNumInsertCount->getNumElements();
    if (steps == 1) {
        return UnitInsertionSpreadMask(P, bixNumInsertCount, pos, expansionRate);
    }
    /* Create a spread mask that adds one spread position for any position
       at which there is at least one item to insert.  */
    StreamSet * spread1_mask = P.CreateStreamSet(1);
    spread1_mask = UnitInsertionSpreadMask(P, bixNumInsertCount, pos, expansionRate);
    /* Spread out the counts so that there are two positions for each nonzero entry. */
    StreamSet * spread_counts = P.CreateStreamSet(steps);
    SpreadByMask(P, spread1_mask, bixNumInsertCount, spread_counts, false, 0, itemsPerOutputUnit);
    /* Divide the count at each original position equally into the
       two positions that were created by the unit spread process. */
    StreamSet * reduced_counts = P.CreateStreamSet(steps - 1);
    P.CreateKernelCall<SpreadMaskStep>(spread_counts, reduced_counts, pos);
    StreamSet * submask = InsertionSpreadMask(P, reduced_counts, pos, itemsPerOutputUnit, expansionRate);
    StreamSet * finalmask = P.CreateStreamSet(1);
    SpreadByMask(P, submask, spread1_mask, finalmask, false, 0, itemsPerOutputUnit);
    return finalmask;
}
}
