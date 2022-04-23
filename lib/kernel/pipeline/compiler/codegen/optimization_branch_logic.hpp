#ifndef OPTIMIZATION_BRANCH_LOGIC_HPP
#define OPTIMIZATION_BRANCH_LOGIC_HPP

#include "../pipeline_compiler.hpp"

namespace kernel {

// The condition stream marker defines regions in which we cannot take the optimization branch and we
// are always safe to take the "normal" (non-optimization) branch with the only observable difference
// being a (single-threaded) performance penalty.

// When we have multiple threads simultaneously executing this kernel, the (K+1)-th thread can only
// begin processing data after the K-th thread starts executing its *final* subsegment (assuming that
// both the final subsegment of the K-th thread and the first subsegment of the (K+1) thread branches
// along the same path.

// Consequently, we do not want to aggressively alternate between optimization and normal path branches
// if only a small amount of work can be done as this would delay other threads from executing. To
// accommodate this, we have a minimum span length critera that defines how many strides of data
// must be permitted by the optimization branch before executing it.

#define MINIMUM_SPAN_LENGTH_OF_OPTIMIZATION_BRANCH 2

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isEitherOptimizationBranchKernelInternallySynchronized
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::isEitherOptimizationBranchKernelInternallySynchronized() const {
    const OptimizationBranch * const optBr = cast<OptimizationBranch>(mKernel);
    const auto a = optBr->getNonZeroKernel()->hasAttribute(AttrId::InternallySynchronized);
    const auto b = optBr->getAllZeroKernel()->hasAttribute(AttrId::InternallySynchronized);
    if (LLVM_UNLIKELY(a != b)) {
        report_fatal_error("PipelineComiler does not currently support OptimizationBranch"
                           " with differing internally synchronized values");
    }
    return a;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeOptimizationBranch
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeOptimizationBranch() {

}

inline bool isConstantOne(const Value * const value) {
    return (isa<ConstantInt>(value) && cast<ConstantInt>(value)->isOne());
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkOptimizationBranchSpanLength
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::checkOptimizationBranchSpanLength(BuilderRef b, Value * const numOfLinearStrides) {

    const OptimizationBranch * const optBr = cast<OptimizationBranch>(mKernel);
    Relationship * const cond = optBr->getCondition();
    if (LLVM_UNLIKELY(!isa<StreamSet>(cond))) {
        report_fatal_error("Optimization branch condition must be a fixed-rate single-stream StreamSet");
    }

    RelationshipType condInput;
    for (const auto e : make_iterator_range(in_edges(mKernelId, mStreamGraph))) {
        const auto binding = source(e, mStreamGraph);
        assert (mStreamGraph[binding].Type == RelationshipNode::IsBinding);
        const auto f = first_in_edge(binding, mStreamGraph);
        assert (mStreamGraph[f].Reason != ReasonType::Reference);
        const auto streamSet = source(f, mStreamGraph);
        const auto & rn = mStreamGraph[streamSet];
        assert (rn.Type == RelationshipNode::IsRelationship);
        if (cast<StreamSet>(rn.Relationship) == cond) {
            condInput = mStreamGraph[e];
            assert (condInput.Type == PortType::Input);
            break;
        }
    }

    const auto streamSetIdx = getInputBufferVertex(condInput);

    const BufferNode & bn = mBufferGraph[streamSetIdx];
    StreamSetBuffer * const buffer = bn.Buffer;

    if (LLVM_UNLIKELY(!isConstantOne(buffer->getStreamSetCount(b)))) {
        report_fatal_error("Optimization branch condition must be a fixed-rate single-stream StreamSet");
    }


    const Binding & condBinding = getInputBinding(condInput);
    const ProcessingRate & condRate = condBinding.getRate();

    if (LLVM_UNLIKELY(!condRate.isFixed())) {
        report_fatal_error("Optimization branch condition must be a fixed-rate single-stream StreamSet");
    }

    const auto bw = b->getBitBlockWidth();

    const auto strideRate = condRate.getRate()
        * (mKernel->getStride() * cast<StreamSet>(cond)->getFieldWidth())
        / Rational{bw};

    assert (strideRate.denominator() == 1);

    const auto blocksPerStride = strideRate.numerator();

    Constant * const BLOCKS_PER_STRIDE = b->getSize(blocksPerStride);

    IntegerType * const sizeTy = b->getSizeTy();
    Constant * const sz_ZERO = b->getSize(0);
    Constant * const sz_ONE = b->getSize(1);
    Constant * const sz_MIN_LENGTH = b->getSize(MINIMUM_SPAN_LENGTH_OF_OPTIMIZATION_BRANCH);
    Constant * const BIT_BLOCK_WIDTH = b->getSize(bw);

    Value * const processed = mAlreadyProcessedPhi[condInput];
    Value * const blockIndex = b->CreateExactUDiv(processed, BIT_BLOCK_WIDTH);

    VectorType * const bitBlockTy = b->getBitBlockType();

    Value * const baseAddress = buffer->getBaseAddress(b);

    const auto prefix = makeKernelName(mKernelId);

    BasicBlock * const entry = b->GetInsertBlock();
    BasicBlock * const scanLengthEndOfRegularSpan = b->CreateBasicBlock(prefix + "_scanLengthEndOfRegularSpan", mKernelLoopCall);
    BasicBlock * const scanLengthFindEndOfOptSpan = b->CreateBasicBlock(prefix + "_scanLengthFindEndOfOptSpan", mKernelLoopCall);
    BasicBlock * const scanLengthCheck = b->CreateBasicBlock(prefix + "_scanLengthCheck", mKernelLoopCall);
    BasicBlock * const scanLengthExit = b->CreateBasicBlock(prefix + "_scanLengthExit", mKernelLoopCall);

    b->CreateUnlikelyCondBr(mOptimizationBranchScanStatePhi, scanLengthFindEndOfOptSpan, scanLengthEndOfRegularSpan);

    // Assume that we just left a regular branch. For us to get back to this loop, we must have
    // located an optimization span of sufficient length. Keep scanning ahead and determine
    // the first
    b->SetInsertPoint(scanLengthFindEndOfOptSpan);
    PHINode * const optNumOfStridesPhi = b->CreatePHI(sizeTy, 3, prefix + "_optNumOfStridesPhi");
    optNumOfStridesPhi->addIncoming(sz_MIN_LENGTH, entry);
    optNumOfStridesPhi->addIncoming(sz_MIN_LENGTH, scanLengthCheck);

    Value * const optScanIndex = b->CreateAdd(blockIndex, optNumOfStridesPhi);
    Value * optAddr = buffer->getStreamBlockPtr(b, baseAddress, sz_ZERO, b->CreateMul(optScanIndex, BLOCKS_PER_STRIDE));
    optAddr = b->CreatePointerCast(optAddr, bitBlockTy->getPointerTo());
    Value * optCondVal = b->CreateLoad(optAddr);
    for (unsigned i = 1; i < blocksPerStride; ++i) {
        Value * const val = b->CreateLoad(b->CreateGEP(optAddr, b->getInt32(i)));
        optCondVal = b->CreateOr(optCondVal, val);
    }
    Value * const foundNonOpt = b->bitblock_any(optCondVal);
    Value * const optNextNumOfStrides = b->CreateAdd(optNumOfStridesPhi, sz_ONE);
    Value * const optAtLimit = b->CreateICmpUGE(optNextNumOfStrides, numOfLinearStrides);
    Value * const optDone = b->CreateOr(foundNonOpt, optAtLimit);
    optNumOfStridesPhi->addIncoming(optNextNumOfStrides, scanLengthFindEndOfOptSpan);
    b->CreateCondBr(optDone, scanLengthExit, scanLengthFindEndOfOptSpan);

    // Assume that we're starting in a regular branch span and locate the end of it, which can
    // happen when we locate an optimization span of sufficient length or the end of the input.
    b->SetInsertPoint(scanLengthEndOfRegularSpan);
    PHINode * const regScanIndexPhi = b->CreatePHI(sizeTy, 2, prefix + "_regScanIndexPhi");
    regScanIndexPhi->addIncoming(sz_ZERO, entry);
    PHINode * const regNumOfStridesPhi = b->CreatePHI(sizeTy, 2, prefix + "_regNumOfStridesPhi");
    regNumOfStridesPhi->addIncoming(sz_ZERO, entry);
    PHINode * const regSpanLengthPhi = b->CreatePHI(sizeTy, 2, prefix + "_regSpanLengthPhi");
    regSpanLengthPhi->addIncoming(sz_ZERO, entry);

    Value * const regScanIndex = b->CreateAdd(blockIndex, regScanIndexPhi);
    Value * regAddr = buffer->getStreamBlockPtr(b, baseAddress, sz_ZERO, b->CreateMul(regScanIndex, BLOCKS_PER_STRIDE));
    regAddr = b->CreatePointerCast(regAddr, bitBlockTy->getPointerTo());
    Value * regCondVal = b->CreateLoad(regAddr);
    for (unsigned i = 1; i < blocksPerStride; ++i) {
        Value * const val = b->CreateLoad(b->CreateGEP(regAddr, b->getInt32(i)));
        regCondVal = b->CreateOr(regCondVal, val);
    }
    regCondVal = b->bitblock_any(regCondVal);



    Value * const regIncSpanLength = b->CreateAdd(regSpanLengthPhi, sz_ONE);
    Value * const regNextOptSpanLength = b->CreateSelect(regCondVal, sz_ZERO, regIncSpanLength);
    Value * const regSpanLargeEnough = b->CreateICmpEQ(regNextOptSpanLength, sz_MIN_LENGTH);
    Value * const regNextScanIndex = b->CreateAdd(regScanIndexPhi, sz_ONE);
    Value * const regNextNumOfStrides = b->CreateSelect(regCondVal, regNextScanIndex, regNumOfStridesPhi);
    Value * const regAtLimit = b->CreateICmpUGE(regNextScanIndex, numOfLinearStrides);
    Value * const regDone = b->CreateOr(regSpanLargeEnough, regAtLimit);
    regScanIndexPhi->addIncoming(regNextScanIndex, scanLengthEndOfRegularSpan);
    regSpanLengthPhi->addIncoming(regNextOptSpanLength, scanLengthEndOfRegularSpan);
    regNumOfStridesPhi->addIncoming(regNextNumOfStrides, scanLengthEndOfRegularSpan);
    b->CreateCondBr(regDone, scanLengthCheck, scanLengthEndOfRegularSpan);

    b->SetInsertPoint(scanLengthCheck);
    Value * const anyRegularStrides = b->CreateICmpNE(regNextNumOfStrides, sz_ZERO);
    Value * const finishedScanning = b->CreateOr(anyRegularStrides, regDone);
    Value * const regNumOfStrudes =
        b->CreateSelect(regSpanLargeEnough, b->CreateSub(regNextScanIndex, sz_MIN_LENGTH), numOfLinearStrides);
    b->CreateCondBr(finishedScanning, scanLengthExit, scanLengthFindEndOfOptSpan);

    b->SetInsertPoint(scanLengthExit);
    PHINode * const finalNumOfStridesPhi = b->CreatePHI(sizeTy, 3, prefix + "_scanLengthNumOfStridesPhi");
    finalNumOfStridesPhi->addIncoming(optNextNumOfStrides, scanLengthFindEndOfOptSpan);
    finalNumOfStridesPhi->addIncoming(regNumOfStrudes, scanLengthCheck);
    PHINode * const chosenBranchPhi = b->CreatePHI(b->getInt1Ty(), 2);
    chosenBranchPhi->addIncoming(b->getFalse(), scanLengthFindEndOfOptSpan);
    chosenBranchPhi->addIncoming(anyRegularStrides, scanLengthCheck);
    mOptimizationBranchSelectedBranch = chosenBranchPhi;
//    mOptimizationBranchSelectedBranch = b->getTrue();
    return finalNumOfStridesPhi;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeOptimizationBranchKernelCall
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeOptimizationBranchKernelCall(BuilderRef b) {



}

}

#endif // OPTIMIZATION_BRANCH_LOGIC_HPP
