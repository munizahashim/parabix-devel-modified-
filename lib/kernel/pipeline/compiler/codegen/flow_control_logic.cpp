#include "../pipeline_compiler.hpp"

// Each partition root determines how much data that it (and consequently its partition) can
// process in a single segment / pipeline iteration. However, it does not necessarily need to
// transfer all of the data provided to it and at times it may be beneficial to withhold data
// to better balance thread workloads.

// The functions here are designed to dynamically managed the maximum segment length of a
// partition root to promote this. By doing so, they may have to malloc a larger thread local
// memory pool or increase the repetition length of repeating streamsets.

#define INCREASE_WEIGHT_FACTOR (5)

#define DECREASE_WEIGHT_FACTOR (3)

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addSegmentLengthSlidingWindowKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addSegmentLengthSlidingWindowKernelProperties(BuilderRef b, const size_t kernelId, const size_t groupId) {
#ifdef USE_DYNAMIC_SEGMENT_LENGTH_SLIDING_WINDOW
    mTarget->addInternalScalar(b->getSizeTy(), SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(kernelId), groupId);
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeInitialSlidingWindowSegmentLengths
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeInitialSlidingWindowSegmentLengths(BuilderRef b, Value * const segmentLengthScalingFactor) {
#ifdef USE_DYNAMIC_SEGMENT_LENGTH_SLIDING_WINDOW
    for (unsigned i = 1U; i < (PartitionCount - 1U); ++i) {
        const auto f = FirstKernelInPartition[i];
        const auto numOfStrides = MaximumNumOfStrides[f];
        Value * const init = b->CreateMul(segmentLengthScalingFactor, b->getSize(numOfStrides));
        b->setScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(f), init);
    }
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeFlowControl
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeFlowControl(BuilderRef b) {
    #ifdef USE_DYNAMIC_SEGMENT_LENGTH_SLIDING_WINDOW
    if (RequiredThreadLocalStreamSetMemory > 0) {
        mThreadLocalMemorySizePtr = b->CreateAllocaAtEntryPoint(b->getSizeTy());
        Value * const reqMem = b->getSize(RequiredThreadLocalStreamSetMemory);
        b->CreateStore(reqMem, mThreadLocalMemorySizePtr);
    }
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadCurrentThreadLocalMemoryRequired
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::loadCurrentThreadLocalMemoryAddress(BuilderRef b) {
    if (RequiredThreadLocalStreamSetMemory > 0) {
        mThreadLocalStreamSetBaseAddress = b->getScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief detemineMaximumNumberOfStrides
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::detemineMaximumNumberOfStrides(BuilderRef b) {
    // The partition root kernel determines the amount of data processed based on the partition input.
    // During normal execution, it always performs max num of strides worth of work. However, if this
    // segment is the last segment, mPartitionSegmentLength is artificially raised to a value of ONE
    // even though we likely can only execute a partial segment worth of work. The root kernel
    // calculates how many strides can be performed and sets mNumOfPartitionStrides to that value.

    // To avoid having every kernel test their I/O during normal execution, the non-root kernels in
    // the same partition refer to the mNumOfPartitionStrides to determine how their segment length.

    if (mIsPartitionRoot) {

        #ifdef USE_DYNAMIC_SEGMENT_LENGTH_SLIDING_WINDOW
        mMaximumNumOfStrides = b->getScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(mKernelId));
        assert (mCurrentPartitionId == KernelPartitionId[mKernelId]);
        assert (mKernelId == FirstKernelInPartition[KernelPartitionId[mKernelId]]);
        const auto firstKernelOfNextPartition = FirstKernelInPartition[mCurrentPartitionId + 1];

        // calculate how much memory is required by this partition relative to max num of strides
        // and determine if the current thread local buffer can fit it.
        size_t maxMemory = 0;
        for (auto kernel = mKernelId; kernel < firstKernelOfNextPartition; ++kernel) {
            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];
                if (bn.isThreadLocal()) {
                    maxMemory = std::max<size_t>(maxMemory, bn.BufferEnd);
                }
            }
        }
        if (maxMemory != 0) {
            assert (RequiredThreadLocalStreamSetMemory > 0);
            Rational memPerStride(maxMemory, MaximumNumOfStrides[mKernelId]);
            Value * const memoryForSegment = b->CreateMulRational(mMaximumNumOfStrides, memPerStride);
            Value * const threadLocalPtr = b->getScalarFieldPtr(BASE_THREAD_LOCAL_STREAMSET_MEMORY);
            BasicBlock * const expandThreadLocalMemory = b->CreateBasicBlock();
            BasicBlock * const afterExpansion = b->CreateBasicBlock();
            Value * const currentMem = b->CreateLoad(mThreadLocalMemorySizePtr);
            Value * const needsExpansion = b->CreateICmpUGT(memoryForSegment, currentMem);
            b->CreateCondBr(needsExpansion, expandThreadLocalMemory, afterExpansion);

            b->SetInsertPoint(expandThreadLocalMemory);

            b->CreateFree(b->CreateLoad(threadLocalPtr));
            // At minimum, we want to double the required space to minimize future reallocs
            Value * expanded = b->CreateRoundUp(memoryForSegment, currentMem);
            b->CreateStore(expanded, mThreadLocalMemorySizePtr);
            #ifdef THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER
            expanded = b->CreateMul(expanded, b->getSize(THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER));
            #endif
            Value * const base = b->CreatePageAlignedMalloc(expanded);
            b->CreateStore(base, threadLocalPtr);
            b->CreateBr(afterExpansion);

            b->SetInsertPoint(afterExpansion);
            mThreadLocalStreamSetBaseAddress = b->CreateLoad(threadLocalPtr);
        } else {
            mThreadLocalStreamSetBaseAddress = nullptr;
        }
        #else
        const auto numOfStrides = MaximumNumOfStrides[mCurrentPartitionRoot];
        mMaximumNumOfStrides = b->CreateMul(mExpectedNumOfStridesMultiplier, b->getSize(numOfStrides));
        #endif
        mThreadLocalScalingFactor = mMaximumNumOfStrides;
    } else {
        const auto ratio = Rational{StrideStepLength[mKernelId], StrideStepLength[mCurrentPartitionRoot]};
        const auto factor = ratio / mPartitionStrideRateScalingFactor;
        mMaximumNumOfStrides = b->CreateMulRational(mNumOfPartitionStrides, factor);
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateNextSlidingWindowSize
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateNextSlidingWindowSize(BuilderRef b, Value * const maxNumOfStrides, Value * const segmentLength) {
    #ifdef USE_DYNAMIC_SEGMENT_LENGTH_SLIDING_WINDOW

    auto makeWeightedVal = [&](Value * A, const unsigned weightA, Value * B) {
        Value * const C = b->CreateMul(A, b->getSize(weightA));
        Value * const E = b->CreateAdd(C, B);
        const auto m = StrideStepLength[mKernelId] * (weightA + 1);
        Value * const F = b->CreateRoundUpRational(E, m);
        return b->CreateUDiv(F, b->getSize(weightA + 1), "wsc");
    };

    Value * const A = makeWeightedVal(segmentLength, INCREASE_WEIGHT_FACTOR, maxNumOfStrides);
    Value * const B = makeWeightedVal(maxNumOfStrides, DECREASE_WEIGHT_FACTOR, segmentLength);
    Value * const higher = b->CreateICmpUGT(segmentLength, maxNumOfStrides);
    Value * const nextSegmentLength = b->CreateSelect(higher, A, B);

    b->setScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(mKernelId), nextSegmentLength);
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateThreadLocalBuffersForSlidingWindow
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateThreadLocalBuffersForSlidingWindow(BuilderRef b) {
    #ifdef USE_DYNAMIC_SEGMENT_LENGTH_SLIDING_WINDOW
//    if (RequiredThreadLocalStreamSetMemory > 0) {
//        BasicBlock * const expandThreadLocalMemory = b->CreateBasicBlock();
//        BasicBlock * const afterExpansion = b->CreateBasicBlock();
//        Value * const nextMemReq = b->CreateLoad(mThreadLocalMemorySizePtr);
//        Value * const needsExpansion = b->CreateICmpULT(mCurrentThreadLocalMemorySize, nextMemReq);
//        b->CreateCondBr(needsExpansion, expandThreadLocalMemory, afterExpansion);

//        b->SetInsertPoint(expandThreadLocalMemory);
//        Value * const threadLocalPtr = b->getScalarFieldPtr(BASE_THREAD_LOCAL_STREAMSET_MEMORY);
//        b->CreateFree(b->CreateLoad(threadLocalPtr));
//        // we want to double the required space to minimize
//        Value * const expanded = b->CreateRoundUp(nextMemReq, mCurrentThreadLocalMemorySize);
//        Value * const base = b->CreatePageAlignedMalloc(expanded);
//        b->CreateStore(expanded, threadLocalPtr);
//        b->CreateBr(afterExpansion);

//        b->SetInsertPoint(afterExpansion);
//    }
    #endif
}

} // end of namespace
