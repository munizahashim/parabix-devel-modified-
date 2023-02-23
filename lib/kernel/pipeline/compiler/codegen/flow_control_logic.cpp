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
    if (MinimumNumOfStrides[kernelId] != MaximumNumOfStrides[kernelId]) {
        mTarget->addInternalScalar(b->getSizeTy(), SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(kernelId), groupId);
    }
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
        if (MinimumNumOfStrides[f] != numOfStrides) {
            Value * const init = b->CreateMul(segmentLengthScalingFactor, b->getSize(numOfStrides));
            b->setScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(f), init);
        }
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

        assert (mCurrentPartitionId == KernelPartitionId[mKernelId]);
        assert (mKernelId == FirstKernelInPartition[KernelPartitionId[mKernelId]]);
        const auto firstKernelOfNextPartition = FirstKernelInPartition[mCurrentPartitionId + 1];
        size_t maxMemory = 0;
        for (auto kernel = mKernelId; kernel < firstKernelOfNextPartition; ++kernel) {
            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];
                if (bn.isThreadLocal()) {
                    maxMemory = std::max<size_t>(maxMemory, bn.BufferEnd);
                    assert (RequiredThreadLocalStreamSetMemory >= maxMemory);
                }
            }
        }

        Value * threadLocalPtr = nullptr;
        if (maxMemory) {
            threadLocalPtr = b->getScalarFieldPtr(BASE_THREAD_LOCAL_STREAMSET_MEMORY);
        }

        #ifdef USE_DYNAMIC_SEGMENT_LENGTH_SLIDING_WINDOW
        // If the min and max num of strides is equal, we almost certainly have strictly fixed
        // rate input into this partition.
        if (MinimumNumOfStrides[mKernelId] != MaximumNumOfStrides[mKernelId]) {

            mMaximumNumOfStrides = b->getScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(mKernelId));

            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, "%s.maxNumOfStrides=%" PRIu64, mCurrentKernelName, mMaximumNumOfStrides);
            #endif

            // calculate how much memory is required by this partition relative to max num of strides
            // and determine if the current thread local buffer can fit it.

            // TODO: suppose the start and end position of every threadlocal streamset is page
            // aligned. Thus the total thread local memory alloced here would be page aligned.
            // We want to expand the buffer such that we scale all the buffer start/end offset
            // by a fixed integer to maintain the alignment but want the smallest such integer
            // that will fit our new memory requirement (that at least doubles the prior one).

            // We then store the capacity modifier and simply rescale it for the new one.

            // TODO: should the start/end positions reflect the minumum amount of work? if they
            // we increased to reflect to a page alignment, this wouldn't be meaningful. So what
            // value ought I be considering here?

            if (maxMemory > 0) {

                // CEIL (  (a + (b/c)) / (x/y) ) = CEIL ( y * (ac + b) / cx )

                const auto & BC = PartitionOverflowStrides[mCurrentPartitionId];
                const auto & XY = PartitionRootStridesPerThreadLocalPage[mCurrentPartitionId];

                Value * V = mMaximumNumOfStrides;
                if (BC.denominator() > 1) {
                    V = b->CreateMul(V, b->getSize(BC.denominator()));
                }
                if (BC.numerator() > 0) {
                    V = b->CreateAdd(V, b->getSize(BC.numerator()));
                }
                if (XY.denominator() > 1) {
                    V = b->CreateMul(V, b->getSize(XY.denominator()));
                }
                const auto cx = BC.denominator() * XY.numerator(); assert (cx > 0);
                if (cx > 1) {
                    V = b->CreateCeilUDiv(V, b->getSize(cx));
                }
                Value * memoryForSegment = b->CreateMul(V, b->getSize(maxMemory));
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
            }

        } else {
            const auto numOfStrides = MaximumNumOfStrides[mCurrentPartitionRoot];
            mMaximumNumOfStrides = b->CreateMul(mExpectedNumOfStridesMultiplier, b->getSize(numOfStrides));
        }

        if (maxMemory > 0) {
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
void PipelineCompiler::updateNextSlidingWindowSize(BuilderRef b, Value * const maxNumOfStrides, Value * const actualNumOfStrides) {
    #ifdef USE_DYNAMIC_SEGMENT_LENGTH_SLIDING_WINDOW
    if (MinimumNumOfStrides[mKernelId] != MaximumNumOfStrides[mKernelId]) {
        ConstantInt * const TWO = b->getSize(2);
        Value * const A = b->CreateMul(maxNumOfStrides, TWO);
        Value * const B = b->CreateAdd(maxNumOfStrides, actualNumOfStrides);
        assert (StrideStepLength[mKernelId] > 0);
        ConstantInt * const stepLength = b->getSize(StrideStepLength[mKernelId] * 2U);
        Value * const C = b->CreateRoundUp(B, stepLength);
        Value * const D = b->CreateUDiv(C, TWO);
        Value * const higher = b->CreateICmpUGT(actualNumOfStrides, maxNumOfStrides);
        Value * const nextMaxNumOfStrides = b->CreateSelect(higher, A, D);
        b->setScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(mKernelId), nextMaxNumOfStrides);
    }
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
