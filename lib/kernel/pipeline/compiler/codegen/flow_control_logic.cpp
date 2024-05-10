#include "../pipeline_compiler.hpp"

// Each partition root determines how much data that it (and consequently its partition) can
// process in a single segment / pipeline iteration. However, it does not necessarily need to
// transfer all of the data provided to it and at times it may be beneficial to withhold data
// to better balance thread workloads.

// The functions here are designed to dynamically managed the maximum segment length of a
// partition root to promote this. By doing so, they may have to malloc a larger thread local
// memory pool or increase the repetition length of repeating streamsets.

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addSegmentLengthSlidingWindowKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addSegmentLengthSlidingWindowKernelProperties(KernelBuilder & b, const size_t kernelId, const size_t groupId) {
    if (MinimumNumOfStrides[kernelId] != MaximumNumOfStrides[kernelId] || mIsNestedPipeline) {
        mTarget->addInternalScalar(b.getSizeTy(), SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(kernelId), groupId);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeInitialSlidingWindowSegmentLengths
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeInitialSlidingWindowSegmentLengths(KernelBuilder & b, Value * const segmentLengthScalingFactor) {
    for (unsigned i = 1U; i < (PartitionCount - 1U); ++i) {
        const auto f = FirstKernelInPartition[i];
        const auto numOfStrides = MaximumNumOfStrides[f];
        if (MinimumNumOfStrides[f] != numOfStrides || mIsNestedPipeline) {
            Value * const init = b.CreateMul(segmentLengthScalingFactor, b.getSize(numOfStrides));
            b.setScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(f), init);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeFlowControl
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeFlowControl(KernelBuilder & b) {
    if (RequiredThreadLocalStreamSetMemory > 0) {
        mThreadLocalMemorySizePtr = b.getScalarFieldPtr(BASE_THREAD_LOCAL_STREAMSET_MEMORY_BYTES).first;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief detemineMaximumNumberOfStrides
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::detemineMaximumNumberOfStrides(KernelBuilder & b) {
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
                    assert (bn.BufferEnd > 0);
                    assert ((bn.BufferStart % b.getPageSize()) == 0);
                    assert ((bn.BufferEnd % b.getPageSize()) == 0);
                    maxMemory = std::max<size_t>(maxMemory, bn.BufferEnd);
                    assert (RequiredThreadLocalStreamSetMemory >= maxMemory);
                }
            }
        }

        Value * threadLocalPtr = nullptr;
        Type * threadLocalTy = nullptr;
        if (maxMemory) {
            std::tie(threadLocalPtr, threadLocalTy) = b.getScalarFieldPtr(BASE_THREAD_LOCAL_STREAMSET_MEMORY);
        }

        // If the min and max num of strides is equal, we almost certainly have strictly fixed
        // rate input into this partition. However if this a nested pipeline, we cannot assume
        // that the outer pipeline will feed data to this at a fixed rate.
        if (MinimumNumOfStrides[mKernelId] != MaximumNumOfStrides[mKernelId] || mIsNestedPipeline) {

            mMaximumNumOfStrides = b.getScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(mKernelId));

            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, "%s.maxNumOfStrides=%" PRIu64, mCurrentKernelName, mMaximumNumOfStrides);
            #endif

            // calculate how much memory is required by this partition relative to max num of strides
            // and determine if the current thread local buffer can fit it.

            if (maxMemory > 0) {

                mThreadLocalScalingFactor =
                    b.CreateCeilUDivRational(mMaximumNumOfStrides, MaximumNumOfStrides[mKernelId]);

                Value * const memoryForSegment = b.CreateMul(mThreadLocalScalingFactor, b.getSize(maxMemory));
                BasicBlock * const expandThreadLocalMemory = b.CreateBasicBlock();
                BasicBlock * const afterExpansion = b.CreateBasicBlock();
                Value * const currentMem = b.CreateLoad(b.getSizeTy(), mThreadLocalMemorySizePtr);
                Value * const needsExpansion = b.CreateICmpUGT(memoryForSegment, currentMem);
                b.CreateCondBr(needsExpansion, expandThreadLocalMemory, afterExpansion);

                b.SetInsertPoint(expandThreadLocalMemory);

                b.CreateFree(b.CreateLoad(threadLocalTy, threadLocalPtr));
                // At minimum, we want to double the required space to minimize future reallocs
                Value * expanded = b.CreateRoundUp(memoryForSegment, currentMem);
                b.CreateStore(expanded, mThreadLocalMemorySizePtr);
                #ifdef THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER
                expanded = b.CreateMul(expanded, b.getSize(THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER));
                #endif
                Value * const base = b.CreatePageAlignedMalloc(expanded);
                b.CreateStore(base, threadLocalPtr);
                b.CreateBr(afterExpansion);

                b.SetInsertPoint(afterExpansion);
            } else {
                mThreadLocalScalingFactor = nullptr;
            }

        } else {
            const auto numOfStrides = MaximumNumOfStrides[mCurrentPartitionRoot];
            mMaximumNumOfStrides = b.CreateMul(mExpectedNumOfStridesMultiplier, b.getSize(numOfStrides));
            mThreadLocalScalingFactor = mExpectedNumOfStridesMultiplier;
        }
        if (maxMemory > 0) {
            mThreadLocalStreamSetBaseAddress = b.CreateLoad(threadLocalTy, threadLocalPtr);
        } else {
            mThreadLocalStreamSetBaseAddress = nullptr;
            mThreadLocalScalingFactor = nullptr;
        }

    } else {
        const auto ratio = Rational{StrideStepLength[mKernelId], StrideStepLength[mCurrentPartitionRoot]};
        const auto factor = ratio / mPartitionStrideRateScalingFactor;
        mMaximumNumOfStrides = b.CreateMulRational(mNumOfPartitionStrides, factor);
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateNextSlidingWindowSize
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateNextSlidingWindowSize(KernelBuilder & b, Value * const maxNumOfStrides, Value * const actualNumOfStrides) {
    if (MinimumNumOfStrides[mKernelId] != MaximumNumOfStrides[mKernelId] || mIsNestedPipeline) {
        ConstantInt * const TWO = b.getSize(2);
        Value * const A = b.CreateMul(maxNumOfStrides, TWO);
        Value * const B = b.CreateAdd(maxNumOfStrides, actualNumOfStrides);
        assert (StrideStepLength[mKernelId] > 0);
        ConstantInt * const stepLength = b.getSize(StrideStepLength[mKernelId] * 2U);
        Value * const C = b.CreateRoundUp(B, stepLength);
        Value * const D = b.CreateUDiv(C, TWO);
        Value * const higher = b.CreateICmpUGT(actualNumOfStrides, maxNumOfStrides);
        Value * const nextMaxNumOfStrides = b.CreateSelect(higher, A, D);
        b.setScalarField(SCALED_SLIDING_WINDOW_SIZE_PREFIX + std::to_string(mKernelId), nextMaxNumOfStrides);
    }
}

} // end of namespace
