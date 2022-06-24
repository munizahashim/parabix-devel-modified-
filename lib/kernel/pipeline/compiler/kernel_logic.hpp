#include "pipeline_compiler.hpp"

#include <llvm/Support/ErrorHandling.h>

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setActiveKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::setActiveKernel(BuilderRef b, const unsigned kernelId, const bool allowThreadLocal) {
    assert (kernelId >= FirstKernel && kernelId <= LastKernel);
    mKernelId = kernelId;
    mKernel = getKernel(kernelId);
    mKernelSharedHandle = nullptr;
    if (LLVM_LIKELY(mKernel->isStateful())) {
        Value * handle = b->getScalarFieldPtr(makeKernelName(kernelId));
        if (LLVM_UNLIKELY(mKernel->externallyInitialized())) {
            handle = b->CreateLoad(b->CreatePointerCast(handle, mKernel->getSharedStateType()->getPointerTo()->getPointerTo()));
        }
        mKernelSharedHandle = handle;
    }
    mKernelThreadLocalHandle = nullptr;
    if (mKernel->hasThreadLocal() && allowThreadLocal) {
        mKernelThreadLocalHandle = getThreadLocalHandlePtr(b, mKernelId);
    }
    mCurrentKernelName = mKernelName[mKernelId];
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeFullyProcessedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::computeFullyProcessedItemCounts(BuilderRef b, Value * const terminated) {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto port = br.Port;
        Value * processed = nullptr;
        if (mUpdatedProcessedDeferredPhi[port]) {
            processed = mUpdatedProcessedDeferredPhi[port];
        } else {
            processed = mUpdatedProcessedPhi[port];
        }
        const Binding & input = br.Binding;
        Value * const fullyProcessed = truncateBlockSize(b, input, processed, terminated);
        mFullyProcessedItemCount[port] = fullyProcessed;
        if (LLVM_UNLIKELY(CheckAssertions)) {
            const auto streamSet = source(e, mBufferGraph);
            const auto producer = parent(streamSet, mBufferGraph);
            if (mCurrentPartitionId == KernelPartitionId[producer]) {
            // if (bn.Locality == BufferLocality::ThreadLocal) {
                Value * const produced = mLocallyAvailableItems[streamSet]; assert (produced);
                // NOTE: static linear buffers are assumed to be threadlocal.
                Value * const fullyConsumed = b->CreateICmpEQ(produced, processed);
                Constant * const fatal = getTerminationSignal(b, TerminationSignal::Fatal);
                Value * const fatalError = b->CreateICmpEQ(mTerminatedAtLoopExitPhi, fatal);
                Value * const valid = b->CreateOr(fullyConsumed, fatalError);
                Constant * const bindingName = b->GetString(input.getName());

                Constant * withOrWithout = nullptr;
                if (mMayLoopToEntry) {
                    withOrWithout = b->GetString("with");
                } else {
                    withOrWithout = b->GetString("without");
                }

                b->CreateAssert(valid,
                                "%s.%s: local available item count (%" PRId64 ") does not match "
                                "its processed item count (%" PRId64 ") in kernel %s loop back support",
                                mCurrentKernelName, bindingName,
                                produced, processed, withOrWithout);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeFullyProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::computeFullyProducedItemCounts(BuilderRef b, Value * const terminated) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto port = br.Port;
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        Value * produced = nullptr;
        if (LLVM_UNLIKELY(bn.OutputItemCountId != streamSet)) {
            produced = mLocallyAvailableItems[bn.OutputItemCountId];
        } else {
            produced = computeFullyProducedItemCount(b, mKernelId, port, mUpdatedProducedPhi[port], terminated);
        }
        assert (isFromCurrentFunction(b, produced, false));
        mFullyProducedItemCount[port]->addIncoming(produced, mKernelLoopExitPhiCatch);
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeFullyProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::computeFullyProducedItemCount(BuilderRef b,
                                                        const size_t kernel,
                                                        const StreamSetPort port,
                                                        Value * produced, Value * const terminationSignal) {

    // TODO: we only need to consider the blocksize attribute if it's possible this
    // stream could be read before being fully written. This might occur if one of
    // it's consumers has a non-Fixed rate that does not have a matching BlockSize
    // attribute.

    assert ("produced cannot be null" && produced);

    const Binding & output = getOutputBinding(kernel, port);
    if (LLVM_UNLIKELY(output.hasAttribute(AttrId::Delayed))) {
        const auto & D = output.findAttribute(AttrId::Delayed);
        Value * const delayed = b->CreateSaturatingSub(produced, b->getSize(D.amount()));
        assert (terminationSignal && terminationSignal->getType()->isIntegerTy(1));
        const auto name = makeBufferName(mKernelId, port) + "_delayedUntilTermination";
        produced = b->CreateSelect(terminationSignal, produced, delayed, name);
    }
    return truncateBlockSize(b, output, produced, terminationSignal);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addLookahead
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::addLookahead(BuilderRef b, const BufferPort & inputPort, Value * const itemCount) const {
    if (LLVM_LIKELY(inputPort.LookAhead == 0)) {
        return itemCount;
    }
    Constant * const lookAhead = b->getSize(inputPort.LookAhead);
    return b->CreateAdd(itemCount, lookAhead);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief subtractLookahead
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::subtractLookahead(BuilderRef b, const BufferPort & inputPort, Value * const itemCount) {
    if (LLVM_LIKELY(inputPort.LookAhead == 0)) {
        return itemCount;
    }
    Constant * const lookAhead = b->getSize(inputPort.LookAhead);
    Value * const closed = isClosed(b, inputPort.Port);
    Value * const reducedItemCount = b->CreateSaturatingSub(itemCount, lookAhead);
    return b->CreateSelect(closed, itemCount, reducedItemCount);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief maskBlockSize
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::truncateBlockSize(BuilderRef b, const Binding & binding, Value * itemCount, Value * const terminationSignal) const {
    // TODO: if we determine all of the inputs of a stream have a blocksize attribute, or the output has one,
    // we can skip masking it on input



    if (LLVM_UNLIKELY(binding.hasAttribute(AttrId::BlockSize))) {
        // If the input rate has a block size attribute then --- for the purpose of determining how many
        // items have been consumed --- we consider a stream set to be fully processed when an entire
        // stride has been processed.
        Constant * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
        Value * const maskedItemCount = b->CreateAnd(itemCount, ConstantExpr::getNeg(BLOCK_WIDTH));
        itemCount = b->CreateSelect(terminationSignal, itemCount, maskedItemCount);
    }
    return itemCount;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getThreadLocalHandlePtr
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getThreadLocalHandlePtr(BuilderRef b, const unsigned kernelIndex) const {
    const Kernel * const kernel = getKernel(kernelIndex);
    assert ("getThreadLocalHandlePtr should not have been called" && kernel->hasThreadLocal());
    const auto prefix = makeKernelName(kernelIndex);
    Value * handle = getScalarFieldPtr(b.get(), prefix + KERNEL_THREAD_LOCAL_SUFFIX);
    if (LLVM_UNLIKELY(kernel->externallyInitialized())) {
        StructType * const localStateTy = kernel->getThreadLocalStateType();
        handle = b->CreatePointerCast(b->CreateLoad(handle), localStateTy->getPointerTo());
    }
    assert (handle->getType()->isPointerTy());
    return handle;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getCommonThreadLocalHandlePtr
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getCommonThreadLocalHandlePtr(BuilderRef b, const unsigned kernelIndex) const {
    const Kernel * const kernel = getKernel(kernelIndex);
    assert ("getThreadLocalHandlePtr should not have been called" && kernel->hasThreadLocal());
    const auto prefix = makeKernelName(kernelIndex);
    Value * handle = getCommonThreadLocalScalarFieldPtr(b.get(), prefix + KERNEL_THREAD_LOCAL_SUFFIX);
    if (LLVM_UNLIKELY(kernel->externallyInitialized())) {
        StructType * const localStateTy = kernel->getThreadLocalStateType();
        handle = b->CreatePointerCast(b->CreateLoad(handle), localStateTy->getPointerTo());
    }
    assert (handle->getType()->isPointerTy());
    return handle;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isBounded
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::isBounded() const {
    assert (mKernelId >= FirstKernel && mKernelId <= LastKernel);
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = source(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isNonThreadLocal()) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & binding = br.Binding;
            const ProcessingRate & rate = binding.getRate();
            switch (rate.getKind()) {
                case RateId::Bounded:
                case RateId::Fixed:
                case RateId::PartialSum:
                    return true;
                case RateId::Greedy:
                    if (rate.getLowerBound() > Rational{0, 1}) {
                        return true;
                    }
                default: break;
            }
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief requiresExplicitFinalStride
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::requiresExplicitFinalStride() const {
    assert (mKernelId >= FirstKernel && mKernelId <= LastKernel);
    if (mKernel->requiresExplicitPartialFinalStride()) {
        return true;
    }
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = source(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isOwned()) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & binding = br.Binding;
            const ProcessingRate & rate = binding.getRate();
            switch (rate.getKind()) {
                case RateId::Fixed:
                case RateId::PartialSum:
                    return true;
                default: break;
            }
        }
    }
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isOwned()) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & binding = br.Binding;
            const ProcessingRate & rate = binding.getRate();
            switch (rate.getKind()) {
                case RateId::Fixed:
                case RateId::PartialSum:
                    return true;
                default: break;
            }
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyPipelineInputs
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::identifyPipelineInputs(const unsigned kernelId) {
    mHasPipelineInput.reset();
    mHasPipelineInput.resize(in_degree(kernelId, mBufferGraph));

    if (LLVM_LIKELY(out_degree(PipelineInput, mBufferGraph) > 0)) {
        for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            const auto producer = parent(streamSet, mBufferGraph);
            if (LLVM_UNLIKELY(producer == PipelineInput)) {
                const BufferPort & br = mBufferGraph[e];
                mHasPipelineInput.set(br.Port.Number);
            }
        }
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasExternalIO
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::hasExternalIO(const size_t kernel) const {
    for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
        const auto streamSet = source(input, mBufferGraph);
        const BufferNode & node = mBufferGraph[streamSet];
        if (node.isExternal()) {
            return true;
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasAtLeastOneNonGreedyInput
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::hasAtLeastOneNonGreedyInput() const {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & bp = mBufferGraph[e];
        const Binding & binding = bp.Binding;
        if (!binding.getRate().isGreedy()) {
            return true;
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isDataParallel
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::isDataParallel(const size_t kernel) const {
    return mIsStatelessKernel.test(kernel) || mIsInternallySynchronized.test(kernel);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isCurrentKernelStateFree
 ** ------------------------------------------------------------------------------------------------------------- */
inline bool PipelineCompiler::isCurrentKernelStateFree() const {
    return PipelineCommonGraphFunctions::isKernelStateFree(mKernelId);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputBufferVertex
 ** ------------------------------------------------------------------------------------------------------------- */
inline unsigned PipelineCompiler::getInputBufferVertex(const StreamSetPort inputPort) const {
    return PipelineCommonGraphFunctions::getInputBufferVertex(mKernelId, inputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputBuffer
 ** ------------------------------------------------------------------------------------------------------------- */
inline StreamSetBuffer * PipelineCompiler::getInputBuffer(const StreamSetPort inputPort) const {
    return PipelineCommonGraphFunctions::getInputBuffer(mKernelId, inputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputBinding
 ** ------------------------------------------------------------------------------------------------------------- */
inline const Binding & PipelineCompiler::getInputBinding(const StreamSetPort inputPort) const {
    return PipelineCommonGraphFunctions::getInputBinding(mKernelId, inputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputBufferVertex
 ** ------------------------------------------------------------------------------------------------------------- */
inline unsigned PipelineCompiler::getOutputBufferVertex(const StreamSetPort outputPort) const {
    return PipelineCommonGraphFunctions::getOutputBufferVertex(mKernelId, outputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputBinding
 ** ------------------------------------------------------------------------------------------------------------- */
inline const Binding & PipelineCompiler::getOutputBinding(const StreamSetPort outputPort) const {
    return PipelineCommonGraphFunctions::getOutputBinding(mKernelId, outputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getOutputBuffer
 ** ------------------------------------------------------------------------------------------------------------- */
inline StreamSetBuffer * PipelineCompiler::getOutputBuffer(const StreamSetPort outputPort) const {
    return PipelineCommonGraphFunctions::getOutputBuffer(mKernelId, outputPort);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getBinding
 ** ------------------------------------------------------------------------------------------------------------- */
inline const Binding & PipelineCompiler::getBinding(const StreamSetPort port) const {
    return PipelineCommonGraphFunctions::getBinding(mKernelId, port);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getReference
 ** ------------------------------------------------------------------------------------------------------------- */
inline const StreamSetPort PipelineCompiler::getReference(const StreamSetPort port) const {
    return PipelineCommonGraphFunctions::getReference(mKernelId, port);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief reset
 ** ------------------------------------------------------------------------------------------------------------- */
template <typename Vec>
inline void reset(Vec & vec, const size_t n) {
    vec.resize(n);
    std::memset(vec.data(), 0, n * sizeof(typename Vec::value_type));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief clearInternalStateForCurrentKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::clearInternalStateForCurrentKernel() {

    // TODO: make it so these are only needed in debug mode for assertion checks?

    mNumOfTruncatedInputBuffers = 0;

    mExecuteStridesIndividually = false;
    mCurrentKernelIsStateFree = false;
    mAllowDataParallelExecution = false;

    mHasMoreInput = nullptr;
    mHasZeroExtendedInput = nullptr;
    mExhaustedPipelineInputPhi = nullptr;
    mExhaustedInputAtJumpPhi = nullptr;

    mAnyClosed[0] = nullptr;
    mAnyClosed[1] = nullptr;

    mKernelInsufficientInput = nullptr;
    mKernelTerminated = nullptr;
    mKernelInitiallyTerminated = nullptr;
    mKernelInitiallyTerminatedExit = nullptr;
    mInitiallyTerminated = nullptr;

    mMaximumNumOfStrides = nullptr;
    mNumOfLinearStridesPhi = nullptr;
    mNumOfLinearStrides = nullptr;
    mFixedRateFactorPhi = nullptr;
    mFinalPartialStrideFixedRateRemainderPhi = nullptr;
    mIsFinalInvocationPhi = nullptr;
    mIsFinalInvocation = nullptr;

    assert (mKernelId >= FirstKernel);
    assert (mKernelId <= LastKernel);

    const auto numOfInputs = in_degree(mKernelId, mBufferGraph);
    reset(mAccessibleInputItems, numOfInputs);
    mInitiallyProcessedItemCount.reset(numOfInputs);
    mInitiallyProcessedDeferredItemCount.reset(numOfInputs);
    mAlreadyProcessedPhi.reset(numOfInputs);
    mAlreadyProcessedDeferredPhi.reset(numOfInputs);
    mIsInputZeroExtended.reset(numOfInputs);
    mInputVirtualBaseAddressPhi.reset(numOfInputs);
    mFirstInputStrideLength.reset(numOfInputs);
    mLinearInputItemsPhi.reset(numOfInputs);
    mReturnedProcessedItemCountPtr.reset(numOfInputs);
    mProcessedItemCountPtr.reset(numOfInputs);
    mProcessedItemCount.reset(numOfInputs);
    mProcessedDeferredItemCountPtr.reset(numOfInputs);
    mProcessedDeferredItemCount.reset(numOfInputs);
    mCurrentProcessedItemCountPhi.reset(numOfInputs);
    mCurrentProcessedDeferredItemCountPhi.reset(numOfInputs);
    mCurrentLinearInputItems.reset(numOfInputs);
    mUpdatedProcessedPhi.reset(numOfInputs);
    mUpdatedProcessedDeferredPhi.reset(numOfInputs);
    mConsumedItemCountsAtLoopExitPhi.reset(numOfInputs);
    mFullyProcessedItemCount.reset(numOfInputs);

    const auto numOfOutputs = out_degree(mKernelId, mBufferGraph);
    reset(mWritableOutputItems, numOfOutputs);
    mAlreadyProducedPhi.reset(numOfOutputs);
    mAlreadyProducedDelayedPhi.reset(numOfOutputs);
    mAlreadyProducedDeferredPhi.reset(numOfOutputs);
    mFirstOutputStrideLength.reset(numOfOutputs);
    mLinearOutputItemsPhi.reset(numOfOutputs);
    mReturnedOutputVirtualBaseAddressPtr.reset(numOfOutputs);
    mReturnedProducedItemCountPtr.reset(numOfOutputs);
    mProducedItemCountPtr.reset(numOfOutputs);
    mProducedItemCount.reset(numOfOutputs);
    mProducedDeferredItemCountPtr.reset(numOfOutputs);
    mProducedDeferredItemCount.reset(numOfOutputs);    
    mCurrentProducedItemCountPhi.reset(numOfOutputs);
    mCurrentProducedDeferredItemCountPhi.reset(numOfOutputs);
    mCurrentLinearOutputItems.reset(numOfOutputs);
    mProducedAtJumpPhi.reset(numOfOutputs);
    mProducedAtTerminationPhi.reset(numOfOutputs);
    mProducedAtTermination.reset(numOfOutputs);
    mUpdatedProducedPhi.reset(numOfOutputs);
    mUpdatedProducedDeferredPhi.reset(numOfOutputs);
    mFullyProducedItemCount.reset(numOfOutputs);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelAssertions
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeKernelAssertions(BuilderRef b) {
    SmallVector<char, 256> tmp;
    for (auto kernel = PipelineInput; kernel <= LastKernel; ++kernel) {
        raw_svector_ostream out(tmp);
        out << kernel << "." << getKernel(kernel)->getName();
        mKernelName[kernel] = b->GetString(out.str());
        tmp.clear();
    }
}

}
