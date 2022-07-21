#ifndef PARTITION_PROCESSING_LOGIC_HPP
#define PARTITION_PROCESSING_LOGIC_HPP

#include "pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makePartitionEntryPoints
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::makePartitionEntryPoints(BuilderRef b) {    


    // the zeroth partition may be a fake one if this pipeline has I/O
    for (unsigned i = 1; i < PartitionCount; ++i) {
        mPartitionEntryPoint[i] = b->CreateBasicBlock("Partition" + std::to_string(i), mPipelineEnd);
    }    
    mPartitionEntryPoint[PartitionCount] = mPipelineEnd;

    const auto ip = b->saveIP();
    IntegerType * const boolTy = b->getInt1Ty();
    IntegerType * const sizeTy = b->getInt64Ty();

    for (unsigned i = 2; i < PartitionCount; ++i) {
        b->SetInsertPoint(mPartitionEntryPoint[i]);
        const auto prefix = std::to_string(i);
        mPartitionPipelineProgressPhi[i] = b->CreatePHI(boolTy, PartitionCount, prefix + ".pipelineProgress");
        mExhaustedPipelineInputAtPartitionEntry[i] = b->CreatePHI(boolTy, PartitionCount, prefix + ".exhaustedInput");
        if (LLVM_UNLIKELY(EnableCycleCounter)) {
            mPartitionStartTimePhi[i] = b->CreatePHI(sizeTy, PartitionCount, prefix + ".startTimeCycleCounter");
        }
    }

    // Create any PHI nodes we need to propogate the current produced/consumed item counts
    // of the kernels we jump over as well as the termination signals for any kernel we may
    // need to check if its closed or not.

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isNonThreadLocal()) {

            const auto output = in_edge(streamSet, mBufferGraph);
            const auto producer = source(output, mBufferGraph);

            const BufferPort & outputPort = mBufferGraph[output];
            const auto prefix = makeBufferName(producer, outputPort.Port);

            const auto k = streamSet - FirstStreamSet;

            auto lastReader = producer;
            if (producer != PipelineInput) {
                for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                    const auto consumer = target(input, mBufferGraph);
                    lastReader = std::max(lastReader, consumer);
                }
                const auto readsPartId = KernelPartitionId[lastReader];
                const auto prodPrefix = prefix + "_produced@partition";
                const auto prodPartId = KernelPartitionId[producer];
                for (auto partitionId = prodPartId + 1; partitionId <= readsPartId; ++partitionId) {
                    b->SetInsertPoint(mPartitionEntryPoint[partitionId]);
                    PHINode * const phi = b->CreatePHI(sizeTy, PartitionCount, prodPrefix + std::to_string(partitionId));
                    mPartitionProducedItemCountPhi[partitionId][k] = phi;
                }
            }

        }
    }


    // any termination signal needs to be phi-ed out if it can be read by a descendent
    // or guards the loop condition at the end of the pipeline loop.

    assert (KernelPartitionId[PipelineInput] == 0);
    assert (KernelPartitionId[PipelineOutput] == PartitionCount - 1);

    BitVector toCheck(PartitionCount, 0);

    for (unsigned i = 0; i < PartitionCount; ++i) {
        if (mTerminationCheck[i]) {
            toCheck.set(i);
        }
    }

    auto partitionId = KernelPartitionId[PipelineOutput];

    for (auto kernel = PipelineOutput; kernel >= FirstKernel; ) {

        const auto lastKernel = kernel;
        for (;;--kernel) {
            if (KernelPartitionId[kernel] != partitionId) {
                break;
            }
            assert (kernel >= FirstKernel);
        }
        assert (partitionId >= 0);

        const auto firstKernel = kernel + 1U;
        assert (KernelPartitionId[firstKernel] == partitionId);
        assert (KernelPartitionId[lastKernel] == partitionId);
        for (auto k = firstKernel; k <= lastKernel; ++k) {
            for (const auto input : make_iterator_range(in_edges(k, mBufferGraph))) {
                const auto streamSet = source(input, mBufferGraph);
                const auto producer = parent(streamSet, mBufferGraph);
                const auto prodPartId = KernelPartitionId[producer];
                toCheck.set(prodPartId);
            }
        }

        auto entryPoint = mPartitionEntryPoint[partitionId];

        const auto prefix = "terminationSignalForPartition" + std::to_string(partitionId) + "@";

        auto termId = toCheck.find_first_in(FirstKernel, partitionId);
        while (termId != -1) {
            PHINode * const phi = PHINode::Create(sizeTy, 2, prefix + std::to_string(termId), entryPoint);
            mPartitionTerminationSignalPhi[partitionId][termId] = phi;
            termId = toCheck.find_first_in(termId + 1, partitionId);
        }

        partitionId = KernelPartitionId[kernel];
    }
    b->restoreIP(ip);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief branchToInitialPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::branchToInitialPartition(BuilderRef b) {

    const auto firstPartition = KernelPartitionId[FirstKernel];
    BasicBlock * const entry = mPartitionEntryPoint[firstPartition];
    b->CreateBr(entry);

    b->SetInsertPoint(entry);
    mCurrentPartitionId = -1U;
    setActiveKernel(b, FirstKernel, true);
    #ifdef ENABLE_PAPI
    readPAPIMeasurement(b, FirstKernel, PAPIReadInitialMeasurementArray);
    #endif
    mKernelStartTime = startCycleCounter(b);
    if (mNumOfThreads != 1 || mIsNestedPipeline) {
        const auto type = isDataParallel(FirstKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        acquireSynchronizationLock(b, FirstKernel, type);
        updateCycleCounter(b, FirstKernel, mKernelStartTime, CycleCounter::KERNEL_SYNCHRONIZATION);
        #ifdef ENABLE_PAPI
        accumPAPIMeasurementWithoutReset(b, PAPIReadInitialMeasurementArray, FirstKernel, PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION);
        #endif
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getPartitionExitPoint
 ** ------------------------------------------------------------------------------------------------------------- */
BasicBlock * PipelineCompiler::getPartitionExitPoint(BuilderRef /* b */) {
    assert (mKernelId >= FirstKernel && mKernelId <= PipelineOutput);
    const auto nextPartitionId = mCurrentPartitionId + 1;
    return mPartitionEntryPoint[nextPartitionId];
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForPartitionEntry
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::checkForPartitionEntry(BuilderRef b) {
    assert (mKernelId >= FirstKernel && mKernelId <= LastKernel);
    mIsPartitionRoot = false;
    const auto partitionId = KernelPartitionId[mKernelId];
    if (partitionId != mCurrentPartitionId) {
        mCurrentPartitionId = partitionId;
        mIsPartitionRoot = true;
        identifyPartitionKernelRange();
        determinePartitionStrideRateScalingFactor();
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "  *** entering partition %" PRIu64, b->getSize(mCurrentPartitionId));
        #endif
    }
    assert (KernelPartitionId[mKernelId - 1U] <= mCurrentPartitionId);
    assert ((KernelPartitionId[mKernelId - 1U] + 1U) >= mCurrentPartitionId);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyPartitionKernelRange
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::identifyPartitionKernelRange() {
    FirstKernelInPartition = mKernelId;
    for (auto i = mKernelId + 1U; i <= PipelineOutput; ++i) {
        if (KernelPartitionId[i] != mCurrentPartitionId) {
            LastKernelInPartition = i - 1U;
            break;
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief determinePartitionStrideRateScalingFactor
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::determinePartitionStrideRateScalingFactor() {
    auto l = StrideStepLength[FirstKernelInPartition];
    auto g = StrideStepLength[FirstKernelInPartition];
    for (auto i = FirstKernelInPartition + 1U; i <= LastKernelInPartition; ++i) {
        l = boost::lcm(l, StrideStepLength[i]);
        g = boost::gcd(g, StrideStepLength[i]);
    }
    assert (l > 0 && g > 0);
    // If a kernel within this partition has a min/max stride value that is greater
    // than the min/max stride of the partition root then when the root kernel
    // executes its final block, its partial stride may actually require the other
    // kernel executes N full strides and a final block. To accomidate this
    // possibility, the partition root scales the num of partition strides
    // full+partial strides by to the following ratio:
    mPartitionStrideRateScalingFactor = Rational{l,g};
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadLastGoodVirtualBaseAddressesOfUnownedBuffersInCurrentPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::loadLastGoodVirtualBaseAddressesOfUnownedBuffersInPartition(BuilderRef b) const {
    for (auto i = mKernelId; i <= LastKernel; ++i) {
        if (KernelPartitionId[i] != mCurrentPartitionId) {
            break;
        }
        loadLastGoodVirtualBaseAddressesOfUnownedBuffers(b, i);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadLastGoodVirtualBaseAddressesOfUnownedBuffersInCurrentPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::phiOutPartitionItemCounts(BuilderRef b, const unsigned kernel,
                                                 const unsigned targetPartitionId,
                                                 const bool fromKernelEntryBlock) {

    BasicBlock * const exitPoint = b->GetInsertBlock();

    for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);

        // When jumping out of a partition to some subsequent one, we may have to
        // phi-out some of the produced item counts. We 3 cases to consider:

        // (1) if we've executed the kernel, we use the fully produced item count.
        // (2) if producer is the current kernel, we use the already produced phi node.
        // (3) if we have yet to execute (and will be jumping over) the kernel, load
        // the prior produced count.


        const unsigned k = streamSet - FirstStreamSet;

        const BufferPort & br = mBufferGraph[e];
        // Select/load the appropriate produced item count
        PHINode * const prodPhi = mPartitionProducedItemCountPhi[targetPartitionId][k];
        if (prodPhi) {
            Value * produced = nullptr;
            if (kernel < mKernelId) {
                produced = mLocallyAvailableItems[streamSet];
            } else if (kernel == mKernelId && !mAllowDataParallelExecution) {
                if (fromKernelEntryBlock) {
                    if (LLVM_UNLIKELY(br.IsDeferred)) {
                        produced = mInitiallyProducedDeferredItemCount[streamSet];
                    } else {
                        produced = mInitiallyProducedItemCount[streamSet];
                    }
                } else if (mProducedAtJumpPhi[br.Port]) {
                    produced = mProducedAtJumpPhi[br.Port];
                } else {
                    if (LLVM_UNLIKELY(br.IsDeferred)) {
                        produced = mAlreadyProducedDeferredPhi[br.Port];
                    } else {
                        produced = mAlreadyProducedPhi[br.Port];
                    }
                }
            } else { // if (kernel > mKernelId) {
                const auto prefix = makeBufferName(kernel, br.Port);
                if (LLVM_UNLIKELY(br.IsDeferred)) {
                    produced = b->getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
                } else {
                    produced = b->getScalarField(prefix + ITEM_COUNT_SUFFIX);
                }
            }

            assert (isFromCurrentFunction(b, produced, false));

            #ifdef PRINT_DEBUG_MESSAGES
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << makeKernelName(mKernelId) << " -> " <<
                   makeBufferName(kernel, br.Port) << "_avail = %" PRIu64;
            debugPrint(b, out.str(), produced);
            #endif

            prodPhi->addIncoming(produced, exitPoint);
        }

    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief phiOutPartitionStatusFlags
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::phiOutPartitionStatusFlags(BuilderRef b, const unsigned targetPartitionId,
                                                  const bool /* fromKernelEntry */) {

    BasicBlock * const exitPoint = b->GetInsertBlock();

    const auto firstPartition = KernelPartitionId[FirstKernel];

    for (auto partitionId = firstPartition; partitionId != targetPartitionId; ++partitionId) {
        PHINode * const termPhi = mPartitionTerminationSignalPhi[targetPartitionId][partitionId];
        if (termPhi) {
            Value * term = nullptr;
            if (partitionId < mCurrentPartitionId) {
                term = mPartitionTerminationSignal[partitionId]; assert (term);
            } else {
                term = readTerminationSignal(b, partitionId); assert (term);
            }
            assert (isFromCurrentFunction(b, term));
            termPhi->addIncoming(term, exitPoint);
        }
    }

    PHINode * const progressPhi = mPartitionPipelineProgressPhi[targetPartitionId];
    assert (progressPhi);
    assert (isFromCurrentFunction(b, progressPhi, false));
    Value * const progress = mPipelineProgress; // fromKernelEntry ? mPipelineProgress : mAlreadyProgressedPhi;
    assert (isFromCurrentFunction(b, progress, false));
    progressPhi->addIncoming(progress, exitPoint);

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getFirstKernelInTargetPartition
 ** ------------------------------------------------------------------------------------------------------------- */
unsigned PipelineCompiler::getFirstKernelInTargetPartition(const unsigned partitionId) const {
    for (auto kernel = PipelineInput; kernel <= PipelineOutput; ++kernel) {
        if (KernelPartitionId[kernel] == partitionId) {
            return kernel;
        }
        assert (KernelPartitionId[kernel] < partitionId);
    }
    llvm_unreachable("unknown partition id?");
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief acquirePartitionSynchronizationLock
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::acquirePartitionSynchronizationLock(BuilderRef b, const unsigned firstKernelInTargetPartition) {
    // Find the first kernel in the partition we're jumping to and acquire the LSN then release
    // all of the kernels we skipped over. However, to safely jump to a partition, we need to
    // know how many items were processed by any consumers of the kernels up to the target
    // kernel; otherwise we run the risk of mangling the buffer state. For safety, wait until we
    // can acquire the last consumer's lock but only release the locks that we end up skipping.
    // TODO: experiment with a mutex lock here.
    #ifdef ENABLE_PAPI
    readPAPIMeasurement(b, mKernelId, PAPIReadBeforeMeasurementArray);
    #endif
    Value * const startTime = startCycleCounter(b);
    assert (firstKernelInTargetPartition <= PipelineOutput);

    const auto targetLock = (firstKernelInTargetPartition == PipelineOutput) ? LastKernel : firstKernelInTargetPartition;
    const auto type = isDataParallel(targetLock) ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;
    acquireSynchronizationLock(b, targetLock, type);

    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        updateCycleCounter(b, mKernelId, startTime, CycleCounter::PARTITION_JUMP_SYNCHRONIZATION);
        BasicBlock * const exitBlock = b->GetInsertBlock();
        mPartitionStartTimePhi[KernelPartitionId[firstKernelInTargetPartition]]->addIncoming(startTime, exitBlock);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief releaseAllSynchronizationLocksUntil
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::releaseAllSynchronizationLocksUntil(BuilderRef b, const unsigned firstKernelInTargetPartition) {
    // Find the first kernel in the partition we're jumping to and acquire the LSN then release
    // all of the kernels we skipped over. However, to safely jump to a partition, we need to
    // know how many items were processed by any consumers of the kernels up to the target
    // kernel; otherwise we run the risk of mangling the buffer state. For safety, wait until we
    // can acquire the last consumer's lock but only release the locks that we end up skipping.

    unsigned releasedPartitionId = 0;
    ConstantInt * const sz_ZERO = b->getSize(0);
    for (auto kernel = mKernelId; kernel < firstKernelInTargetPartition; ++kernel) {
        if (releasedPartitionId != KernelPartitionId[kernel]) {
            releasedPartitionId = KernelPartitionId[kernel];
            recordStridesPerSegment(b, kernel, sz_ZERO);
        }
        if (isDataParallel(kernel)) {
            releaseSynchronizationLock(b, kernel, SYNC_LOCK_PRE_INVOCATION);
            releaseSynchronizationLock(b, kernel, SYNC_LOCK_POST_INVOCATION);
        } else {
            releaseSynchronizationLock(b, kernel, SYNC_LOCK_FULL);
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeInitiallyTerminatedPartitionExit
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeInitiallyTerminatedPartitionExit(BuilderRef b) {

    b->SetInsertPoint(mKernelInitiallyTerminated);

    loadLastGoodVirtualBaseAddressesOfUnownedBuffersInPartition(b);

    if (mIsPartitionRoot) {

        const auto nextPartitionId = mCurrentPartitionId + 1;
        const auto jumpId = PartitionJumpTargetId[mCurrentPartitionId];
        assert (nextPartitionId <= jumpId);

        if (LLVM_LIKELY(nextPartitionId != jumpId)) {

            const auto targetKernelId = getFirstKernelInTargetPartition(nextPartitionId);

            acquirePartitionSynchronizationLock(b, targetKernelId);
            mKernelInitiallyTerminatedExit = b->GetInsertBlock();

            PHINode * const exhaustedInputPhi = mExhaustedPipelineInputAtPartitionEntry[nextPartitionId];
            exhaustedInputPhi->addIncoming(mExhaustedInput, mKernelInitiallyTerminatedExit);

            for (auto kernel = PipelineInput; kernel <= mKernelId; ++kernel) {
                phiOutPartitionItemCounts(b, kernel, nextPartitionId, true);
            }

            for (auto kernel = mKernelId + 1U; kernel <= PipelineOutput; ++kernel) {
                if (KernelPartitionId[kernel] == nextPartitionId) {
                    break;
                }
                phiOutPartitionItemCounts(b, kernel, nextPartitionId, true);
            }
            phiOutPartitionStatusFlags(b, nextPartitionId, true);

            releaseAllSynchronizationLocksUntil(b, targetKernelId);

            zeroAnySkippedTransitoryConsumedItemCountsUntil(b, targetKernelId);

            ensureAnyExternalProcessedAndProducedCountsAreUpdated(b, targetKernelId, true);

            updateCycleCounter(b, mKernelId, mKernelStartTime, CycleCounter::TOTAL_TIME);
            #ifdef ENABLE_PAPI
            accumPAPIMeasurementWithoutReset(b, PAPIReadInitialMeasurementArray, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
            #endif

            b->CreateBr(mNextPartitionEntryPoint);
            return;
        } else if (mKernelJumpToNextUsefulPartition) {
            mKernelInitiallyTerminatedExit = b->GetInsertBlock();
            assert (mKernelJumpToNextUsefulPartition != mKernelInitiallyTerminated);
            assert (mExhaustedInputAtJumpPhi);
            mExhaustedInputAtJumpPhi->addIncoming(mExhaustedInput, mKernelInitiallyTerminatedExit);
            for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
                const auto streamSet = target(e, mBufferGraph);
                const auto & br = mBufferGraph[e];
                Value * produced = nullptr;
                if (LLVM_UNLIKELY(br.IsDeferred)) {
                    produced = mInitiallyProducedDeferredItemCount[streamSet];
                } else {
                    produced = mInitiallyProducedItemCount[streamSet];
                }
                const auto port = br.Port;
                assert (isFromCurrentFunction(b, produced, false));
                mProducedAtJumpPhi[port]->addIncoming(produced, mKernelInitiallyTerminatedExit);
            }
            b->CreateBr(mKernelJumpToNextUsefulPartition);
            return;
        }
    }

    if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
        acquireSynchronizationLock(b, mKernelId, SYNC_LOCK_POST_INVOCATION);
        releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION);
    }

    mKernelInitiallyTerminatedExit = b->GetInsertBlock();
    updateKernelExitPhisAfterInitiallyTerminated(b);
    b->CreateBr(mKernelExit);

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeOnInitialTerminationJumpToNextPartitionToCheck
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeJumpToNextPartition(BuilderRef b) {

    b->SetInsertPoint(mKernelJumpToNextUsefulPartition);
    const auto jumpPartitionId = PartitionJumpTargetId[mCurrentPartitionId];
    assert (mCurrentPartitionId < jumpPartitionId);

    const auto targetKernelId = getFirstKernelInTargetPartition(jumpPartitionId);
    acquirePartitionSynchronizationLock(b, targetKernelId);

    PHINode * const exhaustedInputPhi = mExhaustedPipelineInputAtPartitionEntry[jumpPartitionId];
    if (exhaustedInputPhi) {
        assert (isFromCurrentFunction(b, exhaustedInputPhi, false));
        Value * const exhausted = mIsBounded ? mExhaustedInputAtJumpPhi : mExhaustedInput;
        assert (isFromCurrentFunction(b, exhausted, false));
        exhaustedInputPhi->addIncoming(exhausted, b->GetInsertBlock()); assert (exhausted);
    }
    for (auto kernel = PipelineInput; kernel <= mKernelId; ++kernel) {
        phiOutPartitionItemCounts(b, kernel, jumpPartitionId, false);
    }
    // NOTE: break condition differs from "writeInitiallyTerminatedPartitionExit"
    for (auto kernel = mKernelId + 1U; kernel <= PipelineOutput; ++kernel) {
        //const auto partId = KernelPartitionId[kernel];
        if (KernelPartitionId[kernel] == jumpPartitionId) {
            break;
        }
        phiOutPartitionItemCounts(b, kernel, jumpPartitionId, false);
    }
    phiOutPartitionStatusFlags(b, jumpPartitionId, false);

    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, "** " + makeKernelName(mKernelId) + ".jumping = %" PRIu64, mSegNo);
    #endif

    releaseAllSynchronizationLocksUntil(b, targetKernelId);

    zeroAnySkippedTransitoryConsumedItemCountsUntil(b, targetKernelId);

    ensureAnyExternalProcessedAndProducedCountsAreUpdated(b, targetKernelId, false);

    updateCycleCounter(b, mKernelId, mKernelStartTime, CycleCounter::TOTAL_TIME);
    #ifdef ENABLE_PAPI
    accumPAPIMeasurementWithoutReset(b, PAPIReadInitialMeasurementArray, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
    #endif

    b->CreateBr(mPartitionEntryPoint[jumpPartitionId]);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForPartitionExit
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::checkForPartitionExit(BuilderRef b) {

    assert (mKernelId >= FirstKernel && mKernelId <= LastKernel);
    // TODO: if any statefree kernel exists, swap counter accumulators to be thread local
    // and combine them at the end?
    updateCycleCounter(b, mKernelId, mKernelStartTime, CycleCounter::TOTAL_TIME);
    const auto type = isDataParallel(mKernelId) ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;
    releaseSynchronizationLock(b, mKernelId, type);

    #ifdef ENABLE_PAPI
    accumPAPIMeasurementWithoutReset(b, PAPIReadInitialMeasurementArray, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
    #endif


    const auto nextKernel = mKernelId + 1;
    if (LLVM_LIKELY(nextKernel < PipelineOutput)) {
        #ifdef ENABLE_PAPI
        readPAPIMeasurement(b, nextKernel, PAPIReadInitialMeasurementArray);
        #endif
        mKernelStartTime = startCycleCounter(b);
        const auto type = isDataParallel(nextKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        acquireSynchronizationLock(b, nextKernel, type);
        updateCycleCounter(b, nextKernel, mKernelStartTime, CycleCounter::KERNEL_SYNCHRONIZATION);
        #ifdef ENABLE_PAPI
        accumPAPIMeasurementWithoutReset(b, PAPIReadInitialMeasurementArray, nextKernel, PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION);
        #endif
    }

    const auto nextPartitionId = KernelPartitionId[nextKernel];
    assert (nextKernel != PipelineOutput || (nextPartitionId != mCurrentPartitionId));

    if (nextPartitionId != mCurrentPartitionId) {
        assert (mCurrentPartitionId < nextPartitionId);
        assert (nextPartitionId <= PartitionCount);
        BasicBlock * const exitBlock = b->GetInsertBlock();
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "  *** exiting partition %" PRIu64, b->getSize(mCurrentPartitionId));
        #endif
        b->CreateBr(mNextPartitionEntryPoint);

        b->SetInsertPoint(mNextPartitionEntryPoint);
        PHINode * const progressPhi = mPartitionPipelineProgressPhi[nextPartitionId];
        progressPhi->addIncoming(mPipelineProgress, exitBlock);
        mPipelineProgress = progressPhi;
        // Since there may be multiple paths into this kernel, phi out the start time
        // for each path.
        if (LLVM_UNLIKELY(EnableCycleCounter)) {
            mPartitionStartTimePhi[nextPartitionId]->addIncoming(mKernelStartTime, exitBlock);
            mKernelStartTime = mPartitionStartTimePhi[nextPartitionId];
        }

        PHINode * const exhaustedInputPhi = mExhaustedPipelineInputAtPartitionEntry[nextPartitionId];
        if (exhaustedInputPhi) {
            exhaustedInputPhi->addIncoming(mExhaustedInput, exitBlock);
            mExhaustedInput = exhaustedInputPhi;
        }

        const auto n = LastStreamSet - FirstStreamSet + 1U;

        for (unsigned i = 0; i != n; ++i) {
            PHINode * const phi = mPartitionProducedItemCountPhi[nextPartitionId][i];
            if (phi) {
                assert (isFromCurrentFunction(b, phi, false));
                const auto streamSet = FirstStreamSet + i;
                assert (isFromCurrentFunction(b, mLocallyAvailableItems[streamSet], false));
                phi->addIncoming(mLocallyAvailableItems[streamSet], exitBlock);
                mLocallyAvailableItems[streamSet] = phi;
            }
        }

        for (unsigned i = 0; i <= mCurrentPartitionId; ++i) {
            PHINode * const termPhi = mPartitionTerminationSignalPhi[nextPartitionId][i];
            if (termPhi) {
                assert (isFromCurrentFunction(b, termPhi, false));
                assert (isFromCurrentFunction(b, mPartitionTerminationSignal[i], false));
                termPhi->addIncoming(mPartitionTerminationSignal[i], exitBlock);
                mPartitionTerminationSignal[i] = termPhi;
            }
        }

    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief ensureAnyExternalProcessedAndProducedCountsAreUpdated
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::ensureAnyExternalProcessedAndProducedCountsAreUpdated(BuilderRef b, const unsigned targetKernelId, const bool fromKernelEntry) {
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const auto & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isExternal())) {
            const auto output = in_edge(streamSet, mBufferGraph);
            const auto producer = source(output, mBufferGraph);
            assert (producer >= PipelineInput && producer <= LastKernel);

            if (producer >= mKernelId && producer < targetKernelId) { // is output streamset

                for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                    if (target(e, mBufferGraph) == PipelineOutput) {
                        const BufferPort & outputPort = mBufferGraph[output];
                        assert (outputPort.Port.Type == PortType::Output);


                        const BufferPort & external = mBufferGraph[e];
                        Value * const ptr = getProducedOutputItemsPtr(external.Port.Number); assert (ptr);
                        Value * itemCount = nullptr;
                        if (producer == mKernelId) {
                            if (mMayLoopToEntry && fromKernelEntry) {
                                if (LLVM_UNLIKELY(outputPort.IsDeferred)) {
                                    itemCount = mAlreadyProducedDeferredPhi[outputPort.Port]; assert (itemCount);
                                } else {
                                    itemCount = mAlreadyProducedPhi[outputPort.Port]; assert (itemCount);
                                }
                            } else {
                                if (LLVM_UNLIKELY(outputPort.IsDeferred)) {
                                    itemCount = mInitiallyProducedDeferredItemCount[streamSet]; assert (itemCount);
                                } else {
                                    itemCount = mInitiallyProducedItemCount[streamSet]; assert (itemCount);
                                }
                            }
                        } else {
                            const auto prefix = makeBufferName(producer, outputPort.Port);
                            if (LLVM_UNLIKELY(outputPort.IsDeferred)) {
                                itemCount = b->getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
                            } else {
                                itemCount = b->getScalarField(prefix + ITEM_COUNT_SUFFIX);
                            }
                        }

                        b->CreateStore(itemCount, ptr);
                    }
                }
            } else if (LLVM_UNLIKELY(producer == PipelineInput)) { // is input streamset
                for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                    const auto consumer = target(e, mBufferGraph);
                    assert (consumer >= FirstKernel && consumer <= LastKernel);
                    if (consumer >= mKernelId && consumer < targetKernelId) {
                        const BufferPort & external = mBufferGraph[output];
                        Value * const ptr = getProcessedInputItemsPtr(external.Port.Number); assert (ptr);
                        const auto prefix = makeBufferName(PipelineInput, external.Port);
                        Value * const alreadyConsumed = b->getScalarField(prefix + CONSUMED_ITEM_COUNT_SUFFIX); assert (alreadyConsumed);
                        b->CreateStore(alreadyConsumed, ptr);
                        break;
                    }
                }
            }
        }
    }
}


}

#endif
