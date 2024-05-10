#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makePartitionEntryPoints
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::makePartitionEntryPoints(KernelBuilder & b) {

    for (unsigned i = 1; i < PartitionCount; ++i) {
        mPartitionEntryPoint[i] = b.CreateBasicBlock("Partition" + std::to_string(i), mPipelineEnd);
    }

//    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
//    mPartitionEntryPoint[PartitionCount] = b.CreateBasicBlock("PipelineLoopCond", mPipelineEnd);
//    #else
//    mPartitionEntryPoint[PartitionCount] = mPipelineEnd;
//    #endif

    IntegerType * const boolTy = b.getInt1Ty();
    IntegerType * const sizeTy = b.getInt64Ty();

    for (unsigned i = 2; i < PartitionCount; ++i) {
        mPartitionPipelineProgressPhi[i] =
            PHINode::Create(boolTy, PartitionCount, std::to_string(i) + ".pipelineProgress", mPartitionEntryPoint[i]);
    }

    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        for (unsigned i = 2; i < (PartitionCount - 1); ++i) {
            mPartitionStartTimePhi[i] =
                PHINode::Create(sizeTy, PartitionCount, std::to_string(i) + ".startTimeCycleCounter", mPartitionEntryPoint[i]);
        }
    }

    // Create any PHI nodes we need to propogate the current produced/consumed item counts
    // of the kernels we jump over as well as the termination signals for any kernel we may
    // need to check if its closed or not.

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];

        if (LLVM_UNLIKELY(bn.isConstant())) {
            continue;
        }

        const auto output = in_edge(streamSet, mBufferGraph);
        const auto producer = source(output, mBufferGraph);
        if (LLVM_UNLIKELY(producer == PipelineInput)) {
            continue;
        }

        // TODO: make new buffer type to automatically state this is a cross partition
        // thread local buffer.

        if (bn.isThreadLocal()) {
            const auto prodPartId = KernelPartitionId[producer];
            bool noCrossPartitionConsumer = true;
            for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const auto consPartId = KernelPartitionId[target(e, mBufferGraph)];
                if (prodPartId != consPartId) {
                    noCrossPartitionConsumer = false;
                    break;
                }
            }
            if (noCrossPartitionConsumer) {
                continue;
            }
        }

        const BufferPort & outputPort = mBufferGraph[output];
        const auto prefix = makeBufferName(producer, outputPort.Port);

        const auto k = streamSet - FirstStreamSet;

        auto lastReader = producer;
        for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const auto consumer = target(input, mBufferGraph);
            // TODO: does the graph need a connection to pipeline output?
            if (consumer < PipelineOutput) {
                lastReader = std::max(lastReader, consumer);
            }
        }
        const auto readsPartId = KernelPartitionId[lastReader];
        assert (readsPartId != KernelPartitionId[PipelineOutput]);
        const auto prodPrefix = prefix + "_produced@partition";
        const auto prodPartId = KernelPartitionId[producer];
        for (auto partitionId = prodPartId + 1; partitionId <= readsPartId; ++partitionId) {
            auto entryPoint = mPartitionEntryPoint[partitionId];
            PHINode * const phi = PHINode::Create(sizeTy, PartitionCount, prodPrefix + std::to_string(partitionId), entryPoint);
            mPartitionProducedItemCountPhi[partitionId][k] = phi;
        }
    }

    // any termination signal needs to be phi-ed out if it can be read by a descendent
    // or guards the loop condition at the end of the pipeline loop.

    assert (KernelPartitionId[PipelineInput] == 0);
    assert (KernelPartitionId[PipelineOutput] == PartitionCount - 1);

    const auto firstPartition = KernelPartitionId[FirstKernel];

    BitVector toCheck(LastKernel + 1);

    for (auto partitionId = firstPartition; partitionId < PartitionCount; ++partitionId) {
        auto entryPoint = mPartitionEntryPoint[partitionId];
        const auto prefix = "terminationSignalForPartition" + std::to_string(partitionId) + "@";
        const auto l = FirstKernelInPartition[partitionId];
        toCheck.reset();
        for (auto p = firstPartition; p < partitionId; ++p) {
            if (mTerminationCheck[p]) {
                toCheck.set(FirstKernelInPartition[p]);
            }
        }

        for (auto k = FirstKernel; k < l; ) {
            for (const auto e : make_iterator_range(out_edges(k, mBufferGraph))) {
                const auto streamSet = target(e, mBufferGraph);
                for (const auto f : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                    const auto consumer = target(f, mBufferGraph);
                    const auto conPartId = KernelPartitionId[consumer];
                    if (conPartId >= partitionId) {
                        toCheck.set(getTerminationSignalIndex(k));
                        goto added_to_set;
                    }
                }
            }
added_to_set: ++k;
        }

        for (auto k = FirstKernel; k < l; ++k) {
            if (toCheck.test(k)) {
                PHINode * const phi = PHINode::Create(sizeTy, 2, prefix + std::to_string(k), entryPoint);
                mPartitionTerminationSignalPhi[partitionId][k - FirstKernel] = phi;
            }
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief branchToInitialPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::branchToInitialPartition(KernelBuilder & b) {

    const auto firstPartition = KernelPartitionId[FirstKernel];
    BasicBlock * const entry = mPartitionEntryPoint[firstPartition];
    b.CreateBr(entry);

    b.SetInsertPoint(entry);
    mCurrentPartitionId = -1U;
    setActiveKernel(b, FirstKernel, true);
    #ifdef ENABLE_PAPI
    startPAPIMeasurement(b, {PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION, PAPIKernelCounter::PAPI_KERNEL_TOTAL});
    #endif
    startCycleCounter(b, {CycleCounter::KERNEL_SYNCHRONIZATION, CycleCounter::TOTAL_TIME});
    if (isMultithreaded()) {
        const auto type = isDataParallel(FirstKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        acquireSynchronizationLock(b, FirstKernel, type, mSegNo);
        updateCycleCounter(b, FirstKernel, CycleCounter::KERNEL_SYNCHRONIZATION);
        #ifdef ENABLE_PAPI
        accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION);
        #endif
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getPartitionExitPoint
 ** ------------------------------------------------------------------------------------------------------------- */
BasicBlock * PipelineCompiler::getPartitionExitPoint(KernelBuilder & /* b */) {
    assert (mKernelId >= FirstKernel && mKernelId <= PipelineOutput);
    return mPartitionEntryPoint[mCurrentPartitionId + 1];
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForPartitionEntry
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkForPartitionEntry(KernelBuilder & b) {
    assert (mKernelId >= FirstKernel && mKernelId <= LastKernel);
    mIsPartitionRoot = false;
    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    mUsingNewSynchronizationVariable = false;
    #endif
    const auto partitionId = KernelPartitionId[mKernelId];
    if (partitionId != mCurrentPartitionId) {
        mIsPartitionRoot = true;
        assert (FirstKernelInPartition[partitionId] == mKernelId);
        mCurrentPartitionId = partitionId;
        mCurrentPartitionRoot = mKernelId;
        mNextPartitionEntryPoint = getPartitionExitPoint(b);
        determinePartitionStrideRateScalingFactor();
        #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
        mUsingNewSynchronizationVariable =
            (PartitionJumpTargetId[partitionId] == (PartitionCount - 1)) && (partitionId != (PartitionCount - 2));
        if (LLVM_UNLIKELY(mUsingNewSynchronizationVariable)) {
            ++mCurrentNestedSynchronizationVariable;
            mPartitionExitSegNoPhi = PHINode::Create(b.getSizeTy(), 2, "regionedSegNo", mNextPartitionEntryPoint);
        }
        #endif
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "  *** entering partition %" PRIu64, b.getSize(mCurrentPartitionId));
        #endif
    }
    assert (KernelPartitionId[mKernelId - 1U] <= mCurrentPartitionId);
    assert ((KernelPartitionId[mKernelId - 1U] + 1U) >= mCurrentPartitionId);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief determinePartitionStrideRateScalingFactor
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::determinePartitionStrideRateScalingFactor() {
    auto l = StrideStepLength[mCurrentPartitionRoot];
    auto g = StrideStepLength[mCurrentPartitionRoot];
    const auto firstKernelInNextPartition = FirstKernelInPartition[mCurrentPartitionId + 1];
    for (auto i = mCurrentPartitionRoot + 1U; i < firstKernelInNextPartition; ++i) {
        assert (KernelPartitionId[i] == mCurrentPartitionId);
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
void PipelineCompiler::loadLastGoodVirtualBaseAddressesOfUnownedBuffersInPartition(KernelBuilder & b) const {
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
void PipelineCompiler::phiOutPartitionItemCounts(KernelBuilder & b, const unsigned kernel,
                                                 const unsigned targetPartitionId,
                                                 const bool fromKernelEntryBlock) {

    BasicBlock * const exitPoint = b.GetInsertBlock();

    for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);

        // When jumping out of a partition to some subsequent one, we may have to
        // phi-out some of the produced item counts. We 3 cases to consider:

        // (1) if we've executed the kernel, we use the fully produced item count.
        // (2) if producer is the current kernel, we use the already produced phi node.
        // (3) if we have yet to execute (and will be jumping over) the kernel, load
        // the prior produced count.


        const auto k = streamSet - FirstStreamSet;

        const BufferPort & br = mBufferGraph[e];
        // Select/load the appropriate produced item count
        PHINode * const prodPhi = mPartitionProducedItemCountPhi[targetPartitionId][k];
        if (prodPhi) {
            Value * produced = nullptr;
            if (kernel < mKernelId) {
                produced = mLocallyAvailableItems[streamSet];
            } else if (kernel == mKernelId && !mAllowDataParallelExecution) {
                assert (!mCurrentKernelIsStateFree);
                if (fromKernelEntryBlock) {
                    if (LLVM_UNLIKELY(br.isDeferred())) {
                        produced = mInitiallyProducedDeferredItemCount[streamSet];
                    } else {
                        produced = mInitiallyProducedItemCount[streamSet];
                    }
                } else if (mProducedAtJumpPhi[br.Port]) {
                    produced = mProducedAtJumpPhi[br.Port];
                } else {
                    if (LLVM_UNLIKELY(br.isDeferred())) {
                        produced = mAlreadyProducedDeferredPhi[br.Port];
                    } else {
                        produced = mAlreadyProducedPhi[br.Port];
                    }
                }
            } else { // if (kernel > mKernelId) {
                const auto prefix = makeBufferName(kernel, br.Port);
                Value * ptr = nullptr;
                if (LLVM_UNLIKELY(br.isDeferred() && !fromKernelEntryBlock)) {
                    ptr = b.getScalarFieldPtr(prefix + DEFERRED_ITEM_COUNT_SUFFIX).first;
                } else {
                    ptr = b.getScalarFieldPtr(prefix + ITEM_COUNT_SUFFIX).first;
                }
                produced = b.CreateLoad(b.getSizeTy(), ptr);
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
void PipelineCompiler::phiOutPartitionStatusFlags(KernelBuilder & b, const unsigned targetPartitionId, const bool fromKernelEntryBlock) {

    BasicBlock * const exitPoint = b.GetInsertBlock();

    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);

    const auto firstKernelOfTargetPartition = FirstKernelInPartition[targetPartitionId];
    assert (firstKernelOfTargetPartition <= PipelineOutput);

    for (auto kernel = FirstKernel; kernel < firstKernelOfTargetPartition; ++kernel) {
        PHINode * const termPhi = mPartitionTerminationSignalPhi[targetPartitionId][kernel - FirstKernel];
        if (termPhi) {
            Value * const term = mKernelTerminationSignal[kernel];
            assert (isFromCurrentFunction(b, term));
            termPhi->addIncoming(term ? term : unterminated, exitPoint);
        }
    }

    PHINode * const progressPhi = mPartitionPipelineProgressPhi[targetPartitionId];
    assert (progressPhi);
    assert (isFromCurrentFunction(b, progressPhi, false));
    Value * const progress = mPipelineProgress;
    assert (isFromCurrentFunction(b, progress, false));
    progressPhi->addIncoming(progress, exitPoint);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief phiOutAllStateAndReleaseSynchronizationLocks
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::phiOutPartitionStateAndReleaseSynchronizationLocks(KernelBuilder & b,
    const unsigned targetKernelId, const unsigned targetPartitionId, const bool fromKernelEntryBlock, Value * const afterFirstSegNo) {

    assert (KernelPartitionId[FirstKernel] == 1);

    for (auto kernel = PipelineInput; kernel < mKernelId; ++kernel) {
        phiOutPartitionItemCounts(b, kernel, targetPartitionId, fromKernelEntryBlock);
    }

    phiOutPartitionItemCounts(b, mKernelId, targetPartitionId, fromKernelEntryBlock);

//    flat_set<size_t> consumed;
//    for (auto kernel = targetKernelId; kernel <= PipelineOutput; ++kernel) {
//        for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
//            const auto streamSet = getTruncatedStreamSetSourceId(source(input, mBufferGraph));
//            const BufferNode & bn = mBufferGraph[streamSet];
//            if (bn.isNonThreadLocal() && bn.isOwned() && bn.isInternal() && !bn.hasZeroElementsOrWidth()) {
//                assert (!bn.isTruncated());
//                consumed.insert(streamSet);
//            }
//        }
//    }

//    for (auto kernel = mKernelId; kernel < targetKernelId; ++kernel) {
//        for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
//            const auto streamSet = getTruncatedStreamSetSourceId(target(output, mBufferGraph));
//            if (consumed.count(streamSet)) {
//                Value * consumed = readConsumedItemCount(b, streamSet);
//                const BufferNode & bn = mBufferGraph[streamSet];
//                const StreamSetBuffer * const buffer = bn.Buffer;
//                assert ("buffer cannot be null!" && buffer);
//                assert (isFromCurrentFunction(b, buffer->getHandle()));
//                Value * const baseAddress = buffer->getBaseAddress(b);
//                Value * const vba = buffer->getVirtualBasePtr(b, baseAddress, consumed);
//                const BufferPort & rt = mBufferGraph[output];
//                const auto handleName = makeBufferName(kernel, rt.Port);
//                b.setScalarField(handleName + LAST_GOOD_VIRTUAL_BASE_ADDRESS, vba);
//            }
//        }
//    }


    releaseAllSynchronizationLocksFor(b, mKernelId);

    assert (afterFirstSegNo);

    Value * const curSegNo = mSegNo;
    mSegNo = afterFirstSegNo;




    for (auto kernel = mKernelId + 1; kernel < targetKernelId; ++kernel) {
        phiOutPartitionItemCounts(b, kernel, targetPartitionId, fromKernelEntryBlock);
        if (HasTerminationSignal.test(kernel)) {
            mKernelTerminationSignal[kernel] = readTerminationSignal(b, kernel);
        }



        releaseAllSynchronizationLocksFor(b, kernel);
    }

    mSegNo = curSegNo;

    phiOutPartitionStatusFlags(b, targetPartitionId, fromKernelEntryBlock);
    for (auto kernel = mKernelId + 1; kernel < targetKernelId; ++kernel) {
        mKernelTerminationSignal[kernel] = nullptr;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief acquirePartitionSynchronizationLock
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::acquirePartitionSynchronizationLock(KernelBuilder & b, const unsigned firstKernelInTargetPartition, Value * const segNo) {

    #ifdef ENABLE_PAPI
    startPAPIMeasurement(b, {PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION, PAPIKernelCounter::PAPI_PARTITION_JUMP_SYNCHRONIZATION});
    #endif
    startCycleCounter(b, {CycleCounter::KERNEL_SYNCHRONIZATION, CycleCounter::PARTITION_JUMP_SYNCHRONIZATION});
    assert (firstKernelInTargetPartition <= PipelineOutput);

    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    assert (firstKernelInTargetPartition != PipelineOutput);
    #endif

    if (firstKernelInTargetPartition == PipelineOutput) {
        const auto type = isDataParallel(LastKernel) ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;
        acquireSynchronizationLock(b, LastKernel, type, segNo);
    } else {
        const auto type = isDataParallel(firstKernelInTargetPartition) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        acquireSynchronizationLock(b, firstKernelInTargetPartition, type, segNo);
    }

    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        const auto partId = KernelPartitionId[firstKernelInTargetPartition];
        if (partId < (PartitionCount - 1)) {
            updateCycleCounter(b, mKernelId, CycleCounter::PARTITION_JUMP_SYNCHRONIZATION);
            BasicBlock * const exitBlock = b.GetInsertBlock();
            Value * const startTime = mCycleCounters[CycleCounter::PARTITION_JUMP_SYNCHRONIZATION];
            mPartitionStartTimePhi[partId]->addIncoming(startTime, exitBlock);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief releaseAllSynchronizationLocksUntil
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::releaseAllSynchronizationLocksFor(KernelBuilder & b, const unsigned kernel) {

    if (KernelPartitionId[kernel - 1] != KernelPartitionId[kernel]) {
        recordStridesPerSegment(b, kernel, b.getSize(0));
    }
    if (isDataParallel(kernel)) {
        releaseSynchronizationLock(b, kernel, SYNC_LOCK_PRE_INVOCATION, mSegNo);
        releaseSynchronizationLock(b, kernel, SYNC_LOCK_POST_INVOCATION, mSegNo);
    } else {
        releaseSynchronizationLock(b, kernel, SYNC_LOCK_FULL, mSegNo);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeInitiallyTerminatedPartitionExit
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeInitiallyTerminatedPartitionExit(KernelBuilder & b) {

    b.SetInsertPoint(mKernelInitiallyTerminated);

    loadLastGoodVirtualBaseAddressesOfUnownedBuffersInPartition(b);

    // NOTE: this branches to the next partition regardless of the jump target destination.

    const auto nextPartitionId = mCurrentPartitionId + 1;
    const auto jumpTargetId = PartitionJumpTargetId[mCurrentPartitionId];
    assert (nextPartitionId <= jumpTargetId);

    if (LLVM_LIKELY((nextPartitionId != jumpTargetId) && mIsPartitionRoot)) {

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "** " + makeKernelName(mKernelId) + ".initiallyTerminated (seqexit) = %" PRIu64, mSegNo);
        #endif

        const auto targetKernelId = FirstKernelInPartition[nextPartitionId];

        Value * nextSegNo = mSegNo;
        #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
        if (mPartitionExitSegNoPhi) {
            nextSegNo = obtainNextSegmentNumber(b);
        }
        #endif

        acquirePartitionSynchronizationLock(b, targetKernelId, nextSegNo);
        mKernelInitiallyTerminatedExit = b.GetInsertBlock();

        phiOutPartitionStateAndReleaseSynchronizationLocks(b, targetKernelId, nextPartitionId, true, nextSegNo);

        zeroAnySkippedTransitoryConsumedItemCountsUntil(b, targetKernelId);

        ensureAnyExternalProcessedAndProducedCountsAreUpdated(b, targetKernelId, true);

        updateCycleCounter(b, mKernelId, CycleCounter::TOTAL_TIME);
        #ifdef ENABLE_PAPI
        accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
        #endif

        #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
        if (mPartitionExitSegNoPhi) {
            assert (nextSegNo);
            mPartitionExitSegNoPhi->addIncoming(nextSegNo, mKernelInitiallyTerminatedExit);
        }
        #endif

        b.CreateBr(mNextPartitionEntryPoint);

    } else if (mKernelJumpToNextUsefulPartition) {

        assert (mIsPartitionRoot);

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "** " + makeKernelName(mKernelId) + ".initiallyTerminated (reusejump) = %" PRIu64, mSegNo);
        #endif

        mKernelInitiallyTerminatedExit = b.GetInsertBlock();
        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const auto streamSet = target(e, mBufferGraph);
            const auto & br = mBufferGraph[e];
            Value * produced = nullptr;
            if (LLVM_UNLIKELY(br.isDeferred())) {
                produced = mInitiallyProducedDeferredItemCount[streamSet];
            } else {
                produced = mInitiallyProducedItemCount[streamSet];
            }
            const auto port = br.Port;
            assert (isFromCurrentFunction(b, produced, false));
            mProducedAtJumpPhi[port]->addIncoming(produced, mKernelInitiallyTerminatedExit);
        }

        mMaximumNumOfStridesAtJumpPhi->addIncoming(b.getSize(0), mKernelInitiallyTerminatedExit);

        b.CreateBr(mKernelJumpToNextUsefulPartition);
    } else {

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "** " + makeKernelName(mKernelId) + ".initiallyTerminated (exitdirect) = %" PRIu64, mSegNo);
        #endif

        if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION, mSegNo);
            acquireSynchronizationLock(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
        }
        mKernelInitiallyTerminatedExit = b.GetInsertBlock();
        updateKernelExitPhisAfterInitiallyTerminated(b);
        b.CreateBr(mKernelExit);
    }


}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeOnInitialTerminationJumpToNextPartitionToCheck
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeJumpToNextPartition(KernelBuilder & b) {

    assert (mIsPartitionRoot);

    b.SetInsertPoint(mKernelJumpToNextUsefulPartition);
    const auto jumpPartitionId = PartitionJumpTargetId[mCurrentPartitionId];
    assert (mCurrentPartitionId < jumpPartitionId);
    const auto targetKernelId = FirstKernelInPartition[jumpPartitionId];
    assert (targetKernelId > (mKernelId + 1U));


    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, "** " + makeKernelName(mKernelId) + ".jumping = %" PRIu64, mSegNo);
    #endif

    updateNextSlidingWindowSize(b, mMaximumNumOfStridesAtJumpPhi, b.getSize(0));

    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    if (LLVM_LIKELY(targetKernelId != PipelineOutput)) {
    #endif
        acquirePartitionSynchronizationLock(b, targetKernelId, mSegNo);
        phiOutPartitionStateAndReleaseSynchronizationLocks(b, targetKernelId, jumpPartitionId, false, mSegNo);
        zeroAnySkippedTransitoryConsumedItemCountsUntil(b, targetKernelId);
    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    } else {
        if (LLVM_UNLIKELY(isDataParallel(mKernelId))) {
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION, mSegNo);
            acquireSynchronizationLock(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_POST_INVOCATION, mSegNo);
        } else {
            releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_FULL, mSegNo);
        }
        phiOutPartitionStatusFlags(b, jumpPartitionId, false);
    }
    #endif

    ensureAnyExternalProcessedAndProducedCountsAreUpdated(b, targetKernelId, false);

    updateCycleCounter(b, mKernelId, CycleCounter::TOTAL_TIME);
    #ifdef ENABLE_PAPI
    accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
    #endif

    b.CreateBr(mPartitionEntryPoint[jumpPartitionId]);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkForPartitionExit
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkForPartitionExit(KernelBuilder & b) {

    assert (mKernelId >= FirstKernel && mKernelId <= LastKernel);

    Value * nextSegNo = mSegNo;
    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    if (LLVM_UNLIKELY(mUsingNewSynchronizationVariable)) {
        assert (mIsPartitionRoot);
        nextSegNo = obtainNextSegmentNumber(b);
    }
    #endif

    // TODO: if any statefree kernel exists, swap counter accumulators to be thread local
    // and combine them at the end?
    updateCycleCounter(b, mKernelId, CycleCounter::TOTAL_TIME);
    const auto type = isDataParallel(mKernelId) ? SYNC_LOCK_POST_INVOCATION : SYNC_LOCK_FULL;
    releaseSynchronizationLock(b, mKernelId, type, mSegNo);

    #ifdef ENABLE_PAPI
    accumPAPIMeasurementWithoutReset(b, mKernelId, PAPIKernelCounter::PAPI_KERNEL_TOTAL);
    #endif

    const auto nextKernel = mKernelId + 1;
    if (LLVM_LIKELY(nextKernel < PipelineOutput)) {
        #ifdef ENABLE_PAPI
        startPAPIMeasurement(b, {PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION, PAPIKernelCounter::PAPI_KERNEL_TOTAL});
        #endif
        startCycleCounter(b, {CycleCounter::KERNEL_SYNCHRONIZATION, CycleCounter::TOTAL_TIME});
        const auto type = isDataParallel(nextKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        acquireSynchronizationLock(b, nextKernel, type, nextSegNo);
        updateCycleCounter(b, nextKernel, CycleCounter::KERNEL_SYNCHRONIZATION);
        #ifdef ENABLE_PAPI
        accumPAPIMeasurementWithoutReset(b, nextKernel, PAPIKernelCounter::PAPI_KERNEL_SYNCHRONIZATION);
        #endif
    }

    const auto nextPartitionId = KernelPartitionId[nextKernel];
    assert (nextKernel != PipelineOutput || (nextPartitionId != mCurrentPartitionId));

    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    mSegNo = nextSegNo;
    #endif

    if (nextPartitionId != mCurrentPartitionId) {
        assert (mCurrentPartitionId < nextPartitionId);
        assert (nextPartitionId <= PartitionCount);
        BasicBlock * const exitBlock = b.GetInsertBlock();
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "  *** exiting partition %" PRIu64, b.getSize(mCurrentPartitionId));
        #endif
        b.CreateBr(mNextPartitionEntryPoint);

        b.SetInsertPoint(mNextPartitionEntryPoint);
        PHINode * const progressPhi = mPartitionPipelineProgressPhi[nextPartitionId];
        progressPhi->addIncoming(mPipelineProgress, exitBlock);
        mPipelineProgress = progressPhi;

        #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
        if (LLVM_UNLIKELY(mPartitionExitSegNoPhi)) {
            mPartitionExitSegNoPhi->addIncoming(nextSegNo, exitBlock);
            mSegNo = mPartitionExitSegNoPhi;
            mPartitionExitSegNoPhi = nullptr;
        }
        #endif
        // Since there may be multiple paths into this kernel, phi out the start time
        // for each path.
        if (LLVM_UNLIKELY(EnableCycleCounter && nextPartitionId < (PartitionCount - 1))) {
            mPartitionStartTimePhi[nextPartitionId]->addIncoming(mCycleCounters[TOTAL_TIME], exitBlock);
            mCycleCounters[TOTAL_TIME] = mPartitionStartTimePhi[nextPartitionId];
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

        const auto firstKernelOfNextPartition = FirstKernelInPartition[nextPartitionId];
        assert (firstKernelOfNextPartition <= PipelineOutput);

        for (auto kernel = FirstKernel; kernel < firstKernelOfNextPartition; ++kernel) {
            PHINode * const termPhi = mPartitionTerminationSignalPhi[nextPartitionId][kernel - FirstKernel];
            if (termPhi) {
                assert (isFromCurrentFunction(b, termPhi, false));
                assert (isFromCurrentFunction(b, mKernelTerminationSignal[kernel], false));
                termPhi->addIncoming(mKernelTerminationSignal[kernel], exitBlock);
                mKernelTerminationSignal[kernel] = termPhi;
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief ensureAnyExternalProcessedAndProducedCountsAreUpdated
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::ensureAnyExternalProcessedAndProducedCountsAreUpdated(KernelBuilder & b, const unsigned targetKernelId, const bool fromKernelEntry) {
#if 0
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
                                if (LLVM_UNLIKELY(outputPort.isDeferred())) {
                                    itemCount = mAlreadyProducedDeferredPhi[outputPort.Port]; assert (itemCount);
                                } else {
                                    itemCount = mAlreadyProducedPhi[outputPort.Port]; assert (itemCount);
                                }
                            } else {
                                if (LLVM_UNLIKELY(outputPort.isDeferred())) {
                                    itemCount = mInitiallyProducedDeferredItemCount[streamSet]; assert (itemCount);
                                } else {
                                    itemCount = mInitiallyProducedItemCount[streamSet]; assert (itemCount);
                                }
                            }
                        } else {
                            const auto prefix = makeBufferName(producer, outputPort.Port);
                            if (LLVM_UNLIKELY(outputPort.isDeferred())) {
                                itemCount = b.getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
                            } else {
                                itemCount = b.getScalarField(prefix + ITEM_COUNT_SUFFIX);
                            }
                        }

                        b.CreateStore(itemCount, ptr);
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
                        Value * alreadyConsumedPtr = b.getScalarFieldPtr(prefix + CONSUMED_ITEM_COUNT_SUFFIX);
                        if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
                            FixedArray<Value *, 1> indices;
                            indices[0] = b.getInt32(0);
                            alreadyConsumedPtr = b.CreateGEP0(ptr, indices);
                        }
                        Value * const alreadyConsumed = b.CreateLoad(alreadyConsumedPtr);
                        assert (ptr->getType()->getPointerElementType() == alreadyConsumed->getType());
                        b.CreateStore(alreadyConsumed, ptr);
                        break;
                    }
                }
            }
        }
    }
#endif
}

} // end of namespace kernel
