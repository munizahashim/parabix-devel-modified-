#ifndef PARTITION_PROCESSING_LOGIC_HPP
#define PARTITION_PROCESSING_LOGIC_HPP

#include "pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makePartitionEntryPoints
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::makePartitionEntryPoints(BuilderRef b) {    


    // the zeroth partition may be a fake one if this pipeline has I/O
    for (auto i : ActivePartitions) {
        mPartitionEntryPoint[i] = b->CreateBasicBlock("Partition" + std::to_string(i), mPipelineEnd);
    }    
    mPartitionEntryPoint[PartitionCount] = mPipelineEnd;

    const auto ip = b->saveIP();
    IntegerType * const boolTy = b->getInt1Ty();
    IntegerType * const sizeTy = b->getInt64Ty();
    const auto m = ActivePartitions.size();
    assert (m > 1);
    assert (ActivePartitions[m - 1] == PartitionCount - 1);

    for (unsigned k = 1; k < m; ++k) {
        const auto partId = ActivePartitions[k];
        b->SetInsertPoint(mPartitionEntryPoint[partId]);
        const auto prefix = std::to_string(partId);
        mPartitionPipelineProgressPhi[partId] = b->CreatePHI(boolTy, PartitionCount, prefix + ".pipelineProgress");
        mExhaustedPipelineInputAtPartitionEntry[partId] = b->CreatePHI(boolTy, PartitionCount, prefix + ".exhaustedInput");
        if (LLVM_UNLIKELY(EnableCycleCounter)) {
            mPartitionStartTimePhi[partId] = b->CreatePHI(sizeTy, PartitionCount, prefix + ".startTimeCycleCounter");
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

            if (LLVM_UNLIKELY(KernelOnHybridThread.test(producer) != mCompilingHybridThread)) {
                continue;
            }

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
                    if (PartitionOnHybridThread.test(partitionId) == mCompilingHybridThread) {
                        b->SetInsertPoint(mPartitionEntryPoint[partitionId]);
                        PHINode * const phi = b->CreatePHI(sizeTy, PartitionCount, prodPrefix + std::to_string(partitionId));
                        mPartitionProducedItemCountPhi[partitionId][k] = phi;
                    }
                }
            }

//            auto firstConsumerOrProducer = producer;
//            auto lastConsumer = lastReader;
//            if (LLVM_UNLIKELY(producer == PipelineInput)) {
//                // For the purpose of reporting the consumed item count, the pipeline output
//                // is always a "consumer" of the pipeline input.
//                firstConsumerOrProducer = PipelineOutput;
//                for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
//                    const auto consumer = target(input, mConsumerGraph);
//                    firstConsumerOrProducer = std::min(firstConsumerOrProducer, consumer);
//                }
//                lastConsumer = PipelineOutput;
//            } else {
//                for (const auto input : make_iterator_range(out_edges(streamSet, mConsumerGraph))) {
//                    const auto consumer = target(input, mConsumerGraph);
//                    lastConsumer = std::max(lastConsumer, consumer);
//                }
//            }

//            const auto prodPartId = KernelPartitionId[firstConsumerOrProducer];
//            const auto consPartId = KernelPartitionId[lastConsumer];

//            const auto consPrefix = prefix + "_consumed@partition";

//            for (auto partitionId = prodPartId + 1; partitionId <= consPartId; ++partitionId) {
//                if (PartitionOnHybridThread.test(partitionId) == mCompilingHybridThread) {
//                    b->SetInsertPoint(mPartitionEntryPoint[partitionId]);
//                    PHINode * const phi = b->CreatePHI(sizeTy, PartitionCount, consPrefix + std::to_string(partitionId));
//                    mPartitionConsumedItemCountPhi[partitionId][k] = phi;
//                }
//            }
        }
    }


    // any termination signal needs to be phi-ed out if it can be read by a descendent
    // or guards the loop condition at the end of the pipeline loop.

    assert (KernelPartitionId[PipelineInput] == 0);
    assert (KernelPartitionId[PipelineOutput] == PartitionCount - 1);

    BitVector toCheck(PartitionCount, 0);

    for (unsigned termId : ActivePartitions) {
        if (mTerminationCheck[termId]) {
            toCheck.set(termId);
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

        if (PartitionOnHybridThread.test(partitionId) == mCompilingHybridThread) {

            const auto firstKernel = kernel + 1U;
            assert (KernelPartitionId[firstKernel] == partitionId);
            assert (KernelPartitionId[lastKernel] == partitionId);
            for (auto k = firstKernel; k <= lastKernel; ++k) {
                for (const auto input : make_iterator_range(in_edges(k, mBufferGraph))) {
                    const auto streamSet = source(input, mBufferGraph);
                    const auto producer = parent(streamSet, mBufferGraph);
                    const auto prodPartId = KernelPartitionId[producer];
                    if (PartitionOnHybridThread.test(prodPartId) == mCompilingHybridThread) {
                        toCheck.set(prodPartId);
                    }
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

        }
        partitionId = KernelPartitionId[kernel];
    }

    initializePipelineInputConsumedPhiNodes(b);
    b->restoreIP(ip);


}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief branchToInitialPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::branchToInitialPartition(BuilderRef b) {
    ActivePartitionIndex = 0;
    const auto firstKernel = ActiveKernels[0];
    const auto firstPartition = KernelPartitionId[firstKernel];
    BasicBlock * const entry = mPartitionEntryPoint[firstPartition];
    b->CreateBr(entry);

    b->SetInsertPoint(entry);
    mCurrentPartitionId = -1U;
    setActiveKernel(b, firstKernel, true);
    #ifdef ENABLE_PAPI
    readPAPIMeasurement(b, firstKernel, PAPIReadInitialMeasurementArray);
    #endif
    mKernelStartTime = startCycleCounter(b);
    if (mNumOfThreads != 1 || mIsNestedPipeline) {
        const auto type = isDataParallel(firstKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        acquireSynchronizationLock(b, firstKernel, type);
        acquireHybridThreadSynchronizationLock(b);
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
    const auto nextPartitionId = ActivePartitions[ActivePartitionIndex + 1];
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
        assert (partitionId == ActivePartitions[ActivePartitionIndex]);
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
                } else {
                    produced = mProducedAtJumpPhi[br.Port];
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

        // If we have a nested kernel that can skip past all kernels that consume from an external input,
        // it won't update the value it needs to report to the external pipeline. Can the external
        // pipeline pass in a shared scalar and trust that the nested one will read/write it safely?

//        PHINode * const consPhi = mPartitionConsumedItemCountPhi[targetPartitionId][k];

//        if (consPhi) {



//            Value * const consumed = readConsumedItemCount(b, streamSet);
//            assert (isFromCurrentFunction(b, consumed, false));

//            #ifdef PRINT_DEBUG_MESSAGES
//            SmallVector<char, 256> tmp;
//            raw_svector_ostream out(tmp);
//            out << makeKernelName(mKernelId) << " -> " <<
//                   makeBufferName(kernel, br.Port) << "_consumed = %" PRIu64;
//            debugPrint(b, out.str(), consumed);
//            #endif

//            consPhi->addIncoming(consumed, exitPoint);
//        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief phiOutPartitionStatusFlags
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::phiOutPartitionStatusFlags(BuilderRef b, const unsigned targetPartitionId,
                                                  const bool /* fromKernelEntry */) {

    assert (PartitionOnHybridThread.test(targetPartitionId) == mCompilingHybridThread);

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

    unsigned firstKernelInTargetPartition = 0;
    assert (PartitionOnHybridThread.test(partitionId) == mCompilingHybridThread);
    const auto m = ActiveKernels.size();
    auto k = ActiveKernelIndex;
    for (; k < m; ++k) {
        const auto kernel = ActiveKernels[k];
        if (KernelPartitionId[kernel] == partitionId) {
            firstKernelInTargetPartition = kernel;
            assert (KernelOnHybridThread.test(kernel) == mCompilingHybridThread);
            break;
        }
        assert (KernelPartitionId[kernel] < partitionId);
    }
    assert (firstKernelInTargetPartition <= PipelineOutput);
    assert (firstKernelInTargetPartition > mKernelId);
    return firstKernelInTargetPartition;
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

//    auto lastConsumer = firstKernelInTargetPartition;
//    for (unsigned i = 0; i <= n; ++i) {
//        if (mPartitionConsumedItemCountPhi[partitionId][i]) {
//            for (const auto e : make_iterator_range(out_edges(FirstStreamSet + i, mConsumerGraph))) {
//                const auto consumer = target(e, mConsumerGraph);
//                if (mRequiresSynchronization.test(consumer)) {
//                   auto c = consumer;
//                   if (KernelOnHybridThread.test(consumer)) {
//                        c = std::max<unsigned>(consumer, firstAfterHybridThread);
//                   }
//                   lastConsumer = std::max<unsigned>(lastConsumer, c);
//                }
//            }
//        }
//    }

//    if (lastConsumer == PipelineOutput) {
//        lastConsumer = KernelOnHybridThread.find_last_unset_in(FirstKernel, PipelineOutput);
//        assert (lastConsumer < PipelineOutput);
//    }



    const auto targetLock = (firstKernelInTargetPartition == PipelineOutput) ? LastKernel : firstKernelInTargetPartition;
    assert (!KernelOnHybridThread.test(firstKernelInTargetPartition));
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
        if (KernelOnHybridThread.test(kernel) == mCompilingHybridThread) {
            if (isDataParallel(kernel)) {
                releaseSynchronizationLock(b, kernel, SYNC_LOCK_PRE_INVOCATION);
                releaseSynchronizationLock(b, kernel, SYNC_LOCK_POST_INVOCATION);
            } else {
                releaseSynchronizationLock(b, kernel, SYNC_LOCK_FULL);
            }
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

        const auto nextPartitionId = ActivePartitions[ActivePartitionIndex + 1U];

        assert (mCurrentPartitionId < nextPartitionId);
        assert (PartitionOnHybridThread.test(nextPartitionId) == mCompilingHybridThread);
        const auto jumpId = PartitionJumpTargetId[mCurrentPartitionId];

        assert (nextPartitionId <= jumpId);       
        assert (PartitionOnHybridThread.test(jumpId) == mCompilingHybridThread);

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
        } else {
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
        }

    } else { // if (!mIsPartitionRoot) {

        mKernelInitiallyTerminatedExit = b->GetInsertBlock();
        updateKernelExitPhisAfterInitiallyTerminated(b);
        b->CreateBr(mKernelExit);
    }

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeOnInitialTerminationJumpToNextPartitionToCheck
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeJumpToNextPartition(BuilderRef b) {

    b->SetInsertPoint(mKernelJumpToNextUsefulPartition);
    const auto jumpPartitionId = PartitionJumpTargetId[mCurrentPartitionId];
    assert (PartitionOnHybridThread.test(jumpPartitionId) == mCompilingHybridThread);
    assert(std::find(ActivePartitions.begin(), ActivePartitions.end(), jumpPartitionId) != ActivePartitions.end());
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


    const auto nextKernel = ActiveKernels[ActiveKernelIndex + 1U];
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
    assert (std::find(ActivePartitions.begin(), ActivePartitions.end(), nextPartitionId) != ActivePartitions.end());
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

//        for (unsigned i = 0; i != n; ++i) {
//            PHINode * const phi = mPartitionConsumedItemCountPhi[nextPartitionId][i];
//            if (phi) {
//                assert (isFromCurrentFunction(b, phi, false));
//                const auto streamSet = FirstStreamSet + i;
//                const ConsumerNode & cn = mConsumerGraph[streamSet];
//                assert (cn.Consumed);
//                assert (isFromCurrentFunction(b, cn.Consumed, false));
//                phi->addIncoming(cn.Consumed, exitBlock);
//                cn.Consumed = phi;
//                assert (cn.PhiNode == nullptr);
//                // The consumed phi node propagates the *initial* consumed item count for
//                // each item to reflect the fact we've skipped some kernel. However, since
//                // we might skip the actual producer, we need to constantly update the
//                // initial item count value to construct a legal program. Despite this
//                // the initial consumed item count should be considered as fixed value
//                // per pipeline iteration.
//                mInitialConsumedItemCount[streamSet] = phi;
//            }
//        }

        for (unsigned j = 0; j <= ActivePartitionIndex; ++j) {
            const auto i = ActivePartitions[j];
            PHINode * const termPhi = mPartitionTerminationSignalPhi[nextPartitionId][i];
            if (termPhi) {
                assert (isFromCurrentFunction(b, termPhi, false));
                assert (isFromCurrentFunction(b, mPartitionTerminationSignal[i], false));
                termPhi->addIncoming(mPartitionTerminationSignal[i], exitBlock);
                mPartitionTerminationSignal[i] = termPhi;
            }
        }

       ++ActivePartitionIndex;
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
