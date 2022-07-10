#ifndef CONSUMER_LOGIC_HPP
#define CONSUMER_LOGIC_HPP

#include "pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addConsumerKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::addConsumerKernelProperties(BuilderRef b, const unsigned kernelId) {
    IntegerType * const sizeTy = b->getSizeTy();

//    const auto addInternallySynchronizedInternalCounters = mIsInternallySynchronized.test(kernelId) && !mIsStatelessKernel.test(kernelId) ;

    const auto groupId = getCacheLineGroupId(kernelId);

    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        // If the out-degree for this buffer is zero, then we've proven that its consumption rate
        // is identical to its production rate.
        const auto numOfIndependentConsumers = out_degree(streamSet, mConsumerGraph);
        if (LLVM_UNLIKELY(numOfIndependentConsumers != 0)) {
            const BufferPort & rd = mBufferGraph[e];
            assert (rd.Port.Type == PortType::Output);
            const auto prefix = makeBufferName(kernelId, rd.Port);


            if (out_degree(streamSet, mConsumerGraph) > 1) {
                // Although we can PHI out the thread's current min. consumed summary count for each
                // buffer, in any complex program, we'll inevitably have general register spill/reloads.
                // By keeping these as stack-allocated variables, the LLVM compiler will hopefully be
                // able to make better decisions whether it should PHI-out these variables.
                mTarget->addNonPersistentScalar(sizeTy, TRANSITORY_CONSUMED_ITEM_COUNT_PREFIX + std::to_string(streamSet));
            }

//            if (kernelId != PipelineInput || mTraceIndividualConsumedItemCounts) {
                // If we're tracing the consumer item counts, we need to store one for each
                // (non-nested) consumer. Any nested consumers will have their own trace.
                Type * countTy = sizeTy;
                if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
                    countTy = ArrayType::get(sizeTy, numOfIndependentConsumers + 1);
                }

                const auto name = prefix + CONSUMED_ITEM_COUNT_SUFFIX;
                mTarget->addInternalScalar(countTy, name, groupId);

//                if (LLVM_LIKELY(bn.isOwned() || bn.isInternal() || bn.CrossesHybridThreadBarrier || mTraceIndividualConsumedItemCounts)) {
//                    mTarget->addInternalScalar(countTy, name, groupId);
//                } else {
//                    mTarget->addNonPersistentScalar(countTy, name);
//                }

//                if (LLVM_UNLIKELY(addInternallySynchronizedInternalCounters)) {
//                    mTarget->addInternalScalar(countTy, prefix + INTERNALLY_SYNCHRONIZED_INTERNAL_ITEM_COUNT_SUFFIX, groupId);
//                }

                if (hasUsersOnFixedAndHybridThread(streamSet)) {
                    assert (mNumOfThreads > 1);
                    const auto altId = getCacheLineGroupId(PipelineOutput);
                    mTarget->addInternalScalar(countTy, prefix + HYBRID_THREAD_CONSUMED_ITEM_COUNT_SUFFIX, altId);
                }
//            }

        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasUsersOnFixedAndHybridThread
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::hasUsersOnFixedAndHybridThread(const size_t streamSet) const {
    const BufferNode & node = mBufferGraph[streamSet];
    if (LLVM_UNLIKELY(node.CrossesHybridThreadBarrier)) {
        std::bitset<2> check(0);
        for (const auto input : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const auto kernel = target(input, mBufferGraph);
            check.set(KernelOnHybridThread.test(kernel) ? 1 : 0);
        }
        return check.all();
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readConsumedItemCounts(BuilderRef b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mConsumerGraph))) {
        const auto streamSet = target(e, mConsumerGraph);
        Value * consumed = readConsumedItemCount(b, streamSet);
//        mInitialConsumedItemCount[streamSet] = consumed; assert (consumed);
        #ifdef PRINT_DEBUG_MESSAGES
        const ConsumerEdge & c = mConsumerGraph[e];
        const StreamSetPort port{PortType::Output, c.Port};
        const auto prefix = makeBufferName(mKernelId, port);
        debugPrint(b, prefix + "_consumed = %" PRIu64, consumed);
        #endif
        if (LLVM_UNLIKELY(CheckAssertions)) {
            Value * const produced = mInitiallyProducedItemCount[streamSet];
            Value * valid = b->CreateICmpULE(consumed, produced);
            if (mInitiallyTerminated) {
                valid = b->CreateOr(valid, mInitiallyTerminated);
            }
            constexpr auto msg =
                "Consumed item count (%" PRId64 ") of %s.%s exceeds its produced item count (%" PRId64 ").";
            const ConsumerEdge & c = mConsumerGraph[e];
            const StreamSetPort port{PortType::Output, c.Port};
            Constant * const bindingName = b->GetString(getBinding(mKernelId, port).getName());
            b->CreateAssert(valid, msg,
                consumed, mCurrentKernelName, bindingName, produced);
        }
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readExternalConsumerItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::readExternalConsumerItemCounts(BuilderRef b) {
    for (const auto e : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        const auto streamSet = source(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_LIKELY(bn.isOwned())) {
            const BufferPort & externalPort = mBufferGraph[e];
            const auto numOfIndependentConsumers = out_degree(streamSet, mConsumerGraph);
            const auto producer = parent(streamSet, mBufferGraph);
            if (LLVM_UNLIKELY((numOfIndependentConsumers != 0) || (producer == PipelineInput))) {
                assert (!bn.CrossesHybridThreadBarrier);
                Value * const consumed = getConsumedOutputItems(externalPort.Port.Number); assert (consumed);
                setConsumedItemCount(b, streamSet, consumed, 0);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::readConsumedItemCount(BuilderRef b, const size_t streamSet) {

    Value * itemCount = nullptr;

    if (out_degree(streamSet, mConsumerGraph) == 0) {

        assert (!mBufferGraph[streamSet].CrossesHybridThreadBarrier);

        // This stream either has no consumers or we've proven that
        // its consumption rate is identical to its production rate.
        Value * produced = mInitiallyProducedItemCount[streamSet];
        assert (isFromCurrentFunction(b, produced, false));
        const auto e = in_edge(streamSet, mBufferGraph);
        const BufferPort & port = mBufferGraph[e];
        if (LLVM_UNLIKELY(produced == nullptr)) {
            const auto producer = source(e, mBufferGraph);
            const auto prefix = makeBufferName(producer, port.Port);
            if (LLVM_UNLIKELY(port.IsDeferred)) {
                produced = b->getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
            } else {
                produced = b->getScalarField(prefix + ITEM_COUNT_SUFFIX);
            }
        }
        auto delayOrLookBehind = std::max(port.Delay, port.LookBehind);
        for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const auto d = std::max(br.Delay, br.LookBehind);
            delayOrLookBehind = std::max(delayOrLookBehind, d);
        }
        if (delayOrLookBehind) {
            produced = b->CreateSaturatingSub(produced, b->getSize(delayOrLookBehind));
        }
        itemCount = produced;
    } else {

        const auto e = in_edge(streamSet, mConsumerGraph);
        const ConsumerEdge & c = mConsumerGraph[e];
        const auto producer = source(e, mConsumerGraph);
        if (LLVM_LIKELY(producer != PipelineInput || mTraceIndividualConsumedItemCounts)) {
            const StreamSetPort port{PortType::Output, c.Port};
            const auto prefix = makeBufferName(producer, port);
            Value * consumed0 = nullptr;
            if (hasUsersOnFixedAndHybridThread(streamSet)) {
                assert (mNumOfThreads > 1);
                Value * ptr = b->getScalarFieldPtr(prefix + HYBRID_THREAD_CONSUMED_ITEM_COUNT_SUFFIX);
                if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
                    Constant * const ZERO = b->getInt32(0);
                    ptr = b->CreateInBoundsGEP(ptr, { ZERO, ZERO } );
                }
                consumed0 = b->CreateLoad(ptr, true);
            }
            Value * ptr = b->getScalarFieldPtr(prefix + CONSUMED_ITEM_COUNT_SUFFIX);
            if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
                Constant * const ZERO = b->getInt32(0);
                ptr = b->CreateInBoundsGEP(ptr, { ZERO, ZERO } );
            }
            Value * const consumed1 = b->CreateLoad(ptr, mNumOfThreads > 1);
            itemCount = b->CreateUMin(consumed0, consumed1);
        } else {
            Value * const ptr = getProcessedInputItemsPtr(c.Port);
            assert (isFromCurrentFunction(b, ptr, false));
            itemCount = b->CreateLoad(ptr, mNumOfThreads > 1);
        }
    }

    return itemCount;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeTransitoryConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::writeTransitoryConsumedItemCount(BuilderRef b, const unsigned streamSet, Value * const produced) {
    if (out_degree(streamSet, mConsumerGraph) > 1) {
        b->setScalarField(TRANSITORY_CONSUMED_ITEM_COUNT_PREFIX + std::to_string(streamSet), produced);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief createConsumedPhiNodesAtExit
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::createConsumedPhiNodesAtExit(BuilderRef b) {
//    IntegerType * const sizeTy = b->getSizeTy();
//    for (const auto e : make_iterator_range(in_edges(mKernelId, mConsumerGraph))) {
//        const ConsumerEdge & c = mConsumerGraph[e];
//        if (c.Flags & ConsumerEdge::UpdatePhi) {
//            const auto streamSet = source(e, mConsumerGraph);
//            const ConsumerNode & cn = mConsumerGraph[streamSet];
////            assert (isFromCurrentFunction(b, cn.Consumed, true));
//            if (LLVM_LIKELY(cn.PhiNode == nullptr)) {
//                const ConsumerEdge & c = mConsumerGraph[e];
//                const StreamSetPort port(PortType::Input, c.Port);
//                const auto prefix = makeBufferName(mKernelId, port);
//                PHINode * const consumedPhi = b->CreatePHI(sizeTy, 2, prefix + "_consumed");
//                cn.PhiNode = consumedPhi;
//            }
//        }
//    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief phiOutConsumedItemCountsAfterInitiallyTerminated
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::phiOutConsumedItemCountsAfterInitiallyTerminated(BuilderRef b) {
//    for (const auto e : make_iterator_range(in_edges(mKernelId, mConsumerGraph))) {
//        const ConsumerEdge & c = mConsumerGraph[e];
//        if (c.Flags & ConsumerEdge::UpdatePhi) {
//            const auto streamSet = source(e, mConsumerGraph);
//            const ConsumerNode & cn = mConsumerGraph[streamSet];
//            assert (isFromCurrentFunction(b, cn.PhiNode, false));
//            assert (isFromCurrentFunction(b, mInitialConsumedItemCount[streamSet], false));
//            cn.PhiNode->addIncoming(mInitialConsumedItemCount[streamSet], mKernelInitiallyTerminatedExit);
//        }
//    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeMinimumConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::computeMinimumConsumedItemCounts(BuilderRef b) {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mConsumerGraph))) {
        const ConsumerEdge & c = mConsumerGraph[e];
        if (c.Flags & ConsumerEdge::UpdateConsumedCount) {
            const StreamSetPort port(PortType::Input, c.Port);
            Value * processed = mFullyProcessedItemCount[port];
            assert (isFromCurrentFunction(b, processed, false));
            // To support the lookbehind attribute, we need to withhold the items from
            // our consumed count and rely on the initial buffer underflow to access any
            // items before the start of the physical buffer.
            const auto input = getInput(mKernelId, port);
            const BufferPort & br = mBufferGraph[input];
            if (LLVM_UNLIKELY(br.LookBehind != 0)) {
                ConstantInt * const amount = b->getSize(br.LookBehind);
                processed = b->CreateSaturatingSub(processed, amount);
            }
            const auto streamSet = source(e, mConsumerGraph);
            assert (streamSet >= FirstStreamSet && streamSet <= LastStreamSet);
            if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
                const ConsumerEdge & c = mConsumerGraph[e]; assert (c.Index > 0);
                setConsumedItemCount(b, streamSet, processed, c.Index);
            }

            const auto output = in_edge(streamSet, mBufferGraph);
            Value * minConsumed = processed;

            if (out_degree(streamSet, mConsumerGraph) != 1) {
                Value * const transConsumedPtr = getScalarFieldPtr(b.get(), TRANSITORY_CONSUMED_ITEM_COUNT_PREFIX + std::to_string(streamSet));
                Value * const prior = b->CreateLoad(transConsumedPtr);
                const auto producer = source(output, mBufferGraph);
                const auto prodPrefix = makeBufferName(producer, mBufferGraph[output].Port);
                minConsumed = b->CreateUMin(prior, processed, prodPrefix + "_minConsumed");
                #ifdef PRINT_DEBUG_MESSAGES
                const auto consPrefix = makeBufferName(mKernelId, port);
                debugPrint(b, consPrefix + "_consumed = %" PRIu64 " -> " + prodPrefix + "_consumed' = %" PRIu64, prior, minConsumed);
                #endif
                if ((c.Flags & ConsumerEdge::WriteConsumedCount) == 0) {
                    b->CreateStore(minConsumed, transConsumedPtr);
                }
            }

            // check to see if we've fully finished processing any stream
            if (c.Flags & ConsumerEdge::WriteConsumedCount) {

                BasicBlock * resumeAfterWrite = nullptr;
                if (c.Flags & ConsumerEdge::MayHaveJumpedConsumer) {
                    BasicBlock * const isNonZero = b->CreateBasicBlock("", mKernelLoopExitPhiCatch);
                    resumeAfterWrite = b->CreateBasicBlock("", mKernelLoopExitPhiCatch);
                    b->CreateCondBr(b->CreateIsNotNull(minConsumed), isNonZero, resumeAfterWrite);

                    b->SetInsertPoint(isNonZero);
                }
                #ifdef PRINT_DEBUG_MESSAGES
                const auto output = in_edge(streamSet, mBufferGraph);
                const BufferPort & br = mBufferGraph[output];
                const auto producer = source(output, mBufferGraph);
                const auto prefix = makeBufferName(producer, br.Port);
                debugPrint(b, " * writing " + prefix + "_consumed = %" PRIu64, minConsumed);
                #endif
                setConsumedItemCount(b, streamSet, minConsumed, 0);



                if (resumeAfterWrite) {
                    b->CreateBr(resumeAfterWrite);
                    b->SetInsertPoint(resumeAfterWrite);
                }
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeFinalConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeConsumedItemCounts(BuilderRef b) {
//    for (const auto e : make_iterator_range(in_edges(mKernelId, mConsumerGraph))) {
//        const ConsumerEdge & c = mConsumerGraph[e];
//        const auto streamSet = source(e, mConsumerGraph);
//        // check to see if we've fully finished processing any stream
//        if (c.Flags & ConsumerEdge::WriteConsumedCount) {
//            const BufferNode & bn = mBufferGraph[streamSet];
//            if (LLVM_UNLIKELY(bn.isExternal())) {
//                for (const auto f : make_iterator_range(in_edges(streamSet, mBufferGraph))) {
//                    if (source(f, mBufferGraph) == PipelineInput) {
//                        const BufferPort & rd = mBufferGraph[e];
//                        Value * const ptr = getProcessedInputItemsPtr(rd.Port.Number);
//                        Value * const consumed = mInitialConsumedItemCount[streamSet]; assert (consumed);
//                        b->CreateStore(consumed, ptr);
//                    }
//                }
//            }
//        }
//    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::setConsumedItemCount(BuilderRef b, const size_t streamSet, not_null<Value *> consumed, const unsigned slot) const {
    const auto pe = in_edge(streamSet, mBufferGraph);
    const auto producer = source(pe, mBufferGraph);
    const BufferPort & outputPort = mBufferGraph[pe];
    Value * ptr = nullptr;

    assert (isFromCurrentFunction(b, consumed.get(), false));

    const auto prefix = makeBufferName(producer, outputPort.Port);

    if (mCompilingHybridThread && hasUsersOnFixedAndHybridThread(streamSet)) {
        ptr = b->getScalarFieldPtr(prefix + HYBRID_THREAD_CONSUMED_ITEM_COUNT_SUFFIX);
    } else {
        ptr = b->getScalarFieldPtr(prefix + CONSUMED_ITEM_COUNT_SUFFIX);
    }
    if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
        ptr = b->CreateInBoundsGEP(ptr, { b->getInt32(0), b->getInt32(slot) });
    }

    b->CreateStore(consumed, ptr);

    // update external count
    if (LLVM_UNLIKELY(producer == PipelineInput && slot == 0)) {
        b->CreateStore(consumed, getProcessedInputItemsPtr(outputPort.Port.Number));
    }

    if (LLVM_UNLIKELY(CheckAssertions)) {
        Value * const prior = b->CreateLoad(ptr);
        const Binding & output = outputPort.Binding;
        // TODO: cross reference which slot the traced count is for?

        assert (mCurrentKernelName);

        b->CreateAssert(b->CreateICmpULE(prior, consumed),
                        "%s.%s: consumed item count is not monotonically nondecreasing "
                        "(prior %" PRIu64 " > current %" PRIu64 " updated by %s)",
                        mKernelName[producer], b->GetString(output.getName()),
                        prior, consumed, mCurrentKernelName);

    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief zeroAnySkippedTransitoryConsumedItemCountsUntil
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::zeroAnySkippedTransitoryConsumedItemCountsUntil(BuilderRef b, const unsigned targetKernelId) {
//    ConstantInt * const sz_ZERO = b->getSize(0);
    Constant * const sz_MAX_INT = ConstantInt::getAllOnesValue(b->getSizeTy());

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        if (out_degree(streamSet, mConsumerGraph) < 2) {
            continue;
        }
        size_t maxConsumerInJumpRange = 0U;
        for (const auto e : make_iterator_range(out_edges(streamSet, mConsumerGraph))) {
            const auto consumer = target(e, mConsumerGraph);
            //assert (consumer >= FirstKernel && consumer <= LastKernel);
            if (consumer >= mKernelId && consumer <= targetKernelId) {
                maxConsumerInJumpRange = std::max<size_t>(maxConsumerInJumpRange, consumer);
            }
        }
        if (maxConsumerInJumpRange > 0) { // && (consumerFlags & ConsumerEdge::WriteConsumedCount) == 0
            Value * const transConsumedPtr = getScalarFieldPtr(b.get(), TRANSITORY_CONSUMED_ITEM_COUNT_PREFIX + std::to_string(streamSet));
            mTestConsumedItemCountForZero.set(streamSet - FirstStreamSet);

//            const auto producer = parent(streamSet, mBufferGraph);
//            Value * minConsumed = sz_ZERO;
//            if (producer < mKernelId) {
//                minConsumed = m
//            } else if (producer == mKernelId) {

//            }




//            Value * minConsumed = nullptr;
//            if (LLVM_UNLIKELY(maxConsumerInJumpRange == mKernelId)) {
//                minConsumed = b->CreateLoad(transConsumedPtr);
//                for (const auto input : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
//                    if (LLVM_UNLIKELY(source(input, mBufferGraph) == streamSet)) {
//                         // mMayLoopToEntry
//                        const auto & bp = mBufferGraph[input];
//                        Value * processed = nullptr;
//                        if (LLVM_UNLIKELY(bp.IsDeferred)) {
//                            processed = mAlreadyProcessedDeferredPhi[bp.Port];
//                        } else {
//                            processed = mAlreadyProcessedPhi[bp.Port];
//                        }
//                        minConsumed = b->CreateUMin(processed, minConsumed);
//                    }
//                }
//            } else {
//                minConsumed = sz_ZERO;
//            }
//            b->CreateStore(minConsumed, transConsumedPtr);
            b->CreateStore(sz_MAX_INT, transConsumedPtr);
        }

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializePipelineInputConsumedPhiNodes
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializePipelineInputConsumedPhiNodes(BuilderRef b) {
//    for (const auto e : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
//        const auto streamSet = target(e, mBufferGraph);
//        const BufferPort & br = mBufferGraph[e];
//        mInitialConsumedItemCount[streamSet] = getAvailableInputItems(br.Port.Number);
//    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief resetConsumerGraphState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::resetConsumerGraphState() {
//    for (unsigned i = FirstStreamSet; i <= LastStreamSet; ++i) {
//        const ConsumerNode & cn = mConsumerGraph[i];
//        cn.Consumed = nullptr;
//        cn.PhiNode = nullptr;
//    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getFinalConsumedCount
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getFinalConsumedCount(const unsigned streamSet) const {
    return nullptr;
}

}

#endif // CONSUMER_LOGIC_HPP
