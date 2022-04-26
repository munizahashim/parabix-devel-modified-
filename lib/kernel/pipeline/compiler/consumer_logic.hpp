#ifndef CONSUMER_LOGIC_HPP
#define CONSUMER_LOGIC_HPP

#include "pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addConsumerKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::addConsumerKernelProperties(BuilderRef b, const unsigned kernelId) {
    if (kernelId != PipelineInput || mTraceIndividualConsumedItemCounts) {

        IntegerType * const sizeTy = b->getSizeTy();

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
                const auto name = prefix + CONSUMED_ITEM_COUNT_SUFFIX;

                // If we're tracing the consumer item counts, we need to store one for each
                // (non-nested) consumer. Any nested consumers will have their own trace.
                Type * countTy = sizeTy;
                if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
                    countTy = ArrayType::get(sizeTy, numOfIndependentConsumers + 1);
                }
                if (LLVM_LIKELY(bn.isOwned() || bn.isInternal() || bn.CrossesHybridThreadBarrier || mTraceIndividualConsumedItemCounts)) {
                    mTarget->addInternalScalar(countTy, name, groupId);
                } else {
                    mTarget->addNonPersistentScalar(countTy, name);
                }
                if (hasUsersOnFixedAndHybridThread(streamSet)) {
                    assert (mNumOfThreads > 1);
                    const auto altId = getCacheLineGroupId(PipelineOutput);
                    mTarget->addInternalScalar(countTy, prefix + HYBRID_THREAD_CONSUMED_ITEM_COUNT_SUFFIX, altId);
                }
            }
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
        mInitialConsumedItemCount[streamSet] = consumed; assert (consumed);
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
                "Consumed item count (%" PRId64 ") of %s.%s exceeded its produced item count (%" PRId64 ").";
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
            Value * const consumed = getConsumedOutputItems(externalPort.Port.Number); assert (consumed);
            const auto numOfIndependentConsumers = out_degree(streamSet, mConsumerGraph);
            const auto producer = parent(streamSet, mBufferGraph);
            if (LLVM_UNLIKELY((numOfIndependentConsumers != 0) || (producer == PipelineInput))) {
                assert (!bn.CrossesHybridThreadBarrier);
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
 * @brief initializeConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializeConsumedItemCount(BuilderRef b, const unsigned kernelId, const StreamSetPort outputPort, Value * const produced) {
    Value * initiallyConsumed = produced;
    const auto output = getOutput(kernelId, outputPort);
    const BufferPort & br = mBufferGraph[output];
    if (br.LookBehind || br.Delay) {
        const auto delayOrLookBehind = std::max(br.Delay, br.LookBehind);
        initiallyConsumed = b->CreateSaturatingSub(produced, b->getSize(delayOrLookBehind));
    }
    const auto streamSet = target(output, mBufferGraph);
    const ConsumerNode & cn = mConsumerGraph[streamSet];
    cn.Consumed = initiallyConsumed;


    #ifdef PRINT_DEBUG_MESSAGES
    const auto prefix = makeBufferName(kernelId, outputPort);
    debugPrint(b, prefix + " -> " + prefix + "_initiallyConsumed = %" PRIu64, initiallyConsumed);
    #endif

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief createConsumedPhiNodesAtExit
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::createConsumedPhiNodesAtExit(BuilderRef b) {
    IntegerType * const sizeTy = b->getSizeTy();
    for (const auto e : make_iterator_range(in_edges(mKernelId, mConsumerGraph))) {
        const ConsumerEdge & c = mConsumerGraph[e];
        if (c.Flags & ConsumerEdge::UpdatePhi) {
            const auto streamSet = source(e, mConsumerGraph);
            const ConsumerNode & cn = mConsumerGraph[streamSet];
            if (LLVM_LIKELY(cn.PhiNode == nullptr)) {
                const ConsumerEdge & c = mConsumerGraph[e];
                const StreamSetPort port(PortType::Input, c.Port);
                const auto prefix = makeBufferName(mKernelId, port);
                PHINode * const consumedPhi = b->CreatePHI(sizeTy, 2, prefix + "_consumed");
                assert (isFromCurrentFunction(b, cn.Consumed, false));
                cn.PhiNode = consumedPhi;
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief phiOutConsumedItemCountsAfterInitiallyTerminated
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::phiOutConsumedItemCountsAfterInitiallyTerminated(BuilderRef b) {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mConsumerGraph))) {
        const ConsumerEdge & c = mConsumerGraph[e];
        if (c.Flags & ConsumerEdge::UpdatePhi) {
            const auto streamSet = source(e, mConsumerGraph);
            const ConsumerNode & cn = mConsumerGraph[streamSet];
            assert (isFromCurrentFunction(b, cn.PhiNode, false));
            assert (isFromCurrentFunction(b, mInitialConsumedItemCount[streamSet], false));
            cn.PhiNode->addIncoming(mInitialConsumedItemCount[streamSet], mKernelInitiallyTerminatedExit);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeMinimumConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::computeMinimumConsumedItemCounts(BuilderRef b) {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mConsumerGraph))) {
        const ConsumerEdge & c = mConsumerGraph[e];
        if (c.Flags & ConsumerEdge::UpdatePhi) {
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
            const ConsumerNode & cn = mConsumerGraph[streamSet]; assert (cn.Consumed);
            if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
                const ConsumerEdge & c = mConsumerGraph[e]; assert (c.Index > 0);
                setConsumedItemCount(b, streamSet, processed, c.Index);
            }
            assert (isFromCurrentFunction(b, cn.Consumed, false));
            const auto output = in_edge(streamSet, mBufferGraph);
            const auto producer = source(output, mBufferGraph);
            const auto prodPrefix = makeBufferName(producer, mBufferGraph[output].Port);
            cn.Consumed = b->CreateUMin(cn.Consumed, processed, prodPrefix + "_minConsumed");

            #ifdef PRINT_DEBUG_MESSAGES
            const auto consPrefix = makeBufferName(mKernelId, port);
            debugPrint(b, consPrefix + " -> " + prodPrefix + "_consumed' = %" PRIu64, cn.Consumed);
            #endif
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeFinalConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::writeConsumedItemCounts(BuilderRef b) {

    for (const auto e : make_iterator_range(in_edges(mKernelId, mConsumerGraph))) {
        const ConsumerEdge & c = mConsumerGraph[e];
        const auto streamSet = source(e, mConsumerGraph);
        const ConsumerNode & cn = mConsumerGraph[streamSet];
        if (c.Flags & ConsumerEdge::UpdatePhi) {
            if (LLVM_LIKELY(cn.PhiNode != nullptr)) {
                cn.PhiNode->addIncoming(cn.Consumed, mKernelLoopExitPhiCatch);
                cn.Consumed = cn.PhiNode;
                cn.PhiNode = nullptr;
            }
        }
        // check to see if we've fully finished processing any stream
        if (c.Flags & ConsumerEdge::WriteConsumedCount) {
            #ifdef PRINT_DEBUG_MESSAGES
            const auto output = in_edge(streamSet, mBufferGraph);
            const BufferPort & br = mBufferGraph[output];
            const auto producer = source(output, mBufferGraph);
            const auto prefix = makeBufferName(producer, br.Port);
            debugPrint(b, " * writing " + prefix + "_consumed = %" PRIu64, cn.Consumed);
            #endif
            setConsumedItemCount(b, streamSet, cn.Consumed, 0);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::setConsumedItemCount(BuilderRef b, const size_t streamSet, not_null<Value *> consumed, const unsigned slot) const {
    const auto pe = in_edge(streamSet, mBufferGraph);
    const auto producer = source(pe, mBufferGraph);
    const BufferPort & rd = mBufferGraph[pe];
    Value * ptr = nullptr;
    if (LLVM_LIKELY(producer != PipelineInput || mTraceIndividualConsumedItemCounts)) {
        const auto prefix = makeBufferName(producer, rd.Port);

        const BufferNode & bn = mBufferGraph[streamSet];

        if (mCompilingHybridThread && hasUsersOnFixedAndHybridThread(streamSet)) {
            ptr = b->getScalarFieldPtr(prefix + HYBRID_THREAD_CONSUMED_ITEM_COUNT_SUFFIX);
        } else {
            ptr = b->getScalarFieldPtr(prefix + CONSUMED_ITEM_COUNT_SUFFIX);
        }
        if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
            ptr = b->CreateInBoundsGEP(ptr, { b->getInt32(0), b->getInt32(slot) });
        }

        if (LLVM_UNLIKELY(CheckAssertions)) {
            Value * const prior = b->CreateLoad(ptr);
            const Binding & output = rd.Binding;
            // TODO: cross reference which slot the traced count is for?

            Constant * const bindingName = b->GetString(output.getName());

            assert (mCurrentKernelName);

            b->CreateAssert(b->CreateICmpULE(prior, consumed),
                            "%s.%s: consumed item count is not monotonically nondecreasing "
                            "(prior %" PRIu64 " > current %" PRIu64 ")",
                            mCurrentKernelName, bindingName,
                            prior, consumed);

            if (bn.Locality == BufferLocality::ThreadLocal) {
                Value * const produced = mLocallyAvailableItems[streamSet]; assert (produced);
                // NOTE: static linear buffers are assumed to be threadlocal.
                Value * const fullyConsumed = b->CreateICmpEQ(produced, consumed);
                Constant * const fatal = getTerminationSignal(b, TerminationSignal::Fatal);
                Value * const fatalError = b->CreateICmpEQ(mTerminatedAtLoopExitPhi, fatal);
                Value * const valid = b->CreateOr(fullyConsumed, fatalError);

                Constant * withOrWithout = nullptr;
                if (mMayLoopToEntry) {
                    withOrWithout = b->GetString("with");
                } else {
                    withOrWithout = b->GetString("without");
                }

                b->CreateAssert(valid,
                                "%s.%s: local available item count (%" PRId64 ") does not match "
                                "its consumed item count (%" PRId64 ") in kernel %s loop back support",
                                mCurrentKernelName, bindingName,
                                produced, consumed, withOrWithout);
            }

        }
    } else {
        ptr = getProcessedInputItemsPtr(rd.Port.Number);
    }

    b->CreateStore(consumed, ptr);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializePipelineInputConsumedPhiNodes
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializePipelineInputConsumedPhiNodes(BuilderRef b) {
    for (const auto e : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferPort & br = mBufferGraph[e];
        mInitialConsumedItemCount[streamSet] = getAvailableInputItems(br.Port.Number);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief reportExternalConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::writeExternalConsumedItemCounts(BuilderRef b) {
//    for (const auto e : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
//        const auto streamSet = target(e, mBufferGraph);
//        const BufferPort & rd = mBufferGraph[e];
//        Value * const ptr = getProcessedInputItemsPtr(rd.Port.Number);
//        Value * const consumed = mInitialConsumedItemCount[streamSet]; assert (consumed);
//        b->CreateStore(consumed, ptr);
//    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getLastConsumerOfStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
unsigned PipelineCompiler::getLastConsumerOfStreamSet(const size_t streamSet) const {
    auto lastConsumer = PipelineInput;
   // assert (out_degree(streamSet, mConsumerGraph) > 0);
    for (const auto input : make_iterator_range(out_edges(streamSet, mConsumerGraph))) {
        const auto consumer = target(input, mConsumerGraph);
        if (LLVM_LIKELY(consumer < PipelineOutput)) {
            lastConsumer = std::max<unsigned>(lastConsumer, consumer);
        }
    }
    return lastConsumer;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief resetConsumerGraphState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::resetConsumerGraphState() {
    for (unsigned i = FirstStreamSet; i <= LastStreamSet; ++i) {
        const ConsumerNode & cn = mConsumerGraph[i];
        cn.Consumed = nullptr;
        cn.PhiNode = nullptr;
    }
}


}

#endif // CONSUMER_LOGIC_HPP
