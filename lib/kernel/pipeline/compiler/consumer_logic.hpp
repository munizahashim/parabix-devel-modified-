#ifndef CONSUMER_LOGIC_HPP
#define CONSUMER_LOGIC_HPP

#include "pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addConsumerKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::addConsumerKernelProperties(BuilderRef b, const unsigned kernelId) {
//    if (kernelId != PipelineInput || mTraceIndividualConsumedItemCounts) {
        IntegerType * const sizeTy = b->getSizeTy();
        const auto groupId = getCacheLineGroupId(kernelId);
        const auto prefix = makeKernelName(kernelId) + CONSUMED_ITEM_COUNT_SUFFIX;
        for (const auto e : make_iterator_range(in_edges(kernelId, mConsumerGraph))) {
            const ConsumerEdge & ce = mConsumerGraph[e];
            if (ce.Flags & ConsumerEdge::WriteConsumedCount) {
                const auto streamSet = source(e, mConsumerGraph);
                const auto name = prefix + std::to_string(streamSet);
                const BufferNode & bn = mBufferGraph[streamSet];
                if (LLVM_LIKELY(bn.isOwned() || bn.isInternal() || mTraceIndividualConsumedItemCounts)) {
                    mTarget->addInternalScalar(sizeTy, name, groupId);
                } else {
                    mTarget->addNonPersistentScalar(sizeTy, name);
                }
            }
        }
//    }

#if 0

        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const auto streamSet = target(e, mBufferGraph);
            if (out_degree(streamSet, mConsumerGraph) != 0) {
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
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (LLVM_LIKELY(bn.isOwned() || bn.isInternal() || bn.CrossesHybridThreadBarrier || mTraceIndividualConsumedItemCounts)) {
                        mTarget->addInternalScalar(countTy, name, groupId);
                    } else {
                        mTarget->addNonPersistentScalar(countTy, name);
                    }
                    if (bn.CrossesHybridThreadBarrier) {
                        assert (mNumOfThreads > 1);
                        const auto altId = getCacheLineGroupId(PipelineOutput) + groupId;
                        mTarget->addInternalScalar(countTy, prefix + HYBRID_THREAD_CONSUMED_ITEM_COUNT_SUFFIX, altId);
                    }

                }

            }
        }
    }
#endif

}



/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readExternalConsumerItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readAllConsumerItemCounts(BuilderRef b) {
//    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
//        const BufferNode & bn = mBufferGraph[streamSet];
//        if (LLVM_LIKELY(bn.isOwned() && bn.Locality != BufferLocality::ThreadLocal)) {
//            Value * consumed = readConsumedItemCount(b, streamSet);
//            mInitialConsumedItemCount[streamSet] = consumed; assert (consumed);
//            const ConsumerNode & cn = mConsumerGraph[streamSet];
//            cn.Consumed = consumed;
//        }
//    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readConsumedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readConsumedItemCounts(BuilderRef b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mConsumerGraph))) {
        const auto streamSet = target(e, mConsumerGraph);
        assert (mInitialConsumedItemCount[streamSet] == nullptr);

        Value * consumed = nullptr;
        if (out_degree(streamSet, mConsumerGraph) == 0) {
            if (mInitiallyProducedDeferredItemCount[streamSet]) {
                consumed = mInitiallyProducedDeferredItemCount[streamSet];
            } else {
                consumed = mInitiallyProducedItemCount[streamSet];
            }
            assert (isFromCurrentFunction(b, consumed, false));
            const auto e = in_edge(streamSet, mBufferGraph);
            const BufferPort & port = mBufferGraph[e];
            auto delayOrLookBehind = std::max(port.Delay, port.LookBehind);
            for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const BufferPort & br = mBufferGraph[e];
                const auto d = std::max(br.Delay, br.LookBehind);
                delayOrLookBehind = std::max(delayOrLookBehind, d);
            }
            if (delayOrLookBehind) {
                consumed = b->CreateSaturatingSub(consumed, b->getSize(delayOrLookBehind));
            }
        } else {
            const auto bitWidth = b->getSizeTy()->getBitWidth();

            Value * historyPtr = nullptr;
            if (LLVM_UNLIKELY(mTraceDynamicBuffers)) {
                const auto & bn = mBufferGraph[streamSet];
                if (isa<DynamicBuffer>(bn.Buffer)) {
                    historyPtr = b->getScalarFieldPtr(STATISTICS_BUFFER_EXPANSION_TEMP_STACK);
                }
            }

            for (const auto f : make_iterator_range(out_edges(streamSet, mConsumerGraph))) {
                const ConsumerEdge & c = mConsumerGraph[f];
                if (c.Flags & ConsumerEdge::WriteConsumedCount) {
                    const auto consumer = target(f, mConsumerGraph);
                    const auto prefix = makeKernelName(consumer) + CONSUMED_ITEM_COUNT_SUFFIX;
                    const auto name = prefix + std::to_string(streamSet);
                    Value * const ptr = b->getScalarFieldPtr(name);
                    Value * const immediatelyConsumed = b->CreateAlignedLoad(ptr, bitWidth / 8);
                    if (LLVM_UNLIKELY(historyPtr != nullptr)) {
                        FixedArray<Value *, 2> array;
                        array[0] = b->getInt32(0);
                        array[0] = b->getInt32(c.Index);
                        Value * tempPtr = b->CreateGEP(historyPtr, array);
                        b->CreateAlignedStore(immediatelyConsumed, tempPtr, bitWidth / 8);
                    }
                    consumed = b->CreateUMin(consumed, immediatelyConsumed);
                }
            }
        }

        // check if this could be a program output
        if (in_degree(PipelineOutput, mBufferGraph) > 0) {
            for (const auto e : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
                if (LLVM_UNLIKELY(source(e, mBufferGraph) == streamSet)) {
                    const BufferPort & externalPort = mBufferGraph[e];
                    Value * const external = getConsumedOutputItems(externalPort.Port.Number);
                    consumed = b->CreateUMin(consumed, external);
                }
            }
        }

        if (consumed == nullptr) {
            errs() << "SS: " << streamSet << "\n";
        }

        assert (consumed);

        mInitialConsumedItemCount[streamSet] = consumed;
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
//    for (const auto e : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
//        const auto streamSet = source(e, mBufferGraph);
//        const BufferNode & bn = mBufferGraph[streamSet];
//        if (LLVM_LIKELY(bn.isOwned())) {
//            const BufferPort & externalPort = mBufferGraph[e];
//            Value * const consumed = getConsumedOutputItems(externalPort.Port.Number); assert (consumed);
//            mInitialConsumedItemCount[streamSet] = consumed;
////            const auto numOfIndependentConsumers = out_degree(streamSet, mConsumerGraph);
////            const auto producer = parent(streamSet, mBufferGraph);
////            if (LLVM_UNLIKELY((numOfIndependentConsumers != 0) || (producer == PipelineInput))) {
////                storeConsumedItemCount(b, streamSet, consumed, 0);
////            }
//        }
//    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::readConsumedItemCount(BuilderRef b, const size_t streamSet) {

    const auto bitWidth = b->getSizeTy()->getBitWidth();

    if (out_degree(streamSet, mConsumerGraph) == 0) {

        assert (!mBufferGraph[streamSet].CrossesHybridThreadBarrier);

        // This stream either has no consumers or we've proven that
        // its consumption rate is identical to its production rate.
        Value * produced = mInitiallyProducedItemCount[streamSet];
        assert (isFromCurrentFunction(b, produced, false));
        const auto e = in_edge(streamSet, mBufferGraph);
        const BufferPort & port = mBufferGraph[e];
//        if (LLVM_UNLIKELY(produced == nullptr)) {
//            const auto producer = source(e, mBufferGraph);
//            const auto prefix = makeBufferName(producer, port.Port);
//            Value * ptr = nullptr;
//            if (LLVM_UNLIKELY(port.IsDeferred)) {
//                ptr = b->getScalarFieldPtr(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
//            } else {
//                ptr = b->getScalarFieldPtr(prefix + ITEM_COUNT_SUFFIX);
//            }
//            produced = b->CreateAlignedLoad(ptr, bitWidth / 8);
//        }
        auto delayOrLookBehind = std::max(port.Delay, port.LookBehind);
        for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const auto d = std::max(br.Delay, br.LookBehind);
            delayOrLookBehind = std::max(delayOrLookBehind, d);
        }
        if (delayOrLookBehind) {
            produced = b->CreateSaturatingSub(produced, b->getSize(delayOrLookBehind));
        }
        return produced;
    }

    const auto e = in_edge(streamSet, mConsumerGraph);
    const ConsumerEdge & c = mConsumerGraph[e];
    const auto producer = source(e, mConsumerGraph);
    if (LLVM_LIKELY(producer != PipelineInput || mTraceIndividualConsumedItemCounts)) {

        const StreamSetPort port{PortType::Output, c.Port};
        const auto prefix = makeBufferName(producer, port);
        const BufferNode & bn = mBufferGraph[streamSet];

        Value * consumed = nullptr;
        if (bn.CrossesHybridThreadBarrier) {
            assert (mNumOfThreads > 1);
            Value * ptr = b->getScalarFieldPtr(prefix + HYBRID_THREAD_CONSUMED_ITEM_COUNT_SUFFIX);
            if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
                Constant * const ZERO = b->getInt32(0);
                ptr = b->CreateInBoundsGEP(ptr, { ZERO, ZERO } );
            }
            consumed = b->CreateAlignedLoad(ptr, bitWidth / 8, true);
        }
        Value * ptr = b->getScalarFieldPtr(prefix + CONSUMED_ITEM_COUNT_SUFFIX);
        if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
            Constant * const ZERO = b->getInt32(0);
            ptr = b->CreateInBoundsGEP(ptr, { ZERO, ZERO } );
        }
        return b->CreateUMin(consumed, b->CreateAlignedLoad(ptr, bitWidth / 8, true));
    } else {
        Value * const ptr = getProcessedInputItemsPtr(c.Port);
        assert (isFromCurrentFunction(b, ptr, false));
        return b->CreateLoad(ptr);
    }


}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializeConsumedItemCount(BuilderRef b, const unsigned kernelId, const StreamSetPort outputPort, Value * const produced) {
    const auto output = getOutput(kernelId, outputPort);
    const auto streamSet = target(output, mBufferGraph);
    if (out_degree(streamSet, mConsumerGraph) != 0) {
        Value * initiallyConsumed = produced;
        const BufferPort & br = mBufferGraph[output];
        if (br.LookBehind || br.Delay) {
            const auto delayOrLookBehind = std::max(br.Delay, br.LookBehind);
            initiallyConsumed = b->CreateSaturatingSub(produced, b->getSize(delayOrLookBehind));
        }
        const ConsumerNode & cn = mConsumerGraph[streamSet];
        cn.Consumed = initiallyConsumed;


        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, outputPort);
        debugPrint(b, prefix + " -> " + prefix + "_initiallyConsumed = %" PRIu64, initiallyConsumed);
        #endif
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief createConsumedPhiNodes
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::createConsumedPhiNodes(BuilderRef b) {
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
                assert (cn.Consumed);
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
//            if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {
//                const ConsumerEdge & c = mConsumerGraph[e]; assert (c.Index > 0);
//                storeConsumedItemCount(b, streamSet, processed, c.Index);
//            }
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
        if (c.Flags & ConsumerEdge::UpdatePhi) {
            const auto streamSet = source(e, mConsumerGraph);
            const ConsumerNode & cn = mConsumerGraph[streamSet];
            if (LLVM_LIKELY(cn.PhiNode != nullptr)) {
                cn.PhiNode->addIncoming(cn.Consumed, mKernelLoopExitPhiCatch);
                cn.Consumed = cn.PhiNode;
                cn.PhiNode = nullptr;
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
                storeConsumedItemCount(b, streamSet, cn.Consumed, c.Index);
            }

        }
    }
}



/** ------------------------------------------------------------------------------------------------------------- *
 * @brief storeConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::storeConsumedItemCount(BuilderRef b, const size_t streamSet, not_null<Value *> consumed, const unsigned slot) const {


    const BufferNode & bn = mBufferGraph[streamSet];

    if (bn.Locality == BufferLocality::ThreadLocal) {

        if (LLVM_UNLIKELY(CheckAssertions)) {

            const auto pe = in_edge(streamSet, mBufferGraph);
            const auto producer = source(pe, mBufferGraph);
            const BufferPort & rd = mBufferGraph[pe];
            const Binding & output = rd.Binding;

            Constant * const bindingName = b->GetString(output.getName());

            Value * const produced = mLocallyAvailableItems[streamSet]; assert (produced);
            // NOTE: static linear buffers are assumed to be threadlocal.
            Value * const fullyConsumed = b->CreateICmpEQ(produced, consumed);
            Constant * const fatal = getTerminationSignal(b, TerminationSignal::Fatal);
            Value * const fatalError = b->CreateICmpEQ(mTerminatedAtLoopExitPhi, fatal);
            Value * const valid = b->CreateOr(fullyConsumed, fatalError);

            b->CreateAssert(valid,
                            "%s.%s: local available item count (%" PRId64 ") does not match "
                            "its consumed item count (%" PRId64 ")",
                            mCurrentKernelName, bindingName,
                            produced, consumed);

        }

    } else if (LLVM_LIKELY(out_degree(streamSet, mConsumerGraph) != 0)) {

        const auto pe = in_edge(streamSet, mBufferGraph);
        const auto producer = source(pe, mBufferGraph);
        const BufferPort & rd = mBufferGraph[pe];
        Value * ptr = nullptr;
        if (LLVM_LIKELY(producer != PipelineInput || mTraceIndividualConsumedItemCounts)) {
            const auto prefix = makeKernelName(mKernelId) + CONSUMED_ITEM_COUNT_SUFFIX;
            const auto name = prefix + std::to_string(streamSet);
            ptr = b->getScalarFieldPtr(name);
        } else {
            ptr = getProcessedInputItemsPtr(rd.Port.Number);
        }
        const auto bitWidth = ptr->getType()->getPointerElementType()->getPrimitiveSizeInBits();
        StoreInst * store = b->CreateAlignedStore(consumed, ptr, bitWidth / 8, true);
        // store->setOrdering(AtomicOrdering::Release);

    }

}

#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief storeConsumedItemCount
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::storeConsumedItemCount(BuilderRef b, const size_t streamSet, not_null<Value *> consumed, const unsigned slot) const {


    const BufferNode & bn = mBufferGraph[streamSet];

    if (bn.Locality == BufferLocality::ThreadLocal) {

        if (LLVM_UNLIKELY(CheckAssertions)) {

            const auto pe = in_edge(streamSet, mBufferGraph);
            const auto producer = source(pe, mBufferGraph);
            const BufferPort & rd = mBufferGraph[pe];
            const Binding & output = rd.Binding;

            Constant * const bindingName = b->GetString(output.getName());

            Value * const produced = mLocallyAvailableItems[streamSet]; assert (produced);
            // NOTE: static linear buffers are assumed to be threadlocal.
            Value * const fullyConsumed = b->CreateICmpEQ(produced, consumed);
            Constant * const fatal = getTerminationSignal(b, TerminationSignal::Fatal);
            Value * const fatalError = b->CreateICmpEQ(mTerminatedAtLoopExitPhi, fatal);
            Value * const valid = b->CreateOr(fullyConsumed, fatalError);

            b->CreateAssert(valid,
                            "%s.%s: local available item count (%" PRId64 ") does not match "
                            "its consumed item count (%" PRId64 ")",
                            mCurrentKernelName, bindingName,
                            produced, consumed);

        }

    } else if (LLVM_LIKELY(out_degree(streamSet, mConsumerGraph) != 0)) {

        const auto pe = in_edge(streamSet, mBufferGraph);
        const auto producer = source(pe, mBufferGraph);
        const BufferPort & rd = mBufferGraph[pe];
        Value * ptr = nullptr;

        if (LLVM_LIKELY(producer != PipelineInput || mTraceIndividualConsumedItemCounts)) {
            const auto prefix = makeBufferName(producer, rd.Port);

            if (mCompilingHybridThread && bn.CrossesHybridThreadBarrier) {
                ptr = b->getScalarFieldPtr(prefix + HYBRID_THREAD_CONSUMED_ITEM_COUNT_SUFFIX);
            } else {
                ptr = b->getScalarFieldPtr(prefix + CONSUMED_ITEM_COUNT_SUFFIX);
            }

            ptr = b->getScalarFieldPtr(prefix + CONSUMED_ITEM_COUNT_SUFFIX);
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

            }
        } else {
            ptr = getProcessedInputItemsPtr(rd.Port.Number);
        }
        const auto bitWidth = ptr->getType()->getPointerElementType()->getPrimitiveSizeInBits();
        StoreInst * store = b->CreateAlignedStore(consumed, ptr, bitWidth / 8, true);
        // store->setOrdering(AtomicOrdering::Release);

    }

}

#endif

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializePipelineInputConsumedPhiNodes
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializePipelineInputConsumedPhiNodes(BuilderRef b) {
    for (const auto e : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        if (out_degree(streamSet, mConsumerGraph) != 0) {
            const BufferPort & br = mBufferGraph[e];
            mInitialConsumedItemCount[streamSet] = getAvailableInputItems(br.Port.Number);
        }
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

}

#endif // CONSUMER_LOGIC_HPP
