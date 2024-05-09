#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addHandlesToPipelineKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addBufferHandlesToPipelineKernel(KernelBuilder & b, const unsigned kernelId, const unsigned groupId) {

    bool hasAnyInternalStreamSets = false;

    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isTruncated())) continue;

        const BufferPort & rd = mBufferGraph[e];
        const auto prefix = makeBufferName(kernelId, rd.Port);
        StreamSetBuffer * const buffer = bn.Buffer;

        // external buffers already have a buffer handle
        if (LLVM_LIKELY(bn.isInternal() || bn.isConstant())) {

            Type * const handleType = buffer->getHandleType(b);
            // We automatically assign the buffer memory according to the buffer start position
            if (LLVM_UNLIKELY(bn.isConstant())) {
                if (cast<RepeatingStreamSet>(buffer)->isDynamic()) {
                    mTarget->addInternalScalar(handleType, prefix, groupId);
                } else {
                    mTarget->addNonPersistentScalar(handleType, prefix);
                }
            } else if (bn.isThreadLocal()) {
                hasAnyInternalStreamSets = true;
                mTarget->addNonPersistentScalar(handleType, prefix);
            } else if (LLVM_LIKELY(bn.isOwned() || bn.hasZeroElementsOrWidth())) {
                hasAnyInternalStreamSets = true;
                mTarget->addInternalScalar(handleType, prefix, groupId);
            } else {
                mTarget->addNonPersistentScalar(handleType, prefix);
                mTarget->addInternalScalar(buffer->getPointerType(), prefix + LAST_GOOD_VIRTUAL_BASE_ADDRESS, groupId);
            }
        }

        // Although we'll end up wasting memory, we can avoid memleaks and the issue of multiple threads
        // successively expanding a buffer and wrongly free-ing one that's still in use by allowing each
        // thread to independently retain a pointer to the "old" buffer and free'ing it on a subseqent
        // segment.
        if (bn.isOwned() && isa<DynamicBuffer>(buffer) && isMultithreaded()) {
            assert (bn.isNonThreadLocal());
            mTarget->addThreadLocalScalar(b.getVoidPtrTy(), prefix + PENDING_FREEABLE_BUFFER_ADDRESS, groupId);
            mTarget->addThreadLocalScalar(b.getSizeTy(), prefix + PENDING_FREEABLE_BUFFER_CAPACITY, groupId);
        }
    }

    if (LLVM_UNLIKELY(!mTarget->allocatesInternalStreamSets())) {
        const Kernel * const kernelObj = getKernel(kernelId);
        if (LLVM_UNLIKELY(hasAnyInternalStreamSets)) {
            SmallVector<char, 1024> tmp;
            raw_svector_ostream msg(tmp);
            msg << "Pipeline " << mTarget->getName() << " is not marked as allocating internal streamsets"
            << " but must do so to support " << kernelObj->getName() << ".";
            report_fatal_error(StringRef(msg.str()));
        }
        if (LLVM_UNLIKELY(kernelObj->allocatesInternalStreamSets())) {
            SmallVector<char, 1024> tmp;
            raw_svector_ostream msg(tmp);
            msg << "Pipeline " << mTarget->getName() << " is not marked as allocating internal streamsets"
            << " but " << kernelObj->getName() << " must do so to be correctly initialized.";
            report_fatal_error(StringRef(msg.str()));
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadInternalStreamSetHandles
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::loadInternalStreamSetHandles(KernelBuilder & b, const bool nonLocal) {
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isTruncated())) continue;
        // external buffers already have a buffer handle
        StreamSetBuffer * const buffer = bn.Buffer;
        if (bn.isNonThreadLocal() == nonLocal) {
            if (LLVM_UNLIKELY(bn.isExternal())) {
                assert (isFromCurrentFunction(b, buffer->getHandle(), true));
            } else if (LLVM_UNLIKELY(bn.isConstant())) {
                assert (nonLocal);
                const auto handleName = REPEATING_STREAMSET_HANDLE_PREFIX + std::to_string(streamSet);
                buffer->setHandle(b.getScalarFieldPtr(handleName));
                const auto & sn = mStreamGraph[streamSet];
                assert (sn.Type == RelationshipNode::IsStreamSet);
                if (cast<RepeatingStreamSet>(sn.Relationship)->isDynamic()) {
                    const auto lengthName = REPEATING_STREAMSET_LENGTH_PREFIX + std::to_string(streamSet);
                    Value * const mod = b.getScalarField(lengthName);
                    cast<RepeatingBuffer>(buffer)->setModulus(mod);
                } else {
                    assert(isa<Constant>(cast<RepeatingBuffer>(buffer)->getModulus()));
                }
            } else {
                const auto pe = in_edge(streamSet, mBufferGraph);
                const auto producer = source(pe, mBufferGraph);
                const BufferPort & rd = mBufferGraph[pe];
                const auto handleName = makeBufferName(producer, rd.Port);
                buffer->setHandle(b.getScalarFieldPtr(handleName));
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getReturnedBufferScaleFactor
 ** ------------------------------------------------------------------------------------------------------------- */
Rational PipelineCompiler::getReturnedBufferScaleFactor(const size_t streamSet) const {
    Rational scaleFactor{0, 1};
    auto updateScaleFactorForPort = [&](BufferGraph::edge_descriptor port) {
        const BufferPort & rd = mBufferGraph[port];
        const Binding & bindingRef = rd.Binding;
        const AttributeSet & attrs = bindingRef.getAttributes();
        if (attrs.hasAttribute(AttrId::ReturnedBuffer)) {
            const Attribute & attrRef = attrs.findAttribute(AttrId::ReturnedBuffer);
            scaleFactor = std::max(scaleFactor, attrRef.ratio());
        }
    };
    updateScaleFactorForPort(in_edge(streamSet, mBufferGraph));
    for (const auto port : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
        updateScaleFactorForPort(port);
    }
    return scaleFactor;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief allocateOwnedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::allocateOwnedBuffers(KernelBuilder & b, Value * const expectedNumOfStrides, Value * const expectedSourceOutputSize, const bool nonLocal) {
    assert (expectedNumOfStrides);
    if (LLVM_UNLIKELY(CheckAssertions)) {
        Value * const valid = b.CreateIsNotNull(expectedNumOfStrides);
        b.CreateAssert(valid,
           "%s: expected number of strides for internally allocated buffers is 0",
           b.GetString(mTarget->getName()));
    }

    // recursively allocate any internal buffers for the nested kernels, giving them the correct
    // num of strides it should expect to perform
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernelObj = getKernel(i);

        if (LLVM_UNLIKELY(kernelObj->allocatesInternalStreamSets())) {
            if (nonLocal || kernelObj->hasThreadLocal()) {
                setActiveKernel(b, i, !nonLocal);
                assert (mKernel == kernelObj);
                SmallVector<Value *, 3> params;
                if (LLVM_LIKELY(mKernelSharedHandle)) {
                    params.push_back(mKernelSharedHandle);
                }
                Value * func = nullptr;
                FunctionType * funcTy;
                if (nonLocal) {
                    std::tie(func, funcTy) = getKernelAllocateSharedInternalStreamSetsFunction(b);
                } else {
                    std::tie(func, funcTy) = getKernelAllocateThreadLocalInternalStreamSetsFunction(b);
                    params.push_back(mKernelThreadLocalHandle);
                }

                params.push_back(b.CreateCeilUMulRational(expectedNumOfStrides, MaximumNumOfStrides[i]));

                b.CreateCall(funcTy, func, params);
            }
        }
    }


    // and allocate any output buffers
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isTruncated() || bn.hasZeroElementsOrWidth())) continue;
        if (bn.isNonThreadLocal() == nonLocal && bn.isOwned()) {
            StreamSetBuffer * const buffer = bn.Buffer;

            if (LLVM_UNLIKELY(bn.isConstant())) {
                generateGlobalDataForRepeatingStreamSet(b, streamSet, expectedNumOfStrides);
            } else {
                if (LLVM_LIKELY(bn.isInternal())) {
                    const auto pe = in_edge(streamSet, mBufferGraph);
                    const auto producer = source(pe, mBufferGraph);
                    const BufferPort & rd = mBufferGraph[pe];
                    const auto handleName = makeBufferName(producer, rd.Port);
                    buffer->setHandle(b.getScalarFieldPtr(handleName));
                } else {
                    assert (isFromCurrentFunction(b, buffer->getHandle(), false));
                }
                if (nonLocal) {

                    Value * multiplier = expectedNumOfStrides;

                    if (LLVM_UNLIKELY(bn.isReturned())) {
                        auto scaleFactor = getReturnedBufferScaleFactor(streamSet);
                        if (scaleFactor > 0) {

                            size_t capacity = 1;
                            if (isa<DynamicBuffer>(buffer)) {
                                capacity = cast<DynamicBuffer>(buffer)->getInitialCapacity();
                            } else if (isa<MMapedBuffer>(buffer)) {
                                capacity = cast<MMapedBuffer>(buffer)->getInitialCapacity();
                            }
                            multiplier = b.CreateRoundUp(expectedSourceOutputSize, expectedNumOfStrides);
                            Value * value = b.CreateCeilUDivRational(multiplier, capacity);
                            multiplier = b.CreateUMax(value, expectedNumOfStrides);
                        }
                    }

                    buffer->allocateBuffer(b, multiplier);

                    #ifdef PRINT_DEBUG_MESSAGES
                    const auto pe = in_edge(streamSet, mBufferGraph);
                    const auto producer = source(pe, mBufferGraph);
                    const BufferPort & rd = mBufferGraph[pe];
                    const auto prefix = makeBufferName(producer, rd.Port);
                    Value * start = buffer->getMallocAddress(b);
                    Value * end = b.CreateGEP(buffer->getType(), start, buffer->getCapacity(b));
                    debugPrint(b, prefix + ".inital malloc range = [%" PRIx64 ",%" PRIx64 ")", start, end);
                    #endif

                }
            }
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief releaseOwnedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::releaseOwnedBuffers(KernelBuilder & b) {
    loadInternalStreamSetHandles(b, true);
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_LIKELY(bn.isDeallocatable())) {
            StreamSetBuffer * const buffer = bn.Buffer;
            assert (isFromCurrentFunction(b, buffer->getHandle(), false));
            buffer->releaseBuffer(b);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief freePendingFreeableDynamicBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::freePendingFreeableDynamicBuffers(KernelBuilder & b) {
    if (LLVM_LIKELY(isMultithreaded())) {
        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_LIKELY(bn.isDeallocatable())) {
                StreamSetBuffer * const buffer = bn.Buffer;
                if (LLVM_LIKELY(isa<DynamicBuffer>(buffer))) {
                    const auto pe = in_edge(streamSet, mBufferGraph);
                    const auto p = source(pe, mBufferGraph);
                    const BufferPort & rd = mBufferGraph[pe];
                    assert (rd.Port.Type == PortType::Output);
                    const auto prefix = makeBufferName(p, rd.Port);
                    Value * const addr = b.getScalarField(prefix + PENDING_FREEABLE_BUFFER_ADDRESS);
                    Value * const capacity = b.getScalarField(prefix + PENDING_FREEABLE_BUFFER_CAPACITY);
                    buffer->destroyBuffer(b, addr, capacity);
                }
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateExternalProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateExternalProducedItemCounts(KernelBuilder & b) {
    for (const auto output : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        const auto streamSet = source(output, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isReturned()) {
            const auto pe = in_edge(streamSet, mBufferGraph);
            const auto producer = source(pe, mBufferGraph);
            const BufferPort & br = mBufferGraph[pe];
            const auto prefix = makeBufferName(producer, br.Port);

            const BufferPort & bp = mBufferGraph[output];
            const auto k = bp.Port.Number;

            Value * itemCount = nullptr;
            if (LLVM_UNLIKELY(br.isDeferred())) {
                itemCount = b.getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
            } else {
                itemCount = b.getScalarField(prefix + ITEM_COUNT_SUFFIX);
            }

            assert (isFromCurrentFunction(b, itemCount, false));
            assert (isFromCurrentFunction(b, mProducedOutputItemPtr[k], false));

            assert (mProducedOutputItemPtr[k]->getType()->isPointerTy());
            b.CreateStore(itemCount, mProducedOutputItemPtr[k]);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief resetInternalBufferHandles
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::resetInternalBufferHandles() {
    for (auto i = FirstStreamSet; i <= LastStreamSet; ++i) {
        const BufferNode & bn = mBufferGraph[i];
        if (LLVM_UNLIKELY(bn.isInternal())) {
            StreamSetBuffer * const buffer = bn.Buffer;
            buffer->setHandle(nullptr);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructStreamSetBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::constructStreamSetBuffers(KernelBuilder & /* b */) {

    mStreamSetInputBuffers.clear();
    const auto numOfInputStreams = out_degree(PipelineInput, mBufferGraph);
    mStreamSetInputBuffers.resize(numOfInputStreams);
    for (const auto e : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
        const BufferPort & rd = mBufferGraph[e];
        const auto i = rd.Port.Number;
        const auto streamSet = target(e, mBufferGraph);
        assert (mBufferGraph[streamSet].isExternal());
        const auto j = streamSet - FirstStreamSet;
        StreamSetBuffer * const buffer = mInternalBuffers[j].release();
        assert (buffer == mBufferGraph[streamSet].Buffer);
        mStreamSetInputBuffers[i].reset(buffer);
    }

    mStreamSetOutputBuffers.clear();
    const auto numOfOutputStreams = in_degree(PipelineOutput, mBufferGraph);
    mStreamSetOutputBuffers.resize(numOfOutputStreams);
    for (const auto e : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
        const BufferPort & rd = mBufferGraph[e];
        const auto i = rd.Port.Number;
        const auto streamSet = source(e, mBufferGraph);
        assert (mBufferGraph[streamSet].isExternal());
        const auto j = streamSet - FirstStreamSet;
        StreamSetBuffer * const buffer = mInternalBuffers[j].release();
        assert (buffer == mBufferGraph[streamSet].Buffer);
        mStreamSetOutputBuffers[i].reset(buffer);
    }

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readAvailableItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readAvailableItemCounts(KernelBuilder & b) {

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = source(e, mBufferGraph);
        if (mLocallyAvailableItems[streamSet] == nullptr) {
            const BufferNode & bn = mBufferGraph[streamSet];
            Value * produced = nullptr;
            if (LLVM_UNLIKELY(bn.isConstant())) {
                produced = ConstantInt::getAllOnesValue(b.getSizeTy());
            } else {
                const auto f = in_edge(streamSet, mBufferGraph);
                const auto producer = source(f, mBufferGraph);
                const BufferPort & outputPort = mBufferGraph[f];
                assert (outputPort.Port.Type == PortType::Output);
                if (LLVM_UNLIKELY(producer == PipelineInput)) {
                    assert (bn.isExternal());
                    // the output port of the pipeline input is an input streamset of the pipeline kernel.
                    produced = getAvailableInputItems(outputPort.Port.Number);
                    writeTransitoryConsumedItemCount(b, streamSet, produced);
                } else {
                    const auto prefix = makeBufferName(producer, outputPort.Port);
                    if (LLVM_UNLIKELY(outputPort.isDeferred())) {
                        produced = b.getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
                    } else {
                        produced = b.getScalarField(prefix + ITEM_COUNT_SUFFIX);
                    }
                }
            }
            mLocallyAvailableItems[streamSet] = produced; assert (produced);
        }
    }
}



/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readProcessedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readProcessedItemCounts(KernelBuilder & b) {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto inputPort = br.Port;
        const auto prefix = makeBufferName(mKernelId, inputPort);
        const auto & suffix = (mCurrentKernelIsStateFree) ?
            STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX : ITEM_COUNT_SUFFIX;

        auto prodRef = b.getScalarFieldPtr(prefix + suffix);
        mProcessedItemCountPtr[inputPort] = prodRef.first;
        Value * itemCount = b.CreateLoad(prodRef.second, prodRef.first);
        mInitiallyProcessedItemCount[inputPort] = itemCount;
        if (br.isDeferred()) {
            auto defRef = b.getScalarFieldPtr(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
            mProcessedDeferredItemCountPtr[inputPort] = defRef.first;
            itemCount = b.CreateLoad(defRef.second, defRef.first);
            mInitiallyProcessedDeferredItemCount[inputPort] = itemCount;
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readProducedItemCounts(KernelBuilder & b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {

        const BufferPort & br = mBufferGraph[e];
        const auto outputPort = br.Port;
        const auto prefix = makeBufferName(mKernelId, outputPort);
        const auto & suffix = (mCurrentKernelIsStateFree) ?
            STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX : ITEM_COUNT_SUFFIX;

        auto prodRef = b.getScalarFieldPtr(prefix + suffix);
        mProducedItemCountPtr[outputPort] = prodRef.first;
        Value * const itemCount = b.CreateLoad(prodRef.second, prodRef.first);
        const auto streamSet = target(e, mBufferGraph);
        mInitiallyProducedItemCount[streamSet] = itemCount;
        if (br.isDeferred()) {
            auto defRef = b.getScalarFieldPtr(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
            mProducedDeferredItemCountPtr[outputPort] = defRef.first;
            Value * const itemCount = b.CreateLoad(defRef.second, defRef.first);
            mInitiallyProducedDeferredItemCount[streamSet] = itemCount;
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeUpdatedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeUpdatedItemCounts(KernelBuilder & b) {

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const StreamSetPort inputPort = br.Port;
//        const Binding & binding = br.Binding;

//        if (br.IsDeferred || isAddressable(binding)) {
////            if (LLVM_UNLIKELY(mKernelIsInternallySynchronized)) {
////                continue;
////            }
//        } else if (LLVM_UNLIKELY(!isCountable(binding))) {
//            continue;
//        }

//        const auto streamSet = source(e, mBufferGraph);

        Value * ptr = nullptr;
        if (mCurrentKernelIsStateFree) {
            const auto prefix = makeBufferName(mKernelId, inputPort);
            ptr = b.getScalarFieldPtr(prefix + ITEM_COUNT_SUFFIX).first;
        } else {
            ptr = mProcessedItemCountPtr[inputPort];
        }
        b.CreateStore(mUpdatedProcessedPhi[inputPort], ptr);
        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, inputPort);
        debugPrint(b, " @ writing " + prefix + "_processed = %" PRIu64, mUpdatedProcessedPhi[inputPort]);
        #endif
        if (br.isDeferred()) {
            assert (!mCurrentKernelIsStateFree);
            b.CreateStore(mUpdatedProcessedDeferredPhi[inputPort], mProcessedDeferredItemCountPtr[inputPort]);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, " @ writing " + prefix + "_processed(deferred) = %" PRIu64, mUpdatedProcessedDeferredPhi[inputPort]);
            #endif
        }
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const StreamSetPort outputPort = br.Port;
//        const Binding & binding = br.Binding;

//        if (br.IsDeferred || isAddressable(binding)) {
////            if (LLVM_UNLIKELY(mKernelIsInternallySynchronized)) {
////                continue;
////            }
//        } else if (LLVM_UNLIKELY(!isCountable(binding))) {
//            continue;
//        }

//        const auto streamSet = target(e, mBufferGraph);

        Value * ptr = nullptr;
        if (mCurrentKernelIsStateFree) {
            const auto prefix = makeBufferName(mKernelId, outputPort);
            ptr = b.getScalarFieldPtr(prefix + ITEM_COUNT_SUFFIX).first;
        } else {
            ptr = mProducedItemCountPtr[outputPort];
        }
        b.CreateStore(mUpdatedProducedPhi[outputPort], ptr);
        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, outputPort);
        debugPrint(b, " @ writing " + prefix + "_produced = %" PRIu64, mUpdatedProducedPhi[outputPort]);
        #endif
        if (br.isDeferred()) {
            assert (!mCurrentKernelIsStateFree);
            b.CreateStore(mUpdatedProducedDeferredPhi[outputPort], mProducedDeferredItemCountPtr[outputPort]);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, " @ writing " + prefix + "_produced(deferred) = %" PRIu64, mUpdatedProducedDeferredPhi[outputPort]);
            #endif
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordFinalProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordFinalProducedItemCounts(KernelBuilder & b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto outputPort = br.Port;
        Value * const fullyProduced = mFullyProducedItemCount[outputPort]; assert (fullyProduced);

        #ifdef PRINT_DEBUG_MESSAGES
        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        const auto prefix = makeBufferName(mKernelId, outputPort);
        out << " * -> " << prefix << "_avail = %" PRIu64;
        debugPrint(b, out.str(), fullyProduced);
        #endif

        const auto streamSet = target(e, mBufferGraph);
        mLocallyAvailableItems[streamSet] = fullyProduced;

        writeTransitoryConsumedItemCount(b, streamSet, fullyProduced);

        // update any external output port(s)
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isExternal())) {
            for (const auto f : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const BufferPort & external = mBufferGraph[f];
                Value * const ptr = getProducedOutputItemsPtr(external.Port.Number);
                b.CreateStore(mLocallyAvailableItems[streamSet], ptr);
            }
        }

        #ifdef PRINT_DEBUG_MESSAGES
        Value * const producedDelta = b.CreateSub(fullyProduced, mInitiallyProducedItemCount[streamSet]);
        debugPrint(b, prefix + "_producedΔ = %" PRIu64, producedDelta);
        #endif

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readReturnedOutputVirtualBaseAddresses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readReturnedOutputVirtualBaseAddresses(KernelBuilder & b) const {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & rd = mBufferGraph[e];
        assert (rd.Port.Type == PortType::Output);
        const StreamSetPort port(PortType::Output, rd.Port.Number);
        if (rd.isManaged()) {
            const auto streamSet = target(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            assert (bn.isNonThreadLocal());
            Value * const ptr = mReturnedOutputVirtualBaseAddressPtr[port]; assert (ptr);
            StreamSetBuffer * const buffer = bn.Buffer;
            Value * vba = b.CreateLoad(buffer->getPointerType(), ptr);
            buffer->setBaseAddress(b, vba);
//            if (CheckAssertions) {
//                b.CreateAssert(vba, "%s.%s returned virtual base addresss cannot be null",
//                                mCurrentKernelName, b.GetString(rd.Binding.get().getName()));
//            }
            buffer->setCapacity(b, mProducedItemCount[port]);
            const auto handleName = makeBufferName(mKernelId, port);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, "%s_updatedVirtualBaseAddress = 0x%" PRIx64, b.GetString(handleName), buffer->getBaseAddress(b));
            #endif
            b.setScalarField(handleName + LAST_GOOD_VIRTUAL_BASE_ADDRESS, vba);
        } else {
            assert (mReturnedOutputVirtualBaseAddressPtr[port] == nullptr);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadLastGoodVirtualBaseAddressesOfUnownedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::loadLastGoodVirtualBaseAddressesOfUnownedBuffers(KernelBuilder & b, const size_t kernelId) const {
    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        // owned or external buffers do not have a mutable vba
        if (LLVM_LIKELY(bn.isOwned() || bn.isExternal() || bn.hasZeroElementsOrWidth())) {
            continue;
        }
        assert (bn.isNonThreadLocal());
        const BufferPort & rd = mBufferGraph[e];
        const auto handleName = makeBufferName(kernelId, rd.Port);
        Value * const vba = b.getScalarField(handleName + LAST_GOOD_VIRTUAL_BASE_ADDRESS);
        StreamSetBuffer * const buffer = bn.Buffer;
        buffer->setBaseAddress(b, vba);
//        if (CheckAssertions) {
//            b.CreateAssert(vba, "%s.%s last good virtual base addresss cannot be null",
//                            mCurrentKernelName, b.GetString(rd.Binding.get().getName()));
//        }
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "%s_loadPriorVirtualBaseAddress = 0x%" PRIx64, b.GetString(handleName), buffer->getBaseAddress(b));
        #endif
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief assignThreadLocalBufferMemoryForPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::remapThreadLocalBufferMemory(KernelBuilder & b) {

    ConstantInt * const BLOCK_WIDTH = b.getSize(b.getBitBlockWidth());

    DataLayout DL(b.getModule());

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isThreadLocal()) {
            assert (!bn.isTruncated());
            assert (RequiredThreadLocalStreamSetMemory > 0);
            assert (mThreadLocalStreamSetBaseAddress);
            assert (mThreadLocalStreamSetBaseAddress->getType() == b.getInt8PtrTy());
            auto start = bn.BufferStart;
            assert ((start % b.getCacheAlignment()) == 0);
            #ifdef THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER
            start *= THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER;
            #endif

            assert (mThreadLocalScalingFactor);
            Value * const startOffset = b.CreateMul(mThreadLocalScalingFactor, b.getSize(start));

            ExternalBuffer * const buffer = cast<ExternalBuffer>(bn.Buffer);
            Value * const produced = mInitiallyProducedItemCount[streamSet];
            PointerType * const ptrTy = buffer->getPointerType();

            Constant * const bytesPerPack = b.getTypeSize(buffer->getType());
            Value * const producedBytes = b.CreateMul(b.CreateUDiv(produced, BLOCK_WIDTH), bytesPerPack);

            Value * const offset = b.CreateSub(startOffset, producedBytes);
            Value * ba = b.CreateGEP(b.getInt8Ty(), mThreadLocalStreamSetBaseAddress, offset);
            ba = b.CreatePointerCast(ba, ptrTy);
            buffer->setBaseAddress(b, ba);

        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getVirtualBaseAddress
 *
 * Returns the address of the "zeroth" item of the (logically-unbounded) stream set.
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getVirtualBaseAddress(KernelBuilder & b,
                                                const BufferPort & rateData,
                                                const BufferNode & bufferNode,
                                                Value * position,
                                                const bool prefetch,
                                                const bool write) const {

    const StreamSetBuffer * const buffer = bufferNode.Buffer;
    assert ("buffer cannot be null!" && buffer);
    assert (isFromCurrentFunction(b, buffer->getHandle()));
    assert (position);

    Value * const baseAddress = buffer->getBaseAddress(b);
    if (bufferNode.isUnowned() || bufferNode.hasZeroElementsOrWidth()) {
        assert (bufferNode.isNonThreadLocal());
        assert (!bufferNode.isConstant());
        assert (!bufferNode.isTruncated());
        return baseAddress;
    }

    Value * const addr = buffer->getVirtualBasePtr(b, baseAddress, position);
    if (prefetch) {
        ExternalBuffer tmp(0, b, buffer->getBaseType(), true, buffer->getAddressSpace());
        Constant * const LOG_2_BLOCK_WIDTH = b.getSize(floor_log2(b.getBitBlockWidth()));
        Value * const blockIndex = b.CreateLShr(position, LOG_2_BLOCK_WIDTH);
        Value * const prefetchAddr = tmp.getStreamBlockPtr(b, addr, b.getSize(0), blockIndex);
        prefetchAtLeastThreeCacheLinesFrom(b, prefetchAddr, write);
    }
    return addr;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief prefetchThreeCacheLinesFrom
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::prefetchAtLeastThreeCacheLinesFrom(KernelBuilder & b, Value * const addr, const bool write) const {
#if 0
    Module * const m = b.getModule();
    Function * const prefetchFunc = Intrinsic::getDeclaration(m, Intrinsic::prefetch);

    DataLayout dl(m);
    Type * const elemTy = addr->getType()->getPointerElementType();
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
    const auto typeSize = dl.getTypeAllocSize(elemTy);
    #else
    const auto typeSize = dl.getTypeAllocSize(elemTy).getFixedSize();
    #endif
    assert (typeSize > 0);

    IntegerType * const int32Ty = b.getInt32Ty();
    FixedArray<Value *, 4> args;
    args[1] = ConstantInt::get(int32Ty, write ? 1 : 0); // write flag
    args[2] = ConstantInt::get(int32Ty, 3); // locality
    args[3] = ConstantInt::get(int32Ty, 1); // cache type?

    const auto cl = b.getCacheAlignment();
    const auto toFetch = round_up_to<unsigned>(cl * 3, typeSize);
    Value * const baseAddr = b.CreatePointerCast(addr, b.getInt8PtrTy());
    for (unsigned i = 0; i < toFetch; i += cl) {
        args[0] = b.CreateGEP0(baseAddr, b.getSize(i));
        b.CreateCall(prefetchFunc->getFunctionType(), prefetchFunc, args);
    }
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputVirtualBaseAddresses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::getInputVirtualBaseAddresses(KernelBuilder & b, Vec<Value *> & baseAddresses) const {
    for (const auto input : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & inputPort = mBufferGraph[input];
        PHINode * processed = nullptr;
        if (mCurrentProcessedDeferredItemCountPhi[inputPort.Port]) {
            processed = mCurrentProcessedDeferredItemCountPhi[inputPort.Port];
        } else {
            processed = mCurrentProcessedItemCountPhi[inputPort.Port];
        }
        const auto streamSet = source(input, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];

        if (LLVM_UNLIKELY(bn.isUnowned() && bn.isInternal())) {
            const auto output = in_edge(streamSet, mBufferGraph);
            const auto producer = source(output, mBufferGraph);
            assert (producer < mKernelId);
            const BufferPort & outputPort = mBufferGraph[output];
            const auto handleName = makeBufferName(producer, outputPort.Port);
            Value * const vba = b.getScalarField(handleName + LAST_GOOD_VIRTUAL_BASE_ADDRESS);
            bn.Buffer->setBaseAddress(b, vba);
        }

        Value * addr = getVirtualBaseAddress(b, inputPort, bn, processed, bn.isNonThreadLocal(), false);
        baseAddresses[inputPort.Port.Number] = addr;
    }
}


} // end of kernel namespace
