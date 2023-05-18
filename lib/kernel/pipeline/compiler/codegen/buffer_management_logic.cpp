#include "../pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addHandlesToPipelineKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addBufferHandlesToPipelineKernel(BuilderRef b, const unsigned kernelId, const unsigned groupId) {

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
            if (LLVM_UNLIKELY(isa<RepeatingStreamSet>(buffer))) {
                if (cast<RepeatingStreamSet>(buffer)->isDynamic()) {
                    mTarget->addInternalScalar(handleType, prefix, groupId);
                } else {
                    mTarget->addNonPersistentScalar(handleType, prefix);
                }
            } else if (bn.Locality == BufferLocality::ThreadLocal) {
                hasAnyInternalStreamSets = true;
                mTarget->addNonPersistentScalar(handleType, prefix);
            } else if (LLVM_LIKELY(bn.isOwned())) {
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
            assert (bn.Locality != BufferLocality::ThreadLocal);
            mTarget->addThreadLocalScalar(b->getVoidPtrTy(), prefix + PENDING_FREEABLE_BUFFER_ADDRESS, groupId);
        }
    }

    if (LLVM_UNLIKELY(!mTarget->allocatesInternalStreamSets())) {
        const Kernel * const kernelObj = getKernel(kernelId);
        if (LLVM_UNLIKELY(hasAnyInternalStreamSets)) {
            SmallVector<char, 1024> tmp;
            raw_svector_ostream msg(tmp);
            msg << "Pipeline " << mTarget->getName() << " is not marked as allocating internal streamsets"
            << " but must do so to support " << kernelObj->getName() << ".";
            report_fatal_error(msg.str());
        }
        if (LLVM_UNLIKELY(kernelObj->allocatesInternalStreamSets())) {
            SmallVector<char, 1024> tmp;
            raw_svector_ostream msg(tmp);
            msg << "Pipeline " << mTarget->getName() << " is not marked as allocating internal streamsets"
            << " but " << kernelObj->getName() << " must do so to be correctly initialized.";
            report_fatal_error(msg.str());
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadInternalStreamSetHandles
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::loadInternalStreamSetHandles(BuilderRef b, const bool nonLocal) {
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
                Value * const handle = b->getScalarFieldPtr(handleName);
                buffer->setHandle(handle);
                const auto & sn = mStreamGraph[streamSet];
                assert (sn.Type == RelationshipNode::IsStreamSet);
                if (cast<RepeatingStreamSet>(sn.Relationship)->isDynamic()) {
                    const auto lengthName = REPEATING_STREAMSET_LENGTH_PREFIX + std::to_string(streamSet);
                    Value * const mod = b->getScalarField(lengthName);
                    cast<RepeatingBuffer>(buffer)->setModulus(mod);
                } else {
                    assert(isa<Constant>(cast<RepeatingBuffer>(buffer)->getModulus()));
                }
            } else {
                const auto pe = in_edge(streamSet, mBufferGraph);
                const auto producer = source(pe, mBufferGraph);
                const BufferPort & rd = mBufferGraph[pe];
                const auto handleName = makeBufferName(producer, rd.Port);
                Value * const handle = b->getScalarFieldPtr(handleName);
                buffer->setHandle(handle);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief allocateOwnedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::allocateOwnedBuffers(BuilderRef b, Value * const expectedNumOfStrides, const bool nonLocal) {
    assert (expectedNumOfStrides);
    if (LLVM_UNLIKELY(CheckAssertions)) {
        Value * const valid = b->CreateIsNotNull(expectedNumOfStrides);
        b->CreateAssert(valid,
           "%s: expected number of strides for internally allocated buffers is 0",
           b->GetString(mTarget->getName()));
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
                if (nonLocal) {
                    func = getKernelAllocateSharedInternalStreamSetsFunction(b);
                } else {
                    func = getKernelAllocateThreadLocalInternalStreamSetsFunction(b);
                    params.push_back(mKernelThreadLocalHandle);
                }

                const auto scale = MaximumNumOfStrides[i] * Rational{mNumOfThreads};
                params.push_back(b->CreateCeilUMulRational(expectedNumOfStrides, scale));

                FunctionType * const funcType = cast<FunctionType>(func->getType()->getPointerElementType());

                b->CreateCall(funcType, func, params);
            }
        }
    }

    // and allocate any output buffers
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isTruncated())) continue;
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
                    Value * const handle = b->getScalarFieldPtr(handleName);
                    buffer->setHandle(handle);
                } else {
                    assert (isFromCurrentFunction(b, buffer->getHandle(), false));
                }
                if (nonLocal) {
                    buffer->allocateBuffer(b, expectedNumOfStrides);

                    #ifdef PRINT_DEBUG_MESSAGES
                    const auto pe = in_edge(streamSet, mBufferGraph);
                    const auto producer = source(pe, mBufferGraph);
                    const BufferPort & rd = mBufferGraph[pe];
                    const auto prefix = makeBufferName(producer, rd.Port);
                    debugPrint(b, prefix + ".inital malloc range = [%" PRIx64 ",%" PRIx64 ")",
                               buffer->getMallocAddress(b), buffer->getOverflowAddress(b));
                    #endif

                }
            }
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief releaseOwnedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::releaseOwnedBuffers(BuilderRef b) {
    loadInternalStreamSetHandles(b, true);
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (LLVM_UNLIKELY(bn.isTruncated())) continue;
        if (bn.isNonThreadLocal() && bn.isOwned() && !bn.isReturned() && !bn.isConstant()) {
            StreamSetBuffer * const buffer = bn.Buffer;
            assert (isFromCurrentFunction(b, buffer->getHandle(), false));
            buffer->releaseBuffer(b);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief freePendingFreeableDynamicBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::freePendingFreeableDynamicBuffers(BuilderRef b) {
    if (LLVM_LIKELY(isMultithreaded())) {
        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isTruncated())) continue;
            if (bn.isNonThreadLocal() && bn.isOwned()) {
                StreamSetBuffer * const buffer = bn.Buffer;
                if (LLVM_LIKELY(isa<DynamicBuffer>(buffer))) {
                    const auto pe = in_edge(streamSet, mBufferGraph);
                    const auto p = source(pe, mBufferGraph);
                    const BufferPort & rd = mBufferGraph[pe];
                    assert (rd.Port.Type == PortType::Output);
                    const auto prefix = makeBufferName(p, rd.Port);
                    b->CreateFree(b->getScalarField(prefix + PENDING_FREEABLE_BUFFER_ADDRESS));
                }
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateExternalPipelineIO
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateExternalPipelineIO(BuilderRef b) {
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
                itemCount = b->getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
            } else {
                itemCount = b->getScalarField(prefix + ITEM_COUNT_SUFFIX);
            }

            assert (isFromCurrentFunction(b, itemCount, false));
            assert (isFromCurrentFunction(b, mProducedOutputItemPtr[k], false));

            assert (mProducedOutputItemPtr[k]->getType()->isPointerTy());
            b->CreateStore(itemCount, mProducedOutputItemPtr[k]);
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
void PipelineCompiler::constructStreamSetBuffers(BuilderRef /* b */) {

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
void PipelineCompiler::readAvailableItemCounts(BuilderRef b) {

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = source(e, mBufferGraph);
        if (mLocallyAvailableItems[streamSet] == nullptr) {
            const BufferNode & bn = mBufferGraph[streamSet];
            Value * produced = nullptr;
            if (LLVM_UNLIKELY(bn.isConstant())) {
                produced = ConstantInt::getAllOnesValue(b->getSizeTy());
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
                        produced = b->getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
                    } else {
                        produced = b->getScalarField(prefix + ITEM_COUNT_SUFFIX);
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
void PipelineCompiler::readProcessedItemCounts(BuilderRef b) {
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto inputPort = br.Port;
        const auto prefix = makeBufferName(mKernelId, inputPort);

        const auto streamSet = source(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        const auto & suffix = (mCurrentKernelIsStateFree &&  bn.isInternal()) ?
            STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX : ITEM_COUNT_SUFFIX;

        Value * const processedPtr = b->getScalarFieldPtr(prefix + suffix);
        mProcessedItemCountPtr[inputPort] = processedPtr;
        Value * itemCount = b->CreateLoad(processedPtr);
        mInitiallyProcessedItemCount[inputPort] = itemCount;
        if (br.isDeferred()) {
            Value * const deferred = b->getScalarFieldPtr(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
            mProcessedDeferredItemCountPtr[inputPort] = deferred;
            itemCount = b->CreateLoad(deferred);
            mInitiallyProcessedDeferredItemCount[inputPort] = itemCount;
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readProducedItemCounts(BuilderRef b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {

        const BufferPort & br = mBufferGraph[e];
        const auto outputPort = br.Port;
        const auto prefix = makeBufferName(mKernelId, outputPort);

        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        const auto & suffix = (mCurrentKernelIsStateFree &&  bn.isInternal()) ?
            STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX : ITEM_COUNT_SUFFIX;

        Value * const produced = b->getScalarFieldPtr(prefix + suffix);
        mProducedItemCountPtr[outputPort] = produced;
        Value * const itemCount = b->CreateLoad(produced);
        mInitiallyProducedItemCount[streamSet] = itemCount;
        if (br.isDeferred()) {
            Value * const deferred = b->getScalarField(prefix + DEFERRED_ITEM_COUNT_SUFFIX);
            mProducedDeferredItemCountPtr[outputPort] = deferred;
            Value * const itemCount = b->CreateLoad(deferred);
            mInitiallyProducedDeferredItemCount[streamSet] = itemCount;
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeUpdatedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeUpdatedItemCounts(BuilderRef b) {

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

        const auto streamSet = source(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];

        Value * ptr = nullptr;
        if (mCurrentKernelIsStateFree) {
            const auto prefix = makeBufferName(mKernelId, inputPort);
            ptr = b->getScalarFieldPtr(prefix + ITEM_COUNT_SUFFIX);
        } else {
            ptr = mProcessedItemCountPtr[inputPort];
        }
        b->CreateStore(mUpdatedProcessedPhi[inputPort], ptr);
        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, inputPort);
        debugPrint(b, " @ writing " + prefix + "_processed = %" PRIu64, mUpdatedProcessedPhi[inputPort]);
        #endif
        if (br.isDeferred()) {
            assert (!mCurrentKernelIsStateFree);
            b->CreateStore(mUpdatedProcessedDeferredPhi[inputPort], mProcessedDeferredItemCountPtr[inputPort]);
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

        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];

        Value * ptr = nullptr;
        if (mCurrentKernelIsStateFree) {
            const auto prefix = makeBufferName(mKernelId, outputPort);
            ptr = b->getScalarFieldPtr(prefix + ITEM_COUNT_SUFFIX);
        } else {
            ptr = mProducedItemCountPtr[outputPort];
        }
        b->CreateStore(mUpdatedProducedPhi[outputPort], ptr);
        #ifdef PRINT_DEBUG_MESSAGES
        const auto prefix = makeBufferName(mKernelId, outputPort);
        debugPrint(b, " @ writing " + prefix + "_produced = %" PRIu64, mUpdatedProducedPhi[outputPort]);
        #endif
        if (br.isDeferred()) {
            assert (!mCurrentKernelIsStateFree);
            b->CreateStore(mUpdatedProducedDeferredPhi[outputPort], mProducedDeferredItemCountPtr[outputPort]);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, " @ writing " + prefix + "_produced(deferred) = %" PRIu64, mUpdatedProducedDeferredPhi[outputPort]);
            #endif
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordFinalProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordFinalProducedItemCounts(BuilderRef b) {
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
                b->CreateStore(mLocallyAvailableItems[streamSet], ptr);
            }
        }

        #ifdef PRINT_DEBUG_MESSAGES
        Value * const producedDelta = b->CreateSub(fullyProduced, mInitiallyProducedItemCount[streamSet]);
        debugPrint(b, prefix + "_producedΔ = %" PRIu64, producedDelta);
        #endif

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readReturnedOutputVirtualBaseAddresses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readReturnedOutputVirtualBaseAddresses(BuilderRef b) const {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & rd = mBufferGraph[e];
        assert (rd.Port.Type == PortType::Output);
        const StreamSetPort port(PortType::Output, rd.Port.Number);
        if (rd.isManaged()) {
            const auto streamSet = target(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            assert (bn.isNonThreadLocal());
            Value * const ptr = mReturnedOutputVirtualBaseAddressPtr[port]; assert (ptr);
            Value * vba = b->CreateLoad(ptr);
            StreamSetBuffer * const buffer = bn.Buffer;
            vba = b->CreatePointerCast(vba, buffer->getPointerType());
            buffer->setBaseAddress(b.get(), vba);
//            if (CheckAssertions) {
//                b->CreateAssert(vba, "%s.%s returned virtual base addresss cannot be null",
//                                mCurrentKernelName, b->GetString(rd.Binding.get().getName()));
//            }
            buffer->setCapacity(b.get(), mProducedItemCount[port]);
            const auto handleName = makeBufferName(mKernelId, port);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, "%s_updatedVirtualBaseAddress = 0x%" PRIx64, b->GetString(handleName), buffer->getBaseAddress(b));
            #endif
            b->setScalarField(handleName + LAST_GOOD_VIRTUAL_BASE_ADDRESS, vba);
        } else {
            assert (mReturnedOutputVirtualBaseAddressPtr[port] == nullptr);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief loadLastGoodVirtualBaseAddressesOfUnownedBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::loadLastGoodVirtualBaseAddressesOfUnownedBuffers(BuilderRef b, const size_t kernelId) const {
    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        // owned or external buffers do not have a mutable vba
        if (LLVM_LIKELY(bn.isOwned() || bn.isExternal())) {
            continue;
        }
        assert (bn.Locality != BufferLocality::ThreadLocal);
        const BufferPort & rd = mBufferGraph[e];
        const auto handleName = makeBufferName(kernelId, rd.Port);
        Value * const vba = b->getScalarField(handleName + LAST_GOOD_VIRTUAL_BASE_ADDRESS);
        StreamSetBuffer * const buffer = bn.Buffer;
        buffer->setBaseAddress(b.get(), vba);
//        if (CheckAssertions) {
//            b->CreateAssert(vba, "%s.%s last good virtual base addresss cannot be null",
//                            mCurrentKernelName, b->GetString(rd.Binding.get().getName()));
//        }
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "%s_loadPriorVirtualBaseAddress = 0x%" PRIx64, b->GetString(handleName), buffer->getBaseAddress(b));
        #endif
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeLookBehindLogic
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeLookBehindLogic(BuilderRef b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        const StreamSetBuffer * const buffer = bn.Buffer;
        if (bn.LookBehind) {
            const BufferPort & br = mBufferGraph[e];
            Constant * const underflow = b->getSize(bn.LookBehind);
            Value * const produced = mCurrentProducedItemCountPhi[br.Port];
            Value * const capacity = buffer->getCapacity(b);
            Value * const producedOffset = b->CreateURem(produced, capacity);
            Value * const needsCopy = b->CreateICmpULE(producedOffset, underflow);
            #ifdef PRINT_DEBUG_MESSAGES
            const auto handleName = makeBufferName(mKernelId, br.Port);
            debugPrint(b, "%s_needsLookBehind = %" PRIx8, b->GetString(handleName), needsCopy);
            #endif
            copy(b, CopyMode::LookBehind, needsCopy, br.Port, buffer, bn.LookBehind);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeLookBehindReflectionLogic
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeDelayReflectionLogic(BuilderRef b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        if (br.Delay) {
            const auto streamSet = target(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            const StreamSetBuffer * const buffer = bn.Buffer;
            Value * const capacity = buffer->getCapacity(b);
            Value * const produced = mCurrentProducedItemCountPhi[br.Port];
            const auto size = round_up_to(br.Delay, b->getBitBlockWidth());
            Constant * const reflection = b->getSize(size);
            Value * const producedOffset = b->CreateURem(produced, capacity);
            Value * const needsCopy = b->CreateICmpULT(producedOffset, reflection);
            copy(b, CopyMode::Delay, needsCopy, br.Port, buffer, br.Delay);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeCopyBackLogic
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeCopyBackLogic(BuilderRef b) {
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.CopyBack) {
            const StreamSetBuffer * const buffer = bn.Buffer;
            const BufferPort & br = mBufferGraph[e];
            Value * const capacity = buffer->getCapacity(b);
            Value * const alreadyProduced = mCurrentProducedItemCountPhi[br.Port];
            Value * const priorOffset = b->CreateURem(alreadyProduced, capacity);
            Value * const produced = mProducedItemCount[br.Port];
            Value * const producedOffset = b->CreateURem(produced, capacity);
            Value * const nonCapacityAlignedWrite = b->CreateIsNotNull(producedOffset);
            Value * const wroteToOverflow = b->CreateICmpULT(producedOffset, priorOffset);
            Value * const needsCopy = b->CreateAnd(nonCapacityAlignedWrite, wroteToOverflow);
            #ifdef PRINT_DEBUG_MESSAGES
            const auto handleName = makeBufferName(mKernelId, br.Port);
            debugPrint(b, "%s_needsCopyBack = %" PRIx8, b->GetString(handleName), needsCopy);
            #endif
            copy(b, CopyMode::CopyBack, needsCopy, br.Port, buffer, bn.CopyBack);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeLookAheadLogic
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeLookAheadLogic(BuilderRef b) {
    // Unless we modified the portion of data that ought to be reflected in the overflow region, do not copy
    // any data. To do so would incur extra writes and pollute the cache with potentially unnecessary data.
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.CopyForwards) {

            const StreamSetBuffer * const buffer = bn.Buffer;
            const BufferPort & br = mBufferGraph[e];
            Value * const capacity = buffer->getCapacity(b);
            Value * const initial = mInitiallyProducedItemCount[streamSet];
            Value * const produced = mUpdatedProducedPhi[br.Port];

            // If we wrote anything and it was not our first write to the buffer ...
            Value * overwroteData = b->CreateICmpUGT(produced, capacity);
            const Binding & output = getOutputBinding(br.Port);
            const ProcessingRate & rate = output.getRate();
            const Rational ONE(1, 1);
            bool mayProduceZeroItems = false;
            if (rate.getLowerBound() < ONE) {
                mayProduceZeroItems = true;
            } else if (rate.isRelative()) {
                const Binding & ref = getBinding(getReference(br.Port));
                const ProcessingRate & refRate = ref.getRate();
                mayProduceZeroItems = (rate.getLowerBound() * refRate.getLowerBound()) < ONE;
            }
            if (mayProduceZeroItems) {
                Value * const producedAnyOutput = b->CreateICmpNE(initial, produced);
                overwroteData = b->CreateAnd(overwroteData, producedAnyOutput);
            }

            // And we started writing within the first block ...
            Constant * const overflowSize = b->getSize(bn.CopyForwards);
            Value * const initialOffset = b->CreateURem(initial, capacity);
            Value * const startedWithinFirstBlock = b->CreateICmpULT(initialOffset, overflowSize);
            Value * const wroteToFirstBlock = b->CreateAnd(overwroteData, startedWithinFirstBlock);

            // And we started writing at the end of the buffer but wrapped over to the start of it,
            Value * const producedOffset = b->CreateURem(produced, capacity);
            Value * const wroteFromEndToStart = b->CreateICmpULT(producedOffset, initialOffset);

            // Then mirror the data in the overflow region.
            Value * const needsCopy = b->CreateOr(wroteToFirstBlock, wroteFromEndToStart);

            // TODO: optimize this further to ensure that we don't copy data that was just copied back from
            // the overflow. Should be enough just to have a "copyback flag" phi node to say it that was the
            // last thing it did to the buffer.
            #ifdef PRINT_DEBUG_MESSAGES
            const auto handleName = makeBufferName(mKernelId, br.Port);
            debugPrint(b, "%s_needsLookAhead = %" PRIx8, b->GetString(handleName), needsCopy);
            #endif
            copy(b, CopyMode::LookAhead, needsCopy, br.Port, buffer, bn.CopyForwards);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeOverflowCopy
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::copy(BuilderRef b, const CopyMode mode, Value * cond,
                            const StreamSetPort outputPort, const StreamSetBuffer * const buffer,
                            const unsigned itemsToCopy) {
    auto makeSuffix = [](CopyMode mode) {
        switch (mode) {
            case CopyMode::LookAhead: return "LookAhead";
            case CopyMode::CopyBack: return "CopyBack";
            case CopyMode::LookBehind: return "LookBehind";
            case CopyMode::Delay: return "Delay";
        }
        llvm_unreachable("unknown copy mode!");
    };

    const auto prefix = makeBufferName(mKernelId, outputPort) + "_" + makeSuffix(mode);

    const auto itemWidth = getItemWidth(buffer->getBaseType());
    assert (is_power_2(itemWidth));
    const auto blockWidth = b->getBitBlockWidth();

    const auto bitsToCopy = round_up_to(itemsToCopy * itemWidth, blockWidth);
    const auto bitsPerStream = round_up_to(itemsToCopy, blockWidth) * itemWidth;

    Value * const numOfStreams = buffer->getStreamSetCount(b);
    ConstantInt * const bytesToCopy = b->getSize(bitsToCopy / 8);

    BasicBlock * const copyStart = b->CreateBasicBlock(prefix, mKernelExit);
    BasicBlock * copyLoop = nullptr;
    if ((bitsToCopy < bitsPerStream) && !(isa<ConstantInt>(numOfStreams) && cast<ConstantInt>(numOfStreams)->isOne())) {
        copyLoop = b->CreateBasicBlock(prefix + "Loop", mKernelExit);
    }
    BasicBlock * const copyExit = b->CreateBasicBlock(prefix + "Exit", mKernelExit);

    b->CreateUnlikelyCondBr(cond, copyStart, copyExit);

    b->SetInsertPoint(copyStart);
    #ifdef ENABLE_PAPI
    readPAPIMeasurement(b, mKernelId, PAPIReadBeforeMeasurementArray);
    #endif
    Value * const beforeCopy = startCycleCounter(b);

    Value * source = buffer->getOverflowAddress(b);
    Value * target = buffer->getMallocAddress(b);

    PointerType * const int8PtrTy = b->getInt8PtrTy();
    source = b->CreatePointerCast(source, int8PtrTy);
    target = b->CreatePointerCast(target, int8PtrTy);

    ConstantInt * const bytesPerStream = b->getSize(bitsPerStream / 8);

    Value * const totalBytesPerStreamSetBlock = b->CreateMul(bytesPerStream, numOfStreams);

    if (mode == CopyMode::LookBehind || mode == CopyMode::Delay) {
        Value * const offset = b->CreateNeg(totalBytesPerStreamSetBlock);
        source = b->CreateInBoundsGEP(source, offset);
        target = b->CreateInBoundsGEP(target, offset);
    }

    if (mode == CopyMode::LookAhead || mode == CopyMode::Delay) {
        std::swap(target, source);
    }

    assert (bitsToCopy >= blockWidth);

    const auto align = blockWidth / 8;

    if (copyLoop) {

        BasicBlock * recordCopyCycleCount = nullptr;
        if (EnableCycleCounter || EnablePAPICounters) {
            recordCopyCycleCount = b->CreateBasicBlock(prefix + "RecordCycleCount", copyExit);
        }

        b->CreateBr(copyLoop);

        b->SetInsertPoint(copyLoop);
        PHINode * const idx = b->CreatePHI(b->getSizeTy(), 2);
        idx->addIncoming(b->getSize(0), copyStart);
        Value * const offset = b->CreateMul(idx, bytesPerStream);
        Value * const sourcePtr = b->CreateGEP(source, offset);
        Value * const targetPtr = b->CreateGEP(target, offset);

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, prefix + "_copying %" PRIu64 " bytes from %" PRIx64 " to %" PRIx64 " (align=%" PRIu64 ")", bytesToCopy, sourcePtr, targetPtr, b->getSize(align));
        #endif

        b->CreateMemCpy(targetPtr, sourcePtr, bytesToCopy, align);

        Value * const nextIdx = b->CreateAdd(idx, b->getSize(1));
        idx->addIncoming(nextIdx, copyLoop);
        Value * const done = b->CreateICmpEQ(nextIdx, numOfStreams);

        BasicBlock * const loopExit = EnableCycleCounter ? recordCopyCycleCount : copyExit;
        b->CreateCondBr(done, loopExit, copyLoop);

        if (EnableCycleCounter || EnablePAPICounters) {
            b->SetInsertPoint(recordCopyCycleCount);
            updateCycleCounter(b, mKernelId, beforeCopy, CycleCounter::BUFFER_COPY);
            #ifdef ENABLE_PAPI
            accumPAPIMeasurementWithoutReset(b, PAPIReadBeforeMeasurementArray, mKernelId, PAPIKernelCounter::PAPI_BUFFER_COPY);
            #endif
            b->CreateBr(copyExit);
        }

    } else {

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, prefix + "_segment_copying %" PRIu64 "x%" PRIu64 "=%" PRIu64 " bytes "
                      "from %" PRIx64 " to %" PRIx64 " (align=%" PRIu64 ")",
                   bytesPerStream, numOfStreams, totalBytesPerStreamSetBlock, source, target, b->getSize(align));
        #endif

        b->CreateMemCpy(target, source, totalBytesPerStreamSetBlock, align);
        if (EnableCycleCounter || EnablePAPICounters) {
            updateCycleCounter(b, mKernelId, beforeCopy, CycleCounter::BUFFER_COPY);
            #ifdef ENABLE_PAPI
            accumPAPIMeasurementWithoutReset(b, PAPIReadBeforeMeasurementArray, mKernelId, PAPIKernelCounter::PAPI_BUFFER_COPY);
            #endif
        }
        b->CreateBr(copyExit);

    }

    b->SetInsertPoint(copyExit);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief assignThreadLocalBufferMemoryForPartition
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::remapThreadLocalBufferMemory(BuilderRef b) {

    DataLayout DL(b->getModule());

    IntegerType * const int8Ty = b->getInt8Ty();

    auto getTypeSize = [&](Type * const type) -> uint64_t {
        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
        return DL.getTypeAllocSize(type);
        #else
        return DL.getTypeAllocSize(type).getFixedSize();
        #endif
    };

    ConstantInt * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isThreadLocal()) {
            assert (!bn.isTruncated());
            assert (RequiredThreadLocalStreamSetMemory > 0);
            assert (mThreadLocalStreamSetBaseAddress);
            assert (mThreadLocalStreamSetBaseAddress->getType()->getPointerElementType() == int8Ty);
            auto start = bn.BufferStart;
            assert ((start % b->getCacheAlignment()) == 0);
            #ifdef THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER
            start *= THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER;
            #endif

            assert (mThreadLocalScalingFactor);
            Value * const startOffset = b->CreateMul(mThreadLocalScalingFactor, b->getSize(start));

            ExternalBuffer * const buffer = cast<ExternalBuffer>(bn.Buffer);
            Value * const produced = mInitiallyProducedItemCount[streamSet];
            PointerType * const ptrTy = buffer->getPointerType();
            Constant * const bytesPerPack = ConstantExpr::getSizeOf(ptrTy->getElementType());
            Value * const producedBytes = b->CreateMul(b->CreateUDiv(produced, BLOCK_WIDTH), bytesPerPack);

            Value * const offset = b->CreateSub(startOffset, producedBytes);
            Value * ba = b->CreateGEP(mThreadLocalStreamSetBaseAddress, offset);
            ba = b->CreatePointerCast(ba, ptrTy);
            buffer->setBaseAddress(b, ba);

        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getVirtualBaseAddress
 *
 * Returns the address of the "zeroth" item of the (logically-unbounded) stream set.
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::getVirtualBaseAddress(BuilderRef b,
                                                const BufferPort & rateData,
                                                const BufferNode & bufferNode,
                                                Value * position,
                                                const bool prefetch,
                                                const bool write) const {

    const StreamSetBuffer * const buffer = bufferNode.Buffer;
    assert ("buffer cannot be null!" && buffer);
    assert (isFromCurrentFunction(b, buffer->getHandle()));


    Value * const baseAddress = buffer->getBaseAddress(b);

    if (bufferNode.isUnowned()) {
        assert (bufferNode.Locality != BufferLocality::ThreadLocal);
        assert (!bufferNode.isConstant());
        return baseAddress;
    }

    Constant * const LOG_2_BLOCK_WIDTH = b->getSize(floor_log2(b->getBitBlockWidth()));
    Constant * const ZERO = b->getSize(0);
    PointerType * const bufferType = buffer->getPointerType();
    Value * const blockIndex = b->CreateLShr(position, LOG_2_BLOCK_WIDTH);

    Value * const address = buffer->getStreamLogicalBasePtr(b, baseAddress, ZERO, blockIndex);
    Value * const addr = b->CreatePointerCast(address, bufferType);
    if (prefetch) {
        Value * const prefetchAddr = buffer->getStreamBlockPtr(b, addr, ZERO, blockIndex);
        prefetchAtLeastThreeCacheLinesFrom(b, prefetchAddr, write);
    }
    return addr;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief prefetchThreeCacheLinesFrom
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::prefetchAtLeastThreeCacheLinesFrom(BuilderRef b, Value * const addr, const bool write) const {
#if 0
    Module * const m = b->getModule();
    Function * const prefetchFunc = Intrinsic::getDeclaration(m, Intrinsic::prefetch);

    DataLayout dl(m);
    Type * const elemTy = addr->getType()->getPointerElementType();
    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
    const auto typeSize = dl.getTypeAllocSize(elemTy);
    #else
    const auto typeSize = dl.getTypeAllocSize(elemTy).getFixedSize();
    #endif
    assert (typeSize > 0);

    IntegerType * const int32Ty = b->getInt32Ty();
    FixedArray<Value *, 4> args;
    args[1] = ConstantInt::get(int32Ty, write ? 1 : 0); // write flag
    args[2] = ConstantInt::get(int32Ty, 3); // locality
    args[3] = ConstantInt::get(int32Ty, 1); // cache type?

    const auto cl = b->getCacheAlignment();
    const auto toFetch = round_up_to<unsigned>(cl * 3, typeSize);
    Value * const baseAddr = b->CreatePointerCast(addr, b->getInt8PtrTy());
    for (unsigned i = 0; i < toFetch; i += cl) {
        args[0] = b->CreateGEP(baseAddr, b->getSize(i));
        b->CreateCall(prefetchFunc->getFunctionType(), prefetchFunc, args);
    }
#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getInputVirtualBaseAddresses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::getInputVirtualBaseAddresses(BuilderRef b, Vec<Value *> & baseAddresses) const {
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

        assert ((bn.Locality != BufferLocality::ConstantShared) ^ isa<RepeatingBuffer>(bn.Buffer));

        if (LLVM_UNLIKELY(bn.isUnowned() && bn.isInternal())) {
            const auto output = in_edge(streamSet, mBufferGraph);
            const auto producer = source(output, mBufferGraph);
            assert (producer < mKernelId);
            const BufferPort & outputPort = mBufferGraph[output];
            const auto handleName = makeBufferName(producer, outputPort.Port);
            Value * const vba = b->getScalarField(handleName + LAST_GOOD_VIRTUAL_BASE_ADDRESS);
            bn.Buffer->setBaseAddress(b.get(), vba);
        }

        Value * addr = getVirtualBaseAddress(b, inputPort, bn, processed, bn.isNonThreadLocal(), false);
        baseAddresses[inputPort.Port.Number] = addr;
    }
}

} // end of kernel namespace
