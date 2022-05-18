#ifndef KERNEL_EXECUTION_LOGIC_HPP
#define KERNEL_EXECUTION_LOGIC_HPP

#include "pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeKernelCall
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeKernelCall(BuilderRef b) {

    // TODO: add MProtect to buffers and their handles.

    // TODO: send in the # of output items we want in the external buffers

    Value * const doSegment = getKernelDoSegmentFunction(b);

    FunctionType * const doSegFuncType = cast<FunctionType>(doSegment->getType()->getPointerElementType());

    #ifndef NDEBUG
    mKernelDoSegmentFunctionType = doSegFuncType;
    #endif

    const auto args = buildKernelCallArgumentList(b);

    #ifdef PRINT_DEBUG_MESSAGES
    debugHalt(b);
    #endif

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
        b->CreateMProtect(mKernelSharedHandle, CBuilder::Protect::WRITE);
    }

    if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {

        readAndUpdateInternalProcessedAndProducedItemCounts(b);

        BasicBlock * resumeKernelExecution = nullptr;

        // If we can loop back to the entry, we assume that its to handle the final block.
        if (mMayLoopToEntry) {

            mHasMoreInput = hasMoreInput(b);

            // if this works correctly, the only time we won't release the pre-invocation lock is when we're
            // going to end up terminating.

            Value * const waitToRelease = b->CreateOr(mHasMoreInput, b->CreateIsNotNull(mIsFinalInvocation));
            const auto prefix = makeKernelName(mKernelId);
            BasicBlock * const releaseSyncLock =
                b->CreateBasicBlock(prefix + "_releasePreInvocationLock", mKernelCompletionCheck);
            resumeKernelExecution =
                b->CreateBasicBlock(prefix + "_resumeKernelExecution", mKernelCompletionCheck);
            b->CreateUnlikelyCondBr(waitToRelease, resumeKernelExecution, releaseSyncLock);

            b->SetInsertPoint(releaseSyncLock);
        }
        releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION);
        if (mMayLoopToEntry) {
            b->CreateBr(resumeKernelExecution);

            b->SetInsertPoint(resumeKernelExecution);
        }
    }

    #ifdef PRINT_DEBUG_MESSAGES
    const auto prefix = makeKernelName(mKernelId);
    debugPrint(b, "* " + prefix + "_isFinal = %" PRIu64, mIsFinalInvocation);
    debugPrint(b, "* " + prefix + "_executing = %" PRIu64, mNumOfLinearStrides);
    #endif

    #ifdef ENABLE_PAPI
    readPAPIMeasurement(b, mKernelId, PAPIReadBeforeMeasurementArray);
    #endif
    Value * const beforeKernelCall = startCycleCounter(b);
    Value * doSegmentRetVal = nullptr;
    if (mRethrowException) {
        const auto prefix = makeKernelName(mKernelId);
        BasicBlock * const invokeOk = b->CreateBasicBlock(prefix + "_invokeOk", mKernelCompletionCheck);        
        #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(11, 0, 0)
        doSegmentRetVal = b->CreateInvoke(doSegFuncType, doSegment, invokeOk, mRethrowException, args);
        #else
        doSegmentRetVal = b->CreateInvoke(doSegment, invokeOk, mRethrowException, args);
        #endif
        b->SetInsertPoint(invokeOk);
    } else {
        doSegmentRetVal = b->CreateCall(doSegFuncType, doSegment, args);
    }
    updateCycleCounter(b, mKernelId, beforeKernelCall, CycleCounter::KERNEL_EXECUTION);
    #ifdef ENABLE_PAPI
    accumPAPIMeasurementWithoutReset(b, PAPIReadBeforeMeasurementArray, mKernelId, PAPIKernelCounter::PAPI_KERNEL_EXECUTION);
    #endif

    #ifdef PRINT_DEBUG_MESSAGES
    debugResume(b);
    #endif

    if (mKernelCanTerminateEarly) {
        mTerminatedExplicitly = doSegmentRetVal;
    } else {
        mTerminatedExplicitly = nullptr;
    }

    if (LLVM_UNLIKELY(mAllowDataParallelExecution)) {
        acquireSynchronizationLock(b, mKernelId, SYNC_LOCK_POST_INVOCATION);
    }
    if (LLVM_LIKELY(!mCurrentKernelIsStateFree)) {
        updateProcessedAndProducedItemCounts(b);
    }
    readReturnedOutputVirtualBaseAddresses(b);

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
        b->CreateMProtect(mKernelSharedHandle, CBuilder::Protect::NONE);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief buildKernelCallArgumentList
 ** ------------------------------------------------------------------------------------------------------------- */
ArgVec PipelineCompiler::buildKernelCallArgumentList(BuilderRef b) {

    // WARNING: any change to this must be reflected in Kernel::addDoSegmentDeclaration, Kernel::getDoSegmentFields,
    // Kernel::setDoSegmentProperties and Kernel::getDoSegmentProperties.

    const auto numOfInputs = in_degree(mKernelId, mBufferGraph);
    const auto numOfOutputs = out_degree(mKernelId, mBufferGraph);

    ArgVec args;

    auto addNextArg = [&](Value * arg) {

        #ifndef NDEBUG
        assert ("null argument" && arg);
        const auto n = mKernelDoSegmentFunctionType->getNumParams();
        if (LLVM_UNLIKELY(args.size() >= n)) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << mKernel->getName() << ": "
                   "was given too many arguments";
            throw std::runtime_error(out.str().str());
        }

        Type * const argTy = mKernelDoSegmentFunctionType->getParamType(args.size());
        if (LLVM_UNLIKELY(argTy != arg->getType())) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);

            Function * const func = mKernel->getDoSegmentFunction(b, true);

            out << mKernel->getName() << ": "
                "invalid argument type for ";

            auto argItr = func->arg_begin();
            std::advance(argItr, args.size());
            out << argItr->getName();

            out << " (#" << args.size()
                << "): expected ";
            argTy->print(out);
            out << " but got ";
            arg->getType()->print(out);
            report_fatal_error(out.str().str());
        }
        #endif
        args.push_back(arg);
    };

    auto addItemCountArg = [&](const BufferPort & port,
                               const bool forceAddressability,
                               Value * const itemCount) {
        const Binding & binding = port.Binding;
        const ProcessingRate & rate = binding.getRate();

        Value * ptr = nullptr;
        if (LLVM_UNLIKELY(rate.isRelative())) {
            return ptr;
        }
        if (forceAddressability || isAddressable(binding)) {
//            if (LLVM_UNLIKELY(mKernelIsInternallySynchronized)) {
//                if (port.Port.Type == PortType::Input) {
//                    ptr = mProcessedItemCountPtr[port.Port];
//                } else {
//                    ptr = mProducedItemCountPtr[port.Port];
//                }
//            } else {
                if (LLVM_UNLIKELY(mNumOfAddressableItemCount == mAddressableItemCountPtr.size())) {
                    auto aic = b->CreateAllocaAtEntryPoint(b->getSizeTy());
                    mAddressableItemCountPtr.push_back(aic);
                }
                ptr = mAddressableItemCountPtr[mNumOfAddressableItemCount++];
                b->CreateStore(itemCount, ptr);
//            }
            addNextArg(ptr);
        } else if (isCountable(binding)) {
            addNextArg(itemCount);
        }
        return ptr;
    };


    args.reserve(4 + (numOfInputs + numOfOutputs) * 4);
    if (LLVM_LIKELY(mKernelSharedHandle)) {
        addNextArg(mKernelSharedHandle);
    }
    assert (mKernelThreadLocalHandle == nullptr || !mKernelThreadLocalHandle->getType()->isEmptyTy());
    if (LLVM_UNLIKELY(mKernelThreadLocalHandle)) {
        assert (mKernelThreadLocalHandle->getType()->getPointerElementType() == mKernel->getThreadLocalStateType());
        if (LLVM_UNLIKELY(mIsOptimizationBranch)) {
            ConstantInt * i32_ZERO = b->getInt32(0);
            FixedArray<Value *, 3> offset;
            offset[0] = i32_ZERO;
            offset[1] = i32_ZERO;
            offset[2] = i32_ZERO;
            Value * const branchTypePtr = b->CreateGEP(mKernelThreadLocalHandle, offset);
            assert (branchTypePtr->getType()->getPointerElementType() == mOptimizationBranchSelectedBranch->getType());
            b->CreateStore(mOptimizationBranchSelectedBranch, branchTypePtr);
        }
        addNextArg(mKernelThreadLocalHandle);
    }
    if (mKernelIsInternallySynchronized) {
        if (mMayLoopToEntry) {
            const auto prefix = makeKernelName(mKernelId);
            Value * const intSegNoPtr = b->getScalarFieldPtr(prefix + INTERNALLY_SYNCHRONIZED_SUB_SEGMENT_SUFFIX);
            Value * const intSegNo = b->CreateLoad(intSegNoPtr);
            Value * const nextSegNo = b->CreateAdd(intSegNo, b->getSize(1));
            b->CreateStore(nextSegNo, intSegNoPtr);
            addNextArg(intSegNo);
        } else {
            addNextArg(mSegNo);
        }
    }
    addNextArg(mNumOfLinearStrides);
    if (mFixedRateFactorPhi) {
        addNextArg(mFixedRateFactorPhi);
    }

    PointerType * const voidPtrTy = b->getVoidPtrTy();

    for (unsigned i = 0; i < numOfInputs; ++i) {
        const auto port = getInput(mKernelId, StreamSetPort(PortType::Input, i));
        const BufferPort & rt = mBufferGraph[port];

        if (LLVM_LIKELY(rt.Port.Reason == ReasonType::Explicit)) {

            Value * processed = nullptr;
            if (rt.IsDeferred) {
                processed = mAlreadyProcessedDeferredPhi[rt.Port];
            } else {
                processed = mAlreadyProcessedPhi[rt.Port];
            }
            assert (processed);

            Value * const addr = mInputVirtualBaseAddressPhi[rt.Port]; assert (addr);
            addNextArg(b->CreatePointerCast(addr, voidPtrTy));

            mReturnedProcessedItemCountPtr[rt.Port] = addItemCountArg(rt, rt.IsDeferred, processed);

            if (LLVM_UNLIKELY(requiresItemCount(rt.Binding))) {
                // calculate how many linear items are from the *deferred* position
                Value * inputItems = mLinearInputItemsPhi[rt.Port]; assert (inputItems);
                if (rt.IsDeferred) {
                    Value * diff = b->CreateSub(mAlreadyProcessedPhi[rt.Port], mAlreadyProcessedDeferredPhi[rt.Port]);
                    inputItems = b->CreateAdd(inputItems, diff);
                }
                addNextArg(inputItems);
            }
        }
    }

    PointerType * const voidPtrPtrTy = voidPtrTy->getPointerTo();

    for (unsigned i = 0; i < numOfOutputs; ++i) {
        const auto port = getOutput(mKernelId, StreamSetPort(PortType::Output, i));
        const BufferPort & rt = mBufferGraph[port];

        assert (rt.Port.Reason == ReasonType::Explicit);
        assert (rt.Port.Type == PortType::Output);

        const auto streamSet = target(port, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        const StreamSetBuffer * const buffer = bn.Buffer;

        Value * produced = mAlreadyProducedPhi[rt.Port];

        if (LLVM_UNLIKELY(rt.IsShared)) {
            if (CheckAssertions) {
                b->CreateAssert(buffer->getHandle(), "handle?");
            }
            addNextArg(b->CreatePointerCast(buffer->getHandle(), voidPtrTy));
        } else if (LLVM_UNLIKELY(rt.IsManaged)) {
            if (LLVM_UNLIKELY(mNumOfVirtualBaseAddresses == mVirtualBaseAddressPtr.size())) {
                auto vba = b->CreateAllocaAtEntryPoint(voidPtrTy);
                mVirtualBaseAddressPtr.push_back(vba);
            }
            Value * ptr = mVirtualBaseAddressPtr[mNumOfVirtualBaseAddresses++];
            ptr = b->CreatePointerCast(ptr, buffer->getPointerType()->getPointerTo());
            b->CreateStore(buffer->getBaseAddress(b.get()), ptr);
            addNextArg(b->CreatePointerCast(ptr, voidPtrPtrTy));
            mReturnedOutputVirtualBaseAddressPtr[rt.Port] = ptr;
        } else {
            Value * const vba = getVirtualBaseAddress(b, rt, bn, produced, bn.isNonThreadLocal(), true);
            addNextArg(b->CreatePointerCast(vba, voidPtrTy));
        }

        mReturnedProducedItemCountPtr[rt.Port] = addItemCountArg(rt, rt.IsDeferred || mKernelCanTerminateEarly, produced);

        if (LLVM_UNLIKELY(rt.IsShared || rt.IsManaged)) {
            addNextArg(readConsumedItemCount(b, streamSet));
        } else if (requiresItemCount(rt.Binding)) {
            addNextArg(mLinearOutputItemsPhi[rt.Port]);
        }

    }

    assert (args.size() == mKernelDoSegmentFunctionType->getNumParams());
    return args;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateProcessedAndProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateProcessedAndProducedItemCounts(BuilderRef b) {

    const auto numOfInputs = in_degree(mKernelId, mBufferGraph);
    const auto numOfOutputs = out_degree(mKernelId, mBufferGraph);

    // calculate or read the item counts (assuming this kernel did not terminate)
    for (unsigned i = 0; i < numOfInputs; ++i) {
        Value * processed = nullptr;
        const auto inputPort = StreamSetPort{PortType::Input, i};
        if (mProcessedItemCount[inputPort] == nullptr) {
            const Binding & input = getInputBinding(inputPort);
            const ProcessingRate & rate = input.getRate();
            if (LLVM_LIKELY(rate.isFixed() || rate.isPartialSum() || rate.isGreedy())) {
                processed = b->CreateAdd(mAlreadyProcessedPhi[inputPort], mLinearInputItemsPhi[inputPort]);
                if (mAlreadyProcessedDeferredPhi[inputPort]) {
                    assert (mReturnedProcessedItemCountPtr[inputPort]);
                    mProcessedDeferredItemCount[inputPort] = b->CreateLoad(mReturnedProcessedItemCountPtr[inputPort]);
                    #ifdef PRINT_DEBUG_MESSAGES
                    const auto prefix = makeBufferName(mKernelId, inputPort);
                    debugPrint(b, prefix + "_processed_deferred' = %" PRIu64, mProcessedDeferredItemCount[inputPort]);
                    #endif
                    if (LLVM_UNLIKELY(CheckAssertions)) {
                        Value * const deferred = mProcessedDeferredItemCount[inputPort];
                        Value * const isDeferred = b->CreateICmpULE(deferred, processed);
                        Value * const isFinal = mIsFinalInvocationPhi;
                        // TODO: workaround now for ScanMatch; if it ends with a match on a
                        // block-aligned boundary the start of the next match seems to be one
                        // after? Revise the logic to only perform a 0-item final block on
                        // kernels that may produce Add'ed data? Define the final/non-final
                        // contract first.
                        Value * const isDeferredOrFinal = b->CreateOr(isDeferred, b->CreateIsNotNull(isFinal));
                        b->CreateAssert(isDeferredOrFinal,
                                        "%s.%s: deferred processed item count (%" PRIu64 ") "
                                        "exceeds non-deferred (%" PRIu64 ")",
                                        mCurrentKernelName,
                                        b->GetString(input.getName()),
                                        deferred, processed);
                    }
                }
            } else if (rate.isBounded() || rate.isUnknown()) {
                assert (mReturnedProcessedItemCountPtr[inputPort]);
                processed = b->CreateLoad(mReturnedProcessedItemCountPtr[inputPort]);
            } else {
                SmallVector<char, 256> tmp;
                raw_svector_ostream out(tmp);
                out << "Kernel " << mKernel->getName() << ":" << input.getName()
                    << " has an " << "input" << " rate that is not properly handled by the PipelineKernel";
                report_fatal_error(out.str());
            }

            mProcessedItemCount[inputPort] = processed; assert (processed);
            #ifdef PRINT_DEBUG_MESSAGES
            const auto prefix = makeBufferName(mKernelId, inputPort);
            debugPrint(b, prefix + "_processed' = %" PRIu64, mProcessedItemCount[inputPort]);
            #endif
        }
    }

    for (unsigned i = 0; i < numOfOutputs; ++i) {
        const auto outputPort = StreamSetPort{PortType::Output, i};
        if (mProducedItemCount[outputPort] == nullptr) {
            Value * produced = nullptr;
            const Binding & output = getOutputBinding(outputPort);
            const ProcessingRate & rate = output.getRate();
            if (LLVM_LIKELY(rate.isFixed() || rate.isPartialSum())) {
                produced = b->CreateAdd(mAlreadyProducedPhi[outputPort], mLinearOutputItemsPhi[outputPort]);
                if (mAlreadyProducedDeferredPhi[outputPort]) {
                    assert (mReturnedProducedItemCountPtr[outputPort]);
                    mProducedDeferredItemCount[outputPort] = b->CreateLoad(mReturnedProducedItemCountPtr[outputPort]);
                    #ifdef PRINT_DEBUG_MESSAGES
                    const auto prefix = makeBufferName(mKernelId, outputPort);
                    debugPrint(b, prefix + "_produced_deferred' = %" PRIu64, mProcessedDeferredItemCount[outputPort]);
                    #endif
                    if (LLVM_UNLIKELY(CheckAssertions)) {
                        Value * const deferred = mProducedDeferredItemCount[outputPort];
                        Value * const isDeferred = b->CreateICmpULE(deferred, produced);
                        Value * const isFinal = mIsFinalInvocationPhi;
                        // TODO: workaround now for ScanMatch; if it ends with a match on a
                        // block-aligned boundary the start of the next match seems to be one
                        // after? Revise the logic to only perform a 0-item final block on
                        // kernels that may produce Add'ed data? Define the final/non-final
                        // contract first.
                        Value * const isDeferredOrFinal = b->CreateOr(isDeferred, b->CreateIsNotNull(isFinal));
                        b->CreateAssert(isDeferredOrFinal,
                                        "%s.%s: deferred processed item count (%" PRIu64 ") "
                                        "exceeds non-deferred (%" PRIu64 ")",
                                        mCurrentKernelName,
                                        b->GetString(output.getName()),
                                        deferred, produced);
                    }
                }
            } else if (rate.isBounded() || rate.isUnknown()) {
                assert (mReturnedProducedItemCountPtr[outputPort]);
                produced = b->CreateLoad(mReturnedProducedItemCountPtr[outputPort]);
            } else if (rate.isRelative()) {
                auto getRefPort = [&] () {
                    const auto refPort = getReference(outputPort);
                    if (LLVM_LIKELY(refPort.Type == PortType::Input)) {
                        return getInput(mKernelId, refPort);
                    } else {
                        return getOutput(mKernelId, refPort);
                    }
                };
                const BufferPort & ref = mBufferGraph[getRefPort()];
                if (mProducedDeferredItemCount[ref.Port]) {
                    mProducedDeferredItemCount[outputPort] = b->CreateMulRational(mProducedDeferredItemCount[ref.Port], rate.getRate());
                }
                produced = b->CreateMulRational(mProducedItemCount[ref.Port], rate.getRate());
            } else {
                SmallVector<char, 256> tmp;
                raw_svector_ostream out(tmp);
                out << "Kernel " << mKernel->getName() << ":" << output.getName()
                    << " has an " << "output" << " rate that is not properly handled by the PipelineKernel";
                report_fatal_error(out.str());
            }

            #ifdef PRINT_DEBUG_MESSAGES
            const auto prefix = makeBufferName(mKernelId, StreamSetPort{PortType::Output, i});
            debugPrint(b, prefix + "_produced' = %" PRIu64, produced);
            #endif

            if (LLVM_UNLIKELY(CheckAssertions)) {
                if (mReturnedProducedItemCountPtr[outputPort]) {
                    const auto port = getOutput(mKernelId, outputPort);
                    const auto streamSet = target(port, mBufferGraph);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (LLVM_LIKELY(bn.isInternal() && bn.isOwned())) {

                        Value * const writable = getWritableOutputItems(b, mBufferGraph[port], true);
                        Value * const delta = b->CreateSub(produced, mAlreadyProducedPhi[outputPort]);
                        Value * const withinCapacity = b->CreateICmpULE(delta, writable);
                        const Binding & output = getOutputBinding(outputPort);
                        b->CreateAssert(withinCapacity,
                                        "%s.%s: reported produced item count delta (%" PRIu64 ") "
                                        "exceeds writable items (%" PRIu64 ")",
                                        mCurrentKernelName,
                                        b->GetString(output.getName()),
                                        delta, writable);
                    }
                }
            }
            mProducedItemCount[outputPort] = produced;
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readAndUpdateInternalProcessedAndProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readAndUpdateInternalProcessedAndProducedItemCounts(BuilderRef b) {

    const auto numOfInputs = in_degree(mKernelId, mBufferGraph);
    const auto numOfOutputs = out_degree(mKernelId, mBufferGraph);

    // calculate or read the item counts (assuming this kernel did not terminate)
    for (unsigned i = 0; i < numOfInputs; ++i) {
        const auto inputPort = StreamSetPort{PortType::Input, i};
        const Binding & input = getInputBinding(inputPort);
        const ProcessingRate & rate = input.getRate();
        if (LLVM_LIKELY(rate.isFixed() || rate.isPartialSum() || rate.isGreedy())) {
            Value * const ptr = mProcessedItemCountPtr[inputPort];
            Value * const processed = b->CreateAdd(mAlreadyProcessedPhi[inputPort], mLinearInputItemsPhi[inputPort]);
            b->CreateStore(processed, ptr);
            #ifdef PRINT_DEBUG_MESSAGES
            const auto prefix = makeBufferName(mKernelId, inputPort);
            debugPrint(b, prefix + "_internal_processed = %" PRIu64, processed);
            #endif
            mProcessedItemCount[inputPort] = processed;
        }
    }

    for (unsigned i = 0; i < numOfOutputs; ++i) {
        const auto outputPort = StreamSetPort{PortType::Output, i};
        const Binding & output = getOutputBinding(outputPort);
        const ProcessingRate & rate = output.getRate();
        if (LLVM_LIKELY(rate.isFixed() || rate.isPartialSum())) {
            Value * const ptr = mProducedItemCountPtr[outputPort];
            Value * const produced = b->CreateAdd(mAlreadyProducedPhi[outputPort], mLinearOutputItemsPhi[outputPort]);
            b->CreateStore(produced, ptr);
            #ifdef PRINT_DEBUG_MESSAGES
            const auto prefix = makeBufferName(mKernelId, outputPort);
            debugPrint(b, prefix + "_internal_produced = %" PRIu64, produced);
            #endif
            mProducedItemCount[outputPort] = produced;
        }
    }

}

}


#endif // KERNEL_EXECUTION_LOGIC_HPP
