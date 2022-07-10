#include "pipeline_compiler.hpp"
#include "kernel_logic.hpp"
#include "kernel_io_calculation_logic.hpp"
#include "kernel_execution_logic.hpp"
#include "kernel_family_logic.hpp"

// TODO: if we have multiple copies of the same type of kernel executing sequentially, we could avoid
// generating an "execution call" for each and instead pass in different handles/item counts. This
// could improve I-Cache utilization.

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief start
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::start(BuilderRef b) {

    mCurrentKernelName = mKernelName[PipelineInput];
    mPipelineLoop = b->CreateBasicBlock("PipelineLoop");
    mPipelineEnd = b->CreateBasicBlock("PipelineEnd");

    makePartitionEntryPoints(b);

    if (CheckAssertions) {
        mRethrowException = b->WriteDefaultRethrowBlock();
    }

    #ifdef PRINT_DEBUG_MESSAGES
    debugInit(b);
    if (mIsNestedPipeline) {
        debugPrint(b, "------------------------------------------------- START %" PRIx64, getHandle());
    } else {
        debugPrint(b, "================================================= START %" PRIx64, getHandle());
    }
    const auto prefix = mTarget->getName();
    if (mNumOfStrides) {
        debugPrint(b, prefix + " +++ NUM OF STRIDES %" PRIu64 "+++", mNumOfStrides);
    }
    debugPrint(b, prefix + " +++ IS FINAL %" PRIu8 "+++", mIsFinal);
    #endif

    #ifdef ENABLE_PAPI
    createEventSetAndStartPAPI(b);
    #endif

    mExpectedNumOfStridesMultiplier = b->getScalarField(EXPECTED_NUM_OF_STRIDES_MULTIPLIER);
    if (LLVM_LIKELY(RequiredThreadLocalStreamSetMemory > 0)) {
        mThreadLocalStreamSetBaseAddress = b->getScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY);
    }

    readExternalConsumerItemCounts(b);
    loadInternalStreamSetHandles(b, true);
    loadInternalStreamSetHandles(b, false);
    initializePipelineInputTerminationSignal(b);

    mKernel = nullptr;
    mKernelId = 0;
    mAddressableItemCountPtr.clear();
    mVirtualBaseAddressPtr.clear();
    mNumOfTruncatedInputBuffers = 0;
    mTruncatedInputBuffer.clear();

    BasicBlock * const entryBlock = b->GetInsertBlock();
    b->CreateBr(mPipelineLoop);

    b->SetInsertPoint(mPipelineLoop);
    mMadeProgressInLastSegment = b->CreatePHI(b->getInt1Ty(), 2, "madeProgressInLastSegment");
    mMadeProgressInLastSegment->addIncoming(b->getTrue(), entryBlock);
    Constant * const i1_FALSE = b->getFalse();
    mPipelineProgress = i1_FALSE;
    mExhaustedInput = i1_FALSE;
    obtainCurrentSegmentNumber(b, entryBlock);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief executeKernel
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::executeKernel(BuilderRef b) {

    assert (std::find(ActiveKernels.begin(), ActiveKernels.end(), mKernelId) != ActiveKernels.end());
    assert (KernelOnHybridThread.test(mKernelId) == mCompilingHybridThread);
    clearInternalStateForCurrentKernel();
    checkForPartitionEntry(b);
    mFixedRateLCM = getLCMOfFixedRateInputs(mKernel);
    mKernelIsInternallySynchronized = mIsInternallySynchronized.test(mKernelId);
    mKernelCanTerminateEarly = mKernel->canSetTerminateSignal();
    mIsOptimizationBranch = isa<OptimizationBranch>(mKernel);
    mExecuteStridesIndividually = mKernel->hasAttribute(AttrId::ExecuteStridesIndividually) || recordsAnyHistogramData();
    mCurrentKernelIsStateFree = mIsStatelessKernel.test(mKernelId);
    assert (mIsStatelessKernel.test(mKernelId) == isCurrentKernelStateFree());
    #ifndef DISABLE_ALL_DATA_PARALLEL_SYNCHRONIZATION
    #ifdef ALLOW_INTERNALLY_SYNCHRONIZED_KERNELS_TO_BE_DATA_PARALLEL
    mAllowDataParallelExecution = mCurrentKernelIsStateFree || mKernelIsInternallySynchronized;
    #else
    mAllowDataParallelExecution = mCurrentKernelIsStateFree;
    #endif
    #endif
    identifyPipelineInputs(mKernelId);


    mIsBounded = isBounded();
    mHasExplicitFinalPartialStride = requiresExplicitFinalStride();
    mCheckInputChannels = false;
    for (const auto input : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[input];
        if (port.CanModifySegmentLength) {
            mCheckInputChannels = true;
            break;
        }
    }
    bool mayHaveInsufficientIO = mCheckInputChannels;
    for (const auto output : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & port = mBufferGraph[output];
        if (port.CanModifySegmentLength) {
            mayHaveInsufficientIO = true;
            break;
        }
    }

    mMayLoopToEntry = mIsOptimizationBranch;
    if (LLVM_LIKELY(!mKernelIsInternallySynchronized)) {
        if (mHasExplicitFinalPartialStride || (mCheckInputChannels && hasAtLeastOneNonGreedyInput())) {
            mMayLoopToEntry = true;
        }
        if (StrideStepLength[FirstKernelInPartition] > StrideStepLength[mKernelId]) {
            mMayLoopToEntry = true;
        }
    }
    mNextPartitionEntryPoint = getPartitionExitPoint(b);
    assert (mNextPartitionEntryPoint);

    mExhaustedPipelineInputAtExit = mExhaustedInput;

#ifndef NDEBUG
    const auto nextPartitionId = ActivePartitions[ActivePartitionIndex + 1];
    const auto jumpId = PartitionJumpTargetId[mCurrentPartitionId];
//    const auto mIsPartitionRoot = mIsPartitionRoot && (nextPartitionId != jumpId);

    assert (nextPartitionId > mCurrentPartitionId);
    assert (jumpId >= nextPartitionId);
#endif

    const auto prefix = makeKernelName(mKernelId);

    // TODO: if a kernel has circular buffers and the produced/consumption rate is not synchronous
    // and the GCD of the stride step length of the producer/consumer is 1 but the stride step length
    // of the consumer is > 1, we may get a scenario in which the partition root needs to check the
    // raw produced item counts rather than the accessible ones to determine the segment length.
    // We could bypass this by having a larger overflow region but doing so would cause us to memcpy
    // more data than necessary.

    /// -------------------------------------------------------------------------------------
    /// BASIC BLOCK CONSTRUCTION
    /// -------------------------------------------------------------------------------------

    mKernelLoopEntry = b->CreateBasicBlock(prefix + "_loopEntry", mNextPartitionEntryPoint);
    mKernelCheckOutputSpace = b->CreateBasicBlock(prefix + "_checkOutputSpace", mNextPartitionEntryPoint);
    mKernelLoopCall = b->CreateBasicBlock(prefix + "_executeKernel", mNextPartitionEntryPoint);
    mKernelCompletionCheck = b->CreateBasicBlock(prefix + "_normalCompletionCheck", mNextPartitionEntryPoint);
    if (mayHaveInsufficientIO) {
        mKernelInsufficientInput = b->CreateBasicBlock(prefix + "_insufficientInput", mNextPartitionEntryPoint);
    }
    if (mIsPartitionRoot) {
        mKernelInitiallyTerminated = b->CreateBasicBlock(prefix + "_initiallyTerminated", mNextPartitionEntryPoint);
        SmallVector<char, 256> tmp;
        raw_svector_ostream nm(tmp);
        nm << prefix << "_jumpFromPartition_" << mCurrentPartitionId
           << "_to_" << PartitionJumpTargetId[mCurrentPartitionId];
        mKernelJumpToNextUsefulPartition = b->CreateBasicBlock(nm.str(), mNextPartitionEntryPoint);
    } else {
        mKernelInitiallyTerminated = nullptr;
        mKernelJumpToNextUsefulPartition = nullptr;
    }

    mKernelTerminated = b->CreateBasicBlock(prefix + "_terminated", mNextPartitionEntryPoint);
    mKernelLoopExit = b->CreateBasicBlock(prefix + "_loopExit", mNextPartitionEntryPoint);
    // The phi catch simplifies compilation logic by "forward declaring" the loop exit point.
    // Subsequent optimization phases will collapse it into the correct exit block.
    mKernelLoopExitPhiCatch = b->CreateBasicBlock(prefix + "_kernelExitPhiCatch", mNextPartitionEntryPoint);
    mKernelExit = b->CreateBasicBlock(prefix + "_kernelExit", mNextPartitionEntryPoint);

    /// -------------------------------------------------------------------------------------
    /// KERNEL / PARTITION ENTRY BLOCK
    /// -------------------------------------------------------------------------------------

    checkIfKernelIsAlreadyTerminated(b);

    readAvailableItemCounts(b);
    readProcessedItemCounts(b);
    readProducedItemCounts(b);
    readConsumedItemCounts(b);
    prepareLinearThreadLocalOutputBuffers(b);

    incrementNumberOfSegmentsCounter(b);
    recordUnconsumedItemCounts(b);

    detemineMaximumNumberOfStrides(b);

    mFinalPartialStrideFixedRateRemainderPhi = nullptr;
    if (mIsPartitionRoot) {
        b->CreateUnlikelyCondBr(mInitiallyTerminated, mKernelInitiallyTerminated, mKernelLoopEntry);
    } else {
        b->CreateBr(mKernelLoopEntry);
    }
    mKernelLoopStart = b->GetInsertBlock();

    /// -------------------------------------------------------------------------------------
    /// PHI NODE INITIALIZATION
    /// -------------------------------------------------------------------------------------

    // Set up some PHI nodes early to simplify accumulating their incoming values.
    initializeKernelLoopEntryPhis(b);
    initializeKernelCheckOutputSpacePhis(b);
    if (mIsPartitionRoot) {
        initializeJumpToNextUsefulPartitionPhis(b);
    }
    if (mayHaveInsufficientIO) {
        initializeKernelInsufficientIOExitPhis(b);
    }
    initializeKernelTerminatedPhis(b);
    initializeKernelLoopExitPhis(b);
    initializeKernelExitPhis(b);

    /// -------------------------------------------------------------------------------------
    /// KERNEL LOOP ENTRY
    /// -------------------------------------------------------------------------------------

    b->SetInsertPoint(mKernelLoopEntry);
    determineNumOfLinearStrides(b);
    mIsFinalInvocation = mIsFinalInvocationPhi;

    // When tracing blocking I/O, test all I/O streams but do not execute the
    // kernel if any stream is insufficient.
    if (mayHaveInsufficientIO && TraceIO) {
        b->CreateUnlikelyCondBr(mBranchToLoopExit, mKernelInsufficientInput, mKernelLoopCall);
        BasicBlock * const exitBlock = b->GetInsertBlock();
        mExhaustedPipelineInputPhi->addIncoming(mExhaustedInput, exitBlock);
    } else {
        b->CreateBr(mKernelLoopCall);
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL CALL
    /// -------------------------------------------------------------------------------------

    b->SetInsertPoint(mKernelLoopCall);
    writeLookBehindLogic(b);
    writeKernelCall(b);
    writeCopyBackLogic(b);
    writeDelayReflectionLogic(b);

    /// -------------------------------------------------------------------------------------
    /// KERNEL EXPLICIT TERMINATION CHECK
    /// -------------------------------------------------------------------------------------

    if (mKernelCanTerminateEarly) {

        Value * const aborted = b->CreateIsNotNull(mTerminatedExplicitly);
        BasicBlock * const explicitTermination =
            b->CreateBasicBlock(prefix + "_explicitTermination", mKernelCompletionCheck);
        b->CreateUnlikelyCondBr(aborted, explicitTermination, mKernelCompletionCheck);

        b->SetInsertPoint(explicitTermination);
        // If the kernel explicitly terminates, it must set its processed/produced item counts.
        // Otherwise, the pipeline will update any countable rates, even upon termination.
        readCountableItemCountsAfterAbnormalTermination(b);
        // TODO: We could have a *fixed-rate* source kernel be a partition root but will need to
        // calculate how many items are the stride "remainder" here.
        signalAbnormalTermination(b);
        b->CreateBr(mKernelTerminated);

    } else { // kernel cannot terminate early

        b->CreateBr(mKernelCompletionCheck);
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL NORMAL COMPLETION CHECK
    /// -------------------------------------------------------------------------------------

    b->SetInsertPoint(mKernelCompletionCheck);
    normalCompletionCheck(b);

    /// -------------------------------------------------------------------------------------
    /// KERNEL TERMINATED
    /// -------------------------------------------------------------------------------------

    b->SetInsertPoint(mKernelTerminated);
    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, "** " + prefix + ".terminated at segment %" PRIu64, mSegNo);
    #endif
    informInputKernelsOfTermination(b);
    clearUnwrittenOutputData(b);
    splatMultiStepPartialSumValues(b);
    writeTerminationSignal(b, mTerminatedSignalPhi);
    if (LLVM_UNLIKELY(mAllowDataParallelExecution && mMayLoopToEntry)) {
        releaseSynchronizationLock(b, mKernelId, SYNC_LOCK_PRE_INVOCATION);
    }
    updatePhisAfterTermination(b);
    b->CreateBr(mKernelLoopExit);

    /// -------------------------------------------------------------------------------------
    /// KERNEL INSUFFICIENT IO EXIT
    /// -------------------------------------------------------------------------------------

    if (mayHaveInsufficientIO) {
        writeInsufficientIOExit(b);
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL LOOP EXIT
    /// -------------------------------------------------------------------------------------

    b->SetInsertPoint(mKernelLoopExit);
    #ifdef PRINT_DEBUG_MESSAGES
    debugPrint(b, "** " + prefix + ".loopExit = %" PRIu64, mSegNo);
    #endif
    writeUpdatedItemCounts(b);
    assert (mTerminatedAtLoopExitPhi);
    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);
    Value * const terminated = b->CreateICmpNE(mTerminatedAtLoopExitPhi, unterminated);
    computeFullyProcessedItemCounts(b, terminated);
    computeMinimumConsumedItemCounts(b);
    writeLookAheadLogic(b);
    computeFullyProducedItemCounts(b, terminated);
    replacePhiCatchWithCurrentBlock(b, mKernelLoopExitPhiCatch, mKernelExit);
    b->CreateBr(mKernelExit);

    /// -------------------------------------------------------------------------------------
    /// KERNEL INITIALLY TERMINATED EXIT
    /// -------------------------------------------------------------------------------------

    if (mIsPartitionRoot) {
        writeInitiallyTerminatedPartitionExit(b);
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL PREPARE FOR PARTITION JUMP
    /// -------------------------------------------------------------------------------------

    if (mIsPartitionRoot) {
        writeJumpToNextPartition(b);
    }

    /// -------------------------------------------------------------------------------------
    /// KERNEL EXIT
    /// -------------------------------------------------------------------------------------

    b->SetInsertPoint(mKernelExit);
    recordFinalProducedItemCounts(b);
    writeConsumedItemCounts(b);
    setCurrentTerminationSignal(b, mTerminatedAtExitPhi);
    if (mIsPartitionRoot) {
        recordStridesPerSegment(b, mKernelId, mTotalNumOfStridesAtExitPhi);
    }
    recordProducedItemCountDeltas(b);

    // chain the progress state so that the next one carries on from this one
    mExhaustedInput = mExhaustedPipelineInputAtExit;
    mPipelineProgress = mAnyProgressedAtExitPhi;
    if (mIsPartitionRoot) {
        assert (mTotalNumOfStridesAtExitPhi);
        mNumOfPartitionStrides = mTotalNumOfStridesAtExitPhi;

        assert (mFinalPartitionSegmentAtExitPhi);
        mFinalPartitionSegment = mFinalPartitionSegmentAtExitPhi;
    }

    if (LLVM_UNLIKELY(CheckAssertions)) {        
        verifyPostInvocationTerminationSignal(b);
    }

    checkForPartitionExit(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief normalCompletionCheck
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::normalCompletionCheck(BuilderRef b) {

    ConstantInt * const i1_TRUE = b->getTrue();

    if (LLVM_LIKELY(mMayLoopToEntry)) {

        if (LLVM_LIKELY(!mAllowDataParallelExecution)) {
            mHasMoreInput = hasMoreInput(b);
        }

        assert (mHasMoreInput);

        BasicBlock * const exitBlockAfterLoopAgainTest = b->GetInsertBlock();

        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const auto port = mBufferGraph[e].Port;
            assert (mProcessedItemCount[port]);
            mAlreadyProcessedPhi[port]->addIncoming(mProcessedItemCount[port], exitBlockAfterLoopAgainTest);
            if (mAlreadyProcessedDeferredPhi[port]) {
                assert (mProcessedDeferredItemCount[port]);
                mAlreadyProcessedDeferredPhi[port]->addIncoming(mProcessedDeferredItemCount[port], exitBlockAfterLoopAgainTest);
            }
        }

        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const auto port = mBufferGraph[e].Port;
            assert (mProducedItemCount[port]);
            mAlreadyProducedPhi[port]->addIncoming(mProducedItemCount[port], exitBlockAfterLoopAgainTest);
            if (mAlreadyProducedDeferredPhi[port]) {
                assert (mProducedDeferredItemCount[port]);
                mAlreadyProducedDeferredPhi[port]->addIncoming(mProducedDeferredItemCount[port], exitBlockAfterLoopAgainTest);
            }
        }

        mAlreadyProgressedPhi->addIncoming(i1_TRUE, exitBlockAfterLoopAgainTest);
        mExecutedAtLeastOnceAtLoopEntryPhi->addIncoming(i1_TRUE, exitBlockAfterLoopAgainTest);
        mCurrentNumOfStridesAtLoopEntryPhi->addIncoming(mUpdatedNumOfStrides, exitBlockAfterLoopAgainTest);

        if (LLVM_UNLIKELY(mIsOptimizationBranch)) {
            assert (mOptimizationBranchSelectedBranch);
            mOptimizationBranchPriorScanStatePhi->addIncoming(mOptimizationBranchSelectedBranch, exitBlockAfterLoopAgainTest);
        }

        const auto prefix = makeKernelName(mKernelId);
        BasicBlock * const isFinalCheck = b->CreateBasicBlock(prefix + "_isFinalCheck", mKernelTerminated);
        b->CreateUnlikelyCondBr(mHasMoreInput, mKernelLoopEntry, isFinalCheck);

        b->SetInsertPoint(isFinalCheck);
    }

    Value * terminationSignal = mIsFinalInvocationPhi;

    assert (terminationSignal);

    BasicBlock * const exitBlock = b->GetInsertBlock();

    // update KernelTerminated phi nodes
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        assert (mProducedItemCount[port]);
        mProducedAtTerminationPhi[port]->addIncoming(mProducedItemCount[port], exitBlock);
    }
    mTerminatedSignalPhi->addIncoming(terminationSignal, exitBlock);

    if (mIsPartitionRoot) {
        assert (mUpdatedNumOfStrides);
        Value * const updatedNumOfStrides = b->CreateMulRational(mUpdatedNumOfStrides, mPartitionStrideRateScalingFactor);
        mTotalNumOfStridesAtLoopExitPhi->addIncoming(updatedNumOfStrides, exitBlock);
    }

    Value * const isFinal = b->CreateIsNotNull(terminationSignal);
    if (mIsPartitionRoot) {
        mFinalPartitionSegmentAtLoopExitPhi->addIncoming(b->getFalse(), exitBlock);
    }
    b->CreateUnlikelyCondBr(isFinal, mKernelTerminated, mKernelLoopExit);
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        mUpdatedProcessedPhi[port]->addIncoming(mProcessedItemCount[port], exitBlock);
        if (mUpdatedProcessedDeferredPhi[port]) {
            assert (mProcessedDeferredItemCount[port]);
            mUpdatedProcessedDeferredPhi[port]->addIncoming(mProcessedDeferredItemCount[port], exitBlock);
        }
    }
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        mUpdatedProducedPhi[port]->addIncoming(mProducedItemCount[port], exitBlock);
        if (mUpdatedProducedDeferredPhi[port]) {
            assert (mProducedDeferredItemCount[port]);
            mUpdatedProducedDeferredPhi[port]->addIncoming(mProducedDeferredItemCount[port], exitBlock);
        }
    }
    mTerminatedAtLoopExitPhi->addIncoming(terminationSignal, exitBlock);
    mAnyProgressedAtLoopExitPhi->addIncoming(i1_TRUE, exitBlock);
    mExhaustedPipelineInputAtLoopExitPhi->addIncoming(mExhaustedInput, exitBlock);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelLoopEntryPhis
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializeKernelLoopEntryPhis(BuilderRef b) {
    IntegerType * const sizeTy = b->getSizeTy();
    IntegerType * const boolTy = b->getInt1Ty();

    assert ("kernel loop start must be created before initializing loop entry phi nodes" && mKernelLoopStart);

    b->SetInsertPoint(mKernelLoopEntry);

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto port = br.Port;
        const auto prefix = makeBufferName(mKernelId, port);
        PHINode * const phi = b->CreatePHI(sizeTy, 2, prefix + "_alreadyProcessed");
        mAlreadyProcessedPhi[port] = phi;
        mCurrentProcessedItemCountPhi[port] = phi;
        assert (mInitiallyProcessedItemCount[port]);
        phi->addIncoming(mInitiallyProcessedItemCount[port], mKernelLoopStart);
        Value * const value = mInitiallyProcessedDeferredItemCount[port];
        if (value) {
            PHINode * const phi = b->CreatePHI(sizeTy, 2, prefix + "_alreadyProcessedDeferred");
            assert (phi);
            phi->addIncoming(value, mKernelLoopStart);
            mAlreadyProcessedDeferredPhi[port] = phi;
            mCurrentProcessedDeferredItemCountPhi[port] = phi;
        }
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto port = br.Port;
        const auto prefix = makeBufferName(mKernelId, port);
        const auto streamSet = target(e, mBufferGraph);
        PHINode * const phi = b->CreatePHI(sizeTy, 2, prefix + "_alreadyProduced");
        mAlreadyProducedPhi[port] = phi;
        mCurrentProducedItemCountPhi[port] = phi;
        assert (mInitiallyProducedItemCount[streamSet]);
        phi->addIncoming(mInitiallyProducedItemCount[streamSet], mKernelLoopStart);
        if (mInitiallyProducedDeferredItemCount[streamSet]) {
            PHINode * const phi = b->CreatePHI(sizeTy, 2, prefix + "_alreadyProducedDeferred");
            mAlreadyProducedDeferredPhi[port] = phi;
            mCurrentProducedDeferredItemCountPhi[port] = phi;
            phi->addIncoming(mInitiallyProducedDeferredItemCount[streamSet], mKernelLoopStart);
        }
    }
    const auto prefix = makeKernelName(mKernelId);
    mAlreadyProgressedPhi = b->CreatePHI(boolTy, 2, prefix + "_madeProgress");
    assert (mPipelineProgress);
    mAlreadyProgressedPhi->addIncoming(mPipelineProgress, mKernelLoopStart);

    if (mMayLoopToEntry) {
        // Since we may loop and call the kernel again, we want to mark that we've progressed
        // if we execute any kernel even if we could not complete a full segment.
        mExecutedAtLeastOnceAtLoopEntryPhi = b->CreatePHI(boolTy, 2, prefix + "_executedAtLeastOnce");
        mExecutedAtLeastOnceAtLoopEntryPhi->addIncoming(b->getFalse(), mKernelLoopStart);
        mCurrentNumOfStridesAtLoopEntryPhi = b->CreatePHI(sizeTy, 2, prefix + "_currentNumOfStrides");
        mCurrentNumOfStridesAtLoopEntryPhi->addIncoming(b->getSize(0), mKernelLoopStart);
    } else {
        mExecutedAtLeastOnceAtLoopEntryPhi = nullptr;
        mCurrentNumOfStridesAtLoopEntryPhi = nullptr;
    }
    if (mIsOptimizationBranch) {
        mOptimizationBranchPriorScanStatePhi = b->CreatePHI(boolTy, 2, prefix + "_optBrScanState");
        mOptimizationBranchPriorScanStatePhi->addIncoming(b->getFalse(), mKernelLoopStart);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelCheckOutputSpacePhis
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializeKernelCheckOutputSpacePhis(BuilderRef b) {
    b->SetInsertPoint(mKernelCheckOutputSpace);
    IntegerType * const sizeTy = b->getSizeTy();
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto inputPort = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, inputPort);
        PHINode * const phi = b->CreatePHI(sizeTy, 2, prefix + "_linearlyAccessible");
        mLinearInputItemsPhi[inputPort] = phi;
        mCurrentLinearInputItems[inputPort] = phi;
        Type * const bufferTy = getInputBuffer(inputPort)->getPointerType();
        mInputVirtualBaseAddressPhi[inputPort] = b->CreatePHI(bufferTy, 2, prefix + "_baseAddress");
    }
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto outputPort = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, outputPort);
        PHINode * const phi = b->CreatePHI(sizeTy, 2, prefix + "_linearlyWritable");
        mLinearOutputItemsPhi[outputPort] = phi;
        mCurrentLinearOutputItems[outputPort] = phi;
    }
    const auto prefix = makeKernelName(mKernelId);
    mNumOfLinearStridesPhi = b->CreatePHI(sizeTy, 2, prefix + "_numOfLinearStridesPhi");
    if (LLVM_LIKELY(mKernel->hasFixedRateInput())) {
        mFixedRateFactorPhi = b->CreatePHI(sizeTy, 2, prefix + "_fixedRateFactorPhi");
    }
    mCurrentFixedRateFactor = mFixedRateFactorPhi;
    mIsFinalInvocationPhi = b->CreatePHI(sizeTy, 2, prefix + "_isFinalPhi");
    if (mIsPartitionRoot) {
        mFinalPartialStrideFixedRateRemainderPhi = b->CreatePHI(sizeTy, 2, prefix + "_partialPartitionStridesPhi");
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelTerminatedPhis
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializeKernelTerminatedPhis(BuilderRef b) {
    b->SetInsertPoint(mKernelTerminated);
    Type * const sizeTy = b->getSizeTy();
    const auto prefix = makeKernelName(mKernelId);
    mTerminatedSignalPhi = b->CreatePHI(sizeTy, 2, prefix + "_terminatedSignal");

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto outputPort = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, outputPort);
        PHINode * const phi = b->CreatePHI(sizeTy, 2, prefix + "_finalProduced");
        mProducedAtTerminationPhi[outputPort] = phi;
        mProducedAtTermination[outputPort] = phi;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelTerminatedPhis
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializeJumpToNextUsefulPartitionPhis(BuilderRef b) {
    assert (mKernelJumpToNextUsefulPartition);
    b->SetInsertPoint(mKernelJumpToNextUsefulPartition);
    const auto prefix = makeKernelName(mKernelId);
    Type * const boolTy = b->getInt1Ty();
    mExhaustedInputAtJumpPhi = b->CreatePHI(boolTy, 2, prefix + "_exhaustedInputAtJumpPhi");
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, port);
        mProducedAtJumpPhi[port] = b->CreatePHI(b->getSizeTy(), 2, prefix + "_producedAtJumpPhi");
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelInsufficientIOExitPhis
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializeKernelInsufficientIOExitPhis(BuilderRef b) {
    b->SetInsertPoint(mKernelInsufficientInput);
    const auto prefix = makeKernelName(mKernelId);
    IntegerType * const boolTy = b->getInt1Ty();
    mExhaustedPipelineInputPhi = b->CreatePHI(boolTy, 2, prefix + "_exhaustedInput");
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelLoopExitPhis
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializeKernelLoopExitPhis(BuilderRef b) {
    b->SetInsertPoint(mKernelLoopExit);
    const auto prefix = makeKernelName(mKernelId);
    IntegerType * const sizeTy = b->getSizeTy();
    IntegerType * const boolTy = b->getInt1Ty();
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, port);
        mUpdatedProcessedPhi[port] = b->CreatePHI(sizeTy, 2, prefix + "_updatedProcessedAtLoopExit");
        if (mAlreadyProcessedDeferredPhi[port]) {
            mUpdatedProcessedDeferredPhi[port] = b->CreatePHI(sizeTy, 2, prefix + "_updatedProcessedDeferredAtLoopExit");
        }
    }
    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, port);
        mUpdatedProducedPhi[port] = b->CreatePHI(sizeTy, 2, prefix + "_updatedProducedAtLoopExit");
        if (mAlreadyProducedDeferredPhi[port]) {
            mUpdatedProducedDeferredPhi[port] = b->CreatePHI(sizeTy, 2, prefix + "_updatedProcessedDeferredAtLoopExit");
        }
    }
    mTerminatedAtLoopExitPhi = b->CreatePHI(sizeTy, 2, prefix + "_terminatedAtLoopExit");
    mAnyProgressedAtLoopExitPhi = b->CreatePHI(boolTy, 2, prefix + "_anyProgressAtLoopExit");
    mExhaustedPipelineInputAtLoopExitPhi = b->CreatePHI(boolTy, 2, prefix + "_exhaustedInputAtLoopExit");
    if (mIsPartitionRoot) {
        mTotalNumOfStridesAtLoopExitPhi = b->CreatePHI(sizeTy, 2, prefix + "_totalNumOfStridesAtLoopExit");
        mFinalPartitionSegmentAtLoopExitPhi = b->CreatePHI(boolTy, 2, prefix + "_finalPartitionSegmentAtLoopExitPhi");
    } else {
        mTotalNumOfStridesAtLoopExitPhi = nullptr;
        mFinalPartitionSegmentAtLoopExitPhi = nullptr;
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeInsufficientIOExit
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeInsufficientIOExit(BuilderRef b) {

    // A partition root will always have an insufficient I/O check since they control how many strides the
    // other kernels in the partition will execute. If a kernel has non-linear I/O, however, we need to test
    // whether we've finished executing.

    b->SetInsertPoint(mKernelInsufficientInput);

    if (LLVM_UNLIKELY(CheckAssertions && mAllowDataParallelExecution && mMayLoopToEntry)) {
        b->CreateAssert(b->CreateNot(mExecutedAtLeastOnceAtLoopEntryPhi),
                        "%s: is a statefree kernel with an invalid loop again check",
                        mCurrentKernelName);
    }


    BasicBlock * const exitBlock = b->GetInsertBlock();

    if (mMayLoopToEntry) {
        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const auto port = mBufferGraph[e].Port;
            mUpdatedProcessedPhi[port]->addIncoming(mAlreadyProcessedPhi[port], exitBlock);
            if (mAlreadyProcessedDeferredPhi[port]) {
                mUpdatedProcessedDeferredPhi[port]->addIncoming(mAlreadyProcessedDeferredPhi[port], exitBlock);
            }
        }

        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const auto port = mBufferGraph[e].Port;
            mUpdatedProducedPhi[port]->addIncoming(mAlreadyProducedPhi[port], exitBlock);
            if (mAlreadyProducedDeferredPhi[port]) {
                mUpdatedProducedDeferredPhi[port]->addIncoming(mAlreadyProducedDeferredPhi[port], exitBlock);
            }
        }
    }

    if (mMayLoopToEntry || !mIsPartitionRoot) {
        assert (mExhaustedPipelineInputPhi);
        mExhaustedPipelineInputAtLoopExitPhi->addIncoming(mExhaustedPipelineInputPhi, exitBlock);
        assert (mAlreadyProgressedPhi);
        mAnyProgressedAtLoopExitPhi->addIncoming(mAlreadyProgressedPhi, exitBlock);
        mTerminatedAtLoopExitPhi->addIncoming(mInitialTerminationSignal, exitBlock);
    }

    if (mIsPartitionRoot) {
        assert (mInitialTerminationSignal);
        if (mExhaustedInputAtJumpPhi) {
            mExhaustedInputAtJumpPhi->addIncoming(mExhaustedPipelineInputPhi, exitBlock);
            for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
                const auto & br = mBufferGraph[e];
                const auto port = br.Port;
                Value * produced = nullptr;
                if (LLVM_UNLIKELY(br.IsDeferred)) {
                    produced = mAlreadyProducedDeferredPhi[port];
                } else {
                    produced = mAlreadyProducedPhi[port];
                }
                assert (isFromCurrentFunction(b, produced, false));
                mProducedAtJumpPhi[port]->addIncoming(produced, exitBlock);
            }
        }
        if (mMayLoopToEntry) {
            mFinalPartitionSegmentAtLoopExitPhi->addIncoming(b->getFalse(), exitBlock);
            assert (mCurrentNumOfStridesAtLoopEntryPhi);
            Value * const currentNumOfStrides = b->CreateMulRational(mCurrentNumOfStridesAtLoopEntryPhi, mPartitionStrideRateScalingFactor);
            mTotalNumOfStridesAtLoopExitPhi->addIncoming(currentNumOfStrides, exitBlock);
            b->CreateLikelyCondBr(mExecutedAtLeastOnceAtLoopEntryPhi, mKernelLoopExit, mKernelJumpToNextUsefulPartition);
        } else {
            b->CreateBr(mKernelJumpToNextUsefulPartition);
        }
    } else {
        // if this is not a partition root, it is not responsible for determining
        // whether the partition is out of input
        b->CreateBr(mKernelLoopExit);
    }

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernelExitPhis
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::initializeKernelExitPhis(BuilderRef b) {
    b->SetInsertPoint(mKernelExit);
    const auto prefix = makeKernelName(mKernelId);
    IntegerType * const sizeTy = b->getSizeTy();
    IntegerType * const boolTy = b->getInt1Ty();

    mTerminatedAtExitPhi = b->CreatePHI(sizeTy, 2, prefix + "_terminatedAtKernelExit");
    assert (mTerminatedAtLoopExitPhi);
    mTerminatedAtExitPhi->addIncoming(mTerminatedAtLoopExitPhi, mKernelLoopExitPhiCatch);
    mTotalNumOfStridesAtExitPhi = nullptr;
    if (mIsPartitionRoot) {
        assert (mTotalNumOfStridesAtLoopExitPhi);
        mTotalNumOfStridesAtExitPhi = b->CreatePHI(sizeTy, 2, prefix + "_totalNumOfStridesAtExit");
        mTotalNumOfStridesAtExitPhi->addIncoming(mTotalNumOfStridesAtLoopExitPhi, mKernelLoopExitPhiCatch);
    }

    createConsumedPhiNodesAtExit(b);

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        const auto prefix = makeBufferName(mKernelId, port);
        PHINode * const fullyProduced = b->CreatePHI(sizeTy, 2, prefix + "_fullyProducedAtKernelExit");
        mFullyProducedItemCount[port] = fullyProduced;
    }

    PHINode * const progress = b->CreatePHI(boolTy, 2, prefix + "_anyProgressAtKernelExit");
    progress->addIncoming(mAnyProgressedAtLoopExitPhi, mKernelLoopExitPhiCatch);
    mAnyProgressedAtExitPhi = progress;

    PHINode * const exhausted = b->CreatePHI(boolTy, 2, prefix + "_exhaustedPipelineInputAtKernelExit");
    exhausted->addIncoming(mExhaustedPipelineInputAtLoopExitPhi, mKernelLoopExitPhiCatch);
    mExhaustedPipelineInputAtExit = exhausted;

    if (mIsPartitionRoot) {
        mFinalPartitionSegmentAtExitPhi = b->CreatePHI(boolTy, 2, prefix + "_anyProgressAtKernelExit");
        mFinalPartitionSegmentAtExitPhi->addIncoming(mFinalPartitionSegmentAtLoopExitPhi, mKernelLoopExitPhiCatch);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateKernelExitPhisAfterInitiallyTerminated
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::updateKernelExitPhisAfterInitiallyTerminated(BuilderRef b) {
    Constant * const completed = getTerminationSignal(b, TerminationSignal::Completed);
    mTerminatedAtExitPhi->addIncoming(completed, mKernelInitiallyTerminatedExit);
    if (mIsPartitionRoot) {
        ConstantInt * const sz_ZERO = b->getSize(0);
        mTotalNumOfStridesAtExitPhi->addIncoming(sz_ZERO, mKernelInitiallyTerminatedExit);
    }

    phiOutConsumedItemCountsAfterInitiallyTerminated(b);

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        const BufferPort & br = mBufferGraph[e];
        Value * produced = nullptr;
        if (LLVM_UNLIKELY(br.IsDeferred)) {
            produced = mInitiallyProducedDeferredItemCount[streamSet];
        } else {
            produced = mInitiallyProducedItemCount[streamSet];
        }
        const auto port = br.Port;
        assert (isFromCurrentFunction(b, produced, false));
        if (mKernelJumpToNextUsefulPartition != mKernelInitiallyTerminated) {
            mProducedAtJumpPhi[port]->addIncoming(produced, mKernelInitiallyTerminatedExit);
        }
        mFullyProducedItemCount[port]->addIncoming(produced, mKernelInitiallyTerminatedExit);
    }

    mAnyProgressedAtExitPhi->addIncoming(mPipelineProgress, mKernelInitiallyTerminatedExit);
    cast<PHINode>(mExhaustedPipelineInputAtExit)->addIncoming(mExhaustedInput, mKernelInitiallyTerminatedExit);
    if (mIsPartitionRoot) {
        mFinalPartitionSegmentAtExitPhi->addIncoming(b->getTrue(), mKernelInitiallyTerminatedExit);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updatePhiCountAfterTermination
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::updatePhisAfterTermination(BuilderRef b) {
    BasicBlock * const exitBlock = b->GetInsertBlock();
    mTerminatedAtLoopExitPhi->addIncoming(mTerminatedSignalPhi, exitBlock);
    mAnyProgressedAtLoopExitPhi->addIncoming(b->getTrue(), exitBlock);
    mExhaustedPipelineInputAtLoopExitPhi->addIncoming(mExhaustedInput, exitBlock);
    if (mIsPartitionRoot) {
        Value * finalNumOfStrides = mUpdatedNumOfStrides; assert (mUpdatedNumOfStrides);
        if (mFinalPartialStrideFixedRateRemainderPhi) {
            const Rational fixedRateFactor = mFixedRateLCM * Rational{mKernel->getStride()};
            Value * fixedRateItems = b->CreateMulRational(finalNumOfStrides, fixedRateFactor);
            fixedRateItems = b->CreateAdd(fixedRateItems, mFinalPartialStrideFixedRateRemainderPhi);
            finalNumOfStrides = b->CreateMulRational(fixedRateItems, mPartitionStrideRateScalingFactor / fixedRateFactor);
        } else {
            finalNumOfStrides = b->CreateMulRational(finalNumOfStrides, mPartitionStrideRateScalingFactor);
        }
        mTotalNumOfStridesAtLoopExitPhi->addIncoming(finalNumOfStrides, exitBlock);
    }
    if (mIsPartitionRoot) {
        mFinalPartitionSegmentAtLoopExitPhi->addIncoming(b->getTrue(), exitBlock);
    }
    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        Value * const totalCount = getLocallyAvailableItemCount(b, port);
        mUpdatedProcessedPhi[port]->addIncoming(totalCount, exitBlock);
        if (mUpdatedProcessedDeferredPhi[port]) {
            mUpdatedProcessedDeferredPhi[port]->addIncoming(totalCount, exitBlock);
        }
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const auto port = mBufferGraph[e].Port;
        Value * const produced = mProducedAtTermination[port];

        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, makeBufferName(mKernelId, port) + "_producedAtTermination = %" PRIu64, produced);
        #endif

        mUpdatedProducedPhi[port]->addIncoming(produced, exitBlock);
        if (mUpdatedProducedDeferredPhi[port]) {
            mUpdatedProducedDeferredPhi[port]->addIncoming(produced, exitBlock);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief end
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::end(BuilderRef b) {



    // A pipeline will end for one or two reasons:

    // 1) Process has *halted* due to insufficient external I/O.

    // 2) All pipeline sinks have terminated (i.e., any kernel that writes
    // to a pipeline output, is marked as having a side-effect, or produces
    // an input for some call in which no dependent kernels is a pipeline
    // sink).

    // TODO: if we determine that all of the pipeline I/O is consumed in one invocation of the
    // pipeline, we can avoid testing at the end whether its terminated.

    Value * terminated = nullptr;
    if (mIsNestedPipeline) {
        if (mCurrentThreadTerminationSignalPtr) {
            terminated = hasPipelineTerminated(b);
        }
        b->CreateBr(mPipelineEnd);
    } else {

        terminated = hasPipelineTerminated(b);

        Value * const done = b->CreateIsNotNull(terminated);
        if (LLVM_UNLIKELY(CheckAssertions && !mCompilingHybridThread)) {
            Value * const progressedOrFinished = b->CreateOr(mPipelineProgress, done);
            Value * const live = b->CreateOr(mMadeProgressInLastSegment, progressedOrFinished);
            b->CreateAssert(live, "Dead lock detected: pipeline could not progress after two iterations");
        }
        releaseHybridThreadSynchronizationLock(b);
        BasicBlock * const exitBlock = b->GetInsertBlock();
        mMadeProgressInLastSegment->addIncoming(mPipelineProgress, exitBlock);
        incrementCurrentSegNo(b, exitBlock);
        b->CreateUnlikelyCondBr(done, mPipelineEnd, mPipelineLoop);
    }
    b->SetInsertPoint(mPipelineEnd);
//    writeFinalHybridThreadSynchronizationNumber(b);
//    writeExternalConsumedItemCounts(b);
//    writeExternalProducedItemCounts(b);
    if (mCurrentThreadTerminationSignalPtr) {
        b->CreateStore(terminated, mCurrentThreadTerminationSignalPtr);
    }
    // free any truncated input buffers
    for (Value * const bufferPtr : mTruncatedInputBuffer) {
        b->CreateFree(b->CreateLoad(bufferPtr));
    }
    #ifdef PRINT_DEBUG_MESSAGES
    if (mIsNestedPipeline) {
        debugPrint(b, "------------------------------------------------- END %" PRIx64, getHandle());
    } else {
        debugPrint(b, "================================================= END %" PRIx64, getHandle());
    }
    debugClose(b);
    #endif

    #ifdef ENABLE_PAPI
    stopPAPIAndDestroyEventSet(b);
    #endif

    if (LLVM_UNLIKELY(canSetTerminateSignal())) {
        Constant * const unterminated = b->getSize(KernelBuilder::TerminationCode::None);
        Constant * const terminated = b->getSize(KernelBuilder::TerminationCode::Terminated);
        Value * const retVal = b->CreateSelect(mPipelineProgress, unterminated, terminated);
        b->setTerminationSignal(retVal);
    }
    mExpectedNumOfStridesMultiplier = nullptr;
    mThreadLocalStreamSetBaseAddress = nullptr;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeExternalProducedItemCounts
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeExternalProducedItemCounts(BuilderRef b) {
//    for (const auto e : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
//        const BufferPort & external = mBufferGraph[e];
//        const auto streamSet = source(e, mBufferGraph);
//        Value * const ptr = getProducedOutputItemsPtr(external.Port.Number);
//        b->CreateStore(mLocallyAvailableItems[streamSet], ptr);
//    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeExternalConsumedItemCounts
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
