#include "../pipeline_compiler.hpp"
#include <pthread.h>
#include <llvm/IR/Verifier.h>

#if BOOST_OS_LINUX
#include <sched.h>
#endif

#if BOOST_OS_MACOS
#include <mach/mach_init.h>
#include <mach/thread_act.h>
#endif

namespace llvm {
#if BOOST_OS_MACOS
template<> class TypeBuilder<pthread_t, false> {
public:
  static Type *get(LLVMContext& C) {
    return IntegerType::getIntNTy(C, sizeof(pthread_t) * CHAR_BIT);
  }
};
#endif

template<> class TypeBuilder<pthread_attr_t, false> {
public:
  static Type *get(LLVMContext& C) {
    return IntegerType::getIntNTy(C, sizeof(pthread_attr_t) * CHAR_BIT);
  }
};
}

enum PipelineStateObjectField : unsigned {
    SHARED_STATE_PARAM
    , THREAD_LOCAL_PARAM
    , PIPELINE_PARAMS
    , INITIAL_SEG_NO
    , FIXED_NUMBER_OF_THREADS
    , ACCUMULATED_SEGMENT_TIME
    , ACCUMULATED_SYNCHRONIZATION_TIME
    , CURRENT_THREAD_ID
    , TERMINATION_SIGNAL
    , CURRENT_THREAD_STATUS_FLAG
    // -------------------
    , THREAD_STRUCT_SIZE
};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief concat
 ** ------------------------------------------------------------------------------------------------------------- */
StringRef concat(StringRef A, StringRef B, SmallVector<char, 256> & tmp) {
    Twine C = A + B;
    tmp.clear();
    C.toVector(tmp);
    return StringRef(tmp.data(), tmp.size());
}

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateMultiThreadKernelMethod
 *
 * Given a computation expressed as a logical pipeline of K kernels k0, k_1, ...k_(K-1)
 * operating over an input stream set S, a segment-parallel implementation divides the input
 * into segments and coordinates a set of T <= K threads to each process one segment at a time.
 * Let S_0, S_1, ... S_N be the segments of S.   Segments are assigned to threads in a round-robin
 * fashion such that processing of segment S_i by the full pipeline is carried out by thread i mod T.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateMultiThreadKernelMethod(KernelBuilder & b) {

    Module * const m = b.getModule();
    PointerType * const voidPtrTy = b.getVoidPtrTy();
    IntegerType * const boolTy = b.getInt1Ty();
    IntegerType * const sizeTy = b.getSizeTy();

    const auto storedState = storeDoSegmentState();

    StructType * const threadStructTy = getThreadStuctType(b, storedState);

    ConstantInt * const i32_ZERO = b.getInt32(0);
    ConstantInt * const sz_ZERO = b.getSize(0);
    ConstantInt * const sz_ONE = b.getSize(1);
    ConstantInt * const sz_TWO = b.getSize(2);

    SmallVector<char, 256> tmp;
    const auto threadName = concat(mTarget->getName(), "_MultithreadedDoSegment", tmp);

    FunctionType * const threadFuncType = FunctionType::get(voidPtrTy, {voidPtrTy}, false);
    Function * const threadFunc = Function::Create(threadFuncType, Function::InternalLinkage, threadName, m);
    if (LLVM_UNLIKELY(CheckAssertions)) {
        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
        threadFunc->setHasUWTable();
        #else
        threadFunc->setUWTableKind(UWTableKind::Default);
        #endif
    }
    Value * const initialSharedState = getHandle();
    Value * const initialThreadLocal = getThreadLocalHandle();
    Value * const initialTerminationSignalPtr = getTerminationSignalPtr();

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE DRIVER
    // -------------------------------------------------------------------------------------------------------------------------

    // use the process thread to handle the initial segment function after spawning
    // (n - 1) threads to handle the subsequent offsets

    Function * const pthreadCreateFn = m->getFunction("pthread_create");
    Function * const pthreadExitFn = m->getFunction("pthread_exit");
    Function * const pthreadJoinFn = m->getFunction("pthread_join");

    Type * const pThreadTy = TypeBuilder<pthread_t, false>::get(b.getContext());

    Value * minimumNumOfThreads = nullptr;
    Value * const maximumNumOfThreads = b.getScalarField(MAXIMUM_NUM_OF_THREADS);
    if (mUseDynamicMultithreading) {
        minimumNumOfThreads = b.getScalarField(MINIMUM_NUM_OF_THREADS);
    } else {
        minimumNumOfThreads = maximumNumOfThreads;
    }
    Value * const threadStateArray =
        b.CreateAlignedMalloc(threadStructTy, maximumNumOfThreads, 0, b.getCacheAlignment());

    DataLayout DL(b.getModule());
    Type * const intPtrTy = DL.getIntPtrType(voidPtrTy);
    BasicBlock * const constructThread = b.CreateBasicBlock("constructThread", mPipelineEnd);
    BasicBlock * const constructedThreads = b.CreateBasicBlock("constructedThreads", mPipelineEnd);

    Value * const moreThanOneThread = b.CreateICmpNE(maximumNumOfThreads, sz_ONE);

    BasicBlock * const constructThreadEntry = b.GetInsertBlock();

    // construct and start the threads

    b.CreateCondBr(moreThanOneThread, constructThread, constructedThreads);

    b.SetInsertPoint(constructThread);
    PHINode * const threadIndex = b.CreatePHI(sizeTy, 2);
    threadIndex->addIncoming(sz_ONE, constructThreadEntry);

    FixedArray<Value *, 2> fieldIndex;
    fieldIndex[0] = threadIndex;
    Value * cThreadLocal = nullptr;
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        SmallVector<Value *, 2> args;
        if (initialSharedState) {
            args.push_back(initialSharedState);
        }
        args.push_back(ConstantPointerNull::get(cast<PointerType>(initialThreadLocal->getType())));
        cThreadLocal = mTarget->initializeThreadLocalInstance(b, args);
        
        if (LLVM_LIKELY(mTarget->allocatesInternalStreamSets())) {
            Function * const allocInternal = mTarget->getAllocateThreadLocalInternalStreamSetsFunction(b, false);
            SmallVector<Value *, 3> allocArgs;
            if (LLVM_LIKELY(mTarget->isStateful())) {
                allocArgs.push_back(initialSharedState);
            }
            allocArgs.push_back(cThreadLocal);
            allocArgs.push_back(sz_ONE);
            b.CreateCall(allocInternal->getFunctionType(), allocInternal, allocArgs);
        }
    }
    Value * const cThreadState = b.CreateGEP(threadStructTy, threadStateArray, threadIndex);

    writeThreadStructObject(b, threadStructTy, cThreadState, initialSharedState, cThreadLocal, storedState, threadIndex, maximumNumOfThreads);
    Value * const nextThreadIndex = b.CreateAdd(threadIndex, sz_ONE);
    BasicBlock * constructNextThread = nullptr;
    if (mUseDynamicMultithreading) {
        BasicBlock * const startThread = b.CreateBasicBlock("startThread", constructedThreads);
        constructNextThread = b.CreateBasicBlock("constructNextThread", constructedThreads);
        Value * const start = b.CreateICmpULT(nextThreadIndex, minimumNumOfThreads);

        b.CreateCondBr(start, startThread, constructNextThread);

        b.SetInsertPoint(startThread);
    }
    FixedArray<Value *, 4> pthreadCreateArgs;
    FunctionType * const pthreadCreateFnTy = pthreadCreateFn->getFunctionType();
    if (mUseDynamicMultithreading) {
        fieldIndex[1] = b.getInt32(CURRENT_THREAD_STATUS_FLAG);
        Value * initThreadStateFlagPtr = b.CreateInBoundsGEP(threadStructTy, threadStateArray, fieldIndex);
        b.CreateStore(sz_ONE, initThreadStateFlagPtr);
    }
    fieldIndex[1] = b.getInt32(CURRENT_THREAD_ID);
    pthreadCreateArgs[0] = b.CreateInBoundsGEP(threadStructTy, threadStateArray, fieldIndex);
    pthreadCreateArgs[1] = ConstantPointerNull::get(cast<PointerType>(pthreadCreateFnTy->getParamType(1)));
    pthreadCreateArgs[2] = b.CreatePointerCast(threadFunc, pthreadCreateFnTy->getParamType(2));
    pthreadCreateArgs[3] = b.CreatePointerCast(cThreadState, voidPtrTy);
    b.CreateCall(pthreadCreateFnTy, pthreadCreateFn, pthreadCreateArgs);
    if (mUseDynamicMultithreading) {
        b.CreateBr(constructNextThread);

        b.SetInsertPoint(constructNextThread);
    }

    Value * const createMoreThreads = b.CreateICmpULT(nextThreadIndex, maximumNumOfThreads);
    threadIndex->addIncoming(nextThreadIndex, b.GetInsertBlock());
    b.CreateCondBr(createMoreThreads, constructThread, constructedThreads);

    b.SetInsertPoint(constructedThreads);

    // execute the process thread
    Value * const processState = threadStateArray;
    writeThreadStructObject(b, threadStructTy, processState, initialSharedState, initialThreadLocal, storedState, sz_ZERO, maximumNumOfThreads);
    fieldIndex[0] = i32_ZERO;
    fieldIndex[1] = b.getInt32(CURRENT_THREAD_ID);
    b.CreateStore(Constant::getNullValue(pThreadTy), b.CreateInBoundsGEP(threadStructTy, threadStateArray, fieldIndex));

    PointerType * const threadStructPtrTy = cast<PointerType>(processState->getType());

    // store where we'll resume compiling the DoSegment method
    const auto resumePoint = b.saveIP();

    const auto anyDebugOptionIsSet = codegen::AnyDebugOptionIsSet();

    // -------------------------------------------------------------------------------------------------------------------------
    // GENERATE DO SEGMENT (KERNEL EXECUTION) FUNCTION CODE
    // -------------------------------------------------------------------------------------------------------------------------

    SmallVector<Type *, 3> csRetValFields;
    Type * csRetValType = nullptr;
    if (CheckAssertions) {
        csRetValType = boolTy; // hasProgressed
    } else {
        csRetValType = b.getVoidTy();
    }

    FixedArray<Type *, 2> csParams;
    csParams[0] = threadStructPtrTy; // thread state
    csParams[1] = sizeTy; // segment number

    FunctionType * const csFuncType = FunctionType::get(csRetValType, csParams, false);
    const auto outerFuncName = concat(mTarget->getName(), "_MultithreadedThread", tmp);
    Function * const csFunc = Function::Create(csFuncType, Function::InternalLinkage, outerFuncName, m);
    csFunc->setCallingConv(CallingConv::C);
    if (!mUseDynamicMultithreading) {
        csFunc->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);
    }
    if (LLVM_UNLIKELY(CheckAssertions)) {
        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
        csFunc->setHasUWTable();
        #else
        csFunc->setUWTableKind(UWTableKind::Default);
        #endif
    }
    b.SetInsertPoint(BasicBlock::Create(m->getContext(), "entry", csFunc));
    auto args = csFunc->arg_begin();
    Value * const threadStruct = &*args++;
    assert (threadStruct->getType() == threadStructPtrTy);
    readThreadStuctObject(b, threadStructTy, threadStruct);
    assert (isFromCurrentFunction(b, getHandle(), !mTarget->isStateful()));
    readDoSegmentState(b, threadStructTy, threadStruct);
    initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);
    mSegNo = &*args;
    #ifdef PRINT_DEBUG_MESSAGES
    debugInit(b);
    #endif
    #ifdef ENABLE_PAPI
    createPAPIMeasurementArrays(b);
    getPAPIEventSet(b);
    #endif
    Value * segmentStartTime = nullptr;
    if (mUseDynamicMultithreading) {
        segmentStartTime = b.CreateReadCycleCounter();
    }

    // generate the pipeline logic for this thread
    start(b);
    branchToInitialPartition(b);
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        setActiveKernel(b, i, true);
        executeKernel(b);
    }
    mKernel = nullptr;
    mKernelId = 0;
    mSegNo = nullptr;
    if (mUseDynamicMultithreading) {
        Value * const segmentEndTime = b.CreateReadCycleCounter();
        Value * const totalSegmentTime = b.CreateSub(segmentEndTime, segmentStartTime);

        FixedArray<Value *, 2> indices2;
        indices2[0] = i32_ZERO;
        indices2[1] = b.getInt32(ACCUMULATED_SEGMENT_TIME);
        Value * const segPtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
        Value * const current = b.CreateLoad(b.getSizeTy(), segPtr);
        Value * const accum = b.CreateAdd(current, totalSegmentTime);
        b.CreateStore(accum, segPtr);
    }
    const auto hasTermSignal = !mIsNestedPipeline || PipelineHasTerminationSignal;
    if (LLVM_LIKELY(hasTermSignal)) {
        writeTerminationSignalToLocalState(b, threadStructTy, threadStruct, hasPipelineTerminated(b));
    }
    if (LLVM_UNLIKELY(CheckAssertions)) {
        b.CreateRet(mPipelineProgress);
    } else {
        b.CreateRetVoid();
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE THREAD
    // -------------------------------------------------------------------------------------------------------------------------
    auto makeThreadFunction = [&](Function * const threadFunc, const bool processThreadForDynamicScheduling) {
        assert (threadFunc);
        threadFunc->setCallingConv(CallingConv::C);
        auto arg = threadFunc->arg_begin();
        arg->setName("threadStruct");

        b.SetInsertPoint(BasicBlock::Create(m->getContext(), "entry", threadFunc));

        Value * const threadStruct = b.CreatePointerCast(arg, threadStructPtrTy);
        readThreadStuctObject(b, threadStructTy, threadStruct);
        assert (isFromCurrentFunction(b, getHandle(), !mTarget->isStateful()));
        assert (isFromCurrentFunction(b, getThreadLocalHandle(), !mTarget->hasThreadLocal()));
        initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);

        Value * minimumThreads = nullptr;
        Value * maximumThreads = nullptr;
        Value * synchronizationCostCheckPeriod = nullptr;
        Value * syncAddThreadThreadhold = nullptr;
        Value * syncRemoveThreadThreadhold = nullptr;

        if (processThreadForDynamicScheduling) {
            minimumThreads = b.getScalarField(MINIMUM_NUM_OF_THREADS);
            maximumThreads = b.getScalarField(MAXIMUM_NUM_OF_THREADS);
            synchronizationCostCheckPeriod =
                b.getScalarField(DYNAMIC_MULTITHREADING_SEGMENT_PERIOD);
            Type * const floatTy = b.getFloatTy();
            Constant * const f100 = ConstantFP::get(floatTy, 100.0);
            syncAddThreadThreadhold =
                b.getScalarField(DYNAMIC_MULTITHREADING_ADDITIONAL_THREAD_SYNCHRONIZATION_THRESHOLD);
            syncAddThreadThreadhold = b.CreateFDiv(syncAddThreadThreadhold, f100);
            syncRemoveThreadThreadhold =
                b.getScalarField(DYNAMIC_MULTITHREADING_REMOVE_THREAD_SYNCHRONIZATION_THRESHOLD);
            syncRemoveThreadThreadhold = b.CreateFDiv(syncRemoveThreadThreadhold, f100);
        }

        #ifdef ENABLE_PAPI
        Value * PAPIPipelineStartMeasurementArray = nullptr;
        if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {
            createPAPIMeasurementArrays(b);
            getPAPIEventSet(b);
            registerPAPIThread(b);
        }
        #endif

        startCycleCounter(b, CycleCounter::FULL_PIPELINE_TIME);
        #ifdef ENABLE_PAPI
        startPAPIMeasurement(b, PAPIKernelCounter::PAPI_FULL_PIPELINE_TIME);
        #endif

        #ifdef PRINT_DEBUG_MESSAGES
        debugInit(b);
        if (mIsNestedPipeline) {
            debugPrint(b, "------------------------------------------------- START %" PRIx64, getHandle());
        } else {
            debugPrint(b, "================================================= START %" PRIx64, getHandle());
        }
        #endif

        // generate the pipeline logic for this thread
        mPipelineLoop = b.CreateBasicBlock("PipelineLoop");
        mPipelineEnd = b.CreateBasicBlock("PipelineEnd");
        BasicBlock * const entryBlock = b.GetInsertBlock();
        b.CreateBr(mPipelineLoop);

        b.SetInsertPoint(mPipelineLoop);
        if (LLVM_UNLIKELY(CheckAssertions)) {
            mMadeProgressInLastSegment = b.CreatePHI(b.getInt1Ty(), 2, "madeProgressInLastSegment");
            mMadeProgressInLastSegment->addIncoming(b.getTrue(), entryBlock);
        }
        PHINode * nextCheckSegmentPhi = nullptr;
        PHINode * activeThreadsPhi = nullptr;
        if (processThreadForDynamicScheduling) {
            nextCheckSegmentPhi = b.CreatePHI(sizeTy, 2, "nextCheckPhi");
            nextCheckSegmentPhi->addIncoming(synchronizationCostCheckPeriod, entryBlock);
            activeThreadsPhi = b.CreatePHI(sizeTy, 2, "activeThreadsPhi");
            activeThreadsPhi->addIncoming(minimumThreads, entryBlock);
        }

        obtainCurrentSegmentNumber(b, entryBlock);

        SmallVector<Value *, 3> args(2);
        args[0] = threadStruct;
        args[1] = mSegNo; assert (mSegNo);
        Value * const csRetVal = b.CreateCall(csFuncType, csFunc, args);

        Value * terminated = nullptr;
        Value * done = nullptr;
        Value * madeProgress = nullptr;

        if (LLVM_LIKELY(hasTermSignal)) {
            terminated = readTerminationSignalFromLocalState(b, threadStructTy, threadStruct);
            done = b.CreateIsNotNull(terminated);
        }
        if (LLVM_UNLIKELY(CheckAssertions)) {
            madeProgress = csRetVal;
            if (LLVM_LIKELY(hasTermSignal)) {
                madeProgress = b.CreateOr(madeProgress, done);
            }
            Value * const live = b.CreateOr(mMadeProgressInLastSegment, madeProgress);
            b.CreateAssert(live, "Dead lock detected: pipeline could not progress after two iterations");
        }

        PHINode * startOfNextPeriodPhi = nullptr;
        PHINode * currentNumOfThreadsPhi = nullptr;

        if (processThreadForDynamicScheduling) {
            assert (mUseDynamicMultithreading);

            // if a thread got stalled or the period was set so low, we could reenter this check prior to
            // the thread itself stopping,

            BasicBlock * checkSynchronizationCostLoop = b.CreateBasicBlock("checkSynchronizationCostLoop", mPipelineEnd);
            BasicBlock * checkToAddThread = b.CreateBasicBlock("checkToAddThread", mPipelineEnd);

            BasicBlock * selectThreadStructToUseForAddThread = b.CreateBasicBlock("selectThreadStructToUseForAddThread", mPipelineEnd);
            BasicBlock * checkIfThreadIsCancelled = b.CreateBasicBlock("checkIfThreadIsCancelled", mPipelineEnd);
            BasicBlock * joinCancelledThread = b.CreateBasicBlock("joinCancelledThread", mPipelineEnd);
            BasicBlock * addThread = b.CreateBasicBlock("addThread", mPipelineEnd);

            BasicBlock * checkToRemoveThread = b.CreateBasicBlock("checkToRemoveThread", mPipelineEnd);
            BasicBlock * selectThreadToRemove = b.CreateBasicBlock("selectThreadToRemove", mPipelineEnd);

            BasicBlock * removeThread = b.CreateBasicBlock("removeThread", mPipelineEnd);
            BasicBlock * nextSegment = b.CreateBasicBlock("nextSegment", mPipelineEnd);
            BasicBlock * recordBeforeNextSegment = nextSegment;
            if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
                recordBeforeNextSegment = b.CreateBasicBlock("recordBeforeNextSegment", nextSegment);
            }


            Value * const check = b.CreateICmpUGE(mSegNo, nextCheckSegmentPhi);
            FixedArray<Value *, 2> indices2;

            BasicBlock * const loopEntry = b.GetInsertBlock();
            b.CreateUnlikelyCondBr(check, checkSynchronizationCostLoop, nextSegment);

            b.SetInsertPoint(checkSynchronizationCostLoop);
            PHINode * const indexPhi = b.CreatePHI(sizeTy, 2);
            indexPhi->addIncoming(sz_ZERO, loopEntry);
            PHINode * const segmentTimeAccumPhi = b.CreatePHI(sizeTy, 2);
            segmentTimeAccumPhi->addIncoming(sz_ZERO, loopEntry);
            PHINode * const synchronizationTimeAccumPhi = b.CreatePHI(sizeTy, 2);
            synchronizationTimeAccumPhi->addIncoming(sz_ZERO, loopEntry);

            indices2[0] = indexPhi;
            indices2[1] = b.getInt32(ACCUMULATED_SEGMENT_TIME);
            Value * const segTimePtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            Value * const nextSegTime = b.CreateAdd(segmentTimeAccumPhi, b.CreateLoad(sizeTy, segTimePtr));

            segmentTimeAccumPhi->addIncoming(nextSegTime, checkSynchronizationCostLoop);

            indices2[1] = b.getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
            Value * const syncTimePtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            Value * const nextSyncTime = b.CreateAdd(synchronizationTimeAccumPhi, b.CreateLoad(sizeTy, syncTimePtr));
            synchronizationTimeAccumPhi->addIncoming(nextSyncTime, checkSynchronizationCostLoop);

            Value * const nextIndex = b.CreateAdd(indexPhi, b.getSize(1));
            indexPhi->addIncoming(nextIndex, checkSynchronizationCostLoop);
            Value * const hasMore = b.getFalse(); // b.CreateICmpNE(nextIndex, activeThreadsPhi);
            b.CreateCondBr(hasMore, checkSynchronizationCostLoop, checkToAddThread);

            b.SetInsertPoint(checkToAddThread);
            Type * const floatTy = b.getFloatTy();
            Value * const fSegTime = b.CreateUIToFP(nextSegTime, floatTy);
            Value * const fSyncTime = b.CreateUIToFP(nextSyncTime, floatTy);
            Value * const fSyncOverhead = b.CreateFDiv(fSyncTime, fSegTime);

            // subtract out the values so that we can keep each check
            indices2[0] = sz_ZERO;
            indices2[1] = b.getInt32(ACCUMULATED_SEGMENT_TIME);

            Value * const baseSegTimePtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            b.CreateStore(sz_ZERO, baseSegTimePtr);
            indices2[1] = b.getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
            Value * const baseSyncTimePtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);

            b.CreateStore(sz_ZERO, baseSyncTimePtr);

            Value * const syncOverheadLow = b.CreateFCmpULT(fSyncOverhead, syncAddThreadThreadhold);
            Value * const canAddMoreThreads = b.CreateICmpULT(activeThreadsPhi, maximumThreads);
            Value * const canAdd = b.CreateAnd(syncOverheadLow, canAddMoreThreads);

            Value * const startOfNextPeriod = b.CreateAdd(mSegNo, synchronizationCostCheckPeriod);
            b.CreateCondBr(canAdd, selectThreadStructToUseForAddThread, checkToRemoveThread);

            b.SetInsertPoint(selectThreadStructToUseForAddThread);
            PHINode * const selectToAddPhi = b.CreatePHI(sizeTy, 2);
            selectToAddPhi->addIncoming(sz_ONE, checkToAddThread);
            indices2[0] = selectToAddPhi;
            indices2[1] = b.getInt32(CURRENT_THREAD_STATUS_FLAG);
            Value * addThreadStateFlagPtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            Value * addThreadStateFlag = b.CreateLoad(sizeTy, addThreadStateFlagPtr);
            Value * canUseThreadStruct = b.CreateICmpNE(addThreadStateFlag, sz_ONE);
            Value * const nextToCheckForAdd = b.CreateAdd(selectToAddPhi, sz_ONE);
            selectToAddPhi->addIncoming(nextToCheckForAdd, selectThreadStructToUseForAddThread);
            b.CreateCondBr(canUseThreadStruct, checkIfThreadIsCancelled, selectThreadStructToUseForAddThread);

            b.SetInsertPoint(checkIfThreadIsCancelled);
            Value * isCancelled = b.CreateICmpEQ(addThreadStateFlag, sz_TWO);
            indices2[1] = b.getInt32(CURRENT_THREAD_ID);
            Value * const threadIdPtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            b.CreateCondBr(isCancelled, joinCancelledThread, addThread);

            b.SetInsertPoint(joinCancelledThread);
            FixedArray<Value *, 2> pthreadJoinArgs;
            Value * threadId = b.CreateLoad(pThreadTy, threadIdPtr);
            pthreadJoinArgs[0] = threadId;
            pthreadJoinArgs[1] = b.CreateAllocaAtEntryPoint(voidPtrTy);
            b.CreateCall(pthreadJoinFn->getFunctionType(), pthreadJoinFn, pthreadJoinArgs);
            b.CreateBr(addThread);

            b.SetInsertPoint(addThread);
            b.CreateStore(sz_ONE, addThreadStateFlagPtr);
            pthreadCreateArgs[0] = threadIdPtr;
            Value * const ts = b.CreateInBoundsGEP(threadStructTy, threadStruct, selectToAddPhi);
            pthreadCreateArgs[3] = b.CreatePointerCast(ts, voidPtrTy);
            b.CreateCall(pthreadCreateFn->getFunctionType(), pthreadCreateFn, pthreadCreateArgs);
            Value * numOfThreadsAfterAdd = b.CreateAdd(activeThreadsPhi, b.getSize(1));
            b.CreateBr(recordBeforeNextSegment);

            b.SetInsertPoint(checkToRemoveThread);
            Value * const syncOverheadHigh = b.CreateFCmpUGT(fSyncOverhead, syncRemoveThreadThreadhold);
            Value * const canRemoveMoreThreads = b.CreateICmpUGT(activeThreadsPhi, minimumThreads);
            Value * const canRemove = b.CreateAnd(syncOverheadHigh, canRemoveMoreThreads);
            b.CreateCondBr(canRemove, selectThreadToRemove, recordBeforeNextSegment);

            b.SetInsertPoint(selectThreadToRemove);
            PHINode * const selectedThreadPhi = b.CreatePHI(sizeTy, 2);
            selectedThreadPhi->addIncoming(sz_ONE, checkToRemoveThread);
            indices2[0] = selectedThreadPhi;
            indices2[1] = b.getInt32(CURRENT_THREAD_STATUS_FLAG);
            Value * const cancelFlagPtr = b.CreateInBoundsGEP(threadStructTy, threadStruct, indices2);
            Value * const isActive = b.CreateICmpEQ(b.CreateLoad(sizeTy, cancelFlagPtr), sz_ONE);
            Value * const nextThreadToCheckForRemove = b.CreateAdd(selectedThreadPhi, sz_ONE);
            selectedThreadPhi->addIncoming(nextThreadToCheckForRemove, selectThreadToRemove);
            b.CreateCondBr(isActive, removeThread, selectThreadToRemove);

            b.SetInsertPoint(removeThread);
            // mark this thread to terminate when it reaches the end of a segment iteration
            b.CreateStore(sz_TWO, cancelFlagPtr);
            Value * const numOfThreadsAfterRemoval = b.CreateSub(activeThreadsPhi, sz_ONE);
            b.CreateBr(recordBeforeNextSegment);

            PHINode * numOfThreadsPhi = nullptr;
            BasicBlock * recordBeforeNextSegmentExit = nullptr;
            if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
                b.SetInsertPoint(recordBeforeNextSegment);

                numOfThreadsPhi = b.CreatePHI(sizeTy, 2);
                numOfThreadsPhi->addIncoming(numOfThreadsAfterAdd, addThread);
                numOfThreadsPhi->addIncoming(activeThreadsPhi, checkToRemoveThread);
                numOfThreadsPhi->addIncoming(numOfThreadsAfterRemoval, removeThread);

                recordDynamicThreadingState(b, mSegNo, fSyncOverhead, numOfThreadsPhi);

                recordBeforeNextSegmentExit = b.GetInsertBlock();

                b.CreateBr(nextSegment);
            }

            b.SetInsertPoint(nextSegment);
            startOfNextPeriodPhi = b.CreatePHI(sizeTy, 3, "startOfNextPeriodPhi");
            startOfNextPeriodPhi->addIncoming(nextCheckSegmentPhi, loopEntry);
            if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
                startOfNextPeriodPhi->addIncoming(startOfNextPeriod, recordBeforeNextSegmentExit);
            } else {
                startOfNextPeriodPhi->addIncoming(startOfNextPeriod, addThread);
                startOfNextPeriodPhi->addIncoming(startOfNextPeriod, removeThread);
                startOfNextPeriodPhi->addIncoming(startOfNextPeriod, checkToRemoveThread);
            }


            currentNumOfThreadsPhi = b.CreatePHI(sizeTy, 3, "currentNumOfThreadsPhi");
            currentNumOfThreadsPhi->addIncoming(activeThreadsPhi, loopEntry);
            if (LLVM_UNLIKELY(TraceDynamicMultithreading)) {
                currentNumOfThreadsPhi->addIncoming(numOfThreadsPhi, recordBeforeNextSegmentExit);
            } else {
                currentNumOfThreadsPhi->addIncoming(numOfThreadsAfterAdd, addThread);
                currentNumOfThreadsPhi->addIncoming(activeThreadsPhi, checkToRemoveThread);
                currentNumOfThreadsPhi->addIncoming(numOfThreadsAfterRemoval, removeThread);
            }
        }

        if (mIsNestedPipeline) {
            b.CreateBr(mPipelineEnd);
        } else {
            BasicBlock * const exitBlock = b.GetInsertBlock();
            if (LLVM_UNLIKELY(CheckAssertions)) {
                mMadeProgressInLastSegment->addIncoming(madeProgress, exitBlock);
            }
            if (processThreadForDynamicScheduling) {
                nextCheckSegmentPhi->addIncoming(startOfNextPeriodPhi, exitBlock);
                activeThreadsPhi->addIncoming(currentNumOfThreadsPhi, exitBlock);
            } else if (mUseDynamicMultithreading) {
                FixedArray<Value *, 2> indices2;
                indices2[0] = sz_ZERO;
                indices2[1] = b.getInt32(CURRENT_THREAD_STATUS_FLAG);
                Value * const statusFlagPtr = b.CreateGEP(threadStructTy, threadStruct, indices2);
                Value * const cancelled = b.CreateLoad(sizeTy, statusFlagPtr);
                done = b.CreateOr(done, b.CreateICmpEQ(cancelled, sz_TWO));
            } else {
                incrementCurrentSegNo(b, exitBlock);
            }
            assert (hasTermSignal);
            b.CreateUnlikelyCondBr(done, mPipelineEnd, mPipelineLoop);
        }

        b.SetInsertPoint(mPipelineEnd);
        assert (isFromCurrentFunction(b, getHandle(), !mTarget->isStateful()));
        assert (isFromCurrentFunction(b, getThreadLocalHandle(), !mTarget->hasThreadLocal()));
        #ifdef PRINT_DEBUG_MESSAGES
        if (mIsNestedPipeline) {
            debugPrint(b, "------------------------------------------------- END %" PRIx64, getHandle());
        } else {
            debugPrint(b, "================================================= END %" PRIx64, getHandle());
        }
        #endif

        #ifdef ENABLE_PAPI
        recordTotalPAPIMeasurement(b);
        #endif
        updateTotalCycleCounterTime(b);

        mExpectedNumOfStridesMultiplier = nullptr;
        mThreadLocalStreamSetBaseAddress = nullptr;
        #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
        mSegNo = mBaseSegNo;
        #endif

        BasicBlock * exitThread  = nullptr;
        BasicBlock * exitFunction  = nullptr;
        // only call pthread_exit() within spawned threads; otherwise it'll be equivalent to calling exit() within the process
        exitThread = b.CreateBasicBlock("ExitThread");
        exitFunction = b.CreateBasicBlock("ExitProcessFunction");
        Value * retVal = nullptr;
        if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
            retVal = b.CreateIntToPtr(b.CreateZExt(mSegNo, intPtrTy), voidPtrTy);
        } else {
            retVal = ConstantPointerNull::getNullValue(voidPtrTy);
        }
        b.CreateCondBr(isProcessThread(b, threadStructTy, threadStruct), exitFunction, exitThread);
        b.SetInsertPoint(exitThread);
        #ifdef ENABLE_PAPI
        unregisterPAPIThread(b);
        #endif
        b.CreateCall(pthreadExitFn->getFunctionType(), pthreadExitFn, retVal);
        b.CreateBr(exitFunction);
        b.SetInsertPoint(exitFunction);
        b.CreateRet(retVal);

    };

    makeThreadFunction(threadFunc, false);

    Function * processThreadFunc = nullptr;

    if (mUseDynamicMultithreading) {
        const auto outerFuncName = concat(mTarget->getName(), "_MultithreadedProcessThread", tmp);
        processThreadFunc = Function::Create(threadFuncType, Function::InternalLinkage, outerFuncName, m);
        processThreadFunc->setCallingConv(CallingConv::C);
        processThreadFunc->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);

        makeThreadFunction(processThreadFunc, true);
    } else {
        processThreadFunc = threadFunc;
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE DRIVER CONTINUED
    // -------------------------------------------------------------------------------------------------------------------------

    b.restoreIP(resumePoint);

    FixedArray<Value *, 1> processArgs;
    processArgs[0] = b.CreatePointerCast(processState, voidPtrTy);
    Value * const mainThreadRetVal =
        b.CreateCall(threadFuncType, processThreadFunc, processArgs);

    Value * firstSegNo = nullptr;
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        firstSegNo = b.CreatePtrToInt(mainThreadRetVal, intPtrTy);
    }

    assert (isFromCurrentFunction(b, processState));
    assert (isFromCurrentFunction(b, initialSharedState));
    assert (isFromCurrentFunction(b, initialThreadLocal));

    setHandle(initialSharedState);
    setThreadLocalHandle(initialThreadLocal);
    initializeScalarMap(b, InitializeOptions::DoNotIncludeThreadLocalScalars);

    Value * firstTerminationSignal = nullptr;
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        firstTerminationSignal = readTerminationSignalFromLocalState(b, threadStructTy, processState);
        assert (firstTerminationSignal);
    }

    // wait for all other threads to complete
    AllocaInst * const status = b.CreateAlloca(voidPtrTy);

    BasicBlock * const checkStatusOfThread = b.CreateBasicBlock("checkStatusOfThread");
    BasicBlock * const joinThread = b.CreateBasicBlock("joinThread");
    BasicBlock * const finalizeAfterJoinThread = b.CreateBasicBlock("finalizeAfterJoinThread");
    BasicBlock * const joinedThreads = b.CreateBasicBlock("joinedThreads");

    BasicBlock * const joinThreadEntry = b.GetInsertBlock();

    // join the threads and destroy any state objects
    b.CreateCondBr(moreThanOneThread, checkStatusOfThread, joinedThreads);

    b.SetInsertPoint(checkStatusOfThread);
    PHINode * const joinThreadIndex = b.CreatePHI(sizeTy, 2);
    joinThreadIndex->addIncoming(sz_ONE, joinThreadEntry);
    PHINode * finalTerminationSignalPhi = nullptr;
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        finalTerminationSignalPhi = b.CreatePHI(firstTerminationSignal->getType(), 2);
        finalTerminationSignalPhi->addIncoming(firstTerminationSignal, joinThreadEntry);
    }
    PHINode * finalSegNoPhi = nullptr;
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        finalSegNoPhi = b.CreatePHI(firstSegNo->getType(), 2);
        finalSegNoPhi->addIncoming(firstSegNo, joinThreadEntry);
    }

    fieldIndex[0] = joinThreadIndex;
    if (mUseDynamicMultithreading) {
        fieldIndex[1] = b.getInt32(CURRENT_THREAD_STATUS_FLAG);
        Value * const statusFlag = b.CreateLoad(sizeTy, b.CreateInBoundsGEP(threadStructTy, threadStateArray, fieldIndex));
        b.CreateCondBr(b.CreateICmpNE(statusFlag, sz_ZERO), joinThread, finalizeAfterJoinThread);
    } else {
        b.CreateBr(joinThread);
    }

    b.SetInsertPoint(joinThread);
    fieldIndex[1] = b.getInt32(CURRENT_THREAD_ID);
    FixedArray<Value *, 2> pthreadJoinArgs;
    Value * threadId = b.CreateLoad(pThreadTy, b.CreateInBoundsGEP(threadStructTy, threadStateArray, fieldIndex));
    pthreadJoinArgs[0] = threadId;
    pthreadJoinArgs[1] = status;
    b.CreateCall(pthreadJoinFn->getFunctionType(), pthreadJoinFn, pthreadJoinArgs);
    b.CreateBr(finalizeAfterJoinThread);

    b.SetInsertPoint(finalizeAfterJoinThread);

    // calculate the last segment # used by any kernel in case any reports require it.
    Value * finalSegNo = nullptr;
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        // Value * const retVal = b.CreatePointerCast(status, intPtrPtrTy);
        Value * const retVal = b.CreatePtrToInt(b.CreateLoad(voidPtrTy, status), intPtrTy);
        finalSegNo = b.CreateUMax(finalSegNoPhi, retVal);
    }

    Value * const jThreadState = b.CreateGEP(threadStructTy, threadStateArray, joinThreadIndex);
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        fieldIndex[1] = b.getInt32(THREAD_LOCAL_PARAM);
        Type * const handlePtrTy = getThreadLocalHandle()->getType();
        Value * const jThreadLocal = b.CreateLoad(handlePtrTy, b.CreateGEP(threadStructTy, threadStateArray, fieldIndex));
        SmallVector<Value *, 3> threadLocalArgs;
        if (LLVM_LIKELY(mTarget->isStateful())) {
            threadLocalArgs.push_back(initialSharedState);
        }
        threadLocalArgs.push_back(initialThreadLocal);
        threadLocalArgs.push_back(jThreadLocal);
        mTarget->finalizeThreadLocalInstance(b, threadLocalArgs);
        b.CreateFree(jThreadLocal);
        threadLocalArgs.pop_back();
    }
    Value * finalTerminationSignal = nullptr;
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        Value * const terminatedSignal = readTerminationSignalFromLocalState(b, threadStructTy, jThreadState);
        assert (terminatedSignal);
        finalTerminationSignal = b.CreateUMax(finalTerminationSignalPhi, terminatedSignal);
    }

    Value * const nextJoinIndex = b.CreateAdd(joinThreadIndex, sz_ONE);
    BasicBlock * const joinThreadExit = b.GetInsertBlock();
    joinThreadIndex->addIncoming(nextJoinIndex, joinThreadExit);
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        finalTerminationSignalPhi->addIncoming(finalTerminationSignal, joinThreadExit);
    }
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        finalSegNoPhi->addIncoming(finalSegNo, joinThreadExit);
    }
    Value * const joinMoreThreads = b.CreateICmpULT(nextJoinIndex, maximumNumOfThreads);
    b.CreateCondBr(joinMoreThreads, checkStatusOfThread, joinedThreads);

    b.SetInsertPoint(joinedThreads);
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        finalTerminationSignalPhi = b.CreatePHI(firstTerminationSignal->getType(), 2);
        finalTerminationSignalPhi->addIncoming(firstTerminationSignal, joinThreadEntry);
        finalTerminationSignalPhi->addIncoming(finalTerminationSignal, joinThreadExit);
    }
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        PHINode * phi = b.CreatePHI(firstSegNo->getType(), 2);
        phi->addIncoming(firstSegNo, joinThreadEntry);
        phi->addIncoming(finalSegNo, joinThreadExit);
        mSegNo = phi;
    } else {
        mSegNo = nullptr;
    }
    b.CreateFree(threadStateArray);
    restoreDoSegmentState(storedState);
    if (PipelineHasTerminationSignal) {
        assert (initialTerminationSignalPtr);
        b.CreateStore(finalTerminationSignalPhi, initialTerminationSignalPtr);
    }

    assert (getHandle() == initialSharedState);
    assert (getThreadLocalHandle() == initialThreadLocal);
    assert (b.getCompiler() == this);

    updateExternalConsumedItemCounts(b);
    updateExternalProducedItemCounts(b);

    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        const auto type = isDataParallel(FirstKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        Value * const ptr = getSynchronizationLockPtrForKernel(b, FirstKernel, type);
        assert (isFromCurrentFunction(b, ptr));
        b.CreateStore(mSegNo, ptr);
        concludeStridesPerSegmentRecording(b);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief start
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::start(KernelBuilder & b) {

    mCurrentKernelName = mKernelName[PipelineInput];
    makePartitionEntryPoints(b);

    if (CheckAssertions) {
        mRethrowException = b.WriteDefaultRethrowBlock();
    }


    mExpectedNumOfStridesMultiplier = b.getScalarField(EXPECTED_NUM_OF_STRIDES_MULTIPLIER);
    initializeFlowControl(b);
    readExternalConsumerItemCounts(b);
    loadInternalStreamSetHandles(b, true);
    loadInternalStreamSetHandles(b, false);

    mKernel = nullptr;
    mKernelId = 0;
    mAddressableItemCountPtr.clear();
    mVirtualBaseAddressPtr.clear();
    mPipelineProgress = b.getFalse();

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getThreadStateType
 ** ------------------------------------------------------------------------------------------------------------- */
StructType * PipelineCompiler::getThreadStuctType(KernelBuilder & b, const std::vector<Value *> & props) const {
    FixedArray<Type *, THREAD_STRUCT_SIZE + 1> fields; // +1 for the cache line padding
    LLVMContext & C = b.getContext();
    IntegerType * const sizeTy = b.getSizeTy();
    Type * const emptyTy = StructType::get(C);

    // NOTE: both the shared and thread local objects are parameters to the kernel.
    // They get automatically set by reading in the appropriate params.

    if (LLVM_LIKELY(mTarget->isStateful())) {
        fields[SHARED_STATE_PARAM] = getHandle()->getType();
        assert (fields[SHARED_STATE_PARAM]->isPointerTy());
    } else {
        fields[SHARED_STATE_PARAM] = emptyTy;
    }

    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        fields[THREAD_LOCAL_PARAM] = getThreadLocalHandle()->getType();
        assert (fields[THREAD_LOCAL_PARAM]->isPointerTy());
    } else {
        fields[THREAD_LOCAL_PARAM] = emptyTy;
    }

    const auto n = props.size();
    std::vector<Type *> paramType(n);
    for (unsigned i = 0; i < n; ++i) {
        paramType[i] = props[i]->getType();
    }
    fields[PIPELINE_PARAMS] = StructType::get(b.getContext(), paramType);

    if (mUseDynamicMultithreading) {
        fields[INITIAL_SEG_NO] = emptyTy;
        fields[FIXED_NUMBER_OF_THREADS] = emptyTy;
        fields[ACCUMULATED_SEGMENT_TIME] = sizeTy;
        fields[ACCUMULATED_SYNCHRONIZATION_TIME] = sizeTy;
    } else {
        fields[INITIAL_SEG_NO] = sizeTy;
        fields[FIXED_NUMBER_OF_THREADS] = sizeTy;
        fields[ACCUMULATED_SEGMENT_TIME] = emptyTy;
        fields[ACCUMULATED_SYNCHRONIZATION_TIME] = emptyTy;
    }

    Type * const pthreadTy = TypeBuilder<pthread_t, false>::get(b.getContext());
    assert (pthreadTy == b.getModule()->getFunction("pthread_self")->getReturnType());
    fields[CURRENT_THREAD_ID] = pthreadTy;
    const auto hasTermSignal = !mIsNestedPipeline || PipelineHasTerminationSignal;
    if (LLVM_LIKELY(hasTermSignal)) {
        fields[TERMINATION_SIGNAL] = sizeTy;
    } else {
        fields[TERMINATION_SIGNAL] = emptyTy;
    }
    if (mUseDynamicMultithreading) {
        fields[CURRENT_THREAD_STATUS_FLAG] = sizeTy;
    } else {
        fields[CURRENT_THREAD_STATUS_FLAG] = emptyTy;
    }

    DataLayout dl(b.getModule());
    // add padding to force this struct to be cache-line-aligned
    uint64_t structSize = 0UL;
    for (unsigned i = 0; i < THREAD_STRUCT_SIZE; ++i) {
        structSize += b.getTypeSize(dl, fields[i]);
    }
    const auto cl = b.getCacheAlignment();
    const auto paddingBytes = (2 * cl) - (structSize % cl);
    IntegerType * const int8Ty = b.getInt8Ty();
    fields[THREAD_STRUCT_SIZE] = ArrayType::get(int8Ty, paddingBytes);
    return StructType::get(C, fields);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initThreadStructObject
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeThreadStructObject(KernelBuilder & b,
                                               StructType * const threadStateTy,
                                               Value * const threadState,
                                               Value * const shared, Value * const threadLocal,
                                               const std::vector<Value *> & props,
                                               Value * const threadNum, Value * const numOfThreads) const {

    FixedArray<Value *, 2> indices2;
    indices2[0] = b.getInt32(0);
    if (LLVM_LIKELY(mTarget->isStateful())) {
        indices2[1] = b.getInt32(SHARED_STATE_PARAM);
        b.CreateStore(shared, b.CreateInBoundsGEP(threadStateTy, threadState, indices2));
    }
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        indices2[1] = b.getInt32(THREAD_LOCAL_PARAM);
        b.CreateStore(threadLocal, b.CreateInBoundsGEP(threadStateTy, threadState, indices2));
    }
    const auto n = props.size();
    assert (threadStateTy->getStructElementType(PIPELINE_PARAMS)->getStructNumElements() == n);
    FixedArray<Value *, 3> indices3;
    indices3[0] = indices2[0];
    indices3[1] = b.getInt32(PIPELINE_PARAMS);
    for (unsigned i = 0; i < n; ++i) {
        indices3[2] = b.getInt32(i);
        b.CreateStore(props[i], b.CreateInBoundsGEP(threadStateTy, threadState, indices3));
    }

    if (mUseDynamicMultithreading) {
        Constant * const sz_ZERO = b.getSize(0);
        indices2[1] = b.getInt32(ACCUMULATED_SEGMENT_TIME);
        b.CreateStore(sz_ZERO, b.CreateInBoundsGEP(threadStateTy, threadState, indices2));
        indices2[1] = b.getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
        b.CreateStore(sz_ZERO, b.CreateInBoundsGEP(threadStateTy, threadState, indices2));
    } else {
        indices2[1] = b.getInt32(INITIAL_SEG_NO);
        b.CreateStore(threadNum, b.CreateInBoundsGEP(threadStateTy, threadState, indices2));
        indices2[1] = b.getInt32(FIXED_NUMBER_OF_THREADS);
        b.CreateStore(numOfThreads, b.CreateInBoundsGEP(threadStateTy, threadState, indices2));
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readThreadStuctObject
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readThreadStuctObject(KernelBuilder & b, StructType * const threadStateTy, Value * threadState) {
    Constant * const i32_ZERO = b.getInt32(0);
    IntegerType * const sizeTy = b.getSizeTy();
    FixedArray<Value *, 2> indices2;
    indices2[0] = i32_ZERO;
    if (mTarget->isStateful()) {
        indices2[1] = b.getInt32(SHARED_STATE_PARAM);
        Type * ty = mTarget->getSharedStateType()->getPointerTo();
        setHandle(b.CreateLoad(ty, b.CreateInBoundsGEP(threadStateTy, threadState, indices2)));
    }
    if (mTarget->hasThreadLocal()) {
        indices2[1] = b.getInt32(THREAD_LOCAL_PARAM);
        Type * ty = mTarget->getThreadLocalStateType()->getPointerTo();
        setThreadLocalHandle(b.CreateLoad(ty, b.CreateInBoundsGEP(threadStateTy, threadState, indices2)));
    }
    if (mUseDynamicMultithreading) {
        indices2[1] = b.getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
        mAccumulatedSynchronizationTimePtr = b.CreateInBoundsGEP(threadStateTy, threadState, indices2);
    } else {
        indices2[1] = b.getInt32(INITIAL_SEG_NO);
        mSegNo = b.CreateLoad(sizeTy, b.CreateInBoundsGEP(threadStateTy, threadState, indices2));
        mAccumulatedSynchronizationTimePtr = nullptr;
        indices2[1] = b.getInt32(FIXED_NUMBER_OF_THREADS);
        mNumOfFixedThreads = b.CreateLoad(sizeTy, b.CreateInBoundsGEP(threadStateTy, threadState, indices2));
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isProcessThread
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::isProcessThread(KernelBuilder & b, StructType * const threadStateTy, Value * const threadState) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(CURRENT_THREAD_ID);
    Value * const ptr = b.CreateInBoundsGEP(threadStateTy, threadState, indices);
    Type * const pthreadTy = TypeBuilder<pthread_t, false>::get(b.getContext());
    return b.CreateIsNull(b.CreateLoad(pthreadTy, ptr));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkPThreadLibrary
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::linkPThreadLibrary(KernelBuilder & b) {

    Type * const voidPtrTy = b.getVoidPtrTy();
    IntegerType * const intTy = IntegerType::getIntNTy(b.getContext(), sizeof(int) * CHAR_BIT);
    IntegerType * const pthreadTy = IntegerType::getIntNTy(b.getContext(), sizeof(pthread_t) * CHAR_BIT);

    BEGIN_SCOPED_REGION
    FunctionType * funTy = FunctionType::get(pthreadTy, false);
    b.LinkFunction("pthread_self", funTy, (void*)&pthread_self);
    END_SCOPED_REGION

    BEGIN_SCOPED_REGION
    FixedArray<Type *, 4> params;
    params[0] = pthreadTy->getPointerTo();
    params[1] = voidPtrTy;
    params[2] = voidPtrTy;
    params[3] = voidPtrTy;
    FunctionType * funTy = FunctionType::get(intTy, params, false);
    b.LinkFunction("pthread_create", funTy, (void*)&pthread_create);
    END_SCOPED_REGION

    BEGIN_SCOPED_REGION
    FixedArray<Type *, 2> params;
    params[0] = pthreadTy;
    params[1] = voidPtrTy->getPointerTo();
    FunctionType * funTy = FunctionType::get(intTy, params, false);
    b.LinkFunction("pthread_join", funTy, (void*)&pthread_join);
    END_SCOPED_REGION

    BEGIN_SCOPED_REGION
    FixedArray<Type *, 1> pthreadExitArgs;
    pthreadExitArgs[0] = voidPtrTy;
    FunctionType * pthreadExitFnTy = FunctionType::get(b.getVoidTy(), pthreadExitArgs, false);
    b.LinkFunction("pthread_exit", pthreadExitFnTy, (void*)pthread_exit); // ->addAttribute(0, llvm::Attribute::AttrKind::NoReturn);
    END_SCOPED_REGION
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateSingleThreadKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateSingleThreadKernelMethod(KernelBuilder & b) {
    assert (!mUseDynamicMultithreading);
    if (LLVM_UNLIKELY(mIsNestedPipeline)) {
        mSegNo = mExternalSegNo; assert (mExternalSegNo);
    } else {
        mSegNo = b.getSize(0);
    }
    mNumOfFixedThreads = b.getSize(1);

    #ifdef ENABLE_PAPI
    createPAPIMeasurementArrays(b);
    getPAPIEventSet(b);
    #endif
    startCycleCounter(b, CycleCounter::FULL_PIPELINE_TIME);
    #ifdef ENABLE_PAPI
    startPAPIMeasurement(b, PAPIKernelCounter::PAPI_FULL_PIPELINE_TIME);
    #endif
    start(b);

    mPipelineLoop = b.CreateBasicBlock("PipelineLoop");
    mPipelineEnd = b.CreateBasicBlock("PipelineEnd");
    BasicBlock * const entryBlock = b.GetInsertBlock();
    b.CreateBr(mPipelineLoop);

    b.SetInsertPoint(mPipelineLoop);
    mMadeProgressInLastSegment = b.CreatePHI(b.getInt1Ty(), 2, "madeProgressInLastSegment");
    mMadeProgressInLastSegment->addIncoming(b.getTrue(), entryBlock);
    obtainCurrentSegmentNumber(b, entryBlock);

    branchToInitialPartition(b);
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        setActiveKernel(b, i, true);
        executeKernel(b);
    }
    end(b);

    updateExternalConsumedItemCounts(b);
    updateExternalProducedItemCounts(b);

    #ifdef ENABLE_PAPI
    recordTotalPAPIMeasurement(b);
    #endif
    updateTotalCycleCounterTime(b);

    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet())) {
        // TODO: this isn't fully correct when this is a nested pipeline
        concludeStridesPerSegmentRecording(b);
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief end
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::end(KernelBuilder & b) {

    // A pipeline will end for one or two reasons:

    // 1) Process has *halted* due to insufficient external I/O.

    // 2) All pipeline sinks have terminated (i.e., any kernel that writes
    // to a pipeline output, is marked as having a side-effect, or produces
    // an input for some call in which no dependent kernels is a pipeline
    // sink).

    // TODO: if we determine that all of the pipeline I/O is consumed in one invocation of the
    // pipeline, we can avoid testing at the end whether its terminated.

    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    b.CreateBr(mPartitionEntryPoint[PartitionCount]);

    b.SetInsertPoint(mPartitionEntryPoint[PartitionCount]);
    #endif
    Value * terminated = nullptr;
    if (mIsNestedPipeline || mUseDynamicMultithreading) {
        if (PipelineHasTerminationSignal) {
            terminated = hasPipelineTerminated(b);
        }
        b.CreateBr(mPipelineEnd);
    } else {
        terminated = hasPipelineTerminated(b);
        Value * const done = b.CreateIsNotNull(terminated);
        if (LLVM_UNLIKELY(CheckAssertions)) {
            Value * const progressedOrFinished = b.CreateOr(mPipelineProgress, done);
            Value * const live = b.CreateOr(mMadeProgressInLastSegment, progressedOrFinished);
            b.CreateAssert(live, "Dead lock detected: pipeline could not progress after two iterations");
        }
        BasicBlock * const exitBlock = b.GetInsertBlock();
        mMadeProgressInLastSegment->addIncoming(mPipelineProgress, exitBlock);
        incrementCurrentSegNo(b, exitBlock);
        b.CreateUnlikelyCondBr(done, mPipelineEnd, mPipelineLoop);
    }
    b.SetInsertPoint(mPipelineEnd);

    if (PipelineHasTerminationSignal) {
        Value * const ptr = getTerminationSignalPtr();
        b.CreateStore(terminated, ptr);
    }

    #ifdef PRINT_DEBUG_MESSAGES
    if (mIsNestedPipeline) {
        debugPrint(b, "------------------------------------------------- END %" PRIx64, getHandle());
    } else {
        debugPrint(b, "================================================= END %" PRIx64, getHandle());
    }
    #endif

    mExpectedNumOfStridesMultiplier = nullptr;
    mThreadLocalStreamSetBaseAddress = nullptr;
    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    mSegNo = mBaseSegNo;
    #endif

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readTerminationSignalFromLocalState
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::readTerminationSignalFromLocalState(KernelBuilder & b, StructType * const threadStateTy, Value * const threadState) const {
    // TODO: generalize a OR/ADD/etc "combination" mechanism for thread-local to output scalars?
    assert (threadState);
    assert (PipelineHasTerminationSignal || !mIsNestedPipeline);
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(TERMINATION_SIGNAL);
    return b.CreateLoad(b.getSizeTy(), b.CreateInBoundsGEP(threadStateTy, threadState, indices));
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeTerminationSignalToLocalState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeTerminationSignalToLocalState(KernelBuilder & b, StructType * const threadStateTy, Value * const threadState, Value * const terminated) const {
    // TODO: generalize a OR/ADD/etc "combination" mechanism for thread-local to output scalars?
    assert (threadState);
    assert (PipelineHasTerminationSignal || !mIsNestedPipeline);
    FixedArray<Value *, 2> indices;
    indices[0] = b.getInt32(0);
    indices[1] = b.getInt32(TERMINATION_SIGNAL);
    b.CreateStore(terminated, b.CreateInBoundsGEP(threadStateTy, threadState, indices));
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief copyInternalState
 ** ------------------------------------------------------------------------------------------------------------- */
std::vector<Value *> PipelineCompiler::storeDoSegmentState() const {

    const auto numOfInputs = getNumOfStreamInputs();
    const auto numOfOutputs = getNumOfStreamOutputs();

    assert (!mTarget->hasAttribute(AttrId::InternallySynchronized));

    std::vector<Value *> S;
    S.reserve(4 + numOfInputs * 5 + numOfOutputs * 6);

    auto append = [&](Value * v) {
        if (v) S.push_back(v);
    };

    append(mIsFinal);
    append(mNumOfStrides);
    append(mFixedRateFactor);
    append(mExternalSegNo);

    auto copy = [&](const Vec<llvm::Value *> & V, const size_t n) {
        for (unsigned i = 0; i < n; ++i) {
            append(V[i]);
        }
    };

    copy(mInputIsClosed, numOfInputs);
    copy(mProcessedInputItemPtr, numOfInputs);
    copy(mAccessibleInputItems, numOfInputs);
    copy(mAvailableInputItems, numOfInputs);
    for (unsigned i = 0; i < numOfInputs; ++i) {
        assert(getInputStreamSetBuffer(i)->getHandle());
        append(getInputStreamSetBuffer(i)->getHandle());
    }

    copy(mProducedOutputItemPtr, numOfOutputs);
    copy(mUpdatableOutputBaseVirtualAddressPtr, numOfOutputs);
    copy(mInitiallyProducedOutputItems, numOfOutputs);
    copy(mWritableOutputItems, numOfOutputs);
    copy(mConsumedOutputItems, numOfOutputs);
    for (unsigned i = 0; i < numOfOutputs; ++i) {
        assert(getOutputStreamSetBuffer(i)->getHandle());
        append(getOutputStreamSetBuffer(i)->getHandle());
    }

    return S;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief restoreInternalState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readDoSegmentState(KernelBuilder & b, StructType * const threadStructTy, Value * const propertyState) {
    FixedArray<Value *, 3> indices3;
    indices3[0] = b.getInt32(0);
    indices3[1] = b.getInt32(PIPELINE_PARAMS);

    StructType * const paramType = cast<StructType>(threadStructTy->getStructElementType(PIPELINE_PARAMS));

    unsigned i = 0;
    #ifndef NDEBUG
    const auto n = paramType->getStructNumElements();
    #endif

    auto revertOne = [&](Value *& v, const bool accept) {
        if (accept) {
            assert (i < n);
            indices3[2] = b.getInt32(i);
            Value * ptr = b.CreateInBoundsGEP(threadStructTy, propertyState, indices3);
            v = b.CreateLoad(paramType->getStructElementType(i), ptr);
            ++i;
        }
    };

    revertOne(mIsFinal, mIsFinal != nullptr);
    revertOne(mNumOfStrides, mNumOfStrides != nullptr);
    revertOne(mFixedRateFactor, mFixedRateFactor != nullptr);
    revertOne(mExternalSegNo, mExternalSegNo != nullptr);

    auto revert = [&](Vec<llvm::Value *> & V, const size_t n) {
        for (unsigned j = 0; j < n; ++j) {
            revertOne(V[j], V[j] != nullptr);
        }
    };

    const auto numOfInputs = getNumOfStreamInputs();
    revert(mInputIsClosed, numOfInputs);
    revert(mProcessedInputItemPtr, numOfInputs);
    revert(mAccessibleInputItems, numOfInputs);
    revert(mAvailableInputItems, numOfInputs);
    for (unsigned j = 0; j < numOfInputs; ++j) {
        Value * handle = nullptr;
        revertOne(handle, true);
        getInputStreamSetBuffer(j)->setHandle(handle);
    }

    const auto numOfOutputs = getNumOfStreamOutputs();
    revert(mProducedOutputItemPtr, numOfOutputs);
    revert(mUpdatableOutputBaseVirtualAddressPtr, numOfOutputs);
    revert(mInitiallyProducedOutputItems, numOfOutputs);
    revert(mWritableOutputItems, numOfOutputs);
    revert(mConsumedOutputItems, numOfOutputs);

    for (unsigned j = 0; j < numOfOutputs; ++j) {
        Value * handle = nullptr;
        revertOne(handle, true);
        getOutputStreamSetBuffer(j)->setHandle(handle);
    }

    assert (i == n);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief restoreInternalState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::restoreDoSegmentState(const std::vector<Value *> & S) {

    auto o = S.begin();

    auto revertOne = [&](Value *& v, const bool accept) {
        if (accept) {
            assert (o != S.end());
            v = *o++;
        }
    };

    revertOne(mIsFinal, mIsFinal != nullptr);
    revertOne(mNumOfStrides, mNumOfStrides != nullptr);
    revertOne(mFixedRateFactor, mFixedRateFactor != nullptr);
    revertOne(mExternalSegNo, mExternalSegNo != nullptr);

    auto revert = [&](Vec<llvm::Value *> & V, const size_t n) {
        for (unsigned j = 0; j < n; ++j) {
            revertOne(V[j], V[j] != nullptr);
        }
    };

    const auto numOfInputs = getNumOfStreamInputs();
    revert(mInputIsClosed, numOfInputs);
    revert(mProcessedInputItemPtr, numOfInputs);
    revert(mAccessibleInputItems, numOfInputs);
    revert(mAvailableInputItems, numOfInputs);
    for (unsigned i = 0; i < numOfInputs; ++i) {
        Value * handle = nullptr;
        revertOne(handle, true);
        getInputStreamSetBuffer(i)->setHandle(handle);
    }

    const auto numOfOutputs = getNumOfStreamOutputs();
    revert(mProducedOutputItemPtr, numOfOutputs);
    revert(mUpdatableOutputBaseVirtualAddressPtr, numOfOutputs);
    revert(mInitiallyProducedOutputItems, numOfOutputs);
    revert(mWritableOutputItems, numOfOutputs);
    revert(mConsumedOutputItems, numOfOutputs);

    for (unsigned i = 0; i < numOfOutputs; ++i) {
        Value * handle = nullptr;
        revertOne(handle, true);
        getOutputStreamSetBuffer(i)->setHandle(handle);
    }

    assert (o == S.end());

}


}
