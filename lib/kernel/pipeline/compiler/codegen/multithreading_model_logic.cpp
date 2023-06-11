#include "../pipeline_compiler.hpp"
#include <pthread.h>

#if BOOST_OS_LINUX
#include <sched.h>
#endif

#if BOOST_OS_MACOS
#include <mach/mach_init.h>
#include <mach/thread_act.h>
#endif

#if BOOST_OS_MACOS
namespace llvm {
template<> class TypeBuilder<pthread_t, false> {
public:
  static Type *get(LLVMContext& C) {
    return IntegerType::getIntNTy(C, sizeof(pthread_t) * CHAR_BIT);
  }
};

template<> class TypeBuilder<pthread_attr_t, false> {
public:
  static Type *get(LLVMContext& C) {
    return IntegerType::getIntNTy(C, sizeof(pthread_attr_t) * CHAR_BIT);
  }
};
}
#endif

enum PipelineStateObjectField : unsigned {
    PIPELINE_PARAMS
    , INITIAL_SEG_NO
    , FIXED_NUMBER_OF_THREADS
    , ACCUMULATED_SEGMENT_TIME
    , ACCUMULATED_SYNCHRONIZATION_TIME
    , PROCESS_THREAD_ID
    , TERMINATION_SIGNAL
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
void PipelineCompiler::generateMultiThreadKernelMethod(BuilderRef b) {

    Module * const m = b->getModule();
    PointerType * const voidPtrTy = b->getVoidPtrTy();
    IntegerType * const boolTy = b->getInt1Ty();
    IntegerType * const sizeTy = b->getSizeTy();
    Type * const emptyTy = StructType::get(m->getContext());

    StructType * const threadStructTy = getThreadStuctType(b);

    ConstantInt * const i32_ZERO = b->getInt32(0);
    ConstantInt * const sz_ZERO = b->getSize(0);
    ConstantInt * const sz_ONE = b->getSize(1);

    SmallVector<char, 256> tmp;
    const auto threadName = concat(mTarget->getName(), "_DoFixedDataSegmentThread", tmp);

    FunctionType * const threadFuncType = FunctionType::get(voidPtrTy, {voidPtrTy}, false);
    Function * const threadFunc = Function::Create(threadFuncType, Function::InternalLinkage, threadName, m);

    Value * const initialSharedState = getHandle();
    Value * const initialThreadLocal = getThreadLocalHandle();
    Value * const initialTerminationSignalPtr = getTerminationSignalPtr();

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE DRIVER
    // -------------------------------------------------------------------------------------------------------------------------

    // use the process thread to handle the initial segment function after spawning
    // (n - 1) threads to handle the subsequent offsets

    Function * const pthreadSelfFn = m->getFunction("pthread_self");
    Function * const pthreadCreateFn = m->getFunction("pthread_create");
    Function * const pthreadExitFn = m->getFunction("pthread_exit");
    Function * const pthreadJoinFn = m->getFunction("pthread_join");

    Type * const pThreadTy = pthreadSelfFn->getReturnType();

    Value * const minimumNumOfThreads = b->getScalarField(MINIMUM_NUM_OF_THREADS);
    Value * maximumNumOfThreads = nullptr;
    if (mUseDynamicMultithreading) {
        maximumNumOfThreads = b->getScalarField(MAXIMUM_NUM_OF_THREADS);
    } else {
        maximumNumOfThreads = minimumNumOfThreads;
    }

    // TODO: probably isn't allowed unless we pass the scalar in as a function arg

    AllocaInst * const baseThreadIds =
        b->CreateCacheAlignedAlloca(pThreadTy, maximumNumOfThreads);

    AllocaInst * const baseThreadStateArray =
        b->CreateCacheAlignedAlloca(threadStructTy, maximumNumOfThreads);

    AllocaInst * baseThreadLocalArray = nullptr;
    PointerType * threadLocalHandlePtrTy = nullptr;
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        threadLocalHandlePtrTy = mTarget->getThreadLocalStateType()->getPointerTo();
        baseThreadLocalArray = b->CreateCacheAlignedAlloca(threadLocalHandlePtrTy, maximumNumOfThreads);
        b->CreateStore(initialThreadLocal, b->CreateGEP(baseThreadLocalArray, i32_ZERO));
    }

    DataLayout DL(b->getModule());
    Type * const intPtrTy = DL.getIntPtrType(voidPtrTy);
    PointerType * const intPtrPtrTy = intPtrTy->getPointerTo();

    Value * const processThreadId = b->CreateCall(pthreadSelfFn->getFunctionType(), pthreadSelfFn, {});

    BasicBlock * const constructThread = b->CreateBasicBlock("constructThread", mPipelineEnd);
    BasicBlock * const constructedThreads = b->CreateBasicBlock("constructedThreads", mPipelineEnd);

    Value * const moreThanOneThread = b->CreateICmpNE(maximumNumOfThreads, sz_ONE);

    BasicBlock * const constructThreadEntry = b->GetInsertBlock();

    // construct and start the threads

    b->CreateCondBr(moreThanOneThread, constructThread, constructedThreads);

    b->SetInsertPoint(constructThread);
    PHINode * const threadIndex = b->CreatePHI(sizeTy, 2);
    threadIndex->addIncoming(sz_ONE, constructThreadEntry);

    Value * cThreadLocal = nullptr;
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {

        SmallVector<Value *, 2> args;
        if (initialSharedState) {
            args.push_back(initialSharedState);
        }
        args.push_back(ConstantPointerNull::get(threadLocalHandlePtrTy));
        cThreadLocal = mTarget->initializeThreadLocalInstance(b, args);
        b->CreateStore(cThreadLocal, b->CreateGEP(baseThreadLocalArray, threadIndex));
        assert (isFromCurrentFunction(b, cThreadLocal));
        if (LLVM_LIKELY(mTarget->allocatesInternalStreamSets())) {
            Function * const allocInternal = mTarget->getAllocateThreadLocalInternalStreamSetsFunction(b, false);
            SmallVector<Value *, 3> allocArgs;
            if (LLVM_LIKELY(mTarget->isStateful())) {
                allocArgs.push_back(initialSharedState);
            }
            allocArgs.push_back(cThreadLocal);
            allocArgs.push_back(sz_ONE);
            b->CreateCall(allocInternal->getFunctionType(), allocInternal, allocArgs);
        }
    }
    Value * const cThreadState = b->CreateGEP(baseThreadStateArray, threadIndex);
    initThreadStructObject(b, cThreadState, processThreadId, cThreadLocal, threadIndex, maximumNumOfThreads);

    Value * const nextThreadIndex = b->CreateAdd(threadIndex, sz_ONE);

    BasicBlock * constructNextThread = nullptr;

    if (mUseDynamicMultithreading) {

        BasicBlock * const startThread = b->CreateBasicBlock("startThread", constructedThreads);

        constructNextThread = b->CreateBasicBlock("constructNextThread", constructedThreads);

        Value * const start = b->CreateICmpULT(nextThreadIndex, minimumNumOfThreads);

        b->CreateCondBr(start, startThread, constructNextThread);

    }

    FixedArray<Value *, 4> pthreadCreateArgs;
    FunctionType * const pthreadCreateFnTy = pthreadCreateFn->getFunctionType();
    pthreadCreateArgs[0] = b->CreateInBoundsGEP(baseThreadIds, threadIndex);
    Constant * const nullVoidPtrVal =
        ConstantPointerNull::get(cast<PointerType>(pthreadCreateFnTy->getParamType(1)));
    pthreadCreateArgs[1] = nullVoidPtrVal;
    pthreadCreateArgs[2] = threadFunc;
    pthreadCreateArgs[3] = b->CreatePointerCast(cThreadState, voidPtrTy);
    b->CreateCall(pthreadCreateFnTy, pthreadCreateFn, pthreadCreateArgs);

    if (mUseDynamicMultithreading) {
        b->CreateBr(constructNextThread);

        b->SetInsertPoint(constructNextThread);
    }

    Value * const createMoreThreads = b->CreateICmpULT(nextThreadIndex, maximumNumOfThreads);
    threadIndex->addIncoming(nextThreadIndex, b->GetInsertBlock());
    b->CreateCondBr(createMoreThreads, constructThread, constructedThreads);

    b->SetInsertPoint(constructedThreads);

    // execute the process thread
    Value * const pty_ZERO = Constant::getNullValue(pThreadTy);
    Value * const processState = b->CreateGEP(baseThreadStateArray, i32_ZERO);
    initThreadStructObject(b, processState, pty_ZERO, initialThreadLocal, sz_ZERO, maximumNumOfThreads);
    PointerType * const threadStructPtrTy = cast<PointerType>(processState->getType());

    // store where we'll resume compiling the DoSegment method
    const auto resumePoint = b->saveIP();
    const auto storedState = storeDoSegmentState();

    const auto anyDebugOptionIsSet = codegen::AnyDebugOptionIsSet();

    // -------------------------------------------------------------------------------------------------------------------------
    // GENERATE DO SEGMENT (KERNEL EXECUTION) FUNCTION CODE
    // -------------------------------------------------------------------------------------------------------------------------

    SmallVector<Type *, 3> csRetValFields;
    Type * csRetValType = nullptr;
    if (CheckAssertions) {
        csRetValType = boolTy; // hasProgressed
    } else {
        csRetValType = b->getVoidTy();
    }

    FixedArray<Type *, 2> csParams;
    csParams[0] = threadStructPtrTy; // thread state
    csParams[1] = sizeTy; // segment number

    FunctionType * const csFuncType = FunctionType::get(csRetValType, csParams, false);
    Function * const csFunc = Function::Create(csFuncType, Function::InternalLinkage, threadName, m);
    csFunc->setCallingConv(CallingConv::C);
    if (!mUseDynamicMultithreading) {
        csFunc->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);
    }

    b->SetInsertPoint(BasicBlock::Create(m->getContext(), "entry", csFunc));
    auto args = csFunc->arg_begin();
    Value * const threadStruct = &*args++;
    assert (threadStruct->getType() == threadStructPtrTy);
    readThreadStuctObject(b, threadStruct);
    assert (isFromCurrentFunction(b, getHandle()));
    assert (isFromCurrentFunction(b, getThreadLocalHandle()));
    mSegNo = &*args;
    #ifdef PRINT_DEBUG_MESSAGES
    debugInit(b);
    #endif
    Value * segmentStartTime = nullptr;
    if (mUseDynamicMultithreading) {
        segmentStartTime = b->CreateReadCycleCounter();
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
        Value * const segmentEndTime = b->CreateReadCycleCounter();
        Value * const totalSegmentTime = b->CreateSub(segmentEndTime, segmentStartTime);

        FixedArray<Value *, 2> indices2;
        Constant * const ZERO = b->getInt32(0);
        indices2[0] = b->getInt32(0);
        indices2[1] = b->getInt32(ACCUMULATED_SEGMENT_TIME);
        Value * const segPtr = b->CreateInBoundsGEP(threadStruct, indices2);
        Value * const current = b->CreateLoad(segPtr);
        Value * const accum = b->CreateAdd(current, totalSegmentTime);
        b->CreateStore(accum, segPtr);
    }
    const auto hasTermSignal = !mIsNestedPipeline || PipelineHasTerminationSignal;
    if (LLVM_LIKELY(hasTermSignal)) {
        writeTerminationSignalToLocalState(b, threadStruct, hasPipelineTerminated(b));
    }
    if (LLVM_UNLIKELY(CheckAssertions)) {
        b->CreateRet(mPipelineProgress);
    } else {
        b->CreateRetVoid();
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE THREAD
    // -------------------------------------------------------------------------------------------------------------------------
    auto makeThreadFunction = [&](Function * const threadFunc, const bool processThreadForDynamicScheduling) {
        threadFunc->setCallingConv(CallingConv::C);
        auto arg = threadFunc->arg_begin();
        arg->setName("threadStruct");

        b->SetInsertPoint(BasicBlock::Create(m->getContext(), "entry", threadFunc));

        Value * threadStructArray = nullptr;
        Value * threadStruct = nullptr;
        Value * segmentsPerCheck = nullptr;
        Value * minimumThreads = nullptr;
        Value * maximumThreads = nullptr;
        Value * fSyncAddThreadThreadhold = nullptr;
        Value * threadIds = nullptr; // pthreadsTyPtr

        if (processThreadForDynamicScheduling) {
            FixedArray<Value *, 2> offset;
            offset[0] = i32_ZERO;
            offset[1] = i32_ZERO;
            threadStructArray = b->CreatePointerCast(arg++, threadStructPtrTy);
            threadStruct = b->CreateGEP(threadStructArray, offset);
            assert (threadStruct->getType() == threadStructPtrTy);
            segmentsPerCheck = arg; // mDynamicMultithreadingSegmentsPerCheck
        } else {
            threadStruct = b->CreatePointerCast(arg, threadStructPtrTy);
        }

        readThreadStuctObject(b, threadStruct);

        #ifdef ENABLE_PAPI
        registerPAPIThread(b);
        #endif

        startCycleCounter(b, CycleCounter::FULL_PIPELINE_TIME);

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
        if (mIsFinal) {
            debugPrint(b, prefix + " +++ IS FINAL %" PRIu8 "+++", mIsFinal);
        }
        #endif

        #ifdef ENABLE_PAPI
        createEventSetAndStartPAPI(b);
        #endif

        // generate the pipeline logic for this thread
        mPipelineLoop = b->CreateBasicBlock("PipelineLoop");
        mPipelineEnd = b->CreateBasicBlock("PipelineEnd");
        BasicBlock * const entryBlock = b->GetInsertBlock();
        b->CreateBr(mPipelineLoop);

        b->SetInsertPoint(mPipelineLoop);
        if (LLVM_UNLIKELY(CheckAssertions)) {
            mMadeProgressInLastSegment = b->CreatePHI(b->getInt1Ty(), 2, "madeProgressInLastSegment");
            mMadeProgressInLastSegment->addIncoming(b->getTrue(), entryBlock);
        }
        PHINode * nextCheckSegmentPhi = nullptr;
        PHINode * activeThreadsPhi = nullptr;
        if (processThreadForDynamicScheduling) {
            nextCheckSegmentPhi = b->CreatePHI(sizeTy, 2, "nextCheckPhi");
            nextCheckSegmentPhi->addIncoming(segmentsPerCheck, entryBlock);
            activeThreadsPhi = b->CreatePHI(sizeTy, 2, "activeThreadsPhi");
            activeThreadsPhi->addIncoming(minimumThreads, entryBlock);
        }

        obtainCurrentSegmentNumber(b, entryBlock);

        SmallVector<Value *, 3> args(2);
        args[0] = threadStruct;
        args[1] = mSegNo; assert (mSegNo);
        Value * const csRetVal = b->CreateCall(csFuncType, csFunc, args);

        Value * terminated = nullptr;
        Value * done = nullptr;
        Value * madeProgress = nullptr;

        if (LLVM_LIKELY(hasTermSignal)) {
            terminated = readTerminationSignalFromLocalState(b, threadStruct);
            done = b->CreateIsNotNull(terminated);
        }
        if (LLVM_UNLIKELY(CheckAssertions)) {
            madeProgress = csRetVal;
            if (LLVM_LIKELY(hasTermSignal)) {
                madeProgress = b->CreateOr(madeProgress, done);
            }
            Value * const live = b->CreateOr(mMadeProgressInLastSegment, madeProgress);
            b->CreateAssert(live, "Dead lock detected: pipeline could not progress after two iterations");
        }

        if (mIsNestedPipeline) {
            b->CreateBr(mPipelineEnd);
        } else if (processThreadForDynamicScheduling) {
            assert (mUseDynamicMultithreading);

            BasicBlock * checkSynchronizationCostLoop = b->CreateBasicBlock("checkSynchronizationCostLoop", mPipelineEnd);
            BasicBlock * checkSynchronizationCost = b->CreateBasicBlock("checkToSynchronizationCost", mPipelineEnd);
            BasicBlock * addThread = b->CreateBasicBlock("addThread", mPipelineEnd);
            BasicBlock * nextSegment = b->CreateBasicBlock("nextSegment", mPipelineEnd);

            Value * const check = b->CreateICmpUGE(mSegNo, nextCheckSegmentPhi);
            BasicBlock * const loopEntry = b->GetInsertBlock();
            b->CreateUnlikelyCondBr(check, checkSynchronizationCostLoop, nextSegment);

            Constant * const sz_ZERO = b->getSize(0);

            b->SetInsertPoint(checkSynchronizationCostLoop);
            PHINode * const indexPhi = b->CreatePHI(sizeTy, 2);
            indexPhi->addIncoming(sz_ZERO, loopEntry);
            PHINode * const segmentTimeAccumPhi = b->CreatePHI(sizeTy, 2);
            segmentTimeAccumPhi->addIncoming(sz_ZERO, loopEntry);
            PHINode * const synchronizationTimeAccumPhi = b->CreatePHI(sizeTy, 2);
            synchronizationTimeAccumPhi->addIncoming(sz_ZERO, loopEntry);


            FixedArray<Value *, 2> indices3;
            indices3[0] = i32_ZERO;
            indices3[1] = indexPhi;
            indices3[2] = b->getInt32(ACCUMULATED_SEGMENT_TIME);
            Value * const segTimePtr = b->CreateInBoundsGEP(threadStructArray, indices3);
            Value * const nextSegTime = b->CreateAdd(segmentTimeAccumPhi, b->CreateLoad(segTimePtr));
            segmentTimeAccumPhi->addIncoming(nextSegTime, checkSynchronizationCostLoop);

            indices3[2] = b->getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
            Value * const syncTimePtr = b->CreateInBoundsGEP(threadStructArray, indices3);
            Value * const nextSyncTime = b->CreateAdd(synchronizationTimeAccumPhi, b->CreateLoad(syncTimePtr));
            synchronizationTimeAccumPhi->addIncoming(nextSyncTime, checkSynchronizationCostLoop);

            Value * const nextIndex = b->CreateAdd(indexPhi, b->getSize(1));
            indexPhi->addIncoming(nextIndex, checkSynchronizationCostLoop);
            Value * const hasMore = b->CreateICmpNE(nextIndex, activeThreadsPhi);
            b->CreateCondBr(hasMore, checkSynchronizationCostLoop, checkSynchronizationCost);

            b->SetInsertPoint(checkSynchronizationCost);
            Type * const dblTy = b->getDoubleTy();
            Value * const fSegTime = b->CreateUIToFP(nextSegTime, dblTy);
            Value * const fSyncTime = b->CreateUIToFP(nextSyncTime, dblTy);
            Value * const fSyncOverhead = b->CreateFDiv(fSyncTime, fSegTime);

            Value * const add = b->CreateFCmpULT(fSyncOverhead, fSyncAddThreadThreadhold);
            b->CreateCondBr(add, addThread, nextSegment);

            b->SetInsertPoint(addThread);

            FixedArray<Value *, 2> indices2;
            indices2[0] = i32_ZERO;
            indices2[1] = nextIndex;
            pthreadCreateArgs[0] = b->CreateInBoundsGEP(threadIds, indices2);
            // pthreadCreateArgs[1] = nullVoidPtrVal; // already set
            // pthreadCreateArgs[2] = threadFunc;
            Value * const ts = b->CreateInBoundsGEP(threadStructArray, indices2);
            pthreadCreateArgs[3] = b->CreatePointerCast(ts, voidPtrTy);
            b->CreateCall(pthreadCreateFn->getFunctionType(), pthreadCreateFn, pthreadCreateArgs);

        } else {
            BasicBlock * const exitBlock = b->GetInsertBlock();
            if (LLVM_UNLIKELY(CheckAssertions)) {
                mMadeProgressInLastSegment->addIncoming(madeProgress, exitBlock);
            }
            incrementCurrentSegNo(b, exitBlock);
            assert (hasTermSignal);
            b->CreateUnlikelyCondBr(done, mPipelineEnd, mPipelineLoop);
        }

        b->SetInsertPoint(mPipelineEnd);
        #ifdef PRINT_DEBUG_MESSAGES
        if (mIsNestedPipeline) {
            debugPrint(b, "------------------------------------------------- END %" PRIx64, getHandle());
        } else {
            debugPrint(b, "================================================= END %" PRIx64, getHandle());
        }
        #endif

        #ifdef ENABLE_PAPI
        stopPAPIAndDestroyEventSet(b);
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
        exitThread = b->CreateBasicBlock("ExitThread");
        exitFunction = b->CreateBasicBlock("ExitProcessFunction");
        Value * retVal = nullptr;
        if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
            retVal = b->CreateIntToPtr(b->CreateZExt(mSegNo, intPtrTy), voidPtrTy);
        } else {
            retVal = ConstantPointerNull::getNullValue(voidPtrTy);
        }
        b->CreateCondBr(isProcessThread(b, threadStruct), exitFunction, exitThread);
        b->SetInsertPoint(exitThread);
        #ifdef ENABLE_PAPI
        unregisterPAPIThread(b);
        #endif
        b->CreateCall(pthreadExitFn->getFunctionType(), pthreadExitFn, retVal);
        b->CreateBr(exitFunction);
        b->SetInsertPoint(exitFunction);
        b->CreateRet(retVal);
    };

    makeThreadFunction(threadFunc, false);

    Function * processThreadFunc = nullptr;

    if (mUseDynamicMultithreading) {

//        //    const static std::string MINIMUM_NUM_OF_THREADS = "MIN.T";
//        //    const static std::string MAXIMUM_NUM_OF_THREADS = "MAX.T";
//        //    const static std::string SEGMENTS_PER_CHECK = "SEG.C";
//        //    const static std::string ADDITIONAL_THREAD_SYNCHRONIZATION_THRESHOLD = "TST.A";

//        Value * const minimumNumOfThreads = b->getScalarField(MINIMUM_NUM_OF_THREADS);
//        Value * maximumNumOfThreads = nullptr;
//        if (mUseDynamicMultithreading) {
//            maximumNumOfThreads = b->getScalarField(MAXIMUM_NUM_OF_THREADS);
//        } else {
//            maximumNumOfThreads = minimumNumOfThreads;
//        }


        FixedArray<Type *, 2> param;
        param[0] = baseThreadStateArray->getType(); // thread state
        param[1] = sizeTy; // segments per check

        FunctionType * const csFuncType = FunctionType::get(b->getVoidTy(), param, false);
        Function * const csFunc = Function::Create(csFuncType, Function::InternalLinkage, threadName, m);
        csFunc->setCallingConv(CallingConv::C);
        csFunc->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);

        makeThreadFunction(processThreadFunc, true);
    } else {
        processThreadFunc = threadFunc;
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE DRIVER CONTINUED
    // -------------------------------------------------------------------------------------------------------------------------

    b->restoreIP(resumePoint);

    assert (isFromCurrentFunction(b, processState));
    assert (isFromCurrentFunction(b, initialSharedState));
    assert (isFromCurrentFunction(b, initialThreadLocal));

    Value * const mainThreadRetVal = b->CreateCall(threadFunc->getFunctionType(), threadFunc, b->CreatePointerCast(processState, voidPtrTy));
    Value * firstSegNo = nullptr;
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        firstSegNo = b->CreatePtrToInt(mainThreadRetVal, intPtrTy);
    }

    SmallVector<Value *, 2> threadLocalArgs;
    if (LLVM_LIKELY(mTarget->isStateful())) {
        threadLocalArgs.push_back(initialSharedState);
    }
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        threadLocalArgs.push_back(initialThreadLocal);
    }

    Value * firstTerminationSignal = nullptr;
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        firstTerminationSignal = readTerminationSignalFromLocalState(b, processState);
        assert (firstTerminationSignal);
    }
    destroyStateObject(b, processState);

    // wait for all other threads to complete
    AllocaInst * const status = b->CreateAlloca(voidPtrTy);

    BasicBlock * const joinThread = b->CreateBasicBlock("joinThread");
    BasicBlock * const joinedThreads = b->CreateBasicBlock("joinedThreads");

    BasicBlock * const joinThreadEntry = b->GetInsertBlock();

    // join the threads and destroy any state objects
    b->CreateCondBr(moreThanOneThread, joinThread, joinedThreads);

    b->SetInsertPoint(joinThread);
    PHINode * const joinThreadIndex = b->CreatePHI(sizeTy, 2);
    joinThreadIndex->addIncoming(sz_ONE, joinThreadEntry);
    PHINode * finalTerminationSignalPhi = nullptr;
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        finalTerminationSignalPhi = b->CreatePHI(firstTerminationSignal->getType(), 2);
        finalTerminationSignalPhi->addIncoming(firstTerminationSignal, joinThreadEntry);
    }
    PHINode * finalSegNoPhi = nullptr;
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        finalSegNoPhi = b->CreatePHI(firstSegNo->getType(), 2);
        finalSegNoPhi->addIncoming(firstSegNo, joinThreadEntry);
    }

    FixedArray<Value *, 2> pthreadJoinArgs;
    Value * threadId = b->CreateLoad(b->CreateInBoundsGEP(baseThreadIds, joinThreadIndex));
    pthreadJoinArgs[0] = threadId;
    pthreadJoinArgs[1] = status;
    b->CreateCall(pthreadJoinFn->getFunctionType(), pthreadJoinFn, pthreadJoinArgs);

    // calculate the last segment # used by any kernel in case any reports require it.
    Value * finalSegNo = nullptr;
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        Value * const retVal = b->CreatePointerCast(status, intPtrPtrTy);
        finalSegNo = b->CreateUMax(finalSegNoPhi, b->CreateLoad(retVal));
    }
    Value * const jThreadState = b->CreateGEP(baseThreadStateArray, joinThreadIndex);
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        Value * const jThreadLocal = b->CreateLoad(b->CreateGEP(baseThreadLocalArray, joinThreadIndex));
        threadLocalArgs.push_back(jThreadLocal);
        mTarget->finalizeThreadLocalInstance(b, threadLocalArgs);
        b->CreateFree(jThreadLocal);
        threadLocalArgs.pop_back();
    }
    Value * finalTerminationSignal = nullptr;
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        Value * const terminatedSignal = readTerminationSignalFromLocalState(b, jThreadState);
        assert (terminatedSignal);
        finalTerminationSignal = b->CreateUMax(finalTerminationSignalPhi, terminatedSignal);
    }
    destroyStateObject(b, jThreadState);

    Value * const nextJoinIndex = b->CreateAdd(joinThreadIndex, sz_ONE);
    BasicBlock * const joinThreadExit = b->GetInsertBlock();
    joinThreadIndex->addIncoming(nextJoinIndex, joinThreadExit);
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        finalTerminationSignalPhi->addIncoming(finalTerminationSignal, joinThreadExit);
    }
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        finalSegNoPhi->addIncoming(finalSegNo, joinThreadExit);
    }
    Value * const joinMoreThreads = b->CreateICmpULT(nextJoinIndex, maximumNumOfThreads);
    b->CreateCondBr(joinMoreThreads, joinThread, joinedThreads);

    b->SetInsertPoint(joinedThreads);
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        finalTerminationSignalPhi = b->CreatePHI(firstTerminationSignal->getType(), 2);
        finalTerminationSignalPhi->addIncoming(firstTerminationSignal, joinThreadEntry);
        finalTerminationSignalPhi->addIncoming(finalTerminationSignal, joinThreadExit);
    }
    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        PHINode * phi = b->CreatePHI(firstSegNo->getType(), 2);
        phi->addIncoming(firstSegNo, joinThreadEntry);
        phi->addIncoming(finalSegNo, joinThreadExit);
        mSegNo = phi;
    } else {
        mSegNo = nullptr;
    }
    restoreDoSegmentState(storedState);
    if (PipelineHasTerminationSignal) {
        assert (initialTerminationSignalPtr);
        b->CreateStore(finalTerminationSignalPhi, initialTerminationSignalPtr);
    }

    assert (getHandle() == initialSharedState);
    assert (getThreadLocalHandle() == initialThreadLocal);
    assert (b->getCompiler() == this);

    initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);
    updateExternalConsumedItemCounts(b);
    updateExternalProducedItemCounts(b);

    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        const auto type = isDataParallel(FirstKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        Value * const ptr = getSynchronizationLockPtrForKernel(b, FirstKernel, type);
        assert (isFromCurrentFunction(b, ptr));
        b->CreateStore(mSegNo, ptr);
        concludeStridesPerSegmentRecording(b);
    }

   // b->getModule()->dump();

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief start
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::start(BuilderRef b) {

    mCurrentKernelName = mKernelName[PipelineInput];
    makePartitionEntryPoints(b);

    if (CheckAssertions) {
        mRethrowException = b->WriteDefaultRethrowBlock();
    }

    mExpectedNumOfStridesMultiplier = b->getScalarField(EXPECTED_NUM_OF_STRIDES_MULTIPLIER);
    initializeFlowControl(b);
    readExternalConsumerItemCounts(b);
    loadInternalStreamSetHandles(b, true);
    loadInternalStreamSetHandles(b, false);

    mKernel = nullptr;
    mKernelId = 0;
    mAddressableItemCountPtr.clear();
    mVirtualBaseAddressPtr.clear();
    mNumOfTruncatedInputBuffers = 0;
    mTruncatedInputBuffer.clear();
    mPipelineProgress = b->getFalse();


}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getThreadStateType
 ** ------------------------------------------------------------------------------------------------------------- */
StructType * PipelineCompiler::getThreadStuctType(BuilderRef b) const {
    FixedArray<Type *, THREAD_STRUCT_SIZE + 1> fields;
    LLVMContext & C = b->getContext();
    IntegerType * const sizeTy = b->getSizeTy();
    Type * const emptyTy = StructType::get(C);

    // NOTE: both the shared and thread local objects are parameters to the kernel.
    // They get automatically set by reading in the appropriate params.

    fields[PIPELINE_PARAMS] = StructType::get(C, mTarget->getDoSegmentFields(b));
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

    Function * const pthreadSelfFn = b->getModule()->getFunction("pthread_self");
    fields[PROCESS_THREAD_ID] = pthreadSelfFn->getReturnType();
    const auto hasTermSignal = !mIsNestedPipeline || PipelineHasTerminationSignal;
    if (LLVM_LIKELY(hasTermSignal)) {
        fields[TERMINATION_SIGNAL] = sizeTy;
    } else {
        fields[TERMINATION_SIGNAL] = emptyTy;
    }

    DataLayout dl(b->getModule());

    auto getTypeSize = [&](Type * const type) -> uint64_t { assert (type);
        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
        return dl.getTypeAllocSize(type);
        #else
        return dl.getTypeAllocSize(type).getFixedSize();
        #endif
    };

    // add padding to force this struct to be cache-line-aligned
    uint64_t structSize = 0UL;
    for (unsigned i = 0; i < THREAD_STRUCT_SIZE; ++i) {
        structSize += getTypeSize(fields[i]);
    }
    const auto cl = b->getCacheAlignment();
    const auto paddingBytes = (2 * cl) - (structSize % cl);
    IntegerType * const int8Ty = b->getInt8Ty();
    fields[THREAD_STRUCT_SIZE] = ArrayType::get(int8Ty, paddingBytes);
    return StructType::get(C, fields);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initThreadStructObject
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initThreadStructObject(BuilderRef b, Value * threadState, Value * const threadId, Value * const threadLocal, Value * const threadNum, Value * const numOfThreads) {

    setThreadLocalHandle(threadLocal);
    const auto props = getDoSegmentProperties(b);
    const auto n = props.size();
  //  assert (threadState->getType()->getPointerElementType()->getStructElementType(PIPELINE_PARAMS)->getStructNumElements() == n);

    FixedArray<Value *, 3> indices3;
    indices3[0] = b->getInt32(0);
    indices3[1] = b->getInt32(PIPELINE_PARAMS);
    for (unsigned i = 0; i < n; ++i) {
        indices3[2] = b->getInt32(i);
        b->CreateStore(props[i], b->CreateInBoundsGEP(threadState, indices3));
    }
    FixedArray<Value *, 2> indices2;
    indices2[0] = b->getInt32(0);
    if (mUseDynamicMultithreading) {
        Constant * const sz_ZERO = b->getSize(0);
        indices2[1] = b->getInt32(ACCUMULATED_SEGMENT_TIME);
        b->CreateStore(sz_ZERO, b->CreateInBoundsGEP(threadState, indices2));
        indices2[1] = b->getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
        b->CreateStore(sz_ZERO, b->CreateInBoundsGEP(threadState, indices2));
    } else {
        indices2[1] = b->getInt32(INITIAL_SEG_NO);
        b->CreateStore(threadNum, b->CreateInBoundsGEP(threadState, indices2));
        indices2[1] = b->getInt32(FIXED_NUMBER_OF_THREADS);
        b->CreateStore(numOfThreads, b->CreateInBoundsGEP(threadState, indices2));
    }
    indices2[1] = b->getInt32(PROCESS_THREAD_ID);
    b->CreateStore(threadId, b->CreateInBoundsGEP(threadState, indices2));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readThreadStuctObject
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readThreadStuctObject(BuilderRef b, Value * threadState) {
    FixedArray<Value *, 3> indices3;
    Constant * const ZERO = b->getInt32(0);
    indices3[0] = ZERO;
    indices3[1] = b->getInt32(PIPELINE_PARAMS);
    Type * const kernelStructType = threadState->getType()->getPointerElementType();
    const auto n = kernelStructType->getStructElementType(PIPELINE_PARAMS)->getStructNumElements();
    SmallVector<Value *, 16> args(n);
    for (unsigned i = 0; i != n; ++i) {
        indices3[2] = b->getInt32(i);
        args[i] = b->CreateLoad(b->CreateInBoundsGEP(threadState, indices3));
    }
    setDoSegmentProperties(b, args);

    FixedArray<Value *, 2> indices2;
    indices2[0] = ZERO;
    if (mUseDynamicMultithreading) {
        indices2[1] = b->getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
        mAccumulatedSynchronizationTimePtr = b->CreateInBoundsGEP(threadState, indices2);
    } else {
        indices2[1] = b->getInt32(INITIAL_SEG_NO);
        mSegNo = b->CreateLoad(b->CreateInBoundsGEP(threadState, indices2));
        mAccumulatedSynchronizationTimePtr = nullptr;
        indices2[1] = b->getInt32(FIXED_NUMBER_OF_THREADS);
        mNumOfFixedThreads = b->CreateLoad(b->CreateInBoundsGEP(threadState, indices2));
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isProcessThread
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::isProcessThread(BuilderRef b, Value * const threadState) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(PROCESS_THREAD_ID);
    Value * const ptr = b->CreateInBoundsGEP(threadState, indices);
    return b->CreateIsNull(b->CreateLoad(ptr));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkPThreadLibrary
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::linkPThreadLibrary(BuilderRef b) {
    b->LinkFunction("pthread_self", pthread_self);
   // b->LinkFunction("pthread_setaffinity_np", pthread_setaffinity_np);
    b->LinkFunction("pthread_create", pthread_create);
    b->LinkFunction("pthread_join", pthread_join);
    // pthread_exit seems difficult to resolve in MacOS? manually doing it here but should be looked into
    FixedArray<Type *, 1> pthreadExitArgs;
    pthreadExitArgs[0] = b->getVoidPtrTy();
    FunctionType * pthreadExitFnTy = FunctionType::get(b->getVoidTy(), pthreadExitArgs, false);
    b->LinkFunction("pthread_exit", pthreadExitFnTy, (void*)pthread_exit); // ->addAttribute(0, llvm::Attribute::AttrKind::NoReturn);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateSingleThreadKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateSingleThreadKernelMethod(BuilderRef b) {
    if (LLVM_UNLIKELY(mIsNestedPipeline)) {
        mSegNo = mExternalSegNo; assert (mExternalSegNo);
    } else {
        mSegNo = b->getSize(0);
    }
    mNumOfFixedThreads = b->getSize(1);

    startCycleCounter(b, CycleCounter::FULL_PIPELINE_TIME);

    start(b);

    mPipelineLoop = b->CreateBasicBlock("PipelineLoop");
    mPipelineEnd = b->CreateBasicBlock("PipelineEnd");
    BasicBlock * const entryBlock = b->GetInsertBlock();
    b->CreateBr(mPipelineLoop);

    b->SetInsertPoint(mPipelineLoop);
    mMadeProgressInLastSegment = b->CreatePHI(b->getInt1Ty(), 2, "madeProgressInLastSegment");
    mMadeProgressInLastSegment->addIncoming(b->getTrue(), entryBlock);
    obtainCurrentSegmentNumber(b, entryBlock);

    branchToInitialPartition(b);
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        setActiveKernel(b, i, true);
        executeKernel(b);
    }
    end(b);

    updateExternalConsumedItemCounts(b);
    updateExternalProducedItemCounts(b);
    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet())) {
        // TODO: this isn't fully correct when this is a nested pipeline
        concludeStridesPerSegmentRecording(b);
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

    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    b->CreateBr(mPartitionEntryPoint[PartitionCount]);

    b->SetInsertPoint(mPartitionEntryPoint[PartitionCount]);
    #endif
    Value * terminated = nullptr;
    if (mIsNestedPipeline) {
        if (PipelineHasTerminationSignal) {
            terminated = hasPipelineTerminated(b);
        }
        b->CreateBr(mPipelineEnd);
    } else {
        terminated = hasPipelineTerminated(b);
        Value * const done = b->CreateIsNotNull(terminated);
        if (LLVM_UNLIKELY(CheckAssertions)) {
            Value * const progressedOrFinished = b->CreateOr(mPipelineProgress, done);
            Value * const live = b->CreateOr(mMadeProgressInLastSegment, progressedOrFinished);
            b->CreateAssert(live, "Dead lock detected: pipeline could not progress after two iterations");
        }
        BasicBlock * const exitBlock = b->GetInsertBlock();
        mMadeProgressInLastSegment->addIncoming(mPipelineProgress, exitBlock);
        incrementCurrentSegNo(b, exitBlock);
        b->CreateUnlikelyCondBr(done, mPipelineEnd, mPipelineLoop);
    }
    b->SetInsertPoint(mPipelineEnd);

    if (PipelineHasTerminationSignal) {
        Value * const ptr = getTerminationSignalPtr();
        b->CreateStore(terminated, ptr);
    }

    #ifdef PRINT_DEBUG_MESSAGES
    if (mIsNestedPipeline) {
        debugPrint(b, "------------------------------------------------- END %" PRIx64, getHandle());
    } else {
        debugPrint(b, "================================================= END %" PRIx64, getHandle());
    }
    #endif

    #ifdef ENABLE_PAPI
    stopPAPIAndDestroyEventSet(b);
    #endif

    updateTotalCycleCounterTime(b);

    mExpectedNumOfStridesMultiplier = nullptr;
    mThreadLocalStreamSetBaseAddress = nullptr;
    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    mSegNo = mBaseSegNo;
    #endif

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readTerminationSignalFromLocalState
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::readTerminationSignalFromLocalState(BuilderRef b, Value * const threadState) const {
    // TODO: generalize a OR/ADD/etc "combination" mechanism for thread-local to output scalars?
    assert (threadState);
    assert (PipelineHasTerminationSignal || !mIsNestedPipeline);
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(TERMINATION_SIGNAL);
    Value * const signal = b->CreateLoad(b->CreateInBoundsGEP(threadState, indices));
    assert (signal->getType()->isIntegerTy());
    return signal;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief writeTerminationSignalToLocalState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::writeTerminationSignalToLocalState(BuilderRef b, Value * const threadState, Value * const terminated) const {
    // TODO: generalize a OR/ADD/etc "combination" mechanism for thread-local to output scalars?
    assert (threadState);
    assert (PipelineHasTerminationSignal || !mIsNestedPipeline);
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(TERMINATION_SIGNAL);
    b->CreateStore(terminated, b->CreateInBoundsGEP(threadState, indices));
}

}
