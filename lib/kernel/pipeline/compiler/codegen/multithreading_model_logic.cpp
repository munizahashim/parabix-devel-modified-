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
}
#endif

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

void __report_pthread_create_error(const int r) {
    SmallVector<char, 256> tmp;
    raw_svector_ostream out(tmp);
    out << "Fatal error: pipeline failed to spawn requested number of threads.\n"
           "pthread_create returned error code " << r << ".";
    report_fatal_error(out.str());
}

#if BOOST_OS_MACOS

// TODO: look into thread affinity for osx

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __pipeline_pin_current_thread_to_cpu
 ** ------------------------------------------------------------------------------------------------------------- */
void __pipeline_pin_current_thread_to_cpu(int32_t cpu) {
    mach_port_t mthread = mach_task_self();
    thread_affinity_policy_data_t policy = { cpu };
    thread_policy_set(mthread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, 1);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __pipeline_pthread_create_on_cpu
 ** ------------------------------------------------------------------------------------------------------------- */
void __pipeline_pthread_create_on_cpu(pthread_t * pthread, void *(*start_routine)(void *), void * arg, int32_t cpu) {
    const auto r = pthread_create_suspended_np(pthread, nullptr, start_routine, arg);
    if (LLVM_UNLIKELY(r != 0)) __report_pthread_create_error(r);
    mach_port_t mthread = pthread_mach_thread_np(*pthread);
    thread_affinity_policy_data_t policy = { cpu };
    thread_policy_set(mthread, THREAD_AFFINITY_POLICY, (thread_policy_t)&policy, 1);
    thread_resume(mthread);
}

#elif BOOST_OS_LINUX

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __pipeline_pin_current_thread_to_cpu
 ** ------------------------------------------------------------------------------------------------------------- */
void __pipeline_pin_current_thread_to_cpu(const int32_t cpu) {
    #ifdef PIN_THREADS_TO_INDIVIDUAL_CORES
    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(cpu, &cpu_set);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __pipeline_pthread_create_on_cpu
 ** ------------------------------------------------------------------------------------------------------------- */
void __pipeline_pthread_create_on_cpu(pthread_t * pthread, void *(*start_routine)(void *), void * arg, const int32_t cpu) {
    #ifdef PIN_THREADS_TO_INDIVIDUAL_CORES
    cpu_set_t cpu_set;
    CPU_SET(cpu, &cpu_set);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpu_set);
    const auto r = pthread_create(pthread, &attr, start_routine, arg);
    pthread_attr_destroy(&attr);
    if (LLVM_UNLIKELY(r != 0)) {
    #endif
        const auto r = pthread_create(pthread, nullptr, start_routine, arg);
        if (LLVM_UNLIKELY(r != 0)) __report_pthread_create_error(r);
    #ifdef PIN_THREADS_TO_INDIVIDUAL_CORES
    }
    #endif
}

#endif

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
    ConstantInt * const i32_ZERO = b->getInt32(0);
    IntegerType * const boolTy = b->getInt1Ty();
    IntegerType * const sizeTy = b->getSizeTy();
    Type * const emptyTy = StructType::get(m->getContext());

    Constant * const nullVoidPtrVal = ConstantPointerNull::getNullValue(voidPtrTy);

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
    assert (mNumOfThreads > 1);
    const auto numOfAdditionalThreads = mNumOfThreads - 1U;

    Function * const pthreadSelfFn = m->getFunction("pthread_self");
    Function * const pthreadCreateFn = m->getFunction("__pipeline_pthread_create_on_cpu");
    Function * const pthreadExitFn = m->getFunction("pthread_exit");
    Function * const pthreadJoinFn = m->getFunction("pthread_join");

    Type * const pThreadTy = pthreadSelfFn->getReturnType();

    Type * const pthreadsTy = ArrayType::get(pThreadTy, numOfAdditionalThreads);
    AllocaInst * const pthreads = b->CreateCacheAlignedAlloca(pthreadsTy);
    SmallVector<Value *, 8> threadIdPtr(numOfAdditionalThreads);
    SmallVector<Value *, 8> threadState(numOfAdditionalThreads);
    SmallVector<Value *, 8> threadLocal(numOfAdditionalThreads);

    DataLayout DL(b->getModule());
    Type * const intPtrTy = DL.getIntPtrType(voidPtrTy);
    PointerType * const intPtrPtrTy = intPtrTy->getPointerTo();

    Value * const processThreadId = b->CreateCall(pthreadSelfFn->getFunctionType(), pthreadSelfFn, {});
    // construct and start the threads
    for (unsigned i = 0; i != numOfAdditionalThreads; ++i) {
        if (mTarget->hasThreadLocal()) {

            SmallVector<Value *, 2> args;
            if (initialSharedState) {
                args.push_back(initialSharedState);
            }
            args.push_back(ConstantPointerNull::get(mTarget->getThreadLocalStateType()->getPointerTo()));
            threadLocal[i] = mTarget->initializeThreadLocalInstance(b, args);
            assert (isFromCurrentFunction(b, threadLocal[i]));
            if (LLVM_LIKELY(mTarget->allocatesInternalStreamSets())) {
                Function * const allocInternal = mTarget->getAllocateThreadLocalInternalStreamSetsFunction(b, false);
                SmallVector<Value *, 3> allocArgs;
                if (LLVM_LIKELY(mTarget->isStateful())) {
                    allocArgs.push_back(initialSharedState);
                }
                allocArgs.push_back(threadLocal[i]);
                allocArgs.push_back(b->getSize(1));
                b->CreateCall(allocInternal->getFunctionType(), allocInternal, allocArgs);
            }
        }
        threadState[i] = constructThreadStructObject(b, processThreadId, threadLocal[i], i + 1);
        FixedArray<Value *, 2> indices;
        indices[0] = i32_ZERO;
        indices[1] = b->getInt32(i);
        threadIdPtr[i] = b->CreateInBoundsGEP(pthreads, indices);

        FixedArray<Value *, 4> pthreadCreateArgs;
        pthreadCreateArgs[0] = threadIdPtr[i];
        pthreadCreateArgs[1] = threadFunc;
        pthreadCreateArgs[2] = b->CreatePointerCast(threadState[i], voidPtrTy);
        pthreadCreateArgs[3] = b->getInt32(i + 1);
        b->CreateCall(pthreadCreateFn->getFunctionType(), pthreadCreateFn, pthreadCreateArgs);
    }

    Function * const pinProcessFn = m->getFunction("__pipeline_pin_current_thread_to_cpu");
    FixedArray<Value *, 1> pinProcessArgs;
    pinProcessArgs[0] = b->getInt32(0);
    b->CreateCall(pinProcessFn->getFunctionType(), pinProcessFn, pinProcessArgs);

    // execute the process thread
    assert (isFromCurrentFunction(b, initialThreadLocal));
    Value * const pty_ZERO = Constant::getNullValue(pThreadTy);
    Value * const processState = constructThreadStructObject(b, pty_ZERO, initialThreadLocal, 0);
    PointerType * const threadStructPtrTy = cast<PointerType>(processState->getType());

    Value * const mainThreadRetVal = b->CreateCall(threadFunc->getFunctionType(), threadFunc, b->CreatePointerCast(processState, voidPtrTy));

    // store where we'll resume compiling the DoSegment method
    const auto resumePoint = b->saveIP();
    const auto storedState = storeDoSegmentState();

    BitVector isInputFromAlternateThread(LastStreamSet + 1U);
    BitVector requiresTerminationSignalFromAlternateThread(PartitionCount);

    const auto anyDebugOptionIsSet = codegen::AnyDebugOptionIsSet();

    // -------------------------------------------------------------------------------------------------------------------------
    // GENERATE DO SEGMENT (KERNEL EXECUTION) FUNCTION CODE
    // -------------------------------------------------------------------------------------------------------------------------

    SmallVector<Type *, 3> csRetValFields;
    const auto returnsTerminationSignal = !mIsNestedPipeline || PipelineHasTerminationSignal;
    if (LLVM_LIKELY(mUseDynamicMultithreading)) {
        csRetValFields.push_back(sizeTy); // synchronization time
    }
    if (LLVM_LIKELY(returnsTerminationSignal)) {
        csRetValFields.push_back(sizeTy); // has terminated
    }
    if (LLVM_UNLIKELY(CheckAssertions)) {
        csRetValFields.push_back(boolTy); // has progressed
    }
    Type * const csRetValType = StructType::get(b->getContext(), csRetValFields);

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

    SmallVector<Value *, 2> csRetVal;
    if (LLVM_LIKELY(mUseDynamicMultithreading)) {
        csRetVal.push_back(b->getSize(0)); // sync cost
    }
    if (LLVM_LIKELY(returnsTerminationSignal)) {
        csRetVal.push_back(hasPipelineTerminated(b));
    }
    if (LLVM_UNLIKELY(CheckAssertions)) {
        csRetVal.push_back(mPipelineProgress);
    }
    b->CreateAggregateRet(csRetVal.data(), csRetVal.size());

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE THREAD
    // -------------------------------------------------------------------------------------------------------------------------
    auto makeThreadFunction = [&](Function * const threadFunc) {
        threadFunc->setCallingConv(CallingConv::C);
        auto threadStructArg = threadFunc->arg_begin();
        threadStructArg->setName("threadStruct");

        b->SetInsertPoint(BasicBlock::Create(m->getContext(), "entry", threadFunc));

        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(4, 0, 0)
        Value * const threadStruct = b->CreatePointerCast(&*threadStructArg, threadStructPtrTy);
        #else
        Value * const threadStruct = b->CreatePointerCast(threadStructArg, threadStructPtrTy);
        #endif

        #ifdef ENABLE_PAPI
        registerPAPIThread(b);
        #endif

        mPipelineStartTime = startCycleCounter(b, CycleCounter::FULL_PIPELINE_TIME);

        #ifdef PRINT_DEBUG_MESSAGES
        debugInit(b);
//        if (mIsNestedPipeline) {
//            debugPrint(b, "------------------------------------------------- START %" PRIx64, getHandle());
//        } else {
//            debugPrint(b, "================================================= START %" PRIx64, getHandle());
//        }
//        const auto prefix = mTarget->getName();
//        if (mNumOfStrides) {
//            debugPrint(b, prefix + " +++ NUM OF STRIDES %" PRIu64 "+++", mNumOfStrides);
//        }
//        if (mIsFinal) {
//            debugPrint(b, prefix + " +++ IS FINAL %" PRIu8 "+++", mIsFinal);
//        }
        #endif

        #ifdef ENABLE_PAPI
        createEventSetAndStartPAPI(b);
        #endif

        readInitialSegmentNum(b, threadStruct);

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
        obtainCurrentSegmentNumber(b, entryBlock);

        SmallVector<Value *, 3> args(2);
        args[0] = threadStruct;
        args[1] = mSegNo; assert (mSegNo);
        Value * const csRetVal = b->CreateCall(csFuncType, csFunc, args);

        Value * syncCost = nullptr;
        Value * terminated = nullptr;
        Value * done = nullptr;
        Value * madeProgress = nullptr;

        unsigned index = 0;
        FixedArray<unsigned, 1> idx;
        if (LLVM_LIKELY(mUseDynamicMultithreading)) {
            idx[0] = index++;
            syncCost = b->CreateExtractValue(csRetVal, idx);
        }
        if (LLVM_LIKELY(returnsTerminationSignal)) {
            idx[0] = index++;
            terminated = b->CreateExtractValue(csRetVal, idx);
            done = b->CreateIsNotNull(terminated);
        }
        if (LLVM_UNLIKELY(CheckAssertions)) {
            idx[0] = index++;
            madeProgress = b->CreateExtractValue(csRetVal, idx);
            if (LLVM_LIKELY(returnsTerminationSignal)) {
                madeProgress = b->CreateOr(madeProgress, done);
            }
            Value * const live = b->CreateOr(mMadeProgressInLastSegment, madeProgress);
            b->CreateAssert(live, "Dead lock detected: pipeline could not progress after two iterations");
        }

        if (mIsNestedPipeline) {
            b->CreateBr(mPipelineEnd);
        } else {
            BasicBlock * const exitBlock = b->GetInsertBlock();
            if (LLVM_UNLIKELY(CheckAssertions)) {
                mMadeProgressInLastSegment->addIncoming(madeProgress, exitBlock);
            }
            incrementCurrentSegNo(b, exitBlock);
            assert (returnsTerminationSignal);
            b->CreateUnlikelyCondBr(done, mPipelineEnd, mPipelineLoop);
        }

        b->SetInsertPoint(mPipelineEnd);
        if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
            getThreadedTerminationSignalPtr(b, threadStruct);
            #ifdef PRINT_DEBUG_MESSAGES
            debugPrint(b, "# pipeline terminated = %" PRIu64 " for %" PRIx64, terminated, getHandle());
            #endif
            assert (terminated);
            assert (mCurrentThreadTerminationSignalPtr);
            b->CreateStore(terminated, mCurrentThreadTerminationSignalPtr);
        }

//        #ifdef PRINT_DEBUG_MESSAGES
//        if (mIsNestedPipeline) {
//            debugPrint(b, "------------------------------------------------- END %" PRIx64, getHandle());
//        } else {
//            debugPrint(b, "================================================= END %" PRIx64, getHandle());
//        }
//        #endif

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
            retVal = nullVoidPtrVal;
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

    makeThreadFunction(threadFunc);

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE DRIVER CONTINUED
    // -------------------------------------------------------------------------------------------------------------------------

    b->restoreIP(resumePoint);

    assert (isFromCurrentFunction(b, processState));
    assert (isFromCurrentFunction(b, initialSharedState));
    assert (isFromCurrentFunction(b, initialThreadLocal));

    if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
        mSegNo = b->CreatePtrToInt(mainThreadRetVal, intPtrTy);
    } else {
        mSegNo = nullptr;
    }

    SmallVector<Value *, 2> threadLocalArgs;
    if (LLVM_LIKELY(mTarget->isStateful())) {
        threadLocalArgs.push_back(initialSharedState);
    }
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        threadLocalArgs.push_back(initialThreadLocal);
    }

    Value * finalTerminationSignal = nullptr;
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        finalTerminationSignal = readTerminationSignalFromLocalState(b, processState);
        assert (finalTerminationSignal);
    }
    destroyStateObject(b, processState);

    // wait for all other threads to complete
    AllocaInst * const status = b->CreateAlloca(voidPtrTy);

    FixedArray<Value *, 2> pthreadJoinArgs;
    for (unsigned i = 0; i != numOfAdditionalThreads; ++i) {
        Value * threadId = b->CreateLoad(threadIdPtr[i]);
        pthreadJoinArgs[0] = threadId;
        pthreadJoinArgs[1] = status;
        b->CreateCall(pthreadJoinFn->getFunctionType(), pthreadJoinFn, pthreadJoinArgs);

        // calculate the last segment # used by any kernel in case any reports require it.
        if (LLVM_UNLIKELY(anyDebugOptionIsSet)) {
            Value * const retVal = b->CreatePointerCast(status, intPtrPtrTy);
            mSegNo = b->CreateUMax(mSegNo, b->CreateLoad(retVal));
        }
        if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
            assert (isFromCurrentFunction(b, threadLocal[i]));
            threadLocalArgs.push_back(threadLocal[i]);
            mTarget->finalizeThreadLocalInstance(b, threadLocalArgs);
            b->CreateFree(threadLocal[i]);
            threadLocalArgs.pop_back();
        }
        if (PipelineHasTerminationSignal) {
            Value * const terminatedSignal = readTerminationSignalFromLocalState(b, threadState[i]);
            assert (terminatedSignal);
            finalTerminationSignal = b->CreateUMax(finalTerminationSignal, terminatedSignal);
        }
        destroyStateObject(b, threadState[i]);
    }

    restoreDoSegmentState(storedState);

    if (PipelineHasTerminationSignal) {
        assert (initialTerminationSignalPtr);
        b->CreateStore(finalTerminationSignal, initialTerminationSignalPtr);
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
//    mPipelineLoop = b->CreateBasicBlock("PipelineLoop");
//    mPipelineEnd = b->CreateBasicBlock("PipelineEnd");

    makePartitionEntryPoints(b);

    if (CheckAssertions) {
        mRethrowException = b->WriteDefaultRethrowBlock();
    }

    mExpectedNumOfStridesMultiplier = b->getScalarField(EXPECTED_NUM_OF_STRIDES_MULTIPLIER);
    initializeFlowControl(b);
//    readExternalConsumerItemCounts(b);
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
        if (mCurrentThreadTerminationSignalPtr) {
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
    if (mCurrentThreadTerminationSignalPtr) {
        assert (canSetTerminateSignal());
        #ifdef PRINT_DEBUG_MESSAGES
        debugPrint(b, "# pipeline terminated = %" PRIu64 " for %" PRIx64, terminated, getHandle());
        #endif
        b->CreateStore(terminated, mCurrentThreadTerminationSignalPtr);
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
 * @brief getThreadStateType
 ** ------------------------------------------------------------------------------------------------------------- */
StructType * PipelineCompiler::getThreadStuctType(BuilderRef b) const {
    FixedArray<Type *, THREAD_STRUCT_SIZE> fields;
    LLVMContext & C = b->getContext();

    assert (mNumOfThreads > 1);

    IntegerType * const sizeTy = b->getSizeTy();
    Type * const emptyTy = StructType::get(C);

    // NOTE: both the shared and thread local objects are parameters to the kernel.
    // They get automatically set by reading in the appropriate params.

    fields[PIPELINE_PARAMS] = StructType::get(C, mTarget->getDoSegmentFields(b));
    if (mUseDynamicMultithreading) {
        fields[INITIAL_SEG_NO] = emptyTy;
        fields[ACCUMULATED_SEGMENT_TIME] = sizeTy;
        fields[ACCUMULATED_SYNCHRONIZATION_TIME] = sizeTy;
    } else {
        fields[INITIAL_SEG_NO] = sizeTy;
        fields[ACCUMULATED_SEGMENT_TIME] = emptyTy;
        fields[ACCUMULATED_SYNCHRONIZATION_TIME] = emptyTy;
    }

    Function * const pthreadSelfFn = b->getModule()->getFunction("pthread_self");
    fields[PROCESS_THREAD_ID] = pthreadSelfFn->getReturnType();
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        fields[TERMINATION_SIGNAL] = sizeTy;
    } else {
        fields[TERMINATION_SIGNAL] = emptyTy;
    }

    return StructType::get(C, fields);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructThreadStructObject
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::constructThreadStructObject(BuilderRef b, Value * const threadId, Value * const threadLocal, const unsigned threadNum) {
    StructType * const threadStructType = getThreadStuctType(b);
    Value * const threadState = makeStateObject(b, threadStructType);
    setThreadLocalHandle(threadLocal);
    const auto props = getDoSegmentProperties(b);
    const auto n = props.size();
    assert (threadStructType->getStructElementType(PIPELINE_PARAMS)->getStructNumElements() == n);

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
        b->CreateStore(b->getSize(threadNum), b->CreateInBoundsGEP(threadState, indices2));
    }
    indices2[1] = b->getInt32(PROCESS_THREAD_ID);
    b->CreateStore(threadId, b->CreateInBoundsGEP(threadState, indices2));
    return threadState;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readInitialSegmentNum
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readInitialSegmentNum(BuilderRef b, Value * threadState) {
    assert (mNumOfThreads > 1);
    if (mUseDynamicMultithreading) {
        return;
    }
    FixedArray<Value *, 2> indices2;
    Constant * const ZERO = b->getInt32(0);
    indices2[0] = b->getInt32(0);
    indices2[1] = b->getInt32(INITIAL_SEG_NO);
    mSegNo = b->CreateLoad(b->CreateInBoundsGEP(threadState, indices2));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readThreadStuctObject
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readThreadStuctObject(BuilderRef b, Value * threadState) {
    assert (mNumOfThreads > 1);

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
    assert (!mIsNestedPipeline && mNumOfThreads != 1);
    if (mUseDynamicMultithreading) {
        indices2[1] = b->getInt32(ACCUMULATED_SEGMENT_TIME);
        mAccumulatedFullSegmentTimePtr = b->CreateInBoundsGEP(threadState, indices2);
        indices2[1] = b->getInt32(ACCUMULATED_SYNCHRONIZATION_TIME);
        mAccumulatedSynchronizationTimePtr = b->CreateInBoundsGEP(threadState, indices2);
    } else {
        indices2[1] = b->getInt32(INITIAL_SEG_NO);
        mSegNo = b->CreateLoad(b->CreateInBoundsGEP(threadState, indices2));
        mAccumulatedFullSegmentTimePtr = nullptr;
        mAccumulatedSynchronizationTimePtr = nullptr;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readThreadStuctObject
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::getThreadedTerminationSignalPtr(BuilderRef b, Value * threadState) {
    assert (mNumOfThreads > 1);
    mCurrentThreadTerminationSignalPtr = getTerminationSignalPtr();
    if (PipelineHasTerminationSignal) {
        FixedArray<Value *, 2> indices2;
        indices2[0] = b->getInt32(0);
        indices2[1] = b->getInt32(TERMINATION_SIGNAL);
        mCurrentThreadTerminationSignalPtr = b->CreateInBoundsGEP(threadState, indices2);
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
   // b->LinkFunction("pthread_create", pthread_create);
    b->LinkFunction("pthread_join", pthread_join);
    BEGIN_SCOPED_REGION
    // pthread_exit seems difficult to resolve in MacOS? manually doing it here but should be looked into
    FixedArray<Type *, 1> args;
    args[0] = b->getVoidPtrTy();
    FunctionType * pthreadExitFnTy = FunctionType::get(b->getVoidTy(), args, false);
    b->LinkFunction("pthread_exit", pthreadExitFnTy, (void*)pthread_exit); // ->addAttribute(0, llvm::Attribute::AttrKind::NoReturn);
    END_SCOPED_REGION

    b->LinkFunction("__pipeline_pthread_create_on_cpu",
                    __pipeline_pthread_create_on_cpu);
    b->LinkFunction("__pipeline_pin_current_thread_to_cpu",
                    __pipeline_pin_current_thread_to_cpu);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateSingleThreadKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateSingleThreadKernelMethod(BuilderRef b) {
    if (PipelineHasTerminationSignal) {
        mCurrentThreadTerminationSignalPtr = getTerminationSignalPtr();
    }
    if (LLVM_UNLIKELY(mIsNestedPipeline)) {
        mSegNo = mExternalSegNo; assert (mExternalSegNo);
    } else {
        mSegNo = b->getSize(0);
    }
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

}
