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
    branchToInitialPartition(b);
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        setActiveKernel(b, i, true);
        executeKernel(b);
    }
    end(b);
    updateExternalPipelineIO(b);
    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet())) {
        // TODO: this isn't fully correct when this is a nested pipeline
        concludeStridesPerSegmentRecording(b);
    }
}


namespace {

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

}

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
    ConstantInt * const ZERO = b->getInt32(0);

    Constant * const nullVoidPtrVal = ConstantPointerNull::getNullValue(voidPtrTy);

    SmallVector<char, 256> tmp;
    const auto threadName = concat(mTarget->getName(), "_DoFixedDataSegmentThread", tmp);

    FunctionType * const threadFuncType = FunctionType::get(voidPtrTy, {voidPtrTy}, false);
    Function * const fixedDataThreadFunc = Function::Create(threadFuncType, Function::InternalLinkage, threadName, m);

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
        indices[0] = ZERO;
        indices[1] = b->getInt32(i);
        threadIdPtr[i] = b->CreateInBoundsGEP(pthreads, indices);

        FixedArray<Value *, 4> pthreadCreateArgs;
        pthreadCreateArgs[0] = threadIdPtr[i];
        pthreadCreateArgs[1] = fixedDataThreadFunc;
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

    Value * const mainThreadRetVal = b->CreateCall(fixedDataThreadFunc->getFunctionType(), fixedDataThreadFunc, b->CreatePointerCast(processState, voidPtrTy));

    // store where we'll resume compiling the DoSegment method
    const auto resumePoint = b->saveIP();
    const auto storedState = storeDoSegmentState();

    BitVector isInputFromAlternateThread(LastStreamSet + 1U);
    BitVector requiresTerminationSignalFromAlternateThread(PartitionCount);

    const auto anyDebugOptionIsSet = codegen::AnyDebugOptionIsSet();

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE THREAD
    // -------------------------------------------------------------------------------------------------------------------------
    auto makeThreadFunction = [&](Function * const threadFunc) {
        threadFunc->setCallingConv(CallingConv::C);
        auto threadStructArg = threadFunc->arg_begin();
        threadStructArg->setName("threadStruct");

        b->SetInsertPoint(BasicBlock::Create(m->getContext(), "entry", threadFunc));
        PointerType * const threadStructTy = cast<PointerType>(processState->getType());
        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(4, 0, 0)
        Value * const threadStruct = b->CreatePointerCast(&*threadStructArg, threadStructTy);
        #else
        Value * const threadStruct = b->CreatePointerCast(threadStructArg, threadStructTy);
        #endif
        #ifdef ENABLE_PAPI
        registerPAPIThread(b);
        #endif
        assert (isFromCurrentFunction(b, threadStruct));
        readThreadStuctObject(b, threadStruct);
        assert (isFromCurrentFunction(b, getHandle()));
        assert (isFromCurrentFunction(b, getThreadLocalHandle()));

        // generate the pipeline logic for this thread
        start(b);
        branchToInitialPartition(b);
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            setActiveKernel(b, i, true);
            executeKernel(b);
        }
        mKernel = nullptr;
        mKernelId = 0;
        end(b);

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

    makeThreadFunction(fixedDataThreadFunc);

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
    if (PipelineHasTerminationSignal) {
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
    updateExternalPipelineIO(b);

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



}
