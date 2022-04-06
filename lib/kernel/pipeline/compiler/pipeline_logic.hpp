#ifndef PIPELINE_LOGIC_HPP
#define PIPELINE_LOGIC_HPP

#include "pipeline_compiler.hpp"
#include <pthread.h>

#if BOOST_OS_LINUX
#include <sched.h>
#endif

#if BOOST_OS_MACOS
#include <mach/mach_init.h>
#include <mach/thread_act.h>
namespace llvm {
template<> class TypeBuilder<pthread_t, false> {
public:
  static Type *get(LLVMContext& C) {
    return IntegerType::getIntNTy(C, sizeof(pthread_t) * CHAR_BIT);
  }
};
}
#endif

namespace kernel {

// NOTE: the following is a workaround for an LLVM bug for 32-bit VMs on 64-bit architectures.
// When calculating the address of a local stack allocated object, the size of a pointer is
// 32-bits but when performing the same GEP on a pointer returned by "malloc" or passed as a
// function argument, the size is 64-bits. More investigation is needed to determine which
// versions of LLVM are affected by this bug.

inline LLVM_READNONE bool allocateOnHeap(BuilderRef b) {
    DataLayout DL(b->getModule());
    return (DL.getPointerSizeInBits() != b->getSizeTy()->getBitWidth());
}

inline Value * makeStateObject(BuilderRef b, Type * type) {
    Value * ptr = nullptr;
    if (LLVM_UNLIKELY(allocateOnHeap(b))) {
        ptr = b->CreatePageAlignedMalloc(type);
    } else {
        ptr = b->CreateCacheAlignedAlloca(type);
    }
    b->CreateMemZero(ptr, ConstantExpr::getSizeOf(type), b->getCacheAlignment());
    return ptr;
}

inline void destroyStateObject(BuilderRef b, Value * threadState) {
    if (LLVM_UNLIKELY(allocateOnHeap(b))) {
        b->CreateFree(threadState);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateImplicitKernels
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::generateImplicitKernels(BuilderRef b) {
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        Kernel * const kernel = const_cast<Kernel *>(getKernel(i));
        if (LLVM_LIKELY(kernel->isGenerated())) {
            kernel->ensureLoaded();
            continue;
        }
        if (kernel->getInitializeFunction(b, false)) {
            kernel->loadCachedKernel(b);
        } else {
            kernel->setModule(mTarget->getModule());
            kernel->generateKernel(b);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addPipelineKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::addPipelineKernelProperties(BuilderRef b) {
    // TODO: look into improving cache locality/false sharing of this struct

    // TODO: create a non-persistent / pass through input scalar type to allow the
    // pipeline to pass an input scalar to a kernel rather than recording it needlessly?
    // Non-family kernels can be contained within the shared state but family ones
    // must be allocated dynamically.

    identifyAllInternallySynchronizedKernels();

    IntegerType * const sizeTy = b->getSizeTy();

    #ifndef USE_FIXED_SEGMENT_NUMBER_INCREMENTS
    if (!ExternallySynchronized) {
        mTarget->addInternalScalar(sizeTy, NEXT_LOGICAL_SEGMENT_NUMBER, 0);
    }
    #endif

    mTarget->addInternalScalar(sizeTy, EXPECTED_NUM_OF_STRIDES_MULTIPLIER, 0);

    if (LLVM_LIKELY(RequiredThreadLocalStreamSetMemory > 0)) {
        PointerType * const int8PtrTy = b->getInt8PtrTy();
        mTarget->addThreadLocalScalar(int8PtrTy, BASE_THREAD_LOCAL_STREAMSET_MEMORY, 0);
    }
    // NOTE: both the shared and thread local objects are parameters to the kernel.
    // They get automatically set by reading in the appropriate params.

    if (HasZeroExtendedStream) {
        PointerType * const voidPtrTy = b->getVoidPtrTy();
        mTarget->addThreadLocalScalar(voidPtrTy, ZERO_EXTENDED_BUFFER);
        mTarget->addThreadLocalScalar(sizeTy, ZERO_EXTENDED_SPACE);
    }

    mKernelId = 0;
    mKernel = mTarget;
    mIsStatefree = false;
    auto currentPartitionId = -1U;
    addBufferHandlesToPipelineKernel(b, PipelineInput);
    addConsumerKernelProperties(b, PipelineInput);
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        // Is this the start of a new partition?
        const auto partitionId = KernelPartitionId[i];
        const bool isRoot = (partitionId != currentPartitionId);
        currentPartitionId = partitionId;
        addInternalKernelProperties(b, i, isRoot);
        addCycleCounterProperties(b, i, isRoot);
        #ifdef ENABLE_PAPI
        addPAPIEventCounterKernelProperties(b, i, isRoot);
        #endif
        addProducedItemCountDeltaProperties(b, i);
        addUnconsumedItemCountProperties(b, i);
    }
    #ifdef ENABLE_PAPI
    addPAPIEventCounterPipelineProperties(b);
    #endif
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addInternalKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addInternalKernelProperties(BuilderRef b, const unsigned kernelId, const bool isRoot) {

    mKernelId = kernelId;
    mKernel = getKernel(kernelId);
    mIsStatefree = isCurrentKernelStatefree();

    IntegerType * const sizeTy = b->getSizeTy();

    const auto groupId = getCacheLineGroupId(kernelId);

    if (isRoot) {
        addTerminationProperties(b, kernelId);
    }

    const auto name = makeKernelName(kernelId);
    if (RequiresSynchronization.test(kernelId)) {
        mTarget->addInternalScalar(sizeTy, name + LOGICAL_SEGMENT_SUFFIX, groupId);
    }

    addConsumerKernelProperties(b, kernelId);

    for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto prefix = makeBufferName(kernelId, br.Port);
        if (LLVM_UNLIKELY(br.IsDeferred)) {
            mTarget->addInternalScalar(sizeTy, prefix + DEFERRED_ITEM_COUNT_SUFFIX, groupId);
        }
        mTarget->addInternalScalar(sizeTy, prefix + ITEM_COUNT_SUFFIX, groupId);
        if (mIsStatefree) {
            assert (!br.IsDeferred);
            const auto streamSet = source(e, mBufferGraph);
            if (edge(streamSet, kernelId, mConsumerGraph).second) {
                mTarget->addThreadLocalScalar(sizeTy, prefix + THREAD_LOCAL_CONSUMED_ITEM_COUNT_SUFFIX, groupId);
            }
        }
    }

    bool hasDynamicBuffer = false;

    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto prefix = makeBufferName(kernelId, br.Port);
        if (LLVM_UNLIKELY(br.IsDeferred)) {
            mTarget->addInternalScalar(sizeTy, prefix + DEFERRED_ITEM_COUNT_SUFFIX, groupId);
        }
        mTarget->addInternalScalar(sizeTy, prefix + ITEM_COUNT_SUFFIX, groupId);

        const auto streamSet = target(e, mBufferGraph);
        const BufferNode & bn = mBufferGraph[streamSet];
        if (isa<DynamicBuffer>(bn.Buffer)) {
            hasDynamicBuffer = true;
        }

    }

    addBufferHandlesToPipelineKernel(b, kernelId);

    addFamilyKernelProperties(b, kernelId);

    if (LLVM_LIKELY(mKernel->isStateful())) {
        Type * sharedStateTy = nullptr;
        if (LLVM_UNLIKELY(mKernel->externallyInitialized())) {
            sharedStateTy = b->getVoidPtrTy();
        } else {
            sharedStateTy = mKernel->getSharedStateType();
        }
        mTarget->addInternalScalar(sharedStateTy, name, groupId);
    }

    if (mKernel->hasThreadLocal()) {
        // we cannot statically allocate a "family" thread local object.
        Type * localStateTy = nullptr;
        if (LLVM_UNLIKELY(mKernel->externallyInitialized())) {
            localStateTy = b->getVoidPtrTy();
        } else {
            localStateTy = mKernel->getThreadLocalStateType();
        }
        mTarget->addThreadLocalScalar(localStateTy, name + KERNEL_THREAD_LOCAL_SUFFIX, groupId);
    }

    if (LLVM_UNLIKELY(mIsStatefree && hasDynamicBuffer)) {
        mTarget->addInternalScalar(sizeTy, name + POST_INVOCATION_LOGICAL_SEGMENT_SUFFIX, groupId);
    }

    if (LLVM_UNLIKELY(mTraceDynamicBuffers)) {
        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const auto bufferVertex = target(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[bufferVertex];
            if (isa<DynamicBuffer>(bn.Buffer)) {
                const BufferPort & rd = mBufferGraph[e];
                const auto prefix = makeBufferName(kernelId, rd.Port);
                LLVMContext & C = b->getContext();
                const auto numOfConsumers = out_degree(bufferVertex, mConsumerGraph);

                // segment num  0
                // new capacity 1
                // produced item count 2
                // consumer processed item count [3,n)
                Type * const traceStructTy = ArrayType::get(sizeTy, numOfConsumers + 3);

                FixedArray<Type *, 2> traceStruct;
                traceStruct[0] = traceStructTy->getPointerTo(); // pointer to trace log
                traceStruct[1] = sizeTy; // length of trace log
                mTarget->addInternalScalar(StructType::get(C, traceStruct),
                                                   prefix + STATISTICS_BUFFER_EXPANSION_SUFFIX, groupId);
            }
        }
    }

    if (LLVM_UNLIKELY(isRoot && DebugOptionIsSet(codegen::TraceStridesPerSegment))) {
        LLVMContext & C = b->getContext();
//        FixedArray<Type *, 2> recordStruct;
//        recordStruct[0] = sizeTy; // segment num
//        recordStruct[1] = sizeTy; // # of strides
        Type * const recordStructTy = ArrayType::get(sizeTy, 2);

        FixedArray<Type *, 4> traceStruct;
        traceStruct[0] = sizeTy; // last num of strides (to avoid unnecessary loads of the trace
                                 // log and simplify the logic for first stride)
        traceStruct[1] = recordStructTy->getPointerTo(); // pointer to trace log
        traceStruct[2] = sizeTy; // trace length
        traceStruct[3] = sizeTy; // trace capacity (for realloc)

        mTarget->addInternalScalar(StructType::get(C, traceStruct),
                                           name + STATISTICS_STRIDES_PER_SEGMENT_SUFFIX, groupId);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateInitializeMethod(BuilderRef b) {

    // TODO: if we detect a fatal error at init, we should not execute
    // the pipeline loop.
    initializeForAllKernels();
    #ifdef ENABLE_PAPI
    if (!ExternallySynchronized) {
        initializePAPI(b);
    }
    #endif

    mScalarValue.reset(FirstKernel, LastScalar);

    initializeKernelAssertions(b);

//    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
//        const Kernel * const kernel = getKernel(i);
//        if (LLVM_LIKELY(kernel->isStateful() && !kernel->externallyInitialized())) {
//            Value * const handle = kernel->createInstance(b);
//            b->setScalarField(makeKernelName(i), handle);
//        }
//    }

    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);
    Constant * const aborted = getTerminationSignal(b, TerminationSignal::Aborted);

    Value * terminated = nullptr;
    auto partitionId = KernelPartitionId[PipelineInput];

    for (auto i = FirstKernel; i <= LastKernel; ++i) {

        const auto curPartitionId = KernelPartitionId[i];
        const auto isRoot = (curPartitionId != partitionId);
        partitionId = curPartitionId;




        // Family kernels must be initialized in the "main" method.
        const Kernel * kernel = getKernel(i);

        setActiveKernel(b, i, false);
        if (isRoot) {
            initializeStridesPerSegment(b);
        }

        if (LLVM_LIKELY(!kernel->externallyInitialized())) {
            ArgVec args;
            if (LLVM_LIKELY(mKernel->isStateful())) {
                args.push_back(mKernelSharedHandle);
            }
            #ifndef NDEBUG
            unsigned expected = 0;
            #endif
            for (const auto e : make_iterator_range(in_edges(i, mScalarGraph))) {
                assert (mScalarGraph[e].Type == PortType::Input);
                assert (expected++ == mScalarGraph[e].Number);
                const auto scalar = source(e, mScalarGraph);
                args.push_back(getScalar(b, scalar));
            }
            for (auto i = 0U; i != args.size(); ++i) {
                assert (isFromCurrentFunction(b, args[i], false));
            }

            Value * const signal = callKernelInitializeFunction(b, args);
            Value * const terminatedOnInit = b->CreateICmpNE(signal, unterminated);

            if (terminated) {
                terminated = b->CreateOr(terminated, terminatedOnInit);
            } else {
                terminated = terminatedOnInit;
            }

        }

        // Is this the last kernel in a partition? If so, store the accumulated
        // termination signal.
        const auto nextPartitionId = KernelPartitionId[i + 1];
        if (terminated && partitionId != nextPartitionId) {
            Value * const signal = b->CreateSelect(terminated, aborted, unterminated);
            writeTerminationSignal(b, signal);
            terminated = nullptr;
        }

    }
    resetInternalBufferHandles();
    ActiveKernels.clear();
    ActivePartitions.clear();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateAllocateSharedInternalStreamSetsMethod(BuilderRef b, Value * const expectedNumOfStrides) {
    initializeForAllKernels();
    b->setScalarField(EXPECTED_NUM_OF_STRIDES_MULTIPLIER, expectedNumOfStrides);
    allocateOwnedBuffers(b, expectedNumOfStrides, true);
    initializeBufferExpansionHistory(b);
    resetInternalBufferHandles();
    ActiveKernels.clear();
    ActivePartitions.clear();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateInitializeThreadLocalMethod(BuilderRef b) {
    initializeForAllKernels();
    assert (mTarget->hasThreadLocal());
    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernel = getKernel(i);
        if (kernel->hasThreadLocal()) {
            setActiveKernel(b, i, true);
            assert (mKernel == kernel);
            Value * const handle = callKernelInitializeThreadLocalFunction(b, mKernelSharedHandle);
            b->CreateStore(handle, getThreadLocalHandlePtr(b, i));
        }
    }
    ActiveKernels.clear();
    ActivePartitions.clear();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateThreadLocalInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateAllocateThreadLocalInternalStreamSetsMethod(BuilderRef b, Value * const expectedNumOfStrides) {
    initializeForAllKernels();
    assert (mTarget->hasThreadLocal());
    if (LLVM_LIKELY(RequiredThreadLocalStreamSetMemory > 0)) {
        ConstantInt * const reqMemory = b->getSize(RequiredThreadLocalStreamSetMemory);
        Value * const memorySize = b->CreateMul(reqMemory, expectedNumOfStrides);
        Value * const base = b->CreatePageAlignedMalloc(memorySize);
        PointerType * const int8PtrTy = b->getInt8PtrTy();
        b->setScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY, b->CreatePointerCast(base, int8PtrTy));
    }
    allocateOwnedBuffers(b, expectedNumOfStrides, false);
    resetInternalBufferHandles();
    ActiveKernels.clear();
    ActivePartitions.clear();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::generateKernelMethod(BuilderRef b) {
    initializeKernelAssertions(b);
    verifyBufferRelationships();
    mScalarValue.reset(FirstKernel, LastScalar);
    readPipelineIOItemCounts(b);
    if (LLVM_UNLIKELY(mNumOfThreads == 0)) {
        report_fatal_error("Fatal error: cannot construct a 0-thread pipeline.");
    }
    if (mNumOfThreads == 1) {
        generateSingleThreadKernelMethod(b);
    } else {
        generateMultiThreadKernelMethod(b);
    }
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateSingleThreadKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateSingleThreadKernelMethod(BuilderRef b) {
    mCompilingHybridThread = false;
    createThreadStateForSingleThread(b);
    initializeForAllKernels();
    start(b);
    ActiveKernelIndex = 0;
    branchToInitialPartition(b);
    for (; ActiveKernelIndex < ActiveKernels.size() - 1; ++ActiveKernelIndex) {
        setActiveKernel(b, ActiveKernels[ActiveKernelIndex], true);
        executeKernel(b);
    }
    end(b);
    ActiveKernels.clear();
    ActivePartitions.clear();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief concat
 ** ------------------------------------------------------------------------------------------------------------- */
inline StringRef concat(StringRef A, StringRef B, SmallVector<char, 256> & tmp) {
    Twine C = A + B;
    tmp.clear();
    C.toVector(tmp);
    return StringRef(tmp.data(), tmp.size());
}

enum : unsigned {
    PIPELINE_PARAMS
    #ifdef USE_FIXED_SEGMENT_NUMBER_INCREMENTS
    , INITIAL_SEG_NO
    #endif
    , PROCESS_THREAD_ID    
    , TERMINATION_SIGNAL
    // -------------------
    , THREAD_STRUCT_SIZE
};



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

    Function * hybridThreadFunc = nullptr;
    if (LLVM_UNLIKELY(KernelOnHybridThread.any())) {
        assert (codegen::EnableHybridThreadModel);
        SmallVector<char, 256> tmp;
        const auto hybridThreadName = concat(mTarget->getName(), "_DoHybridSegmentThread", tmp);
        hybridThreadFunc = Function::Create(threadFuncType, Function::InternalLinkage, hybridThreadName, m);
    }

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
  //  const auto numOfHybridThreads = (hybridThreadFunc != nullptr ? 1U : 0U);
  //  const auto numOfFixedDataThreads = numOfAdditionalThreads - numOfHybridThreads;

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

    Value * const processThreadId = b->CreateCall(pthreadSelfFn->getFunctionType(), pthreadSelfFn, {});
    // start the normal threads
    const auto firstThreadedSegNo = hybridThreadFunc ? 0U : 1U;
    for (unsigned i = 0; i != numOfAdditionalThreads; ++i) {
        if (mTarget->hasThreadLocal()) {
            threadLocal[i] = mTarget->initializeThreadLocalInstance(b, initialSharedState);
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
        threadState[i] = constructThreadStructObject(b, processThreadId, threadLocal[i], i + firstThreadedSegNo);
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

    Function * const mainFunc = hybridThreadFunc ? hybridThreadFunc : fixedDataThreadFunc;

    b->CreateCall(mainFunc->getFunctionType(), mainFunc, b->CreatePointerCast(processState, voidPtrTy));

    // store where we'll resume compiling the DoSegment method
    const auto resumePoint = b->saveIP();
    const auto storedState = storeDoSegmentState();

    BitVector isInputFromAlternateThread(LastStreamSet + 1U);
    BitVector requiresTerminationSignalFromAlternateThread(PartitionCount);

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE THREAD
    // -------------------------------------------------------------------------------------------------------------------------
    auto makeThreadFunction = [&](Function * const threadFunc, const bool isHybridThread) {
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

        assert (ActiveKernels.empty());
        assert (ActivePartitions.empty());
        flat_set<unsigned> partitions;
        for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
            if (KernelOnHybridThread.test(i) == isHybridThread) {
                ActiveKernels.push_back(i);
                partitions.insert(KernelPartitionId[i]);
            }
        }
        ActiveKernels.push_back(PipelineOutput);
        ActivePartitions.assign(partitions.begin(), partitions.end());
        assert (KernelPartitionId[PipelineInput] == 0);
        assert (KernelPartitionId[PipelineOutput] == PartitionCount - 1);
        ActivePartitions.push_back(PartitionCount - 1);

        // generate the pipeline logic for this thread
        mCompilingHybridThread = isHybridThread;
        if (isHybridThread) {
            KernelOnHybridThread.set(PipelineInput);
            KernelOnHybridThread.set(PipelineOutput);
            PartitionOnHybridThread.set(0);
            PartitionOnHybridThread.set(PartitionCount - 1);
        } else {
            KernelOnHybridThread.reset(PipelineInput);
            KernelOnHybridThread.reset(PipelineOutput);
            PartitionOnHybridThread.reset(0);
            PartitionOnHybridThread.reset(PartitionCount - 1);
        }

        isInputFromAlternateThread.reset();
        requiresTerminationSignalFromAlternateThread.reset();

        if (KernelOnHybridThread.any()) {
            for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
                if (KernelOnHybridThread.test(kernel) == mCompilingHybridThread) {
                    for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                        const auto streamSet = source(input, mBufferGraph);
                        const auto output = in_edge(streamSet, mBufferGraph);
                        const auto producer = source(output, mBufferGraph);
                        assert (producer < kernel);
                        if (KernelOnHybridThread.test(producer) != mCompilingHybridThread) {
                            isInputFromAlternateThread.set(streamSet);
                            requiresTerminationSignalFromAlternateThread.set(KernelPartitionId[producer]);
                        }
                    }
                }
            }
            assert (KernelPartitionId[PipelineInput] == 0);
            assert (KernelPartitionId[PipelineOutput] == PartitionCount - 1);
        }

        start(b);
        const auto m = ActiveKernels.size() - 1;
        mCurrentPartitionId = 0;
        ActiveKernelIndex = 0;
        branchToInitialPartition(b);
        for (; ActiveKernelIndex < m; ++ActiveKernelIndex) {
            setActiveKernel(b, ActiveKernels[ActiveKernelIndex], true);
            executeKernel(b);
        }
        mKernel = nullptr;
        mKernelId = 0;
        end(b);
        ActiveKernels.clear();
        ActivePartitions.clear();

        BasicBlock * exitThread  = nullptr;
        BasicBlock * exitFunction  = nullptr;
        // only call pthread_exit() within spawned threads; otherwise it'll be equivalent to calling exit() within the process
        exitThread = b->CreateBasicBlock("ExitThread");
        exitFunction = b->CreateBasicBlock("ExitProcessFunction");
        b->CreateCondBr(isProcessThread(b, threadStruct), exitFunction, exitThread);
        b->SetInsertPoint(exitThread);
        #ifdef ENABLE_PAPI
        unregisterPAPIThread(b);
        #endif
        b->CreateCall(pthreadExitFn->getFunctionType(), pthreadExitFn, nullVoidPtrVal);
        b->CreateBr(exitFunction);
        b->SetInsertPoint(exitFunction);
        b->CreateRet(nullVoidPtrVal);
    };

    makeThreadFunction(fixedDataThreadFunc, false);
    if (hybridThreadFunc) {
        clearInternalState(b);
        makeThreadFunction(hybridThreadFunc, true);
    }

    // -------------------------------------------------------------------------------------------------------------------------
    // MAKE PIPELINE DRIVER CONTINUED
    // -------------------------------------------------------------------------------------------------------------------------

    b->restoreIP(resumePoint);

    assert (isFromCurrentFunction(b, processState));
    assert (isFromCurrentFunction(b, initialSharedState));
    assert (isFromCurrentFunction(b, initialThreadLocal));

    // wait for all other threads to complete
    AllocaInst * const status = b->CreateAlloca(voidPtrTy);

    FixedArray<Value *, 2> pthreadJoinArgs;
    for (unsigned i = 0; i != numOfAdditionalThreads; ++i) {
        Value * threadId = b->CreateLoad(threadIdPtr[i]);
        pthreadJoinArgs[0] = threadId;
        pthreadJoinArgs[1] = status;
        b->CreateCall(pthreadJoinFn->getFunctionType(), pthreadJoinFn, pthreadJoinArgs);
    }

    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        const auto n = mTarget->isStateful() ? 2 : 1;
        SmallVector<Value *, 2> args(n);
        if (LLVM_LIKELY(mTarget->isStateful())) {
            args[0] = initialSharedState;
        }
        for (unsigned i = 0; i < numOfAdditionalThreads; ++i) {
            args[n - 1] = threadLocal[i];
            mTarget->finalizeThreadLocalInstance(b, args);
        }
    }
    if (PipelineHasTerminationSignal) {
        assert (initialTerminationSignalPtr);
        assert (mCurrentThreadTerminationSignalPtr);
        Value * terminated = readTerminationSignalFromLocalState(b, processState);
        assert (terminated);
        for (unsigned i = 0; i < numOfAdditionalThreads; ++i) {
            Value * const terminatedSignal = readTerminationSignalFromLocalState(b, threadState[i]);
            assert (terminatedSignal);
            assert (terminated->getType() == terminatedSignal->getType());
            terminated = b->CreateUMax(terminated, terminatedSignal);
        }
        assert (terminated);
        b->CreateStore(terminated, initialTerminationSignalPtr);
    }

    // TODO: the pipeline kernel scalar state is invalid after leaving this function. Best bet would be to copy the
    // scalarmap and replace it.
    for (unsigned i = 0; i < numOfAdditionalThreads; ++i) {
        destroyStateObject(b, threadState[i]);
    }
    destroyStateObject(b, processState);
    restoreDoSegmentState(storedState);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateFinalizeMethod(BuilderRef b) {
    initializeForAllKernels();
    mScalarValue.reset(FirstKernel, LastScalar);
    // calculate the last segment # used by any kernel in case any reports require it.
    mSegNo = nullptr;
    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceBlockedIO) ||
                      DebugOptionIsSet(codegen::TraceStridesPerSegment))) {
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            if (RequiresSynchronization.test(i)) {
                const auto prefix = makeKernelName(i);
                Value * const ptr = getScalarFieldPtr(b.get(), prefix + LOGICAL_SEGMENT_SUFFIX);
                Value * const segNo = b->CreateLoad(ptr);
                mSegNo = b->CreateUMax(mSegNo, segNo);
            }
        }
    }
    printOptionalCycleCounter(b);
    #ifdef ENABLE_PAPI
    printPAPIReportIfRequested(b);
    #endif
    printOptionalBlockingIOStatistics(b);
    printOptionalBlockedIOPerSegment(b);
    printOptionalBufferExpansionHistory(b);
    printOptionalStridesPerSegment(b);
    printProducedItemCountDeltas(b);
    printUnconsumedItemCounts(b);
    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        setActiveKernel(b, i, false);
        SmallVector<Value *, 1> params;
        if (LLVM_LIKELY(mKernel->isStateful())) {
            params.push_back(mKernelSharedHandle);
        }
        mScalarValue[i] = callKernelFinalizeFunction(b, params);
    }
    releaseOwnedBuffers(b, true);
    resetInternalBufferHandles();
    #ifdef ENABLE_PAPI
    if (!ExternallySynchronized) {
        shutdownPAPI(b);
    }
    #endif
    ActiveKernels.clear();
    ActivePartitions.clear();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getThreadStateType
 ** ------------------------------------------------------------------------------------------------------------- */
inline StructType * PipelineCompiler::getThreadStuctType(BuilderRef b) const {
    FixedArray<Type *, THREAD_STRUCT_SIZE> fields;
    LLVMContext & C = b->getContext();

    assert (mNumOfThreads > 1);

    // NOTE: both the shared and thread local objects are parameters to the kernel.
    // They get automatically set by reading in the appropriate params.

    fields[PIPELINE_PARAMS] = StructType::get(C, mTarget->getDoSegmentFields(b));
    #ifdef USE_FIXED_SEGMENT_NUMBER_INCREMENTS
    if (mNumOfThreads > 1) {
        fields[INITIAL_SEG_NO] = b->getSizeTy();
    } else {
        fields[INITIAL_SEG_NO] = StructType::get(C);
    }
    #endif
    Function * const pthreadSelfFn = b->getModule()->getFunction("pthread_self");
    fields[PROCESS_THREAD_ID] = pthreadSelfFn->getReturnType();
    if (LLVM_LIKELY(PipelineHasTerminationSignal)) {
        fields[TERMINATION_SIGNAL] = b->getSizeTy();
    } else {
        fields[TERMINATION_SIGNAL] = StructType::get(C);
    }
    return StructType::get(C, fields);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructThreadStructObject
 ** ------------------------------------------------------------------------------------------------------------- */
inline Value * PipelineCompiler::constructThreadStructObject(BuilderRef b, Value * const threadId, Value * const threadLocal, const unsigned threadNum) {
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
    #ifdef USE_FIXED_SEGMENT_NUMBER_INCREMENTS
    indices2[1] = b->getInt32(INITIAL_SEG_NO);
    b->CreateStore(b->getSize(threadNum), b->CreateInBoundsGEP(threadState, indices2));
    #endif
    indices2[1] = b->getInt32(PROCESS_THREAD_ID);
    b->CreateStore(threadId, b->CreateInBoundsGEP(threadState, indices2));
    return threadState;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readThreadStuctObject
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::readThreadStuctObject(BuilderRef b, Value * threadState) {
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
    #ifdef USE_FIXED_SEGMENT_NUMBER_INCREMENTS
    if (mNumOfThreads > 1) {
        indices2[1] = b->getInt32(INITIAL_SEG_NO);
        mSegNo = b->CreateLoad(b->CreateInBoundsGEP(threadState, indices2));
    }
    #endif
    mCurrentThreadTerminationSignalPtr = getTerminationSignalPtr();
    if (PipelineHasTerminationSignal) {
        indices2[1] = b->getInt32(TERMINATION_SIGNAL);
        mCurrentThreadTerminationSignalPtr = b->CreateInBoundsGEP(threadState, indices2);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief createThreadStateForSingleThread
 ** ------------------------------------------------------------------------------------------------------------- */
inline void PipelineCompiler::createThreadStateForSingleThread(BuilderRef /* b */) {
    if (PipelineHasTerminationSignal) {
        mCurrentThreadTerminationSignalPtr = getTerminationSignalPtr();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isProcessThread
 ** ------------------------------------------------------------------------------------------------------------- */
inline Value * PipelineCompiler::isProcessThread(BuilderRef b, Value * const threadState) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(PROCESS_THREAD_ID);
    Value * const ptr = b->CreateInBoundsGEP(threadState, indices);
    return b->CreateIsNull(b->CreateLoad(ptr));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateFinalizeThreadLocalMethod(BuilderRef b) {
    initializeForAllKernels();
    assert (mTarget->hasThreadLocal());
    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernel = getKernel(i);
        if (kernel->hasThreadLocal()) {
            setActiveKernel(b, i, true);
            assert (mKernel == kernel);
            SmallVector<Value *, 2> args;
            if (LLVM_LIKELY(mKernelSharedHandle != nullptr)) {
                args.push_back(mKernelSharedHandle);
            }
            args.push_back(mKernelThreadLocalHandle);
            callKernelFinalizeThreadLocalFunction(b, args);
        }
    }
    #ifdef ENABLE_PAPI
    accumulateFinalPAPICounters(b);
    #endif
    releaseOwnedBuffers(b, false);
    // Since all of the nested kernels thread local state is contained within
    // this pipeline thread's thread local state, freeing the pipeline's will
    // also free the inner kernels.
    if (LLVM_LIKELY(RequiredThreadLocalStreamSetMemory > 0)) {
        b->CreateFree(b->getScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY));
    }
    b->CreateFree(getThreadLocalHandle());
    ActiveKernels.clear();
    ActivePartitions.clear();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readTerminationSignalFromLocalState
 ** ------------------------------------------------------------------------------------------------------------- */
Value * PipelineCompiler::readTerminationSignalFromLocalState(BuilderRef b, Value * const threadState) const {
    // TODO: generalize a OR/ADD/etc "combination" mechanism for thread-local to output scalars?
    assert (threadState);
    assert (mCurrentThreadTerminationSignalPtr);
    assert (PipelineHasTerminationSignal);
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(TERMINATION_SIGNAL);
    Value * const signal = b->CreateLoad(b->CreateInBoundsGEP(threadState, indices));
    assert (signal->getType()->isIntegerTy());
    return signal;
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
 * @brief verifyBufferRelationships
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::verifyBufferRelationships() const {

    // If this pipeline is internally synchronized, it must own and manage its output buffers; otherwise
    // the outer pipeline would have to be able to manage it without having correct knowledge of its
    // current state in multithreaded mode. Verify that the correct attributes have been set.
    if (LLVM_UNLIKELY(ExternallySynchronized)) {
        for (const auto e : make_iterator_range(in_edges(PipelineOutput, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            const auto producer = parent(streamSet, mBufferGraph);
            const Kernel * const kernelObj = getKernel(producer);
            assert (kernelObj);
            const auto synchronized = kernelObj->hasAttribute(AttrId::InternallySynchronized);
            const BufferPort & br = mBufferGraph[e];
            const Binding & output = br.Binding;

            const auto managed = br.IsManaged;
            const auto sharedManaged = br.IsShared;

            if (LLVM_UNLIKELY(managed && sharedManaged)) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream out(tmp);
                out << mTarget->getName() << "." << output.getName();
                out << " cannot be both a Managed and SharedManaged buffer.";
                report_fatal_error(out.str());
            }

            const auto unmanaged = !(managed | sharedManaged);

            if (LLVM_UNLIKELY(synchronized ^ unmanaged)) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream out(tmp);
                out << mTarget->getName() << "." << output.getName()
                    << " should ";
                if (synchronized) {
                    out << "not ";
                }
                out << "have been marked as a ManagedBuffer.";
                report_fatal_error(out.str());
            }


        }
    }



#if 0

    // verify that the buffer config is valid
    for (unsigned i = FirstStreamSet; i <= LastStreamSet; ++i) {

        const BufferNode & bn = G[i];
        const auto pe = in_edge(i, G);
        const auto producerVertex = source(pe, G);
        const Kernel * const producer = getKernel(producerVertex);
        const BufferRateData & producerRate = G[pe];
        const Binding & output = producerRate.Binding;




        // Type check stream set I/O types.
        Type * const baseType = output.getType();
        for (const auto e : make_iterator_range(out_edges(i, G))) {
            const BufferRateData & consumerRate = G[e];
            const Binding & input = consumerRate.Binding;
            if (LLVM_UNLIKELY(baseType != input.getType())) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream msg(tmp);
                msg << producer->getName() << ':' << output.getName()
                    << " produces a ";
                baseType->print(msg);
                const Kernel * const consumer = getKernel(target(e, G));
                msg << " but "
                    << consumer->getName() << ':' << input.getName()
                    << " expects ";
                input.getType()->print(msg);
                report_fatal_error(msg.str());
            }
        }

        for (const auto ce : make_iterator_range(out_edges(i, G))) {
            const Binding & input = G[ce].Binding;
            if (LLVM_UNLIKELY(requiresLinearAccess(input))) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream out(tmp);
                const auto consumer = target(ce, G);
                out << getKernel(consumer)->getName()
                    << '.' << input.getName()
                    << " requires that "
                    << producer->getName()
                    << '.' << output.getName()
                    << " is a Linear buffer.";
                report_fatal_error(out.str());
            }
        }


    }

#endif

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeForAllKernels
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeForAllKernels() {
    assert (ActiveKernels.empty());
    assert (ActivePartitions.empty());
    KernelOnHybridThread.reset(PipelineInput);
    KernelOnHybridThread.reset(PipelineOutput);
    PartitionOnHybridThread.reset(KernelPartitionId[PipelineInput]);
    PartitionOnHybridThread.reset(KernelPartitionId[PipelineOutput]);
    ActiveKernels.reserve(LastKernel - PipelineOutput + 1);
    for (unsigned i = FirstKernel; i <= PipelineOutput; ++i) {
        ActiveKernels.push_back(i);
    }
    ActivePartitions.reserve(PartitionCount);
    assert (KernelPartitionId[FirstKernel] == 1);
    for (unsigned i = 1; i < PartitionCount; ++i) {
        ActivePartitions.push_back(i);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief clearInternalState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::clearInternalState(BuilderRef b) {

    mPartitionEntryPoint.reset(0, PartitionCount);
    mPartitionTerminationSignal.reset(0, PartitionCount - 1);
    mExhaustedPipelineInputAtPartitionEntry.reset(0, PartitionCount - 1);

    mLocallyAvailableItems.reset(FirstStreamSet, LastStreamSet);


    std::fill_n(mPartitionProducedItemCountPhi.data(), mPartitionProducedItemCountPhi.num_elements(), nullptr);
    std::fill_n(mPartitionConsumedItemCountPhi.data(), mPartitionConsumedItemCountPhi.num_elements(), nullptr);
    std::fill_n(mPartitionTerminationSignalPhi.data(), mPartitionTerminationSignalPhi.num_elements(), nullptr);
    mPartitionPipelineProgressPhi.reset(0, PartitionCount - 1);

    resetInternalBufferHandles();
    resetConsumerGraphState();
}

}

#endif // PIPELINE_LOGIC_HPP
