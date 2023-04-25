#include "../pipeline_compiler.hpp"
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

LLVM_READNONE bool allocateOnHeap(BuilderRef b) {
    DataLayout DL(b->getModule());
    return (DL.getPointerSizeInBits() != b->getSizeTy()->getBitWidth());
}

Value * makeStateObject(BuilderRef b, Type * type) {
    Value * ptr = nullptr;
    if (LLVM_UNLIKELY(allocateOnHeap(b))) {
        ptr = b->CreatePageAlignedMalloc(type);
    } else {
        ptr = b->CreateCacheAlignedAlloca(type);
    }
    b->CreateMemZero(ptr, ConstantExpr::getSizeOf(type), b->getCacheAlignment());
    return ptr;
}

void destroyStateObject(BuilderRef b, Value * threadState) {
    if (LLVM_UNLIKELY(allocateOnHeap(b))) {
        b->CreateFree(threadState);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief bindAdditionalInitializationArguments
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::bindAdditionalInitializationArguments(BuilderRef b, ArgIterator & arg, const ArgIterator & arg_end) const {
    bindFamilyInitializationArguments(b, arg, arg_end);
    const PipelineKernel * const pk = cast<PipelineKernel>(mTarget);
    if (LLVM_UNLIKELY(pk->generatesDynamicRepeatingStreamSets())) {
        bindRepeatingStreamSetInitializationArguments(b, arg, arg_end);
    }
    assert (arg == arg_end);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateImplicitKernels
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateImplicitKernels(BuilderRef b) {
    assert (b->getModule() == mTarget->getModule());
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        const_cast<Kernel *>(getKernel(i))->generateOrLoadKernel(b);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addPipelineKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addPipelineKernelProperties(BuilderRef b) {
    // TODO: look into improving cache locality/false sharing of this struct

    // TODO: create a non-persistent / pass through input scalar type to allow the
    // pipeline to pass an input scalar to a kernel rather than recording it needlessly?
    // Non-family kernels can be contained within the shared state but family ones
    // must be allocated dynamically.

    IntegerType * const sizeTy = b->getSizeTy();

    #ifndef USE_FIXED_SEGMENT_NUMBER_INCREMENTS
    if (!mIsNestedPipeline) {
        mTarget->addInternalScalar(sizeTy, NEXT_LOGICAL_SEGMENT_NUMBER, 0);
    }
    #endif

    mTarget->addInternalScalar(sizeTy, EXPECTED_NUM_OF_STRIDES_MULTIPLIER, 0);

    if (LLVM_LIKELY(RequiredThreadLocalStreamSetMemory > 0)) {
        PointerType * const int8PtrTy = b->getInt8PtrTy();
        mTarget->addThreadLocalScalar(int8PtrTy, BASE_THREAD_LOCAL_STREAMSET_MEMORY, 0);
        mTarget->addThreadLocalScalar(sizeTy, BASE_THREAD_LOCAL_STREAMSET_MEMORY_BYTES, 0);
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
    auto currentPartitionId = -1U;
    addBufferHandlesToPipelineKernel(b, PipelineInput, 0);
    addConsumerKernelProperties(b, PipelineInput);
    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    unsigned nestedSynchronizationVariableCount = 0;
    #endif
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
        #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
        if (isRoot && PartitionJumpTargetId[partitionId] == (PartitionCount - 1)) {
            mTarget->addInternalScalar(sizeTy,
                NESTED_LOGICAL_SEGMENT_NUMBER_PREFIX + std::to_string(++nestedSynchronizationVariableCount), getCacheLineGroupId(i));
        }
        #endif
    }
    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        mTarget->addThreadLocalScalar(b->getInt64Ty(), STATISTICS_CYCLE_COUNT_TOTAL,
                                      getCacheLineGroupId(PipelineOutput), ThreadLocalScalarAccumulationRule::Sum);
    }
    addRepeatingStreamSetBufferProperties(b);
    generateMetaDataForRepeatingStreamSets(b);
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
    const auto isStateless = isKernelStateFree(kernelId);
    if (LLVM_UNLIKELY(isStateless)) {
        mIsStatelessKernel.set(kernelId);
    }
    assert (mIsStatelessKernel.test(kernelId) == isStateless);
    const auto isInternallySynchronized = mKernel->hasAttribute(AttrId::InternallySynchronized);
    if (LLVM_UNLIKELY(isInternallySynchronized)) {
        mIsInternallySynchronized.set(kernelId);
    }
    #if defined(DISABLE_ALL_DATA_PARALLEL_SYNCHRONIZATION)
    const auto allowDataParallelExecution = false;
    #elif defined(ALLOW_INTERNALLY_SYNCHRONIZED_KERNELS_TO_BE_DATA_PARALLEL)
    const auto allowDataParallelExecution = isStateless || isInternallySynchronized;
    #else
    const auto allowDataParallelExecution = isStateless;
    #endif

    IntegerType * const sizeTy = b->getSizeTy();

    const auto groupId = getCacheLineGroupId(kernelId);

    addTerminationProperties(b, kernelId, groupId);

    const auto name = makeKernelName(kernelId);

    const auto syncLockType = allowDataParallelExecution ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
    mTarget->addInternalScalar(sizeTy, name + LOGICAL_SEGMENT_SUFFIX[syncLockType], groupId);

    if (isRoot) {
        addSegmentLengthSlidingWindowKernelProperties(b, kernelId, groupId);
    }

    addConsumerKernelProperties(b, kernelId);

    for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto prefix = makeBufferName(kernelId, br.Port);
        mTarget->addInternalScalar(sizeTy, prefix + ITEM_COUNT_SUFFIX, groupId);
        if (LLVM_UNLIKELY(isStateless)) {
            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            mTarget->addInternalScalar(sizeTy, prefix + STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX, groupId);
        }
        if (LLVM_UNLIKELY(br.isDeferred())) {
            mTarget->addInternalScalar(sizeTy, prefix + DEFERRED_ITEM_COUNT_SUFFIX, groupId);
        }
    }

    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        const auto prefix = makeBufferName(kernelId, br.Port);
        mTarget->addInternalScalar(sizeTy, prefix + ITEM_COUNT_SUFFIX, groupId);
        if (LLVM_UNLIKELY(isStateless)) {
            const auto streamSet = target(e, mBufferGraph);
            mTarget->addInternalScalar(sizeTy, prefix + STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX, groupId);
        }
        if (LLVM_UNLIKELY(br.isDeferred())) {
            mTarget->addInternalScalar(sizeTy, prefix + DEFERRED_ITEM_COUNT_SUFFIX, groupId);
        }
    }

    addBufferHandlesToPipelineKernel(b, kernelId, groupId);

    addFamilyKernelProperties(b, kernelId, groupId);

    if (LLVM_UNLIKELY(isInternallySynchronized)) {
        // TODO: only needed if its possible to loop back or if we are not guaranteed that this kernel will always fire
        mTarget->addInternalScalar(sizeTy, name + INTERNALLY_SYNCHRONIZED_SUB_SEGMENT_SUFFIX, groupId);
    }

    if (LLVM_LIKELY(mKernel->isStateful())) {
        Type * sharedStateTy = nullptr;
        if (LLVM_UNLIKELY(isKernelFamilyCall(kernelId))) {
            sharedStateTy = b->getVoidPtrTy();
        } else {
            sharedStateTy = mKernel->getSharedStateType();
        }
        mTarget->addInternalScalar(sharedStateTy, name, groupId);
    }

    if (mKernel->hasThreadLocal()) {
        // we cannot statically allocate a "family" thread local object.
        Type * localStateTy = nullptr;
        if (LLVM_UNLIKELY(isKernelFamilyCall(kernelId))) {
            localStateTy = b->getVoidPtrTy();
        } else {
            localStateTy = mKernel->getThreadLocalStateType();
        }
        mTarget->addThreadLocalScalar(localStateTy, name + KERNEL_THREAD_LOCAL_SUFFIX, groupId);
    }

    if (LLVM_UNLIKELY(allowDataParallelExecution)) {
        mTarget->addInternalScalar(sizeTy, name + LOGICAL_SEGMENT_SUFFIX[SYNC_LOCK_POST_INVOCATION], groupId);
    }

    if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram || mGenerateDeferredItemCountHistogram)) {
        addHistogramProperties(b, kernelId, groupId);
    }

    if (LLVM_UNLIKELY(mTraceDynamicBuffers)) {
        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const auto bufferVertex = target(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[bufferVertex];
            if (bn.Buffer->isDynamic()) {
                const BufferPort & rd = mBufferGraph[e];
                const auto prefix = makeBufferName(kernelId, rd.Port);
                LLVMContext & C = b->getContext();
                const auto numOfConsumers = std::max(out_degree(bufferVertex, mConsumerGraph), 1UL);

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
    #ifdef ENABLE_PAPI
    if (!mIsNestedPipeline) {
        initializePAPI(b);
    }
    #endif

    mScalarValue.reset(FirstKernel, LastScalar);

    initializeKernelAssertions(b);

    Constant * const unterminated = getTerminationSignal(b, TerminationSignal::None);
    Constant * const aborted = getTerminationSignal(b, TerminationSignal::Aborted);

    Value * terminated = nullptr;
    auto partitionId = KernelPartitionId[PipelineInput];

    for (auto i = FirstKernel; i <= LastKernel; ++i) {

        const auto curPartitionId = KernelPartitionId[i];
        const auto isRoot = (curPartitionId != partitionId);
        partitionId = curPartitionId;
        // Family kernels must be initialized in the "main" method.
        setActiveKernel(b, i, false);
        assert (mKernel->isGenerated());
        if (isRoot) {
            initializeStridesPerSegment(b);
        }

        if (LLVM_LIKELY(!isKernelFamilyCall(i))) {
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
        if (terminated && HasTerminationSignal[mKernelId]) {
            Value * const signal = b->CreateSelect(terminated, aborted, unterminated);
            writeTerminationSignal(b, mKernelId, signal);
            terminated = nullptr;
        }

    }
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateAllocateSharedInternalStreamSetsMethod(BuilderRef b, Value * const expectedNumOfStrides) {
    b->setScalarField(EXPECTED_NUM_OF_STRIDES_MULTIPLIER, expectedNumOfStrides);
    initializeInitialSlidingWindowSegmentLengths(b, expectedNumOfStrides);
    allocateOwnedBuffers(b, expectedNumOfStrides, true);
    initializeBufferExpansionHistory(b);
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateInitializeThreadLocalMethod(BuilderRef b) {
    assert (mTarget->hasThreadLocal());
    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernel = getKernel(i);
        if (kernel->hasThreadLocal()) {
            setActiveKernel(b, i, true);
            assert (mKernel == kernel);
            callKernelInitializeThreadLocalFunction(b);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateThreadLocalInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateAllocateThreadLocalInternalStreamSetsMethod(BuilderRef b, Value * const expectedNumOfStrides) {
    assert (mTarget->hasThreadLocal());
    if (LLVM_LIKELY(RequiredThreadLocalStreamSetMemory > 0)) {
        auto size = RequiredThreadLocalStreamSetMemory;
        #ifdef THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER
        size *= THREADLOCAL_BUFFER_CAPACITY_MULTIPLIER;
        #endif
        ConstantInt * const reqMemory = b->getSize(size);
        Value * const memorySize = b->CreateMul(reqMemory, expectedNumOfStrides);
        Value * const base = b->CreatePageAlignedMalloc(memorySize);
        PointerType * const int8PtrTy = b->getInt8PtrTy();
        b->setScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY, b->CreatePointerCast(base, int8PtrTy));
        b->setScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY_BYTES, memorySize);
    }
    allocateOwnedBuffers(b, expectedNumOfStrides, false);
    resetInternalBufferHandles();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateKernelMethod(BuilderRef b) {
    initializeKernelAssertions(b);
    // verifyBufferRelationships();
    mScalarValue.reset(FirstKernel, LastScalar);
   // readPipelineIOItemCounts(b);
    if (LLVM_UNLIKELY(mNumOfThreads == 0)) {
        report_fatal_error("Fatal error: cannot construct a 0-thread pipeline.");
    }
    if (mNumOfThreads == 1) {
        generateSingleThreadKernelMethod(b);
    } else {
        if (LLVM_UNLIKELY(mIsNestedPipeline)) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "A multi-threaded pipeline is already internally synchronized. "
                "Explicitly annotating " << mTarget->getName() << " with the InternallySynchronized attribute "
                "will prevent an outer pipeline kernel from behaving in the intended manner.";
            report_fatal_error(out.str());
        }
        generateMultiThreadKernelMethod(b);
    }
    resetInternalBufferHandles();
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
    }
    #ifdef USE_FIXED_SEGMENT_NUMBER_INCREMENTS
    else {
        mSegNo = b->getSize(0);
    }
    #endif
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

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief concat
 ** ------------------------------------------------------------------------------------------------------------- */
StringRef concat(StringRef A, StringRef B, SmallVector<char, 256> & tmp) {
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
 * @brief generateFinalizeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateFinalizeMethod(BuilderRef b) {
    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet())) {

        // get the last segment # used by any kernel in case any reports require it.
        const auto type = isDataParallel(FirstKernel) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL;
        Value * const ptr = getSynchronizationLockPtrForKernel(b, FirstKernel, type);
        mSegNo = b->CreateLoad(ptr);

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
        if (mGenerateTransferredItemCountHistogram) {
            printHistogramReport(b, HistogramReportType::TransferredItems);
        }
        if (mGenerateDeferredItemCountHistogram) {
            printHistogramReport(b, HistogramReportType::DeferredItems);
        }
    }

    mScalarValue.reset(FirstKernel, LastScalar);
    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        setActiveKernel(b, i, true);
        SmallVector<Value *, 1> params;
        if (LLVM_LIKELY(mKernel->isStateful())) {
            assert (mTarget->isStateful());
            params.push_back(mKernelSharedHandle);
        }
        if (LLVM_UNLIKELY(mKernel->hasThreadLocal())) {
            assert (mTarget->hasThreadLocal());
            params.push_back(mKernelThreadLocalHandle);
        }
        mScalarValue[i] = callKernelFinalizeFunction(b, params);
    }
    if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram || mGenerateDeferredItemCountHistogram)) {
        freeHistogramProperties(b);
    }
    deallocateRepeatingBuffers(b);
    releaseOwnedBuffers(b);
    resetInternalBufferHandles();
    #ifdef ENABLE_PAPI
    if (!mIsNestedPipeline) {
        shutdownPAPI(b);
    }
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getThreadStateType
 ** ------------------------------------------------------------------------------------------------------------- */
StructType * PipelineCompiler::getThreadStuctType(BuilderRef b) const {
    FixedArray<Type *, THREAD_STRUCT_SIZE> fields;
    LLVMContext & C = b->getContext();

    assert (mNumOfThreads > 1);

    // NOTE: both the shared and thread local objects are parameters to the kernel.
    // They get automatically set by reading in the appropriate params.

    fields[PIPELINE_PARAMS] = StructType::get(C, mTarget->getDoSegmentFields(b));
    #ifdef USE_FIXED_SEGMENT_NUMBER_INCREMENTS
    fields[INITIAL_SEG_NO] = b->getSizeTy();
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
    #ifdef USE_FIXED_SEGMENT_NUMBER_INCREMENTS
    indices2[1] = b->getInt32(INITIAL_SEG_NO);
    mSegNo = b->CreateLoad(b->CreateInBoundsGEP(threadState, indices2));
    #endif
    mCurrentThreadTerminationSignalPtr = getTerminationSignalPtr();
    if (PipelineHasTerminationSignal) {
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
 * @brief generateFinalizeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::generateFinalizeThreadLocalMethod(BuilderRef b) {
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
            args.push_back(getCommonThreadLocalHandlePtr(b, i));
            args.push_back(mKernelThreadLocalHandle);
            callKernelFinalizeThreadLocalFunction(b, args);
            if (LLVM_UNLIKELY(isKernelFamilyCall(i))) {
                b->CreateFree(mKernelThreadLocalHandle);
            }
        }
    }

    #ifdef ENABLE_PAPI
    accumulateFinalPAPICounters(b);
    #endif
    // Since all of the nested kernels thread local state is contained within
    // this pipeline thread's thread local state, freeing the pipeline's will
    // also free the inner kernels.
    if (LLVM_LIKELY(RequiredThreadLocalStreamSetMemory > 0)) {
        b->CreateFree(b->getScalarField(BASE_THREAD_LOCAL_STREAMSET_MEMORY));
    }
    if (LLVM_UNLIKELY(HasZeroExtendedStream)) {
        b->CreateFree(b->getScalarField(ZERO_EXTENDED_BUFFER));
    }
    freePendingFreeableDynamicBuffers(b);
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
 * @brief clearInternalState
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::clearInternalState(BuilderRef b) {

    mPartitionEntryPoint.reset(0, PartitionCount);
    mKernelTerminationSignal.reset(FirstKernel, LastKernel);

    mLocallyAvailableItems.reset(FirstStreamSet, LastStreamSet);

    std::fill_n(mPartitionProducedItemCountPhi.data(), mPartitionProducedItemCountPhi.num_elements(), nullptr);
    std::fill_n(mPartitionConsumedItemCountPhi.data(), mPartitionConsumedItemCountPhi.num_elements(), nullptr);
    std::fill_n(mPartitionTerminationSignalPhi.data(), mPartitionTerminationSignalPhi.num_elements(), nullptr);
    mPartitionPipelineProgressPhi.reset(0, PartitionCount - 1);

    resetInternalBufferHandles();
}

}
