#include <kernel/pipeline/pipeline_kernel.h>
#include <toolchain/toolchain.h>

// #define USE_2020_PIPELINE_COMPILER

#ifdef USE_2020_PIPELINE_COMPILER
#include "2020/compiler/pipeline_compiler.hpp"
#else
#include "compiler/pipeline_compiler.hpp"
// #include "PROP/compiler/pipeline_compiler.hpp"
#endif
#include <llvm/IR/Function.h>
#include <kernel/pipeline/pipeline_builder.h>

// NOTE: the pipeline kernel is primarily a proxy for the pipeline compiler. Ideally, by making some kernels
// a "family", the pipeline kernel will be compiled once for the lifetime of a program. Thus we can avoid even
// constructing any data structures for the pipeline in normal usage.

using IDISA::FixedVectorType;

namespace kernel {

#define COMPILER (static_cast<PipelineCompiler *>(b->getCompiler()))

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addInternalKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::addInternalProperties(BuilderRef b) {
    COMPILER->generateImplicitKernels(b);
    COMPILER->addPipelineKernelProperties(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateInitializeMethod(BuilderRef b) {
    COMPILER->generateInitializeMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateInitializeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateInitializeThreadLocalMethod(BuilderRef b) {
    COMPILER->generateInitializeThreadLocalMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernelMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateKernelMethod(BuilderRef b) {
    COMPILER->generateKernelMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateFinalizeMethod(BuilderRef b) {
    COMPILER->generateFinalizeMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateFinalizeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateFinalizeThreadLocalMethod(BuilderRef b) {
    COMPILER->generateFinalizeThreadLocalMethod(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addKernelDeclarations
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::addKernelDeclarations(BuilderRef b) {
    for (const auto & k : mKernels) {
        k->addKernelDeclarations(b);
    }
    Kernel::addKernelDeclarations(b);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief hasInternalStreamSets
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineKernel::allocatesInternalStreamSets() const {
    return true;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateSharedInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateAllocateSharedInternalStreamSetsMethod(BuilderRef b, Value * expectedNumOfStrides) {
    #ifndef USE_2020_PIPELINE_COMPILER
    COMPILER->generateAllocateSharedInternalStreamSetsMethod(b, expectedNumOfStrides);
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateAllocateThreadLocalInternalStreamSetsMethod
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::generateAllocateThreadLocalInternalStreamSetsMethod(BuilderRef b, Value * expectedNumOfStrides) {
    #ifndef USE_2020_PIPELINE_COMPILER
    COMPILER->generateAllocateThreadLocalInternalStreamSetsMethod(b, expectedNumOfStrides);
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkExternalMethods
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::linkExternalMethods(BuilderRef b) {
    PipelineCompiler::linkPThreadLibrary(b);
    for (const auto & k : mKernels) {
        k->linkExternalMethods(b);
    }
    for (const CallBinding & call : mCallBindings) {
        call.Callee = b->LinkFunction(call.Name, call.Type, call.FunctionPointer);
    }
    #ifdef ENABLE_PAPI
    if (LLVM_UNLIKELY(codegen::PapiCounterOptions.compare(codegen::OmittedOption) != 0)) {
        PipelineCompiler::linkPAPILibrary(b);
    }
    #endif
    #ifndef USE_2020_PIPELINE_COMPILER
    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet())) {
        PipelineCompiler::linkInstrumentationFunctions(b);
        PipelineCompiler::linkHistogramFunctions(b);
    }
    #endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addAdditionalFunctions
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::addAdditionalFunctions(BuilderRef b) {
    if (hasAttribute(AttrId::InternallySynchronized) || externallyInitialized()) {
        return;
    }
    addOrDeclareMainFunction(b, Kernel::AddExternal);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief containsKernelFamilies
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineKernel::externallyInitialized() const {
    if (LLVM_UNLIKELY(hasFamilyName())) {
        return true;
    }
    for (Kernel * k : mKernels) {
        if (k->externallyInitialized()) {
            return true;
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addFamilyInitializationArgTypes
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::addFamilyInitializationArgTypes(BuilderRef b, InitArgTypes & argTypes) const {
    unsigned n = 0;
    for (const Kernel * kernel : mKernels) {
        if (kernel->externallyInitialized()) {
            if (LLVM_LIKELY(kernel->isStateful())) {
                n += 1;
            }
            if (kernel->hasFamilyName()) {
                const auto ai = kernel->allocatesInternalStreamSets();
                const auto k1 = ai ? 3U : 2U;
                const auto tl = kernel->hasThreadLocal();
                const auto k2 = tl ? (k1 * 2U) : k1;
                n += k2;
            }
        }
    }
    if (LLVM_LIKELY(n > 0)) {
        argTypes.append(n, b->getVoidPtrTy());
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recursivelyConstructFamilyKernels
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::recursivelyConstructFamilyKernels(BuilderRef b, InitArgs & args, const ParamMap & params, NestedStateObjs & toFree) const {
    for (const Kernel * const kernel : mKernels) {
        if (LLVM_UNLIKELY(kernel->externallyInitialized())) {
            kernel->constructFamilyKernels(b, args, params, toFree);
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runOptimizationPasses
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::runOptimizationPasses(BuilderRef b) const {
    #ifndef USE_2020_PIPELINE_COMPILER
    COMPILER->runOptimizationPasses(b);
    #endif
}

#define JOIN3(X,Y,Z) BOOST_JOIN(X,BOOST_JOIN(Y,Z))

#define REPLACE_INTERNAL_KERNEL_BINDINGS(BindingType) \
    const auto * const from = JOIN3(m, BindingType, s)[i].getRelationship(); \
    for (auto * K : mKernels) { \
        const auto & B = K->JOIN3(get, BindingType, Bindings)(); \
        for (unsigned j = 0; j < B.size(); ++j) { \
            if (LLVM_UNLIKELY(B[j].getRelationship() == from)) { \
                K->JOIN3(set, BindingType, At)(j, value); } } } \
    JOIN3(m, BindingType, s)[i].setRelationship(value);

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setInputStreamSetAt
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::setInputStreamSetAt(const unsigned i, StreamSet * const value) {
    REPLACE_INTERNAL_KERNEL_BINDINGS(InputStreamSet);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setOutputStreamSetAt
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::setOutputStreamSetAt(const unsigned i, StreamSet * const value) {
    REPLACE_INTERNAL_KERNEL_BINDINGS(OutputStreamSet);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setInputScalarAt
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::setInputScalarAt(const unsigned i, Scalar * const value) {
    REPLACE_INTERNAL_KERNEL_BINDINGS(InputScalar);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setOutputScalarAt
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineKernel::setOutputScalarAt(const unsigned i, Scalar * const value) {
    REPLACE_INTERNAL_KERNEL_BINDINGS(OutputScalar);
}

#undef JOIN3
#undef REPLACE_INTERNAL_KERNEL_BINDINGS

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief instantiateKernelCompiler
 ** ------------------------------------------------------------------------------------------------------------- */
std::unique_ptr<KernelCompiler> PipelineKernel::instantiateKernelCompiler(BuilderRef b) const {
    return std::make_unique<PipelineCompiler>(b, const_cast<PipelineKernel *>(this));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief isCachable
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineKernel::isCachable() const {
    return codegen::EnablePipelineObjectCache;
}

size_t __getStridesPerSegment() {
    return codegen::BufferSegments * codegen::SegmentSize;
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addOrDeclareMainFunction
 ** ------------------------------------------------------------------------------------------------------------- */
Function * PipelineKernel::addOrDeclareMainFunction(BuilderRef b, const MainMethodGenerationType method) const {

    unsigned suppliedArgs = 0;
    if (LLVM_LIKELY(isStateful())) {
        suppliedArgs += 1;
    }
    if (LLVM_LIKELY(hasThreadLocal())) {
        suppliedArgs += 1;
    }

    Module * const m = b->getModule();
    Function * const doSegment = getDoSegmentFunction(b, false); assert (doSegment);
    assert (doSegment->arg_size() >= suppliedArgs);
   //  const auto numOfDoSegArgs = doSegment->arg_size() - suppliedArgs;
    Function * const terminate = getFinalizeFunction(b);

    const auto numOfStreamSets = mInputStreamSets.size() + mOutputStreamSets.size();

    // maintain consistency with the Kernel interface by passing first the stream sets
    // and then the scalars.
    SmallVector<Type *, 32> params;
    params.reserve(getNumOfScalarInputs() + numOfStreamSets);

    PointerType * streamSetPtrTy = nullptr;
    if (numOfStreamSets) {
        // must match streamsetptr.h
        FixedArray<Type *, 2> fields;
        fields[0] = b->getVoidPtrTy();
        fields[1] = b->getInt64Ty();
        StructType * sty = StructType::get(b->getContext(), fields);
        streamSetPtrTy = sty->getPointerTo();
    }

    // The initial params of doSegment are its shared handle, thread-local handle and numOfStrides.
    // (assuming the kernel has both handles). The remaining are the stream set params.

    for (unsigned i = 0; i < numOfStreamSets; ++i) {
        params.push_back(streamSetPtrTy);
    }
    for (const auto & input : getInputScalarBindings()) {
        params.push_back(input.getType());
    }

    const auto linkageType = (method == AddInternal) ? Function::InternalLinkage : Function::ExternalLinkage;

    SmallVector<char, 256> tmp;
    raw_svector_ostream funcNameGen(tmp);
    funcNameGen << getName() << '@' << codegen::SegmentSize << "_main";
    const auto funcName = funcNameGen.str();

    Function * main = m->getFunction(funcName);
    if (LLVM_LIKELY(main == nullptr)) {
        // get the finalize method output type and set its return type as this function's return type
        FunctionType * const mainFunctionType = FunctionType::get(terminate->getReturnType(), params, false);
        main = Function::Create(mainFunctionType, linkageType, funcName, m);
        main->setCallingConv(CallingConv::C);
    }

    // declaration only; exit
    if (method == DeclareExternal) {
        return main;
    }

    assert (main->empty());

    b->SetInsertPoint(BasicBlock::Create(b->getContext(), "entry", main));
    auto arg = main->arg_begin();
    auto nextArg = [&]() {
        assert (arg != main->arg_end());
        Value * const v = &*arg;
        std::advance(arg, 1);
        return v;
    };
    SmallVector<Value *, 16> segmentArgs(doSegment->arg_size());

    if (LLVM_UNLIKELY(numOfStreamSets > 0)) {

        auto argCount = suppliedArgs;

        ConstantInt * const i32_ZERO = b->getInt32(0);
        ConstantInt * const i32_ONE = b->getInt32(1);

        Value * const sz_ZERO = b->getSize(0);

        FixedArray<Value *, 2> fields;
        fields[0] = i32_ZERO;

        for (auto i = mInputStreamSets.size(); i--; ) {
            Value * const streamSetArg = nextArg();
            assert (streamSetArg->getType() == streamSetPtrTy);
            // virtual base input address
            fields[1] = i32_ZERO;
            Value * const vbaPtr = b->CreateGEP(streamSetArg, fields);
            segmentArgs[argCount++] = b->CreateLoad(vbaPtr);
            // processed input items
            fields[1] = i32_ONE;
            Value * const processedPtr = b->CreateAllocaAtEntryPoint(b->getSizeTy());
            b->CreateStore(sz_ZERO, processedPtr);
            segmentArgs[argCount++] = processedPtr; // updatable
            // accessible input items
            segmentArgs[argCount++] = b->CreateLoad(b->CreateGEP(streamSetArg, fields));
        }

        for (auto i = mOutputStreamSets.size(); i--; ) {
            Value * const streamSetArg = nextArg();
            assert (streamSetArg->getType() == streamSetPtrTy);

            // shared dynamic buffer handle or virtual base output address
            fields[1] = i32_ZERO;
            segmentArgs[argCount++] = b->CreateGEP(streamSetArg, fields);

            // produced output items
            fields[1] = i32_ONE;
            Value * const itemPtr = b->CreateGEP(streamSetArg, fields);
            segmentArgs[argCount++] = b->CreateGEP(streamSetArg, fields);
            segmentArgs[argCount++] = b->CreateLoad(itemPtr);
        }

        assert (argCount == doSegment->arg_size());
    }

    Value * sharedHandle = nullptr;
    NestedStateObjs toFree;
    BEGIN_SCOPED_REGION
    ParamMap paramMap;
    for (const auto & input : getInputScalarBindings()) {
        const Scalar * const scalar = cast<Scalar>(input.getRelationship());
        Value * const value = nextArg();
        paramMap.insert(std::make_pair(scalar, value));
    }
    InitArgs args;
    sharedHandle = constructFamilyKernels(b, args, paramMap, toFree);
    END_SCOPED_REGION
    assert (isStateful() || sharedHandle == nullptr);

    size_t argCount = 0;
    if (LLVM_LIKELY(isStateful())) {
        segmentArgs[argCount++] = sharedHandle;
    }
    Value * threadLocalHandle = nullptr;
    if (LLVM_LIKELY(hasThreadLocal())) {
        SmallVector<Value *, 2> args;
        if (LLVM_LIKELY(isStateful())) {
            args.push_back(sharedHandle);
        }
        args.push_back(ConstantPointerNull::get(getThreadLocalStateType()->getPointerTo()));
        threadLocalHandle = initializeThreadLocalInstance(b, args);
        segmentArgs[argCount++] = threadLocalHandle;
        toFree.push_back(threadLocalHandle);
    }
    assert (argCount == suppliedArgs);

    if (LLVM_UNLIKELY(hasAttribute(AttrId::InternallySynchronized))) {
        report_fatal_error(doSegment->getName() + " cannot be externally synchronized");
    }

    ConstantInt * const sz_ONE = b->getSize(1);

    // allocate any internal stream sets
    if (LLVM_LIKELY(allocatesInternalStreamSets())) {
        Function * const allocShared = getAllocateSharedInternalStreamSetsFunction(b);
        SmallVector<Value *, 2> allocArgs;
        if (LLVM_LIKELY(isStateful())) {
            allocArgs.push_back(sharedHandle);
        }
        // pass in the desired number of segments
        //TODO: fix this so BufferSegments is an argument to main
        allocArgs.push_back(sz_ONE);
        b->CreateCall(allocShared->getFunctionType(), allocShared, allocArgs);
        if (LLVM_LIKELY(hasThreadLocal())) {
            Function * const allocThreadLocal = getAllocateThreadLocalInternalStreamSetsFunction(b);
            SmallVector<Value *, 3> allocArgs;
            if (LLVM_LIKELY(isStateful())) {
                allocArgs.push_back(sharedHandle);
            }
            allocArgs.push_back(threadLocalHandle);
            allocArgs.push_back(sz_ONE);
            b->CreateCall(allocThreadLocal->getFunctionType(), allocThreadLocal, allocArgs);
        }
    }

    PHINode * successPhi = nullptr;
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts) ||
                      codegen::DebugOptionIsSet(codegen::EnablePipelineAsserts))) {
        BasicBlock * const handleCatch = b->CreateBasicBlock("");
        BasicBlock * const handleDeallocation = b->CreateBasicBlock("");

        IntegerType * const int32Ty = b->getInt32Ty();
        PointerType * const int8PtrTy = b->getInt8PtrTy();
        LLVMContext & C = b->getContext();
        StructType * const caughtResultType = StructType::get(C, { int8PtrTy, int32Ty });
        Function * const personalityFn = b->getDefaultPersonalityFunction();
        main->setPersonalityFn(personalityFn);

        BasicBlock * const beforeInvoke = b->GetInsertBlock();
        b->CreateInvoke(doSegment, handleDeallocation, handleCatch, segmentArgs);

        b->SetInsertPoint(handleCatch);
        LandingPadInst * const caughtResult = b->CreateLandingPad(caughtResultType, 0);
        caughtResult->addClause(ConstantPointerNull::get(int8PtrTy));
        Function * catchFn = b->getBeginCatch();
        Function * catchEndFn = b->getEndCatch();
        b->CreateCall(catchFn->getFunctionType(), catchFn, {b->CreateExtractValue(caughtResult, 0)});
        b->CreateCall(catchEndFn->getFunctionType(), catchEndFn, {});
        BasicBlock * const afterCatch = b->GetInsertBlock();
        b->CreateBr(handleDeallocation);

        b->SetInsertPoint(handleDeallocation);
        successPhi = b->CreatePHI(b->getInt1Ty(), 2);
        successPhi->addIncoming(b->getTrue(), beforeInvoke);
        successPhi->addIncoming(b->getFalse(), afterCatch);
    } else {
        b->CreateCall(doSegment->getFunctionType(), doSegment, segmentArgs);
    }
    SmallVector<Value *, 3> args;
    if (LLVM_LIKELY(isStateful())) {
        args.push_back(sharedHandle);
    }
    if (LLVM_LIKELY(hasThreadLocal())) {
        args.push_back(threadLocalHandle);
        args.push_back(threadLocalHandle);
        finalizeThreadLocalInstance(b, args);
        args.pop_back();
    }
    Value * const result = finalizeInstance(b, args);
    for (Value * stateObj : toFree) {
        b->CreateFree(stateObj);
    }
    b->CreateRet(result);
    return main;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
PipelineKernel::PipelineKernel(BuilderRef b,
                               std::string && signature,
                               const unsigned numOfThreads,
                               Kernels && kernels, CallBindings && callBindings,
                               Bindings && stream_inputs, Bindings && stream_outputs,
                               Bindings && scalar_inputs, Bindings && scalar_outputs,
                               LengthAssertions && lengthAssertions)
: Kernel(b, TypeId::Pipeline,
         [&] () {
             std::string tmp;
             tmp.reserve(32);
             raw_string_ostream name(tmp);
             name << 'P' << numOfThreads
                  << '_' << Kernel::getStringHash(signature);
             name.flush();
             return tmp;
         } (),
         std::move(stream_inputs), std::move(stream_outputs),
         std::move(scalar_inputs), std::move(scalar_outputs),
         {} /* Internal scalars are generated by the PipelineCompiler */)
, mNumOfThreads(numOfThreads)
, mSignature(std::move(signature))
, mKernels(std::move(kernels))
, mCallBindings(std::move(callBindings))
, mLengthAssertions(std::move(lengthAssertions)) {

}

PipelineKernel::~PipelineKernel() {

}

}
