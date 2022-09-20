#include <kernel/core/kernel_compiler.h>
#include <kernel/core/kernel_builder.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/MDBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/PromoteMemToReg.h>
#include <llvm/IR/Dominators.h>
#ifndef NDEBUG
#include <llvm/IR/Verifier.h>
#endif
#include <boost/intrusive/detail/math.hpp>
#include <boost/container/flat_set.hpp>
#include <kernel/core/streamsetptr.h>
#include <codegen/TypeBuilder.h>

using namespace llvm;
using namespace boost;
using boost::intrusive::detail::floor_log2;
using boost::container::flat_set;

namespace kernel {

using AttrId = Attribute::KindId;
using Rational = ProcessingRate::Rational;
using RateId = ProcessingRate::KindId;
using StreamSetPort = Kernel::StreamSetPort;
using PortType = Kernel::PortType;

const static std::string BUFFER_HANDLE_SUFFIX = "_buffer";
const static std::string TERMINATION_SIGNAL = "__termination_signal";
const static std::string INTERNAL_COMMON_THREAD_LOCAL_PREFIX = "!__ctl__";

#define BEGIN_SCOPED_REGION {
#define END_SCOPED_REGION }

// TODO: this check is a bit too strict in general; if the pipeline could request data/
// EOF padding from the MemorySource kernel, it would be possible to re-enable.

// #define CHECK_IO_ADDRESS_RANGE

// TODO: split the init/final into two methods each, one to do allocation/init, and the
// other final/deallocate? Would potentially allow us to reuse the kernel/stream set
// memory in the nested engine if each init method memzero'ed them. Would need to change
// the "main" method.

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief generateKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::generateKernel(BuilderRef b) {
    // NOTE: make sure to keep and reset the original compiler here. A kernel could generate new kernels and
    // reuse the same KernelBuilder to do so; this could result in unexpected behaviour if the this function
    // exits without restoring the original compiler state.
    auto const oc = b->getCompiler();
    b->setCompiler(this);
    constructStreamSetBuffers(b);
    #ifndef NDEBUG
    for (const auto & buffer : mStreamSetInputBuffers) {
        assert ("input buffer not set by constructStreamSetBuffers" && buffer.get());
    }
    for (const auto & buffer : mStreamSetOutputBuffers) {
        assert ("output buffer not set by constructStreamSetBuffers" && buffer.get());
    }
    #endif
    addBaseInternalProperties(b);
    mTarget->addInternalProperties(b);
    mTarget->constructStateTypes(b);
    mTarget->addKernelDeclarations(b);
    callGenerateInitializeMethod(b);
    callGenerateAllocateSharedInternalStreamSets(b);
    callGenerateInitializeThreadLocalMethod(b);
    callGenerateAllocateThreadLocalInternalStreamSets(b);
    callGenerateDoSegmentMethod(b);
    callGenerateFinalizeThreadLocalMethod(b);
    callGenerateFinalizeMethod(b);
    mTarget->addAdditionalFunctions(b);

    // TODO: we could create a LLVM optimization pass manager here and execute it on this kernel;
    // it would allow the programmer to define a set of optimizations they want executed on the
    // kernel code. However, if compilers are intended to be short lived, we wouldn't be able to
    // easily share it amongst the same type of kernel compiler.

    // What is the cost of generating a pass manager instance for each compiled kernel vs.
    // the complexity of using a factory?

    runInternalOptimizationPasses(b->getModule());
    mTarget->runOptimizationPasses(b);
    b->setCompiler(oc);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructStreamSetBuffers
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::constructStreamSetBuffers(BuilderRef b) {
    mStreamSetInputBuffers.clear();
    const auto numOfInputStreams = mInputStreamSets.size();
    mStreamSetInputBuffers.resize(numOfInputStreams);
    for (unsigned i = 0; i < numOfInputStreams; ++i) {
        const Binding & input = mInputStreamSets[i];
        mStreamSetInputBuffers[i].reset(new ExternalBuffer(i, b, input.getType(), true, 0));
    }
    mStreamSetOutputBuffers.clear();
    const auto numOfOutputStreams = mOutputStreamSets.size();
    mStreamSetOutputBuffers.resize(numOfOutputStreams);
    for (unsigned i = 0; i < numOfOutputStreams; ++i) {
        const Binding & output = mOutputStreamSets[i];
        mStreamSetOutputBuffers[i].reset(new ExternalBuffer(i + numOfInputStreams, b, output.getType(), true, 0));
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addBaseInternalProperties
  ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::addBaseInternalProperties(BuilderRef b) {
     // If an output is a managed buffer, store its handle.
    const auto n = mOutputStreamSets.size();
    for (unsigned i = 0; i < n; ++i) {
        const Binding & output = mOutputStreamSets[i];
        Type * const handleTy = mStreamSetOutputBuffers[i]->getHandleType(b);
        if (LLVM_UNLIKELY(Kernel::isLocalBuffer(output, false))) {
            mTarget->addInternalScalar(handleTy, output.getName() + BUFFER_HANDLE_SUFFIX);
        } else {
            mTarget->addNonPersistentScalar(handleTy, output.getName() + BUFFER_HANDLE_SUFFIX);
        }
    }
    IntegerType * const sizeTy = b->getSizeTy();
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {

        // In multi-threaded mode, given a small file, the pipeline could finish before all of the
        // threads are constructed. Since we cannot detect when this occurs without additional
        // book keeping and the behaviour is safe, we do not guard against double termination.
        // All other kernels are checked to ensure that there are no pipeline errors.

        if (mTarget->getTypeId() != Kernel::TypeId::Pipeline || mTarget->hasAttribute(AttrId::InternallySynchronized)) {
            mTarget->addInternalScalar(sizeTy, TERMINATION_SIGNAL);
        } else {
            mTarget->addNonPersistentScalar(sizeTy, TERMINATION_SIGNAL);
        }
    } else {
        mTarget->addNonPersistentScalar(sizeTy, TERMINATION_SIGNAL);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief reset
 ** ------------------------------------------------------------------------------------------------------------- */
template <typename Vec>
inline void reset(Vec & vec, const size_t n) {
    vec.resize(n);
    std::fill_n(vec.begin(), n, nullptr);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callGenerateInitializeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateInitializeMethod(BuilderRef b) {
    b->setCompiler(this);
    mCurrentMethod = mTarget->getInitializeFunction(b);
    mEntryPoint = BasicBlock::Create(b->getContext(), "entry", mCurrentMethod);
    b->SetInsertPoint(mEntryPoint);
    auto arg = mCurrentMethod->arg_begin();
    const auto arg_end = mCurrentMethod->arg_end();
    auto nextArg = [&]() {
        assert (arg != arg_end);
        Value * const v = &*arg;
        std::advance(arg, 1);
        return v;
    };
    assert (getHandle() == nullptr);
    if (LLVM_LIKELY(mTarget->isStateful())) {
        setHandle(nextArg());
    }
    if (LLVM_LIKELY(mTarget->isStateful())) {
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
            b->CreateMProtect(mSharedHandle, CBuilder::Protect::WRITE);
        }
    }
    initializeScalarMap(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
    for (const auto & binding : mInputScalars) {
        b->setScalarField(binding.getName(), nextArg());
    }
    bindFamilyInitializationArguments(b, arg, arg_end);
    assert (arg == arg_end);    
    // TODO: we could permit shared managed buffers here if we passed in the buffer
    // into the init method. However, since there are no uses of this in any written
    // program, we currently prohibit it.
    initializeOwnedBufferHandles(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
    // any kernel can set termination on initialization
    mTerminationSignalPtr = b->getScalarFieldPtr(TERMINATION_SIGNAL);
    b->CreateStore(b->getSize(KernelBuilder::TerminationCode::None), mTerminationSignalPtr);
    mTarget->generateInitializeMethod(b);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect) && mTarget->isStateful())) {
        b->CreateMProtect(mSharedHandle, CBuilder::Protect::READ);
    }
    b->CreateRet(b->CreateLoad(mTerminationSignalPtr));
    clearInternalStateAfterCodeGen();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief bindFamilyInitializationArguments
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::bindFamilyInitializationArguments(BuilderRef /* b */, ArgIterator & /* arg */, const ArgIterator & /* arg_end */) const {

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callGenerateInitializeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateInitializeThreadLocalMethod(BuilderRef b) {
    if (mTarget->hasThreadLocal()) {
        b->setCompiler(this);
        assert (mSharedHandle == nullptr && mThreadLocalHandle == nullptr);
        mCurrentMethod = mTarget->getInitializeThreadLocalFunction(b);
        mEntryPoint = BasicBlock::Create(b->getContext(), "entry", mCurrentMethod);
        b->SetInsertPoint(mEntryPoint);
        auto arg = mCurrentMethod->arg_begin();
        auto nextArg = [&]() {
            assert (arg != mCurrentMethod->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };
        if (LLVM_LIKELY(mTarget->isStateful())) {
            setHandle(nextArg());
        }

        StructType * const threadLocalTy = mTarget->getThreadLocalStateType();
        Value * const providedState = b->CreatePointerCast(nextArg(), threadLocalTy->getPointerTo());
        BasicBlock * const allocThreadLocal = BasicBlock::Create(b->getContext(), "allocThreadLocalState", mCurrentMethod);
        BasicBlock * const initThreadLocal = BasicBlock::Create(b->getContext(), "initThreadLocalState", mCurrentMethod);
        b->CreateCondBr(b->CreateIsNull(providedState), allocThreadLocal, initThreadLocal);

        b->SetInsertPoint(allocThreadLocal);
        Value * const allocedState = b->CreatePageAlignedMalloc(mTarget->getThreadLocalStateType());
        assert (providedState->getType() == allocedState->getType());
        b->CreateBr(initThreadLocal);

        b->SetInsertPoint(initThreadLocal);
        PHINode * const threadLocal = b->CreatePHI(providedState->getType(), 2);
        threadLocal->addIncoming(providedState, mEntryPoint);
        threadLocal->addIncoming(allocedState, allocThreadLocal);
        mThreadLocalHandle = threadLocal;
        initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);
        mTarget->generateInitializeThreadLocalMethod(b);
        b->CreateRet(threadLocal);
        clearInternalStateAfterCodeGen();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callAllocateSharedInternalStreamSets
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateAllocateSharedInternalStreamSets(BuilderRef b) {
    if (LLVM_UNLIKELY(mTarget->allocatesInternalStreamSets())) {
        b->setCompiler(this);
        assert (mSharedHandle == nullptr && mThreadLocalHandle == nullptr);
        mCurrentMethod = mTarget->getAllocateSharedInternalStreamSetsFunction(b);
        mEntryPoint = BasicBlock::Create(b->getContext(), "entry", mCurrentMethod);
        b->SetInsertPoint(mEntryPoint);
        auto arg = mCurrentMethod->arg_begin();
        auto nextArg = [&]() {
            assert (arg != mCurrentMethod->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };
        if (LLVM_LIKELY(mTarget->isStateful())) {
            setHandle(nextArg());
        }
        Value * const expectedNumOfStrides = nextArg();
        initializeScalarMap(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
        initializeOwnedBufferHandles(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
        mTarget->generateAllocateSharedInternalStreamSetsMethod(b, expectedNumOfStrides);
        b->CreateRetVoid();
        clearInternalStateAfterCodeGen();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callAllocateThreadLocalInternalStreamSets
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateAllocateThreadLocalInternalStreamSets(BuilderRef b) {
    if (LLVM_UNLIKELY(mTarget->allocatesInternalStreamSets() && mTarget->hasThreadLocal())) {
        b->setCompiler(this);
        assert (mSharedHandle == nullptr && mThreadLocalHandle == nullptr);
        mCurrentMethod = mTarget->getAllocateThreadLocalInternalStreamSetsFunction(b);
        mEntryPoint = BasicBlock::Create(b->getContext(), "entry", mCurrentMethod);
        b->SetInsertPoint(mEntryPoint);
        auto arg = mCurrentMethod->arg_begin();
        auto nextArg = [&]() {
            assert (arg != mCurrentMethod->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };
        if (LLVM_LIKELY(mTarget->isStateful())) {
            setHandle(nextArg());
        }
        setThreadLocalHandle(nextArg());
        Value * const expectedNumOfStrides = nextArg();
        initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);
        initializeOwnedBufferHandles(b, InitializeOptions::IncludeThreadLocalScalars);
        mTarget->generateAllocateThreadLocalInternalStreamSetsMethod(b, expectedNumOfStrides);
        b->CreateRetVoid();
        clearInternalStateAfterCodeGen();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getLCMOfFixedRateInputs
 ** ------------------------------------------------------------------------------------------------------------- */
/* static */ Rational KernelCompiler::getLCMOfFixedRateInputs(const Kernel * const target) {
    Rational rateLCM(1);
    bool first = true;
    for (const Binding & input : target->getInputStreamSetBindings()) {
        const ProcessingRate & rate = input.getRate();
        if (LLVM_LIKELY(rate.isFixed())) {
            if (first) {
                rateLCM = rate.getRate();
                first = false;
            } else {
                rateLCM = lcm(rateLCM, rate.getRate());
            }
        }
    }
    return rateLCM;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief setDoSegmentProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::setDoSegmentProperties(BuilderRef b, const ArrayRef<Value *> args) {

    // WARNING: any change to this must be reflected in Kernel::addDoSegmentDeclaration,
    // Kernel::getDoSegmentFields, KernelCompiler::getDoSegmentProperties,
    // PipelineCompiler::buildKernelCallArgumentList and PipelineKernel::addOrDeclareMainFunction

    auto arg = args.begin();
    auto nextArg = [&]() {
        assert (arg != args.end());
        Value * const v = *arg; assert (v);
        std::advance(arg, 1);
        return v;
    };

    const auto enableAsserts = codegen::DebugOptionIsSet(codegen::EnableAsserts);

    clearInternalStateAfterCodeGen();

    if (LLVM_LIKELY(mTarget->isStateful())) {
        setHandle(nextArg());
        assert (mSharedHandle->getType()->getPointerElementType() == mTarget->getSharedStateType());
        if (LLVM_UNLIKELY(enableAsserts)) {
            b->CreateAssert(getHandle(), "%s: shared handle cannot be null", b->GetString(getName()));
        }
    }
    if (LLVM_UNLIKELY(mTarget->hasThreadLocal())) {
        setThreadLocalHandle(nextArg());
        assert (mThreadLocalHandle->getType()->getPointerElementType() == mTarget->getThreadLocalStateType());
        if (LLVM_UNLIKELY(enableAsserts)) {
            b->CreateAssert(getThreadLocalHandle(), "%s: thread local handle cannot be null", b->GetString(getName()));
        }
    }    
    const auto internallySynchronized = mTarget->hasAttribute(AttrId::InternallySynchronized);
    // TODO: the simplest way of ensuring we can allow external I/O to be passed though the main pipeline
    // even if there are multiple consumers of the input with differing processing rates is to special
    // case the outermost pipeline such that all I/O is always addressible. This creates a problem,
    // however, in that we will not be able to optimize a single kernel program by having the main
    // function call it directly instead of a pipeline. Given that this is not a realistic use-case,
    // we're ignoring that limitation for now.
    const auto isMainPipeline = (mTarget->getTypeId() == Kernel::TypeId::Pipeline) && !internallySynchronized;

    Rational fixedRateLCM{0};
    mFixedRateFactor = nullptr;

    if (LLVM_UNLIKELY(isMainPipeline)) {
        mIsFinal = b->getTrue();
        mRawNumOfStrides = nullptr;
        mNumOfStrides = nullptr;
    } else {
        if (LLVM_UNLIKELY(internallySynchronized)) {
            mExternalSegNo = nextArg();
        }
        mRawNumOfStrides = nextArg();
        if (LLVM_UNLIKELY(mTarget->hasAttribute(AttrId::MustExplicitlyTerminate))) {
            mIsFinal = nullptr;
            mNumOfStrides = mRawNumOfStrides;
        } else {
            mIsFinal = b->CreateIsNull(mRawNumOfStrides);
            mNumOfStrides = b->CreateSelect(mIsFinal, b->getSize(1), mRawNumOfStrides);
        }
        if (LLVM_LIKELY(mTarget->hasFixedRateInput())) {
            fixedRateLCM = getLCMOfFixedRateInputs(mTarget);
            mFixedRateFactor = nextArg();
        }
    }

    initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);

    // NOTE: the disadvantage of passing the stream pointers as a parameter is that it becomes more difficult
    // to access a stream set from a LLVM function call. We could create a stream-set aware function creation
    // and call system here but that is not an ideal way of handling this.

    #ifdef CHECK_IO_ADDRESS_RANGE
    auto checkStreamRange = [&](const StreamSetBuffer * const buffer, const Binding & binding, Value * const startItemCount) {

        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        out << "StreamSet " << getName() << ":" << binding.getName();

        PointerType * const int8PtrTy = b->getInt8PtrTy();

        ConstantInt * const ZERO = b->getSize(0);
        ConstantInt * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());

        Value * const fromIndex = b->CreateUDiv(startItemCount, BLOCK_WIDTH);
        Value * const baseAddress = buffer->getBaseAddress(b);
        Value * const startPtr = buffer->getStreamBlockPtr(b, baseAddress, ZERO, fromIndex);
        Value * const start = b->CreatePointerCast(startPtr, int8PtrTy);

        Value * const endPos = b->CreateAdd(startItemCount, buffer->getCapacity(b));
        Value * const toIndex = b->CreateCeilUDiv(endPos, BLOCK_WIDTH);
        Value * const endPtr = buffer->getStreamBlockPtr(b, baseAddress, ZERO, toIndex);
        Value * const end = b->CreatePointerCast(endPtr, int8PtrTy);

        Value * const length = b->CreatePtrDiff(end, start);

        b->CreateAssert(b->CreateICmpULE(start, end),
                        "%s: illegal kernel I/O address range [0x%" PRIx64 ", 0x%" PRIx64 ")",
                        b->GetString(out.str()), start, end);

        b->CheckAddress(start, length, out.str());


    };
    #endif

    const auto numOfInputs = getNumOfStreamInputs();

    IntegerType * const sizeTy = b->getSizeTy();
    for (unsigned i = 0; i < numOfInputs; i++) {

        /// ----------------------------------------------------
        /// virtual base address
        /// ----------------------------------------------------
        StreamSetBuffer * const buffer = mStreamSetInputBuffers[i].get();

        const Binding & input = mInputStreamSets[i];
        Value * const virtualBaseAddress = b->CreatePointerCast(nextArg(), buffer->getPointerType());
        Value * const localHandle = b->CreateAllocaAtEntryPoint(buffer->getHandleType(b));
        buffer->setHandle(localHandle);
        buffer->setBaseAddress(b, virtualBaseAddress);

        if (LLVM_UNLIKELY(internallySynchronized)) {
            mInputIsClosed[i] = nextArg();
            assert (mInputIsClosed[i]->getType() == b->getInt1Ty());
        } else {
            mInputIsClosed[i] = mIsFinal;
        }

        /// ----------------------------------------------------
        /// processed item count
        /// ----------------------------------------------------

        const ProcessingRate & rate = input.getRate();
        Value * processed = nullptr;
        if (isMainPipeline || isAddressable(input)) {
            mProcessedInputItemPtr[i] = nextArg();
            processed = b->CreateLoad(mProcessedInputItemPtr[i]);
        } else {
            if (LLVM_LIKELY(isCountable(input))) {
                processed = nextArg();
            } else { // isRelative
                const auto port = getStreamPort(rate.getReference());
                assert (port.Type == PortType::Input && port.Number < i);
                assert (mProcessedInputItemPtr[port.Number]);
                Value * const ref = b->CreateLoad(mProcessedInputItemPtr[port.Number]);
                processed = b->CreateMulRational(ref, rate.getRate());
            }
            // NOTE: we create a redundant alloca to store the input param so that
            // Mem2Reg can convert it into a PHINode if the item count is updated in
            // a loop; otherwise, it will be discarded in favor of the param itself.
            AllocaInst * const processedItems = b->CreateAllocaAtEntryPoint(sizeTy);
            b->CreateStore(processed, processedItems);
            mProcessedInputItemPtr[i] = processedItems;
        }
        /// ----------------------------------------------------
        /// accessible item count
        /// ----------------------------------------------------
        Value * accessible = nullptr;
        if (LLVM_UNLIKELY(isMainPipeline || requiresItemCount(input))) {
            accessible = nextArg();
        } else {
            accessible = b->CreateCeilUMulRational(mFixedRateFactor, rate.getRate() / fixedRateLCM);
        }
        assert (accessible);
        assert (accessible->getType() == sizeTy);
        mAccessibleInputItems[i] = accessible;

        Value * avail = b->CreateAdd(processed, accessible);
        mAvailableInputItems[i] = avail;
        if (input.hasLookahead()) {
            avail = b->CreateAdd(avail, b->getSize(input.getLookahead()));
        }
        buffer->setCapacity(b, avail);
        #ifdef CHECK_IO_ADDRESS_RANGE
        if (LLVM_UNLIKELY(enableAsserts)) {
            checkStreamRange(buffer, input, processed);
        }
        #endif
    }

    // set all of the output buffers
    const auto numOfOutputs = getNumOfStreamOutputs();

    const auto canTerminate = canSetTerminateSignal();

    for (unsigned i = 0; i < numOfOutputs; i++) {

        /// ----------------------------------------------------
        /// logical buffer base address
        /// ----------------------------------------------------
        StreamSetBuffer * const buffer = mStreamSetOutputBuffers[i].get();

        const Binding & output = mOutputStreamSets[i];
        const auto isShared = output.hasAttribute(AttrId::SharedManagedBuffer);
        const auto isLocal =  Kernel::isLocalBuffer(output, false);

        if (LLVM_UNLIKELY(isShared)) {
            Value * const handle = nextArg();
            assert (isa<DynamicBuffer>(buffer));
            buffer->setHandle(b->CreatePointerCast(handle, buffer->getHandlePointerType(b)));
        } else if (LLVM_UNLIKELY(isMainPipeline || isLocal)) {
            // If an output is a managed buffer, the address is stored within the state instead
            // of being passed in through the function call.
            mUpdatableOutputBaseVirtualAddressPtr[i] = nextArg();
            Value * handle = getScalarFieldPtr(b.get(), output.getName() + BUFFER_HANDLE_SUFFIX);
            buffer->setHandle(handle);
        } else {
            Value * const virtualBaseAddress = b->CreatePointerCast(nextArg(), buffer->getPointerType());
            Value * const localHandle = b->CreateAllocaAtEntryPoint(buffer->getHandleType(b));
            buffer->setHandle(localHandle);
            buffer->setBaseAddress(b, virtualBaseAddress);
            assert (isa<ExternalBuffer>(buffer));
        }

        /// ----------------------------------------------------
        /// produced item count
        /// ----------------------------------------------------
        const ProcessingRate & rate = output.getRate();
        Value * produced = nullptr;
        if (LLVM_LIKELY(canTerminate || isMainPipeline || isAddressable(output))) {
            mProducedOutputItemPtr[i] = nextArg();
            produced = b->CreateLoad(mProducedOutputItemPtr[i]);
        } else {
            if (LLVM_LIKELY(isCountable(output))) {
                produced = nextArg();
            } else { // isRelative
                // For now, if something is produced at a relative rate to another stream in a kernel that
                // may terminate, its final item count is inherited from its reference stream and cannot
                // be set independently. Should they be independent at early termination?
                const auto port = getStreamPort(rate.getReference());
                assert (port.Type == PortType::Input || (port.Type == PortType::Output && port.Number < i));
                const auto & items = (port.Type == PortType::Input) ? mProcessedInputItemPtr : mProducedOutputItemPtr;
                Value * const ref = b->CreateLoad(items[port.Number]);
                produced = b->CreateMulRational(ref, rate.getRate());
            }
            AllocaInst * const producedItems = b->CreateAllocaAtEntryPoint(sizeTy);
            b->CreateStore(produced, producedItems);
            mProducedOutputItemPtr[i] = producedItems;
        }
        assert (produced);
        assert (produced->getType() == sizeTy);
        mInitiallyProducedOutputItems[i] = produced;

        /// ----------------------------------------------------
        /// writable / consumed item count
        /// ----------------------------------------------------
        Value * writable = nullptr;
        if (isShared || isLocal) {
            Value * const consumed = nextArg();
            assert (consumed->getType() == sizeTy);
            mConsumedOutputItems[i] = consumed;
            writable = buffer->getLinearlyWritableItems(b, produced, consumed);
            assert (writable && writable->getType() == sizeTy);
        } else {
            if (isMainPipeline || requiresItemCount(output)) {
                writable = nextArg();
                assert (writable && writable->getType() == sizeTy);
            } else if (mFixedRateFactor) {
                writable = b->CreateCeilUMulRational(mFixedRateFactor, rate.getRate() / fixedRateLCM);
                assert (writable && writable->getType() == sizeTy);
            }
            Value * capacity = nullptr;
            if (writable) {
                capacity = b->CreateAdd(produced, writable);
                buffer->setCapacity(b, capacity);
                #ifdef CHECK_IO_ADDRESS_RANGE
                if (LLVM_UNLIKELY(enableAsserts)) {
                    checkStreamRange(buffer, output, produced);
                }
                #endif
            } else {
                capacity = ConstantExpr::getNeg(b->getSize(1));
                buffer->setCapacity(b, capacity);
            }
        }
        mWritableOutputItems[i] = writable;
    }
    assert (arg == args.end());

    // initialize the termination signal if this kernel can set it
    mTerminationSignalPtr = nullptr;
    if (canTerminate) {
        mTerminationSignalPtr = b->getScalarFieldPtr(TERMINATION_SIGNAL);
        if (LLVM_UNLIKELY(enableAsserts)) {
            Value * const unterminated =
                b->CreateICmpEQ(b->CreateLoad(mTerminationSignalPtr), b->getSize(KernelBuilder::TerminationCode::None));
            b->CreateAssert(unterminated, getName() + ".doSegment was called after termination?");
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getDoSegmentProperties
 *
 * Reverse of the setDoSegmentProperties operation; used by the PipelineKernel when constructing internal threads
 * to simplify passing of the state data.
 ** ------------------------------------------------------------------------------------------------------------- */
std::vector<Value *> KernelCompiler::getDoSegmentProperties(BuilderRef b) const {

    // WARNING: any change to this must be reflected in addDoSegmentDeclaration, getDoSegmentFields,
    // setDoSegmentProperties, and PipelineCompiler::writeKernelCall

    std::vector<Value *> props;
    if (LLVM_LIKELY(mTarget->isStateful())) {
        props.push_back(mSharedHandle); assert (mSharedHandle);
    }
    if (LLVM_UNLIKELY(mTarget->hasThreadLocal())) {
        props.push_back(mThreadLocalHandle); assert (mThreadLocalHandle);
    }
    const auto internallySynchronized = mTarget->hasAttribute(AttrId::InternallySynchronized);
    const auto isMainPipeline = (mTarget->getTypeId() == Kernel::TypeId::Pipeline) && !internallySynchronized;

    if (LLVM_UNLIKELY(internallySynchronized)) {
        props.push_back(mExternalSegNo);
    }

    if (LLVM_LIKELY(!isMainPipeline)) {
        props.push_back(mNumOfStrides); assert (mNumOfStrides);
        if (LLVM_LIKELY(mTarget->hasFixedRateInput())) {
            props.push_back(mFixedRateFactor);
        }
    }

    PointerType * const voidPtrTy = b->getVoidPtrTy();

    const auto numOfInputs = getNumOfStreamInputs();
    for (unsigned i = 0; i < numOfInputs; i++) {
        /// ----------------------------------------------------
        /// logical buffer base address
        /// ----------------------------------------------------
        const auto & buffer = mStreamSetInputBuffers[i];
        props.push_back(b->CreatePointerCast(buffer->getBaseAddress(b), voidPtrTy));
        /// ----------------------------------------------------
        /// is closed
        /// ----------------------------------------------------
        if (LLVM_UNLIKELY(internallySynchronized)) {
            props.push_back(mInputIsClosed[i]);
        }
        /// ----------------------------------------------------
        /// processed item count
        /// ----------------------------------------------------
        const Binding & input = mInputStreamSets[i];
        if (isMainPipeline || isAddressable(input)) {
            props.push_back(mProcessedInputItemPtr[i]);
        } else if (LLVM_LIKELY(isCountable(input))) {
            props.push_back(b->CreateLoad(mProcessedInputItemPtr[i]));
        }
        /// ----------------------------------------------------
        /// accessible item count
        /// ----------------------------------------------------
        if (isMainPipeline || requiresItemCount(input)) {
            props.push_back(mAccessibleInputItems[i]);
        }
    }

    // set all of the output buffers
    const auto numOfOutputs = getNumOfStreamOutputs();
    const auto canTerminate = canSetTerminateSignal();

    for (unsigned i = 0; i < numOfOutputs; i++) {
        /// ----------------------------------------------------
        /// logical buffer base address
        /// ----------------------------------------------------
        const auto & buffer = mStreamSetOutputBuffers[i];
        const Binding & output = mOutputStreamSets[i];

        const auto isShared = output.hasAttribute(AttrId::SharedManagedBuffer);
        const auto isLocal = Kernel::isLocalBuffer(output, false);

        Value * handle = nullptr;
        if (LLVM_UNLIKELY(isShared)) {            
            handle = b->CreatePointerCast(buffer->getHandle(), voidPtrTy);
        } else if (LLVM_UNLIKELY(isMainPipeline || isLocal)) {
            // If an output is a managed buffer, the address is stored within the state instead
            // of being passed in through the function call.
            PointerType * const voidPtrPtrTy = voidPtrTy->getPointerTo();
            handle = b->CreatePointerCast(mUpdatableOutputBaseVirtualAddressPtr[i], voidPtrPtrTy);
        } else {
            handle = b->CreatePointerCast(buffer->getBaseAddress(b), voidPtrTy);
        }
        props.push_back(handle);

        /// ----------------------------------------------------
        /// produced item count
        /// ----------------------------------------------------
        if (LLVM_LIKELY(canTerminate || isMainPipeline ||isAddressable(output))) {
            props.push_back(mProducedOutputItemPtr[i]);
        } else if (LLVM_LIKELY(isCountable(output))) {
            props.push_back(b->CreateLoad(mProducedOutputItemPtr[i]));
        }
        /// ----------------------------------------------------
        /// writable / consumed item count
        /// ----------------------------------------------------
        if (LLVM_UNLIKELY(isShared || isLocal)) {
            props.push_back(mConsumedOutputItems[i]);
        } else if (isMainPipeline || requiresItemCount(output)) {
            props.push_back(mWritableOutputItems[i]);
        }

    }
    return props;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callGenerateDoSegmentMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateDoSegmentMethod(BuilderRef b) {

    assert (mInputStreamSets.size() == mStreamSetInputBuffers.size());
    assert (mOutputStreamSets.size() == mStreamSetOutputBuffers.size());

    b->setCompiler(this);
    mCurrentMethod = mTarget->getDoSegmentFunction(b);
    mEntryPoint = BasicBlock::Create(b->getContext(), "entry", mCurrentMethod);
    b->SetInsertPoint(mEntryPoint);

    BEGIN_SCOPED_REGION
    Vec<Value *, 64> args;
    args.reserve(mCurrentMethod->arg_size());
    for (auto ArgI = mCurrentMethod->arg_begin(); ArgI != mCurrentMethod->arg_end(); ++ArgI) {
        args.push_back(&(*ArgI));
    }
    setDoSegmentProperties(b, args);
    END_SCOPED_REGION

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
        b->CreateMProtect(mSharedHandle, CBuilder::Protect::WRITE);
    }

    mTarget->generateKernelMethod(b);

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
        b->CreateMProtect(mSharedHandle, CBuilder::Protect::READ);
    }

    const auto numOfOutputs = getNumOfStreamOutputs();

    for (unsigned i = 0; i < numOfOutputs; i++) {
        // Write the virtual base address out to inform the pipeline of any changes
        const auto & buffer = mStreamSetOutputBuffers[i];
        if (mUpdatableOutputBaseVirtualAddressPtr[i]) {
            Value * const baseAddress = buffer->getBaseAddress(b);
            if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream out(tmp);
                const Binding & output = mOutputStreamSets[i];
                out << getName() << ":%s is returning a virtual base address "
                                    "computed from a null base address.";
                b->CreateAssert(baseAddress, out.str(), b->GetString(output.getName()));
            }
            Constant * const LOG_2_BLOCK_WIDTH = b->getSize(floor_log2(b->getBitBlockWidth()));
            Constant * const ZERO = b->getSize(0);
            Value * produced = mInitiallyProducedOutputItems[i];
            assert (isFromCurrentFunction(b, produced, false));
            // TODO: will LLVM optimizations replace the following with the already loaded value?
            // If not, re-loading it here may reduce register pressure / compilation time.
            if (mProducedOutputItemPtr[i]) {
                assert (isFromCurrentFunction(b, mProducedOutputItemPtr[i], false));
                produced = b->CreateLoad(mProducedOutputItemPtr[i]);
            }
            assert (isFromCurrentFunction(b, produced, true));
            Value * const blockIndex = b->CreateLShr(produced, LOG_2_BLOCK_WIDTH);
            Value * vba = buffer->getStreamLogicalBasePtr(b.get(), baseAddress, ZERO, blockIndex);

            assert (isFromCurrentFunction(b, vba, true));
            vba = b->CreatePointerCast(vba, b->getVoidPtrTy());

            assert (isFromCurrentFunction(b, mUpdatableOutputBaseVirtualAddressPtr[i], true));

            b->CreateStore(vba, mUpdatableOutputBaseVirtualAddressPtr[i]);
        }
    }

    // return the termination signal (if one exists)
    if (mTerminationSignalPtr) {
        assert (isFromCurrentFunction(b, mTerminationSignalPtr, true));
        b->CreateRet(b->CreateLoad(mTerminationSignalPtr));
        mTerminationSignalPtr = nullptr;
    } else {
        b->CreateRetVoid();
    }
    clearInternalStateAfterCodeGen();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief copyInternalState
 ** ------------------------------------------------------------------------------------------------------------- */
std::vector<Value *> KernelCompiler::storeDoSegmentState() const {

    const auto numOfInputs = getNumOfStreamInputs();
    const auto numOfOutputs = getNumOfStreamOutputs();

    assert (!mTarget->hasAttribute(AttrId::InternallySynchronized));

    std::vector<Value *> S;
    S.resize(8 + numOfInputs * 5 + numOfOutputs * 6);

    auto o = S.begin();

    auto append = [&](Value * v) {
        *o++ = v;
    };

    append(mSharedHandle);
    append(mThreadLocalHandle);
    append(mTerminationSignalPtr);
    append(mIsFinal);
    append(mNumOfStrides);
    append(mFixedRateFactor);
    append(mExternalSegNo);
    append(mNumOfStrides);

    auto copy = [&](const Vec<llvm::Value *> & V, const size_t n) {
        o = std::copy_n(V.begin(), n, o);
    };

    copy(mInputIsClosed, numOfInputs);
    copy(mProcessedInputItemPtr, numOfInputs);
    copy(mAccessibleInputItems, numOfInputs);
    copy(mAvailableInputItems, numOfInputs);
    for (unsigned i = 0; i < numOfInputs; ++i) {
        append(getInputStreamSetBuffer(i)->getHandle());
    }

    copy(mProducedOutputItemPtr, numOfOutputs);
    copy(mUpdatableOutputBaseVirtualAddressPtr, numOfOutputs);
    copy(mInitiallyProducedOutputItems, numOfOutputs);
    copy(mWritableOutputItems, numOfOutputs);
    copy(mConsumedOutputItems, numOfOutputs);
    for (unsigned i = 0; i < numOfOutputs; ++i) {
        append(getOutputStreamSetBuffer(i)->getHandle());
    }

    assert (o == S.end());

    return S;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief restoreInternalState
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::restoreDoSegmentState(const std::vector<Value *> & S) {

    assert (!mTarget->hasAttribute(AttrId::InternallySynchronized));

    clearInternalStateAfterCodeGen();

    auto o = S.begin();

    auto revertOne = [&](Value *& v) {
        assert (o != S.end());
        v = *o++;
    };

    revertOne(mSharedHandle);
    revertOne(mThreadLocalHandle);
    revertOne(mTerminationSignalPtr);
    revertOne(mIsFinal);
    revertOne(mNumOfStrides);
    revertOne(mFixedRateFactor);
    revertOne(mExternalSegNo);
    revertOne(mNumOfStrides);

    auto revert = [&](Vec<llvm::Value *> & V, const size_t n) {
        assert (static_cast<size_t>(std::distance(o, S.end())) >= n);
        assert (V.size() == n);
        std::copy_n(o, n, V.begin());
        o += n;
    };

    const auto numOfInputs = getNumOfStreamInputs();
    revert(mInputIsClosed, numOfInputs);
    revert(mProcessedInputItemPtr, numOfInputs);
    revert(mAccessibleInputItems, numOfInputs);
    revert(mAvailableInputItems, numOfInputs);
    for (unsigned i = 0; i < numOfInputs; ++i) {
        Value * handle;
        revertOne(handle);
        getInputStreamSetBuffer(i)->setHandle(handle);
    }

    const auto numOfOutputs = getNumOfStreamOutputs();
    revert(mProducedOutputItemPtr, numOfOutputs);
    revert(mUpdatableOutputBaseVirtualAddressPtr, numOfOutputs);
    revert(mInitiallyProducedOutputItems, numOfOutputs);
    revert(mWritableOutputItems, numOfOutputs);
    revert(mConsumedOutputItems, numOfOutputs);

    for (unsigned i = 0; i < numOfOutputs; ++i) {
        Value * handle;
        revertOne(handle);
        getOutputStreamSetBuffer(i)->setHandle(handle);
    }

    assert (o == S.end());


}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callGenerateFinalizeThreadLocalMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateFinalizeThreadLocalMethod(BuilderRef b) {
    if (mTarget->hasThreadLocal()) {
        b->setCompiler(this);
        mCurrentMethod = mTarget->getFinalizeThreadLocalFunction(b);
        mEntryPoint = BasicBlock::Create(b->getContext(), "entry", mCurrentMethod);
        b->SetInsertPoint(mEntryPoint);
        auto arg = mCurrentMethod->arg_begin();
        auto nextArg = [&]() {
            assert (arg != mCurrentMethod->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };
        if (LLVM_LIKELY(mTarget->isStateful())) {
            setHandle(nextArg());
        }
        mCommonThreadLocalHandle = nextArg();
        mThreadLocalHandle = nextArg();
        initializeScalarMap(b, InitializeOptions::IncludeAndAutomaticallyAccumulateThreadLocalScalars);
        mTarget->generateFinalizeThreadLocalMethod(b);
        b->CreateRetVoid();
        clearInternalStateAfterCodeGen();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief callGenerateFinalizeMethod
 ** ------------------------------------------------------------------------------------------------------------- */
inline void KernelCompiler::callGenerateFinalizeMethod(BuilderRef b) {
    b->setCompiler(this);
    mCurrentMethod = mTarget->getFinalizeFunction(b);
    mEntryPoint = BasicBlock::Create(b->getContext(), "entry", mCurrentMethod);
    b->SetInsertPoint(mEntryPoint);
    auto args = mCurrentMethod->arg_begin();
    if (LLVM_LIKELY(mTarget->isStateful())) {
        setHandle(&*(args++));
    }
    if (LLVM_LIKELY(mTarget->hasThreadLocal())) {
        setThreadLocalHandle(&*(args++));
    }
    assert (args == mCurrentMethod->arg_end());
    initializeScalarMap(b, InitializeOptions::IncludeThreadLocalScalars);
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableMProtect))) {
        b->CreateMProtect(mSharedHandle,CBuilder::Protect::WRITE);
    }
    initializeOwnedBufferHandles(b, InitializeOptions::DoNotIncludeThreadLocalScalars);
    mTarget->generateFinalizeMethod(b); // may be overridden by the Kernel subtype
    const auto outputs = getFinalOutputScalars(b);
    if (outputs.empty()) {
        b->CreateRetVoid();
    } else {
        const auto n = outputs.size();
        if (n == 1) {
            b->CreateRet(outputs[0]);
        } else {
            b->CreateAggregateRet(outputs.data(), n);
        }
    }
    clearInternalStateAfterCodeGen();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getFinalOutputScalars
 ** ------------------------------------------------------------------------------------------------------------- */
std::vector<Value *> KernelCompiler::getFinalOutputScalars(BuilderRef b) {
    const auto n = mOutputScalars.size();
    std::vector<Value *> outputs(n);
    for (unsigned i = 0; i < n; ++i) {
        outputs[i] = b->CreateLoad(getScalarFieldPtr(b.get(), mOutputScalars[i].getName()));
    }
    return outputs;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeScalarMap
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::initializeScalarMap(BuilderRef b, const InitializeOptions options) {

    FixedArray<Value *, 3> indices;
    indices[0] = b->getInt32(0);

    const auto sharedTy = mTarget->getSharedStateType();

    const auto threadLocalTy = mTarget->getThreadLocalStateType();

    #ifndef NDEBUG
    auto verifyStateType = [](Value * const handle, StructType * const stateType) {
        if (handle == nullptr && stateType == nullptr) {
            return true;
        }
        if (handle == nullptr || stateType == nullptr) {
            return false;
        }
        if (handle->getType() != stateType->getPointerTo()) {
            return false;
        }
        assert (!stateType->isOpaque());
        const auto n = stateType->getStructNumElements();
        assert ((n % 2) == 0);
        for (unsigned i = 0; i < n; i += 2) {
            assert (isa<StructType>(stateType->getStructElementType(i)));
        }
        return true;
    };
    assert ("incorrect shared handle/type!" && verifyStateType(mSharedHandle, sharedTy));
    if (options == InitializeOptions::IncludeThreadLocalScalars) {
        assert ("incorrect thread local handle/type!" && verifyStateType(mThreadLocalHandle, threadLocalTy));
    }
    #undef CHECK_TYPES
    #endif

    mScalarFieldMap.clear();

    auto addToScalarFieldMap = [&](StringRef bindingName, Value * const scalar, Type * const expectedType) {
        const auto i = mScalarFieldMap.insert(std::make_pair(bindingName, scalar));
        if (LLVM_UNLIKELY(!i.second)) {
            SmallVector<char, 256> tmp;
            raw_svector_ostream out(tmp);
            out << "Kernel " << getName() << " contains two scalar or alias fields named " << bindingName;
            report_fatal_error(out.str());
        }
        if (expectedType) {
            Type * const actualType = scalar->getType()->getPointerElementType();
            if (LLVM_UNLIKELY(actualType != expectedType)) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream out(tmp);
                out << "Scalar " << getName() << '.' << bindingName << " was expected to be a ";
                expectedType->print(out);
                out << " but was stored as a ";
                actualType->print(out);
                report_fatal_error(out.str());
            }
        }
    };

    flat_set<unsigned> sharedGroups;
    flat_set<unsigned> threadLocalGroups;

    bool hasThreadLocalAccum = false;

    for (const auto & scalar : mInternalScalars) {
        assert (scalar.getValueType());
        switch (scalar.getScalarType()) {
            case ScalarType::Internal:
                sharedGroups.insert(scalar.getGroup());
                break;
            case ScalarType::ThreadLocal:
                if (options == InitializeOptions::DoNotIncludeThreadLocalScalars) continue;
                threadLocalGroups.insert(scalar.getGroup());
                if (options != InitializeOptions::IncludeAndAutomaticallyAccumulateThreadLocalScalars) continue;
                if (scalar.getAccumulationRule() != Kernel::ThreadLocalScalarAccumulationRule::DoNothing) {
                    assert (mCommonThreadLocalHandle && "no main thread local given?");
                    hasThreadLocalAccum = true;
                }
                break;
            default: break;
        }
    }

    std::vector<unsigned> sharedIndex(sharedGroups.size() + 2, 0);
    std::vector<unsigned> threadLocalIndex(threadLocalGroups.size(), 0);


    BasicBlock * combineToMainThreadLocal = nullptr;

    if (LLVM_UNLIKELY(hasThreadLocalAccum)) {
        combineToMainThreadLocal = b->CreateBasicBlock("combineToMainThreadLocal");
    }

    auto enumerate = [&](const Bindings & bindings, const unsigned groupId) {
        indices[1] = b->getInt32(groupId * 2);
        auto & k = sharedIndex[groupId];
        for (const auto & binding : bindings) {
            assert ((groupId * 2) < sharedTy->getStructNumElements());
            assert (k < sharedTy->getStructElementType(groupId * 2)->getStructNumElements());
            assert (sharedTy->getStructElementType(groupId * 2)->getStructElementType(k) == binding.getType());
            indices[2] = b->getInt32(k++);
            Value * const scalar = b->CreateGEP(sharedTy, mSharedHandle, indices);
            addToScalarFieldMap(binding.getName(), scalar, binding.getType());
        }
    };

    enumerate(mInputScalars, 0);

    BasicBlock * combineExit = combineToMainThreadLocal;

    for (const auto & binding : mInternalScalars) {
        Value * scalar = nullptr;

        auto getGroupIndex = [&](const flat_set<unsigned> & groups) {
            const auto f = groups.find(binding.getGroup());
            assert (f != groups.end());
            return std::distance(groups.begin(), f);
        };

        switch (binding.getScalarType()) {
            case ScalarType::Internal:
                assert (mSharedHandle);
                BEGIN_SCOPED_REGION
                const auto j = getGroupIndex(sharedGroups) + 1;
                indices[1] = b->getInt32(j * 2);
                auto & k = sharedIndex[j];
                assert ((j * 2) < sharedTy->getStructNumElements());
                assert (k < sharedTy->getStructElementType(j * 2)->getStructNumElements());
                assert (sharedTy->getStructElementType(j * 2)->getStructElementType(k) == binding.getValueType());
                indices[2] = b->getInt32(k++);
                scalar = b->CreateGEP(sharedTy, mSharedHandle, indices);
                END_SCOPED_REGION
                break;
            case ScalarType::ThreadLocal:
                if (options == InitializeOptions::DoNotIncludeThreadLocalScalars) continue;
                assert (mThreadLocalHandle);
                BEGIN_SCOPED_REGION
                const auto j = getGroupIndex(threadLocalGroups);
                indices[1] = b->getInt32(j * 2);
                auto & k = threadLocalIndex[j];
                assert ((j * 2) < threadLocalTy->getStructNumElements());
                assert (k < threadLocalTy->getStructElementType(j * 2)->getStructNumElements());
                assert (threadLocalTy->getStructElementType(j * 2)->getStructElementType(k) == binding.getValueType());
                indices[2] = b->getInt32(k++);
                scalar = b->CreateGEP(threadLocalTy, mThreadLocalHandle, indices);
                if (LLVM_UNLIKELY(options == InitializeOptions::IncludeAndAutomaticallyAccumulateThreadLocalScalars)) {
                    Value * const mainScalar = b->CreateGEP(threadLocalTy, mCommonThreadLocalHandle, indices);

                    addToScalarFieldMap(INTERNAL_COMMON_THREAD_LOCAL_PREFIX + binding.getName(), scalar, binding.getValueType());

                    if (binding.getAccumulationRule() != Kernel::ThreadLocalScalarAccumulationRule::DoNothing) {

                        const auto ip = b->saveIP();
                        b->SetInsertPoint(combineExit);

                        Type * const ty = mainScalar->getType()->getPointerElementType();
                        if (isa<ArrayType>(ty)) {
                            ArrayType * const arrayTy = cast<ArrayType>(ty);
                            const auto m = arrayTy->getNumElements();
                            if (LLVM_UNLIKELY(m == 0)) {
                                report_fatal_error(getName() + ": cannot automatically accumulate a 0-element scalar");
                            }

                            ConstantInt * const i32_ZERO = b->getInt32(0);
                            ConstantInt * const i32_ONE = b->getInt32(1);

                            FixedArray<Value *, 2> indices;
                            indices[0] = i32_ZERO;

                            BasicBlock * const entry = b->GetInsertBlock();
                            BasicBlock * const loop = b->CreateBasicBlock();
                            BasicBlock * const exit = b->CreateBasicBlock();
                            b->CreateBr(loop);

                            b->SetInsertPoint(loop);
                            PHINode * const idxPhi = b->CreatePHI(b->getInt32Ty(), 2);
                            idxPhi->addIncoming(i32_ZERO, entry);
                            indices[1] = idxPhi;
                            Value * const scalarVal = b->CreateLoad(b->CreateGEP(scalar, indices));
                            Value * const mainScalarPtr = b->CreateGEP(mainScalar, indices);
                            Value * mainScalarVal = b->CreateLoad(mainScalarPtr);
                            switch (binding.getAccumulationRule()) {
                                case Kernel::ThreadLocalScalarAccumulationRule::Sum:
                                    mainScalarVal = b->CreateAdd(scalarVal, mainScalarVal);
                                    break;
                                default: llvm_unreachable("unexpected thread-local scalar accumulation rule");
                            }
                            b->CreateStore(mainScalarVal, mainScalarPtr);
                            Value * const nextIdx = b->CreateAdd(idxPhi, i32_ONE);
                            idxPhi->addIncoming(nextIdx, loop);
                            b->CreateCondBr(b->CreateICmpNE(nextIdx, b->getInt32(m)), loop, exit);

                            b->SetInsertPoint(exit);
                            combineExit = exit;
                        } else {
                            Value * const scalarVal = b->CreateLoad(scalar);
                            Value * mainScalarVal = b->CreateLoad(mainScalar);
                            switch (binding.getAccumulationRule()) {
                                case Kernel::ThreadLocalScalarAccumulationRule::Sum:
                                    mainScalarVal = b->CreateAdd(scalarVal, mainScalarVal);
                                    break;
                                default: llvm_unreachable("unexpected thread-local scalar accumulation rule");
                            }
                            b->CreateStore(mainScalarVal, mainScalar);
                        }
                        b->restoreIP(ip);
                    }



                }
                END_SCOPED_REGION
                break;
            case ScalarType::NonPersistent:
                scalar = b->CreateAllocaAtEntryPoint(binding.getValueType());
                b->CreateStore(Constant::getNullValue(binding.getValueType()), scalar);
                break;
            default: llvm_unreachable("I/O scalars cannot be internal");
        }
        addToScalarFieldMap(binding.getName(), scalar, binding.getValueType());
    }

    enumerate(mOutputScalars, sharedGroups.size() + 1);

    // finally add any aliases
    for (const auto & alias : mScalarAliasMap) {
        const auto f = mScalarFieldMap.find(alias.second);
        if (f != mScalarFieldMap.end()) {
            addToScalarFieldMap(alias.first, f->second, nullptr);
        }
    }

    if (LLVM_UNLIKELY(hasThreadLocalAccum)) {
        BasicBlock * const exit = b->CreateBasicBlock("afterThreadLocalAccumulation");
        Value * const cond = b->CreateICmpEQ(mThreadLocalHandle, mCommonThreadLocalHandle);
        b->CreateCondBr(cond, exit, combineToMainThreadLocal);
        b->SetInsertPoint(combineExit);
        b->CreateBr(exit);
        b->SetInsertPoint(exit);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getCommonThreadLocalScalarFieldPtr
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelCompiler::getCommonThreadLocalScalarFieldPtr(KernelBuilder * b, const llvm::StringRef name) const {
    return getScalarFieldPtr(b, INTERNAL_COMMON_THREAD_LOCAL_PREFIX + name.str());
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addAlias
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::addAlias(llvm::StringRef alias, llvm::StringRef scalarName) {
    mScalarAliasMap.emplace_back(alias, scalarName);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeBindingMap
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::initializeIOBindingMap() {

    auto enumerate = [&](const Bindings & bindings, const BindingType type) {
        const auto n = bindings.size();
        for (unsigned i = 0; i < n; ++i) {
            const auto & binding = bindings[i];
            mBindingMap.insert(std::make_pair(binding.getName(), BindingMapEntry{type, i}));
        }
    };

    enumerate(mInputScalars, BindingType::ScalarInput);
    enumerate(mOutputScalars, BindingType::ScalarOutput);
    enumerate(mInputStreamSets, BindingType::StreamInput);
    enumerate(mOutputStreamSets, BindingType::StreamOutput);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeOwnedBufferHandles
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::initializeOwnedBufferHandles(BuilderRef b, const InitializeOptions /* options */) {
    const auto numOfOutputs = getNumOfStreamOutputs();
    for (unsigned i = 0; i < numOfOutputs; i++) {
        const Binding & output = mOutputStreamSets[i];
        if (LLVM_UNLIKELY(Kernel::isLocalBuffer(output, false))) {
            Value * const handle = getScalarFieldPtr(b.get(), output.getName() + BUFFER_HANDLE_SUFFIX);
            const auto & buffer = mStreamSetOutputBuffers[i]; assert (buffer.get());
            buffer->setHandle(handle);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getBinding
 ** ------------------------------------------------------------------------------------------------------------- */
const BindingMapEntry & KernelCompiler::getBinding(const BindingType type, const llvm::StringRef name) const {

    const auto f = mBindingMap.find(name);
    if (f != mBindingMap.end()) {
        const BindingMapEntry & entry = f->second;
        assert (entry.Type == type);
        return entry;
    }

    SmallVector<char, 256> tmp;
    raw_svector_ostream out(tmp);
    out << "Kernel " << getName() << " does not contain an ";
    switch (type) {
        case BindingType::ScalarInput:
        case BindingType::StreamInput:
            out << "input"; break;
        case BindingType::ScalarOutput:
        case BindingType::StreamOutput:
            out << "output"; break;
    }
    out << ' ';
    switch (type) {
        case BindingType::ScalarInput:
        case BindingType::ScalarOutput:
            out << "scalar"; break;
        case BindingType::StreamInput:
        case BindingType::StreamOutput:
            out << "streamset"; break;
    }
    out << " named \"" << name << "\"\n"
           "Currently contains:";


    auto listAvailableBindings = [&](const Bindings & bindings) {
        if (LLVM_UNLIKELY(bindings.empty())) {
            out << "<no bindings>";
        } else {
            char joiner = ' ';
            for (const auto & binding : bindings) {
                out << joiner << binding.getName();
                joiner = ',';
            }
        }
        out << '\n';
    };

    switch (type) {
        case BindingType::ScalarInput:
            listAvailableBindings(mInputScalars); break;
        case BindingType::ScalarOutput:
            listAvailableBindings(mOutputScalars); break;
        case BindingType::StreamInput:
            listAvailableBindings(mInputStreamSets); break;
        case BindingType::StreamOutput:
            listAvailableBindings(mOutputStreamSets); break;
    }

    report_fatal_error(out.str());
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getStreamPort
 ** ------------------------------------------------------------------------------------------------------------- */
StreamSetPort KernelCompiler::getStreamPort(const StringRef name) const {

    // NOTE: temporary refactoring step to limit changes outside of the kernel class

    static_assert(static_cast<unsigned>(BindingType::StreamInput) == static_cast<unsigned>(PortType::Input), "");
    static_assert(static_cast<unsigned>(BindingType::StreamOutput) == static_cast<unsigned>(PortType::Output), "");

    const auto f = mBindingMap.find(name);
    if (LLVM_LIKELY(f != mBindingMap.end())) {

        const BindingMapEntry & entry = f->second;
        switch (entry.Type) {
            case BindingType::StreamInput:
            case BindingType::StreamOutput:
                return StreamSetPort(static_cast<PortType>(entry.Type), entry.Index);
            default: break;
        }
    }

    SmallVector<char, 256> tmp;
    raw_svector_ostream out(tmp);
    out << "Kernel " << getName() << " does not contain a streamset named: \"" << name << "\"";
    report_fatal_error(out.str());
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getScalarFieldPtr
 ** ------------------------------------------------------------------------------------------------------------- */
Value * KernelCompiler::getScalarFieldPtr(KernelBuilder * const b, const StringRef name) const {
    assert (this);
    if (LLVM_UNLIKELY(mScalarFieldMap.empty())) {
        SmallVector<char, 256> tmp;
        raw_svector_ostream out(tmp);
        out << "Scalar map for " << getName() << " was not initialized prior to calling getScalarFieldPtr";
        report_fatal_error(out.str());
    } else {
        const auto f = mScalarFieldMap.find(name);
        if (LLVM_UNLIKELY(f == mScalarFieldMap.end())) {
            #ifdef NDEBUG
            SmallVector<char, 1024> tmp;
            raw_svector_ostream out(tmp);
            #else
            auto & out = errs();
            #endif
            out << "Scalar map for " << getName() << " does not contain " << name << "\n\n"
                "Currently contains:";
            char spacer = ' ';
            for (const auto & entry : mScalarFieldMap) {
                out << spacer << entry.getKey();
                spacer = ',';
            }
            #ifdef NDEBUG
            report_fatal_error(out.str());
            #else
            out << "\n";
            assert (false);
            #endif
        }
        Value * const result = f->second;
        assert (isFromCurrentFunction(b, result, false));
        return result;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getScalarValuePtr
 ** ------------------------------------------------------------------------------------------------------------- */
bool KernelCompiler::hasScalarField(const llvm::StringRef name) const {
    return mScalarFieldMap.count(name) == 0;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getLowerBound
 ** ------------------------------------------------------------------------------------------------------------- */
Rational KernelCompiler::getLowerBound(const Binding & binding) const {
    const ProcessingRate & rate = binding.getRate();
    if (rate.hasReference()) {
        return rate.getLowerBound() * getLowerBound(getStreamBinding(rate.getReference()));
    } else {
        return rate.getLowerBound();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getUpperBound
 ** ------------------------------------------------------------------------------------------------------------- */
Rational KernelCompiler::getUpperBound(const Binding & binding) const {
    const ProcessingRate & rate = binding.getRate();
    if (rate.hasReference()) {
        return rate.getUpperBound() * getUpperBound(getStreamBinding(rate.getReference()));
    } else {
        return rate.getUpperBound();
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief requiresOverflow
 ** ------------------------------------------------------------------------------------------------------------- */
bool KernelCompiler::requiresOverflow(const Binding & binding) const {
    const ProcessingRate & rate = binding.getRate();
    if (rate.isFixed() || binding.hasAttribute(AttrId::BlockSize)) {
        return false;
    } else if (rate.isRelative()) {
        return requiresOverflow(getStreamBinding(rate.getReference()));
    } else {
        return true;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief clearInternalStateAfterCodeGen
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::clearInternalStateAfterCodeGen() {
    // TODO: this function is more of a sanity check to ensure we don't have a pointer
    // to an out-of-scope LLVM-value. It should be possible to remove it.
    mScalarFieldMap.clear();
    mSharedHandle = nullptr;
    mThreadLocalHandle = nullptr;
    mCommonThreadLocalHandle = nullptr;
    mExternalSegNo = nullptr;
    mCurrentMethod = nullptr;
    mEntryPoint = nullptr;
    mIsFinal = nullptr;
    mNumOfStrides = nullptr;
    mTerminationSignalPtr = nullptr;
    const auto numOfInputs = getNumOfStreamInputs();
    reset(mInputIsClosed, numOfInputs);
    reset(mProcessedInputItemPtr, numOfInputs);
    reset(mAccessibleInputItems, numOfInputs);
    reset(mAvailableInputItems, numOfInputs);
    const auto numOfOutputs = getNumOfStreamOutputs();
    reset(mProducedOutputItemPtr, numOfOutputs);
    reset(mInitiallyProducedOutputItems, numOfOutputs);
    reset(mWritableOutputItems, numOfOutputs);
    reset(mConsumedOutputItems, numOfOutputs);
    reset(mUpdatableOutputBaseVirtualAddressPtr, numOfOutputs);
    for (const auto & buffer : mStreamSetInputBuffers) {
        buffer->setHandle(nullptr);
    }
    for (const auto & buffer : mStreamSetOutputBuffers) {
        buffer->setHandle(nullptr);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runInternalOptimizationPasses
 ** ------------------------------------------------------------------------------------------------------------- */
void KernelCompiler::runInternalOptimizationPasses(Module * const m) {

    #ifndef NDEBUG
    SmallVector<char, 256> tmp;
    raw_svector_ostream msg(tmp);
    bool BrokenDebugInfo = false;
    if (LLVM_UNLIKELY(verifyModule(*m, &msg, &BrokenDebugInfo))) {
        m->print(errs(), nullptr);
        report_fatal_error(msg.str());
    }
    #endif

    // Attempt to promote all of the allocas in the entry block into PHI nodes
    // and delete any unnecessary Alloca and GEP instructions.

    SmallVector<AllocaInst *, 32> allocas;

    for (Function & f : *m) {
        if (f.empty()) continue;

        BasicBlock & bb = f.getEntryBlock();

        Instruction * inst = bb.getFirstNonPHIOrDbgOrLifetime();
        while (inst) {
            for (unsigned i = 0; i < inst->getNumOperands(); ++i) {
                Value * const op = inst->getOperand(i);
                if (op == nullptr) {
                    report_fatal_error("null operand");
                }
            }


            Instruction * const nextNode = inst->getNextNode();
            if (isa<AllocaInst>(inst) || isa<GetElementPtrInst>(inst)) {
                if (LLVM_UNLIKELY(inst->getNumUses() == 0)) {
                    inst->eraseFromParent();
                    inst = nextNode;
                    continue;
                }
            }
            if (isa<AllocaInst>(inst)) {
                if (isAllocaPromotable(cast<AllocaInst>(inst))) {
                    allocas.push_back(cast<AllocaInst>(inst));
                }
            }
            inst = nextNode;
        }

        if (allocas.empty()) continue;

        DominatorTree dt(f);
        PromoteMemToReg(allocas, dt);
        allocas.clear();
    }



}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
KernelCompiler::KernelCompiler(not_null<Kernel *> kernel) noexcept
: mTarget(kernel)
, mInputStreamSets(kernel->mInputStreamSets)
, mOutputStreamSets(kernel->mOutputStreamSets)
, mInputScalars(kernel->mInputScalars)
, mOutputScalars(kernel->mOutputScalars)
, mInternalScalars(kernel->mInternalScalars) {
    initializeIOBindingMap();
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief destructor
 ** ------------------------------------------------------------------------------------------------------------- */
KernelCompiler::~KernelCompiler() {

}

}
