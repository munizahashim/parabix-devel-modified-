#include <kernel/pipeline/driver/driver.h>

#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <llvm/IR/Module.h>
#include <toolchain/toolchain.h>
#include <objcache/object_cache.h>
#include <llvm/Support/raw_ostream.h>

using namespace kernel;
using namespace llvm;

using RelationshipAllocator = Relationship::Allocator;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makePipelineWithIO
 ** ------------------------------------------------------------------------------------------------------------- */
std::unique_ptr<ProgramBuilder> BaseDriver::makePipelineWithIO(Bindings stream_inputs, Bindings stream_outputs, Bindings scalar_inputs, Bindings scalar_outputs) {
    PipelineKernel * const pipeline =
        new PipelineKernel(getBuilder(),
                           std::move(stream_inputs), std::move(stream_outputs),
                           std::move(scalar_inputs), std::move(scalar_outputs));
    return std::make_unique<ProgramBuilder>(*this, pipeline);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makePipeline
 ** ------------------------------------------------------------------------------------------------------------- */
std::unique_ptr<ProgramBuilder> BaseDriver::makePipeline(Bindings scalar_inputs, Bindings scalar_outputs) {
    PipelineKernel * const pipeline =
        new PipelineKernel(getBuilder(),
                           {}, {},
                           std::move(scalar_inputs), std::move(scalar_outputs));
    return std::make_unique<ProgramBuilder>(*this, pipeline);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
StreamSet * BaseDriver::CreateStreamSet(const unsigned NumElements, const unsigned FieldWidth) noexcept {
    RelationshipAllocator A(mAllocator);
    return new (A) StreamSet(getContext(), NumElements, FieldWidth);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateRepeatingStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
RepeatingStreamSet * BaseDriver::CreateRepeatingStreamSet(const unsigned FieldWidth, std::vector<std::vector<uint64_t>> && stringSet, const bool isDynamic) noexcept {
    RelationshipAllocator A(mAllocator);
    // TODO: the stringSet will probably cause a memleak
    return new (A) RepeatingStreamSet(getContext(), FieldWidth, std::move(stringSet), isDynamic, false);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateUnalignedRepeatingStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
RepeatingStreamSet * BaseDriver::CreateUnalignedRepeatingStreamSet(const unsigned FieldWidth, std::vector<std::vector<uint64_t>> && stringSet, const bool isDynamic) noexcept {
    RelationshipAllocator A(mAllocator);
    // TODO: the stringSet will probably cause a memleak
    return new (A) RepeatingStreamSet(getContext(), FieldWidth, std::move(stringSet), isDynamic, true);
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateTruncatedStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
TruncatedStreamSet * BaseDriver::CreateTruncatedStreamSet(const StreamSet * data) noexcept {
    RelationshipAllocator A(mAllocator);
    return new (A) TruncatedStreamSet(getContext(), data);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateConstant
 ** ------------------------------------------------------------------------------------------------------------- */
Scalar * BaseDriver::CreateScalar(not_null<Type *> scalarType) noexcept {
    RelationshipAllocator A(mAllocator);
    return new (A) Scalar(scalarType);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateConstant
 ** ------------------------------------------------------------------------------------------------------------- */
Scalar * BaseDriver::CreateConstant(not_null<Constant *> value) noexcept {
    RelationshipAllocator A(mAllocator);
    return new (A) ScalarConstant(value);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief CreateCommandLineScalar
 ** ------------------------------------------------------------------------------------------------------------- */
Scalar * BaseDriver::CreateCommandLineScalar(CommandLineScalarType type) noexcept {
    RelationshipAllocator A(mAllocator);
    Type * scalarTy = nullptr;
    switch (type) {

        #ifdef ENABLE_PAPI
        case CommandLineScalarType::PAPIEventSet:
            scalarTy = mBuilder->getInt32Ty(); break;
        case CommandLineScalarType::PAPIEventList:
            scalarTy = mBuilder->getInt32Ty()->getPointerTo(); break;
        #endif
        case CommandLineScalarType::ParabixIllustratorObject:
            scalarTy = mBuilder->getVoidPtrTy(); break;
        case CommandLineScalarType::DynamicMultithreadingAddSynchronizationThreshold:
        case CommandLineScalarType::DynamicMultithreadingRemoveSynchronizationThreshold:
            scalarTy = mBuilder->getFloatTy(); break;
        default:
            scalarTy = mBuilder->getSizeTy(); break;
    }


    return new (A) CommandLineScalar(type, scalarTy);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void BaseDriver::addKernel(not_null<Kernel *> kernel) {

    if (LLVM_UNLIKELY(kernel->isGenerated())) {
        return;
    }

    // Verify the I/O relationships were properly set / defaulted in.

    for (Binding & input : kernel->getInputScalarBindings()) {
        if (input.getRelationship() == nullptr) {
            input.setRelationship(CreateScalar(input.getType()));
        }
    }

    if (LLVM_UNLIKELY(codegen::EnableIllustrator)) {
        // TODO: temporary design choice; need to rethink how we should handle implicit scalars
        auto illustratorObject = CreateCommandLineScalar(CommandLineScalarType::ParabixIllustratorObject);
        kernel->getInputScalarBindings().emplace_back(KERNEL_ILLUSTRATOR_CALLBACK_OBJECT, illustratorObject);
    }

    for (Binding & input : kernel->getInputStreamSetBindings()) {
        if (LLVM_UNLIKELY(input.getRelationship() == nullptr)) {
            report_fatal_error(StringRef(kernel->getName()) + "." + input.getName() + " must be set upon construction");
        }
    }
    for (Binding & output : kernel->getOutputStreamSetBindings()) {
        if (LLVM_UNLIKELY(output.getRelationship() == nullptr)) {
            report_fatal_error(StringRef(kernel->getName()) + "." + output.getName() + " must be set upon construction");
        }
    }
    for (Binding & output : kernel->getOutputScalarBindings()) {
        if (output.getRelationship() == nullptr) {
            output.setRelationship(CreateScalar(output.getType()));
        }
    }

    if (LLVM_LIKELY(mObjectCache.get())) {
        switch (mObjectCache->loadCachedObjectFile(getBuilder(), kernel)) {
            case CacheObjectResult::CACHED:
                mCachedKernel.emplace_back(kernel.get());
                break;
            case CacheObjectResult::COMPILED:
                mCompiledKernel.emplace_back(kernel.get());
                break;
            case CacheObjectResult::UNCACHED:
                mUncachedKernel.emplace_back(kernel.get());
                break;
        }
        assert ("kernel does not contain a module?" && kernel->getModule());
    } else {
        kernel->makeModule(getBuilder());
        mUncachedKernel.emplace_back(kernel.get());
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
BaseDriver::BaseDriver(std::string && moduleName)
: mContext(new LLVMContext())
, mMainModule(new Module(moduleName, *mContext))
, mBuilder(nullptr)
, mObjectCache(nullptr) {
    if (LLVM_UNLIKELY(codegen::EnableObjectCache)) {
        mObjectCache.reset(new ParabixObjectCache());
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief destructor
 ** ------------------------------------------------------------------------------------------------------------- */
BaseDriver::~BaseDriver() {

}
