#pragma once

#include <codegen/FunctionTypeBuilder.h>
#include <codegen/virtual_driver.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <kernel/core/kernel.h>
#include <kernel/core/relationship.h>
#include <util/slab_allocator.h>
#include <boost/integer.hpp>
#include <kernel/illustrator/illustrator.h>
#include <string>
#include <vector>
#include <memory>

namespace llvm { class Function; }
namespace kernel { class KernelBuilder; }
namespace kernel { class ProgramBuilder; }
class CBuilder;
class ParabixObjectCache;

class BaseDriver : public codegen::VirtualDriver {
    friend class CBuilder;
    friend class kernel::ProgramBuilder;
    friend class kernel::Kernel;
public:

    using Kernel = kernel::Kernel;
    using Relationship = kernel::Relationship;
    using Bindings = kernel::Bindings;
    using KernelSet = std::vector<std::unique_ptr<Kernel>>;
    using KernelMap = llvm::StringMap<std::unique_ptr<Kernel>>;


    std::unique_ptr<kernel::ProgramBuilder> makePipelineWithIO(Bindings stream_inputs = {}, Bindings stream_outputs = {}, Bindings scalar_inputs = {}, Bindings scalar_outputs = {});

    std::unique_ptr<kernel::ProgramBuilder> makePipeline(Bindings scalar_inputs = {}, Bindings scalar_outputs = {});

    kernel::KernelBuilder & getBuilder() {
        return *mBuilder;
    }

    kernel::StreamSet * CreateStreamSet(const unsigned NumElements = 1, const unsigned FieldWidth = 1) noexcept;

    kernel::RepeatingStreamSet * CreateRepeatingStreamSet(const unsigned FieldWidth, std::vector<std::vector<uint64_t> > &&stringSet, const bool isDynamic = true) noexcept;

    kernel::TruncatedStreamSet * CreateTruncatedStreamSet(const kernel::StreamSet * data) noexcept;

    kernel::RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(const unsigned FieldWidth, std::vector<std::vector<uint64_t> > &&stringSet, const bool isDynamic = true) noexcept;

    kernel::Scalar * CreateScalar(not_null<llvm::Type *> scalarType) noexcept;

    kernel::Scalar * CreateConstant(not_null<llvm::Constant *> value) noexcept;

    kernel::Scalar * CreateCommandLineScalar(kernel::CommandLineScalarType type) noexcept;

    void addKernel(not_null<Kernel *> kernel);

    virtual bool hasExternalFunction(const llvm::StringRef functionName) const = 0;

    virtual void generateUncachedKernels() = 0;

    virtual void * finalizeObject(kernel::Kernel * pipeline) = 0;

    virtual ~BaseDriver();

    llvm::LLVMContext & getContext() const {
        return *mContext.get();
    }

    llvm::Module * getMainModule() const {
        return mMainModule;
    }

    bool getPreservesKernels() const {
        return mPreservesKernels;
    }

    void setPreserveKernels(const bool value = true) {
        mPreservesKernels = value;
    }

protected:

    BaseDriver(std::string && moduleName);

    template <typename ExternalFunctionType>
    void LinkFunction(not_null<Kernel *> kernel, llvm::StringRef name, ExternalFunctionType & functionPtr) const;

    virtual llvm::Function * addLinkFunction(llvm::Module * mod, llvm::StringRef name, llvm::FunctionType * type, void * functionPtr) const = 0;

protected:

    std::unique_ptr<llvm::LLVMContext>                      mContext;
    llvm::Module * const                                    mMainModule;
    std::unique_ptr<kernel::KernelBuilder>                  mBuilder;
    std::unique_ptr<ParabixObjectCache>                     mObjectCache;

    bool                                                    mPreservesKernels = false;
    KernelSet                                               mUncachedKernel;
    KernelSet                                               mCachedKernel;
    KernelSet                                               mCompiledKernel;
    KernelSet                                               mPreservedKernel;
    SlabAllocator<>                                         mAllocator;
};

template <typename ExternalFunctionType>
void BaseDriver::LinkFunction(not_null<Kernel *> kernel, llvm::StringRef name, ExternalFunctionType & functionPtr) const {
    kernel->link<ExternalFunctionType>(name, functionPtr);
}

