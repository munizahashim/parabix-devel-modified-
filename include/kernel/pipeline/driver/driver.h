#ifndef DRIVER_H
#define DRIVER_H

#include <codegen/FunctionTypeBuilder.h>
#include <codegen/virtual_driver.h>
#include <llvm/ExecutionEngine/GenericValue.h>
#include <kernel/core/kernel.h>
#include <kernel/core/relationship.h>
#include <util/slab_allocator.h>
#include <boost/integer.hpp>
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
    using BuilderRef = Kernel::BuilderRef;
    using KernelSet = std::vector<std::unique_ptr<Kernel>>;
    using KernelMap = llvm::StringMap<std::unique_ptr<Kernel>>;


    std::unique_ptr<kernel::ProgramBuilder> makePipelineWithIO(Bindings stream_inputs = {}, Bindings stream_outputs = {}, Bindings scalar_inputs = {}, Bindings scalar_outputs = {});

    std::unique_ptr<kernel::ProgramBuilder> makePipeline(Bindings scalar_inputs = {}, Bindings scalar_outputs = {});

    BuilderRef getBuilder() {
        return mBuilder;
    }

    kernel::StreamSet * CreateStreamSet(const unsigned NumElements = 1, const unsigned FieldWidth = 1);

    template<unsigned FieldWidth, typename storage_t = typename boost::uint_t<FieldWidth>::fast>
    kernel::RepeatingStreamSet * CreateRepeatingStreamSet(const storage_t * string) {
        static_assert(FieldWidth == 8, "non-8 bit types are not currently supported");
        // TODO: although we could represent 1-bit values with a 0/1 string, and 8-bit with ASCII,
        // supporting other types will require a varadic argument. Should every type be automatically
        // converted to a byte array? What should the interface be for multi-element types?
        return __CreateRepeatingStreamSet8(string, sizeof(string));
    }

    template<unsigned NumElements, unsigned FieldWidth, typename storage_t = typename boost::uint_t<FieldWidth>::fast>
    kernel::RepeatingStreamSet * CreateRepeatingStreamSet(std::array<const storage_t *, NumElements> string) {
        static_assert(FieldWidth == 8, "non-8 bit types are not currently supported");
        // TODO: although we could represent 1-bit values with a 0/1 string, and 8-bit with ASCII,
        // supporting other types will require a varadic argument. Should every type be automatically
        // converted to a byte array? What should the interface be for multi-element types?
        return __CreateRepeatingStreamSet8(string, sizeof(string));
    }


    kernel::Scalar * CreateScalar(not_null<llvm::Type *> scalarType);

    kernel::Scalar * CreateConstant(not_null<llvm::Constant *> value);

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

private:

    kernel::RepeatingStreamSet * __CreateRepeatingStreamSet8(const uint8_t * string, size_t length);

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

#endif // DRIVER_H
