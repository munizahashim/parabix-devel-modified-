#ifndef PIPELINE_BUILDER_H
#define PIPELINE_BUILDER_H

#include <kernel/pipeline/pipeline_kernel.h>
#include <boost/integer.hpp>

class BaseDriver;

namespace kernel {

class OptimizationBranchBuilder;
//class PipelineAnalysis;
//class PipelineCompiler;

class PipelineBuilder {
    friend class PipelineKernel;
    friend class PipelineAnalysis;
    friend class PipelineCompiler;
    friend class OptimizationBranchBuilder;
public:

    using Kernels = PipelineKernel::Kernels;
    using CallBinding = PipelineKernel::CallBinding;
    using CallBindings = PipelineKernel::CallBindings;
    using LengthAssertion = PipelineKernel::LengthAssertion;
    using LengthAssertions = PipelineKernel::LengthAssertions;

    BaseDriver & getDriver() { return mDriver;}

    template<typename KernelType, typename... Args>
    Kernel * CreateKernelCall(Args &&... args) {
        return initializeKernel(new KernelType(mDriver.getBuilder(), std::forward<Args>(args) ...), 0U);
    }

    template<typename KernelType, typename... Args>
    Kernel * CreateKernelFamilyCall(Args &&... args) {
        return initializeKernel(new KernelType(mDriver.getBuilder(), std::forward<Args>(args) ...), PipelineKernel::KernelBindingFlag::Family);
    }

    Kernel * AddKernelCall(Kernel * kernel, const unsigned flags) {
        return initializeKernel(kernel, flags);
    }

    template<typename KernelType, typename... Args>
    PipelineKernel * CreateNestedPipelineCall(Args &&... args) {
        return initializePipeline(new KernelType(mDriver.getBuilder(), std::forward<Args>(args) ...), 0U);
    }

    template<typename KernelType, typename... Args>
    PipelineKernel * CreateNestedPipelineFamilyCall(Args &&... args) {
        return initializePipeline(new KernelType(mDriver.getBuilder(), std::forward<Args>(args) ...), PipelineKernel::KernelBindingFlag::Family);
    }

    std::shared_ptr<OptimizationBranchBuilder>
        CreateOptimizationBranch(Relationship * const condition,
                                 Bindings && stream_inputs = {}, Bindings && stream_outputs = {},
                                 Bindings && scalar_inputs = {}, Bindings && scalar_outputs = {});

    StreamSet * CreateStreamSet(const unsigned NumElements = 1, const unsigned FieldWidth = 1) {
        return mDriver.CreateStreamSet(NumElements, FieldWidth);
    }

    Scalar * CreateConstant(llvm::Constant * value) {
        return mDriver.CreateConstant(value);
    }

    using pattern_t = std::vector<uint64_t>;

    RepeatingStreamSet * CreateRepeatingStreamSet(unsigned FieldWidth, pattern_t string, const bool isDynamic = true) {
        return mDriver.CreateRepeatingStreamSet(FieldWidth, std::vector<pattern_t>{std::move(string)}, isDynamic);
    }

    RepeatingStreamSet * CreateRepeatingStreamSet(unsigned FieldWidth, std::vector<pattern_t> string, const bool isDynamic = true) {
        return mDriver.CreateRepeatingStreamSet(FieldWidth, std::move(string), isDynamic);
    }

    template<unsigned FieldWidth, unsigned NumOfElements>
    RepeatingStreamSet * CreateRepeatingStreamSet(std::array<pattern_t, NumOfElements> & string) {
        return mDriver.CreateRepeatingStreamSet(FieldWidth, std::vector<pattern_t>{string.begin(), string.end()}, true);
    }

    RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(unsigned FieldWidth, pattern_t string, const bool isDynamic = true) {
        return mDriver.CreateUnalignedRepeatingStreamSet(FieldWidth, std::vector<pattern_t>{std::move(string)}, isDynamic);
    }

    RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(unsigned FieldWidth, std::vector<pattern_t> string, const bool isDynamic = true) {
        return mDriver.CreateUnalignedRepeatingStreamSet(FieldWidth, std::move(string), isDynamic);
    }

    template<unsigned FieldWidth, unsigned NumOfElements>
    RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(std::array<pattern_t, NumOfElements> & string) {
        return mDriver.CreateUnalignedRepeatingStreamSet(FieldWidth, std::vector<pattern_t>{string.begin(), string.end()}, true);
    }

    template <typename ExternalFunctionType>
    void CreateCall(std::string name, ExternalFunctionType & functionPtr, std::initializer_list<Scalar *> args) {
        llvm::FunctionType * const type = FunctionTypeBuilder<ExternalFunctionType>::get(mDriver.getContext());
        assert ("FunctionTypeBuilder did not resolve a function type." && type);
        assert ("Function was not provided the correct number of args" && type->getNumParams() == args.size());
        // Since the pipeline kernel module has not been made yet, just record the function info and its arguments.
        mCallBindings.emplace_back(std::move(name), type, reinterpret_cast<void *>(&functionPtr), std::move(args));
    }

    Scalar * getInputScalar(const unsigned i) {
        return llvm::cast<Scalar>(mInputScalars[i].getRelationship());
    }

    Scalar * getInputScalar(const llvm::StringRef name);

    void setInputScalar(const llvm::StringRef name, Scalar * value);

    Scalar * getOutputScalar(const unsigned i) {
        return llvm::cast<Scalar>(mOutputScalars[i].getRelationship());
    }

    Scalar * getOutputScalar(const llvm::StringRef name);

    void setOutputScalar(const llvm::StringRef name, Scalar * value);

    void AssertEqualLength(const StreamSet * A, const StreamSet * B) {
        mLengthAssertions.emplace_back(LengthAssertion{{A, B}});
    }

    PipelineBuilder(BaseDriver & driver,
                    Bindings && stream_inputs, Bindings && stream_outputs,
                    Bindings && scalar_inputs, Bindings && scalar_outputs,
                    const unsigned numOfThreads);

    PipelineBuilder(BaseDriver & driver,
                    llvm::StringRef pipelineName,
                    Bindings && stream_inputs, Bindings && stream_outputs,
                    Bindings && scalar_inputs, Bindings && scalar_outputs,
                    const unsigned numOfThreads);

    virtual ~PipelineBuilder() {}

    virtual Kernel * makeKernel();

    void setExternallySynchronized(const bool value = true) {
        mExternallySynchronized = value;
    }

    void setUniqueName(std::string name) {
        mUniqueName.swap(name);
    }

protected:


    // Internal pipeline constructor uses a zero-length tag struct to prevent
    // overloading errors. This parameter will be dropped by the compiler.
    struct Internal {};
    PipelineBuilder(Internal, BaseDriver & driver,
                    Bindings stream_inputs, Bindings stream_outputs,
                    Bindings scalar_inputs, Bindings scalar_outputs);

    Kernel * initializeKernel(Kernel * const kernel, const unsigned flags);

    PipelineKernel * initializePipeline(PipelineKernel * const kernel, const unsigned flags);

protected:

    BaseDriver &        mDriver;
    // eventual pipeline configuration
    unsigned            mNumOfThreads;
    bool                mExternallySynchronized = false;
    Bindings            mInputStreamSets;
    Bindings            mOutputStreamSets;
    Bindings            mInputScalars;
    Bindings            mOutputScalars;
    Bindings            mInternalScalars;
    Kernels             mKernels;
    CallBindings        mCallBindings;
    LengthAssertions    mLengthAssertions;
    std::string         mUniqueName;
};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief ProgramBuilder
 ** ------------------------------------------------------------------------------------------------------------- */
class ProgramBuilder : public PipelineBuilder {
    friend class PipelineBuilder;
public:

    void * compile();

    void setNumOfThreads(const unsigned threads) {
        mNumOfThreads = threads;
    }

    ProgramBuilder(BaseDriver & driver,
                   Bindings && stream_inputs, Bindings && stream_outputs,
                   Bindings && scalar_inputs, Bindings && scalar_outputs);

private:

    void * compileKernel(Kernel * const kernel);
};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief PipelineBranchBuilder
 ** ------------------------------------------------------------------------------------------------------------- */
class OptimizationBranchBuilder final : public PipelineBuilder {
    friend class PipelineKernel;
    friend class PipelineBuilder;
public:

    const std::unique_ptr<PipelineBuilder> & getNonZeroBranch() const {
        return mNonZeroBranch;
    }

    const std::unique_ptr<PipelineBuilder> & getAllZeroBranch() const {
        return mAllZeroBranch;
    }

    ~OptimizationBranchBuilder();

protected:

    OptimizationBranchBuilder(BaseDriver & driver, Relationship * const condition,
                              Bindings && stream_inputs, Bindings && stream_outputs,
                              Bindings && scalar_inputs, Bindings && scalar_outputs);

    Kernel * makeKernel() override;

private:
    Relationship * const             mCondition;
    std::unique_ptr<PipelineBuilder> mNonZeroBranch;
    std::unique_ptr<PipelineBuilder> mAllZeroBranch;
};

inline std::shared_ptr<OptimizationBranchBuilder> PipelineBuilder::CreateOptimizationBranch (
        Relationship * const condition,
        Bindings && stream_inputs, Bindings && stream_outputs,
        Bindings && scalar_inputs, Bindings && scalar_outputs) {
    std::shared_ptr<OptimizationBranchBuilder> branch(
        new OptimizationBranchBuilder(mDriver, condition,
            std::move(stream_inputs), std::move(stream_outputs),
            std::move(scalar_inputs), std::move(scalar_outputs)));
    return branch;
}

}

#endif // PIPELINE_BUILDER_H
