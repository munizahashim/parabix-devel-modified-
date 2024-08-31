#pragma once

#include <kernel/pipeline/pipeline_kernel.h>
#include <llvm/IR/Constants.h>
#include <boost/integer.hpp>

class BaseDriver;

namespace kernel {

class OptimizationBranchBuilder;

class PipelineBuilder { // : public LLVMTypeSystemInterface
    friend class PipelineKernel;
    friend class PipelineAnalysis;
    friend class PipelineCompiler;
    friend class OptimizationBranchBuilder;
public:

    using Kernels = PipelineKernel::Kernels;
    using Relationships = PipelineKernel::Relationships;
    using CallBinding = PipelineKernel::CallBinding;
    using CallBindings = PipelineKernel::CallBindings;
    using LengthAssertion = PipelineKernel::LengthAssertion;
    using LengthAssertions = PipelineKernel::LengthAssertions;

    BaseDriver & getDriver() { return mDriver;}

    template<typename KernelType, typename... Args>
    Kernel * CreateKernelCall(Args &&... args) {
        return initializeKernel(new KernelType(mDriver, std::forward<Args>(args) ...), 0U);
    }

    template<typename KernelType, typename... Args>
    Kernel * CreateKernelFamilyCall(Args &&... args) {
        return initializeKernel(new KernelType(mDriver, std::forward<Args>(args) ...), PipelineKernel::KernelBindingFlag::Family);
    }

    Kernel * AddKernelCall(Kernel * kernel, const unsigned flags) {
        return initializeKernel(kernel, flags);
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

    Scalar * CreateScalar(llvm::Type * type) {
        return mDriver.CreateScalar(type);
    }

    using pattern_t = std::vector<uint64_t>;

    #define RETURN_REPSTREAMSET(...) \
        RepeatingStreamSet * const ss = mDriver.CreateRepeatingStreamSet(__VA_ARGS__); \
        mTarget->mInternallyGeneratedStreamSets.push_back(ss); \
        return ss

    RepeatingStreamSet * CreateRepeatingStreamSet(unsigned FieldWidth, pattern_t string, const bool isDynamic = true) {
        RETURN_REPSTREAMSET(FieldWidth, std::vector<pattern_t>{std::move(string)}, isDynamic);
    }

    RepeatingStreamSet * CreateRepeatingStreamSet(unsigned FieldWidth, std::vector<pattern_t> string, const bool isDynamic = true) {
        RETURN_REPSTREAMSET(FieldWidth, std::move(string), isDynamic);
    }

    StreamSet * CreateRepeatingBixNum(unsigned bixNumBits, pattern_t nums, const bool isDynamic = true);

    template<unsigned FieldWidth, unsigned NumOfElements>
    RepeatingStreamSet * CreateRepeatingStreamSet(std::array<pattern_t, NumOfElements> & string) {
        RETURN_REPSTREAMSET(FieldWidth, std::vector<pattern_t>{string.begin(), string.end()}, true);
    }

    #undef RETURN_REPSTREAMSET

    #define RETURN_REPSTREAMSET(...) \
        RepeatingStreamSet * const ss = mDriver.CreateUnalignedRepeatingStreamSet(__VA_ARGS__); \
        mTarget->mInternallyGeneratedStreamSets.push_back(ss); \
        return ss

    RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(unsigned FieldWidth, pattern_t string, const bool isDynamic = true) {
        RETURN_REPSTREAMSET(FieldWidth, std::vector<pattern_t>{std::move(string)}, isDynamic);
    }

    RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(unsigned FieldWidth, std::vector<pattern_t> string, const bool isDynamic = true) {
        RETURN_REPSTREAMSET(FieldWidth, std::move(string), isDynamic);
    }

    template<unsigned FieldWidth, unsigned NumOfElements>
    RepeatingStreamSet * CreateUnalignedRepeatingStreamSet(std::array<pattern_t, NumOfElements> & string) {
        RETURN_REPSTREAMSET(FieldWidth, std::vector<pattern_t>{string.begin(), string.end()}, true);
    }

    #undef RETURN_REPSTREAMSET

    TruncatedStreamSet * CreateTruncatedStreamSet(const StreamSet * data) {
        return mDriver.CreateTruncatedStreamSet(data);
    }

    template <typename ExternalFunctionType>
    void CreateCall(std::string name, ExternalFunctionType & functionPtr, std::initializer_list<Scalar *> args) {
        llvm::FunctionType * const type = FunctionTypeBuilder<ExternalFunctionType>::get(mDriver.getContext());
        assert ("FunctionTypeBuilder did not resolve a function type." && type);
        assert ("Function was not provided the correct number of args" && type->getNumParams() == args.size());
        // Since the pipeline kernel module has not been made yet, just record the function info and its arguments.
        mTarget->mCallBindings.emplace_back(std::move(name), type, reinterpret_cast<void *>(&functionPtr), std::move(args));
    }

    StreamSet * getInputStreamSet(const unsigned i) {
        return static_cast<StreamSet *>(mTarget->mInputStreamSets[i].getRelationship());
    }

    StreamSet * getInputStreamSet(const llvm::StringRef name);

    StreamSet * getOutputStreamSet(const unsigned i) {
        return static_cast<StreamSet *>(mTarget->mOutputStreamSets[i].getRelationship());
    }

    StreamSet * getOutputStreamSet(const llvm::StringRef name);

    Scalar * getInputScalar(const unsigned i) {
        return static_cast<Scalar *>(mTarget->mInputScalars[i].getRelationship());
    }

    Scalar * getInputScalar(const llvm::StringRef name);

    Scalar * getOutputScalar(const unsigned i) {
        return static_cast<Scalar *>(mTarget->mOutputScalars[i].getRelationship());
    }

    Scalar * getOutputScalar(const llvm::StringRef name);

    void setOutputScalar(const llvm::StringRef name, Scalar * value);

    void AssertEqualLength(const StreamSet * A, const StreamSet * B) {
        mTarget->mLengthAssertions.emplace_back(LengthAssertion{{A, B}});
    }

    virtual ~PipelineBuilder() {}

    virtual Kernel * makeKernel();

    void setExternallySynchronized(const bool value = true) {
        mExternallySynchronized = value;
    }

    void setUniqueName(std::string name) {
        mTarget->mSignature.swap(name);
    }

    void captureByteData(llvm::StringRef streamName, StreamSet * byteData, char nonASCIIsubstitute = '.');

    void captureBitstream(llvm::StringRef streamName, StreamSet * bitstream, char zeroCh = '.', char oneCh = '1');

    void captureBixNum(llvm::StringRef streamName, StreamSet * bixnum, char hexBase = 'A');

    /// Get a constant value representing either true or false.
    llvm::ConstantInt * LLVM_READNONE getInt1(bool V) {
      return llvm::ConstantInt::get(getInt1Ty(), V);
    }

    /// Get the constant value for i1 true.
    llvm::ConstantInt * LLVM_READNONE getTrue() {
      return llvm::ConstantInt::getTrue(mDriver.getContext());
    }

    /// Get the constant value for i1 false.
    llvm::ConstantInt * LLVM_READNONE getFalse() {
      return llvm::ConstantInt::getFalse(mDriver.getContext());
    }

    /// Get a constant 8-bit value.
    llvm::ConstantInt * LLVM_READNONE getInt8(uint8_t C) {
      return llvm::ConstantInt::get(getInt8Ty(), C);
    }

    /// Get a constant 16-bit value.
    llvm::ConstantInt * LLVM_READNONE getInt16(uint16_t C) {
      return llvm::ConstantInt::get(getInt16Ty(), C);
    }

    /// Get a constant 32-bit value.
    llvm::ConstantInt * LLVM_READNONE getInt32(uint32_t C) {
      return llvm::ConstantInt::get(getInt32Ty(), C);
    }

    /// Get a constant 64-bit value.
    llvm::ConstantInt * getInt64(uint64_t C) {
      return llvm::ConstantInt::get(getInt64Ty(), C);
    }

    /// Get a constant 64-bit value.
    llvm::ConstantInt * getSize(size_t C) {
      return llvm::ConstantInt::get(getSizeTy(), C);
    }

    /// Get a constant N-bit value, zero extended or truncated from
    /// a 64-bit value.
    llvm::ConstantInt *getIntN(unsigned N, uint64_t C) {
      return llvm::ConstantInt::get(getIntNTy(N), C);
    }

    /// Get a constant integer value.
    llvm::ConstantInt *getInt(const llvm::APInt &AI) {
      return llvm::ConstantInt::get(mDriver.getContext(), AI);
    }

    llvm::Constant * getDouble(const double C) {
        return llvm::ConstantFP::get(getDoubleTy(), C);
    }

    llvm::Constant * getFloat(const float C) {
        return llvm::ConstantFP::get(getFloatTy(), C);
    }

    /// Fetch the type representing a single bit
    llvm::IntegerType * getInt1Ty() {
      return llvm::Type::getInt1Ty(mDriver.getContext());
    }

    /// Fetch the type representing an 8-bit integer.
    llvm::IntegerType * getInt8Ty() {
      return llvm::Type::getInt8Ty(mDriver.getContext());
    }

    /// Fetch the type representing a pointer to an 8-bit integer value.
    llvm::PointerType * getInt8PtrTy(unsigned AddrSpace = 0) {
      return llvm::PointerType::getInt8PtrTy(mDriver.getContext(), AddrSpace);
    }

    /// Fetch the type representing a 16-bit integer.
    llvm::IntegerType * getInt16Ty() {
      return llvm::Type::getInt16Ty(mDriver.getContext());
    }

    /// Fetch the type representing a pointer to an 8-bit integer value.
    llvm::PointerType * getInt16PtrTy(unsigned AddrSpace = 0) {
      return llvm::PointerType::getInt16PtrTy(mDriver.getContext(), AddrSpace);
    }

    /// Fetch the type representing a 32-bit integer.
    llvm::IntegerType * getInt32Ty() {
      return llvm::Type::getInt32Ty(mDriver.getContext());
    }

    /// Fetch the type representing a pointer to an 8-bit integer value.
    llvm::PointerType * getInt32PtrTy(unsigned AddrSpace = 0) {
      return llvm::PointerType::getInt32PtrTy(mDriver.getContext(), AddrSpace);
    }

    /// Fetch the type representing a 64-bit integer.
    llvm::IntegerType * getInt64Ty() {
      return llvm::Type::getInt64Ty(mDriver.getContext());
    }

    /// Fetch the type representing a 64-bit integer.
    llvm::IntegerType * getSizeTy() {
      return llvm::IntegerType::get(mDriver.getContext(), sizeof(size_t) * 8);
    }

    /// Fetch the type representing a pointer to an 8-bit integer value.
    llvm::PointerType * getInt64PtrTy(unsigned AddrSpace = 0) {
      return llvm::PointerType::getInt64PtrTy(mDriver.getContext(), AddrSpace);
    }

    /// Fetch the type representing a 128-bit integer.
    llvm::IntegerType * LLVM_READNONE getInt128Ty() {
        return llvm::Type::getInt128Ty(mDriver.getContext());
    }

    /// Fetch the type representing an N-bit integer.
    llvm::IntegerType * LLVM_READNONE getIntNTy(unsigned N) {
      return llvm::Type::getIntNTy(mDriver.getContext(), N);
    }

    /// Fetch the type representing a 16-bit floating point value.
    llvm::Type * LLVM_READNONE getHalfTy() {
      return llvm::Type::getHalfTy(mDriver.getContext());
    }

    /// Fetch the type representing a 16-bit brain floating point value.
    llvm::Type * LLVM_READNONE getBFloatTy() {
      return llvm::Type::getBFloatTy(mDriver.getContext());
    }

    /// Fetch the type representing a 32-bit floating point value.
    llvm::Type * LLVM_READNONE getFloatTy() {
      return llvm::Type::getFloatTy(mDriver.getContext());
    }

    /// Fetch the type representing a 64-bit floating point value.
    llvm::Type * LLVM_READNONE getDoubleTy() {
      return llvm::Type::getDoubleTy(mDriver.getContext());
    }

    /// Fetch the type representing void.
    llvm::Type * LLVM_READNONE getVoidTy() {
      return llvm::Type::getVoidTy(mDriver.getContext());
    }

    /// Fetch the type of an integer with size at least as big as that of a
    /// pointer in the given address space.
    llvm::IntegerType * LLVM_READNONE getIntPtrTy(unsigned AddrSpace = 0) {
        return mDriver.getIntPtrTy(AddrSpace);
    }

    llvm::IntegerType * LLVM_READNONE getIntAddrTy() const {
        return mDriver.getIntAddrTy();
    }

    llvm::LLVMContext & getContext() {
        return mDriver.getContext();
    }

    unsigned getBitBlockWidth() const {
        return mDriver.getBitBlockWidth();
    }

    llvm::VectorType * getBitBlockType() const {
        return mDriver.getBitBlockType();
    }

    llvm::VectorType * getStreamTy(const unsigned FieldWidth = 1) {
        return mDriver.getStreamTy(FieldWidth);
    }

    llvm::ArrayType * getStreamSetTy(const unsigned NumElements = 1, const unsigned FieldWidth = 1) {
        return mDriver.getStreamSetTy(NumElements, FieldWidth);
    }

protected:

    PipelineBuilder(BaseDriver & driver, PipelineKernel * const kernel);

    Kernel * initializeKernel(Kernel * const kernel, const unsigned flags);

protected:

    BaseDriver &            mDriver;
    // eventual pipeline configuration
    PipelineKernel * const  mTarget;

    bool                    mExternallySynchronized = false;
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
                              PipelineKernel * const allZero,
                              PipelineKernel * const nonZero,
                              PipelineKernel * const branch);

    Kernel * makeKernel() override;

private:
    Relationship * const             mCondition;
    std::unique_ptr<PipelineBuilder> mNonZeroBranch;
    std::unique_ptr<PipelineBuilder> mAllZeroBranch;
};

}

