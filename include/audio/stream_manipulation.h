#pragma once
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/relationship.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Constants.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>

using namespace pablo;
using namespace kernel;
using namespace llvm;

namespace audio 
{
    class CreateOnes : public PabloKernel {
    public:
        CreateOnes(KernelBuilder & kb, StreamSet * dataStream, StreamSet * onesStream)
            : PabloKernel(kb, "CreateOnes",
                        {Binding{"dataStream", dataStream}},
                        {Binding{"onesStream", onesStream}}) {}
    protected:
        void generatePabloMethod() override;
    };

    class SplitKernel final : public MultiBlockKernel {
    public:
        SplitKernel(KernelBuilder & b,
                const unsigned int bitsPerSample,
                StreamSet * const inputStreams,
                StreamSet * const outputStreams);
    protected:
        void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    private:
        unsigned int bitsPerSample;
        unsigned int numInputStreams;
    };

    class Split2Kernel final : public MultiBlockKernel {
    public:
        Split2Kernel(KernelBuilder &b, const unsigned int bitsPerSample, StreamSet *const inputStream, StreamSet *const outputStream_1, StreamSet *const outputStream_2);
    protected:
        void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    private:
        unsigned int bitsPerSample;
    };

    class MergeKernel final : public MultiBlockKernel {
    public:
        MergeKernel(KernelBuilder & b,
                const unsigned int bitsPerSample,
                StreamSet * const firstInputStream,
                StreamSet * const secondInputStream,
                StreamSet * const outputStream);
    protected:
        void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    private:
        unsigned int bitsPerSample;
    };
}