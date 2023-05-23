/*
 *  Copyright (c) 2018 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <kernel/core/idisa_target.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Constant.h>
#include <testing/assert.h>
#include <random>

#include <llvm/IR/Instructions.h>

using namespace kernel;
using namespace llvm;

bool verbose = true;

class CopyKernel final : public SegmentOrientedKernel {
public:
    CopyKernel(BuilderRef b, StreamSet * input, StreamSet * output, Scalar * upTo);
protected:
    void generateDoSegmentMethod(BuilderRef b) override;
};

CopyKernel::CopyKernel(BuilderRef b, StreamSet * input, StreamSet * output, Scalar * upTo)
: SegmentOrientedKernel(b, "copykernel",
// input streams
{Binding{"input", input}},
// output stream
{Binding{"output", output}},
// input scalar
{Binding{"upTo", upTo}},
{},
// internal scalar
{}) {
    addAttribute(CanTerminateEarly());
}

void CopyKernel::generateDoSegmentMethod(BuilderRef b) {

    BasicBlock * const copyAll = b->CreateBasicBlock("copyAll");
    BasicBlock * const copyPartial = b->CreateBasicBlock("copyPartial");
    BasicBlock * const segmentExit = b->CreateBasicBlock("segmentExit");

    const auto is = getInputStreamSet(0);
    const auto ne = is->getNumElements();
    const auto fw = is->getFieldWidth();
    const auto bw = b->getBitBlockWidth();

    Value * const upTo = b->getScalarField("upTo");
    Value * const processed = b->getProcessedItemCount("input");
    Value * const avail = b->getAvailableItemCount("input");

    ConstantInt * const sz_ZERO = b->getSize(0);

    Value * const inputPtr = b->getInputStreamBlockPtr("input", sz_ZERO);
    Value * const outputPtr = b->getOutputStreamBlockPtr("output", sz_ZERO);

    b->CreateLikelyCondBr(b->CreateICmpULE(avail, upTo), copyAll, copyPartial);

    b->SetInsertPoint(copyAll);
    Constant * const bytesPerStride = b->getSize(ne * fw * bw  / 8);
    Value * const numBytes = b->CreateMul(bytesPerStride, b->getNumOfStrides());
    b->CreateMemCpy(outputPtr, inputPtr, numBytes, bw / 8);
    b->CreateBr(segmentExit);


    b->SetInsertPoint(copyPartial);
    Value * const remaining = b->CreateSub(upTo, processed);
    Constant * const BLOCK_WIDTH = b->getSize(bw);
    Value * const fullStrides = b->CreateUDiv(remaining, BLOCK_WIDTH);
    Value * const fullStrideBytes = b->CreateMul(bytesPerStride, fullStrides);
    b->CreateMemCpy(outputPtr, inputPtr, fullStrideBytes, bw / 8);

    Value * items = b->CreateURem(remaining, BLOCK_WIDTH);
    if (fw) {
        items = b->CreateMul(items, b->getSize(fw));
    }

    for (unsigned i = 0; i < ne; ++i) {
        Value * inputPtr = nullptr;
        Value * outputPtr = nullptr;
        Constant * const sz_I = b->getSize(i);
        Value * current = items;
        for (unsigned j = 0; j < fw; ++j) {
            if (fw == 1) {
                inputPtr = b->getInputStreamBlockPtr("input", sz_I);
                outputPtr = b->getOutputStreamBlockPtr("output", sz_I);
            } else {
                Constant * const sz_J = b->getSize(j);
                inputPtr = b->getInputStreamPackPtr("input", sz_I, sz_J);
                outputPtr = b->getOutputStreamPackPtr("output", sz_I, sz_J);
            }
            Value * const maskPos = b->CreateUMin(current, BLOCK_WIDTH);
            Value * const mask = b->CreateNot(b->bitblock_mask_from(maskPos));
            Value * val = b->CreateAnd(b->CreateAlignedLoad(inputPtr, bw / 8), mask);
            b->CreateStore(val, outputPtr);
            current = b->CreateSaturatingSub(current, BLOCK_WIDTH);
        }
    }
    b->setProducedItemCount("output", upTo);
    b->setTerminationSignal();
    b->CreateBr(segmentExit);

    b->SetInsertPoint(segmentExit);
}

class PassThroughKernel final : public SegmentOrientedKernel {
public:
    PassThroughKernel(BuilderRef b, TruncatedStreamSet * output, Scalar * upTo);
protected:
    void generateDoSegmentMethod(BuilderRef b) override;
};

PassThroughKernel::PassThroughKernel(BuilderRef b, TruncatedStreamSet * output, Scalar * upTo)
: SegmentOrientedKernel(b, "passThroughKernel",
// input streams
{},
// output stream
{Binding{"output", output}},
// input scalar
{Binding{"upTo", upTo}},
{},
// internal scalar
{}) {
    addAttribute(CanTerminateEarly());
}

void PassThroughKernel::generateDoSegmentMethod(BuilderRef b) {

    BasicBlock * const termKernel = b->CreateBasicBlock("termKernel");
    BasicBlock * const segmentExit = b->CreateBasicBlock("segmentExit");
    Value * const upTo = b->getScalarField("upTo");

    Value * const max = b->getWritableOutputItems("output"); assert (max);
    Value * const avail = b->CreateAdd(b->getProducedItemCount("output"), max);
    b->CreateLikelyCondBr(b->CreateICmpULT(avail, upTo), segmentExit, termKernel);

    b->SetInsertPoint(termKernel);
    b->setProducedItemCount("output", upTo);
    b->setTerminationSignal();
    b->CreateBr(segmentExit);

    b->SetInsertPoint(segmentExit);
}

class StreamEq : public MultiBlockKernel {
    using BuilderRef = BuilderRef;
public:
    enum class Mode { EQ, NE };

    StreamEq(BuilderRef b, StreamSet * x, StreamSet * y, Scalar * outPtr);
    void generateInitializeMethod(BuilderRef b) override;
    void generateMultiBlockLogic(BuilderRef b, llvm::Value * const numOfStrides) override;
    void generateFinalizeMethod(BuilderRef b) override;

};

StreamEq::StreamEq(
    BuilderRef b,
    StreamSet * lhs,
    StreamSet * rhs,
    Scalar * outPtr)
    : MultiBlockKernel(b, [&]() -> std::string {
       std::string backing;
       raw_string_ostream str(backing);
       str << "StreamEq::["
           << "<i" << lhs->getFieldWidth() << ">"
           << "[" << lhs->getNumElements() << "],"
           << "<i" << rhs->getFieldWidth() << ">"
           << "[" << rhs->getNumElements() << "]]";
        str.flush();
        return backing;
    }(),
    {{"lhs", lhs}, {"rhs", rhs}},
    {},
    {{"result_ptr", outPtr}},
    {},
    {InternalScalar(b->getInt1Ty(), "accum")})
{
    assert(lhs->getFieldWidth() == rhs->getFieldWidth());
    assert(lhs->getNumElements() == rhs->getNumElements());
    setStride(b->getBitBlockWidth() / lhs->getFieldWidth());
    addAttribute(SideEffecting());
}

void StreamEq::generateInitializeMethod(BuilderRef b) {
    b->setScalarField("accum", b->getInt1(true));
}

void StreamEq::generateMultiBlockLogic(BuilderRef b, Value * const numOfStrides) {
    auto istreamset = b->getInputStreamSet("lhs");
    const uint32_t FW = istreamset->getFieldWidth();
    const uint32_t COUNT = istreamset->getNumElements();

    BasicBlock * const entryBlock = b->GetInsertBlock();
    BasicBlock * const loopBlock = b->CreateBasicBlock("loop");
    BasicBlock * const exitBlock = b->CreateBasicBlock("exit");

    Value * const initialAccum = b->getScalarField("accum");
    Constant * const sz_ZERO = b->getSize(0);

    Value * const hasMoreItems = b->CreateICmpNE(numOfStrides, sz_ZERO);

    b->CreateLikelyCondBr(hasMoreItems, loopBlock, exitBlock);

    b->SetInsertPoint(loopBlock);
    PHINode * const strideNo = b->CreatePHI(b->getSizeTy(), 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode * const accumPhi = b->CreatePHI(b->getInt1Ty(), 2);
    accumPhi->addIncoming(initialAccum, entryBlock);
    Value * nextAccum = accumPhi;
    for (uint32_t i = 0; i < COUNT; ++i) {
        Value * lhs;
        Value * rhs;
        if (FW == 1) {
            lhs = b->loadInputStreamBlock("lhs", b->getInt32(i), strideNo);
            rhs = b->loadInputStreamBlock("rhs", b->getInt32(i), strideNo);
        } else {
            // TODO: using strideNo in this fashion is technically going to refer to the
            // correct pack in memory but will exceed the number of elements in the pack
            lhs = b->loadInputStreamPack("lhs", b->getInt32(i), strideNo);
            rhs = b->loadInputStreamPack("rhs", b->getInt32(i), strideNo);
        }
        // Perform vector comparison lhs != rhs.
        // Result will be a vector of all zeros if lhs == rhs
        Value * const vComp = b->CreateICmpNE(lhs, rhs);
        Value * const vCompAsInt = b->CreateBitCast(vComp, b->getIntNTy(cast<IDISA::FixedVectorType>(vComp->getType())->getNumElements()));
        // `comp` will be `true` iff lhs == rhs (i.e., `vComp` is a vector of all zeros)
        Value * const comp = b->CreateICmpEQ(vCompAsInt, Constant::getNullValue(vCompAsInt->getType()));
    //    b->CallPrintInt("comp", comp);
        // `and` `comp` into `accum` so that `accum` will be `true` iff lhs == rhs for all blocks in the two streams
        nextAccum = b->CreateAnd(nextAccum, comp);
    }

    Value * const nextStrideNo = b->CreateAdd(strideNo, b->getSize(1));
    strideNo->addIncoming(nextStrideNo, loopBlock);
    accumPhi->addIncoming(nextAccum, loopBlock);
    b->CreateCondBr(b->CreateICmpNE(nextStrideNo, numOfStrides), loopBlock, exitBlock);

    b->SetInsertPoint(exitBlock);
    PHINode * const finalAccum = b->CreatePHI(b->getInt1Ty(), 2);
    finalAccum->addIncoming(initialAccum, entryBlock);
    finalAccum->addIncoming(nextAccum, loopBlock);
    b->setScalarField("accum", finalAccum);
}

void StreamEq::generateFinalizeMethod(BuilderRef b) {
    // a `result` value of `true` means the assertion passed
    Value * result = b->getScalarField("accum");

    // A `ptrVal` value of `0` means that the test is currently passing and a
    // value of `1` means the test is failing. If the test is already failing,
    // then we don't need to update the test state.
    Value * const ptrVal = b->CreateLoad(b->getScalarField("result_ptr"));
    Value * resultState  = b->CreateSelect(result, b->getInt32(0), b->getInt32(1));;

    Value * const newVal = b->CreateSelect(b->CreateICmpEQ(ptrVal, b->getInt32(1)), b->getInt32(1), resultState);
    b->CreateStore(newVal, b->getScalarField("result_ptr"));
}

typedef void (*TestFunctionType)(uint64_t copyCount, uint64_t passCount, uint32_t * output);

bool runRepeatingStreamSetTest(CPUDriver & pxDriver, std::default_random_engine & rng) {

    auto & b = pxDriver.getBuilder();

    IntegerType * const int64Ty = b->getInt64Ty();
    PointerType * const int32PtrTy = b->getInt32Ty()->getPointerTo();

    auto P = pxDriver.makePipeline(
                {Binding{int64Ty, "copyCount"},
                 Binding{int64Ty, "passCount"},
                 Binding{int32PtrTy, "output"}},
                {});

    std::uniform_int_distribution<uint64_t> fwDist(0, 5);

    const auto fieldWidth = fwDist(rng);

    std::uniform_int_distribution<uint64_t> numElemDist(1, 8);

    std::uniform_int_distribution<uint64_t> dist(0ULL, (1ULL << static_cast<uint64_t>(fieldWidth)) - 1ULL);

    std::uniform_int_distribution<uint64_t> patLength(1, 100);


    const auto numElements = numElemDist(rng);

    const auto patternLength = patLength(rng);


    std::vector<std::vector<uint64_t>> pattern(numElements);
    for (unsigned i = 0; i < numElements; ++i) {
        auto & vec = pattern[i];
        vec.resize(patternLength);
        for (unsigned j = 0; j < patternLength; ++j) {
            vec[j] = dist(rng);
        }
    }

    Scalar * const copyCount = P->getInputScalar("copyCount");

    RepeatingStreamSet * const RepeatingStream = P->CreateRepeatingStreamSet(fieldWidth, pattern);

    StreamSet * const Output = P->CreateStreamSet(numElements, fieldWidth);

    P->CreateKernelCall<CopyKernel>(RepeatingStream, Output, copyCount);

    TruncatedStreamSet * const Trunc1 = P->CreateTruncatedStreamSet(RepeatingStream);

    TruncatedStreamSet * const Trunc2 = P->CreateTruncatedStreamSet(Output);

    P->CreateKernelCall<PassThroughKernel>(Trunc1, copyCount);

    Scalar * const passCount = P->getInputScalar("passCount");

    P->CreateKernelCall<PassThroughKernel>(Trunc2, passCount);

    Scalar * output = P->getInputScalar("output");

    P->CreateKernelCall<StreamEq>(Trunc1, Trunc2, output);


    const auto f = reinterpret_cast<TestFunctionType>(P->compile());

    std::uniform_int_distribution<uint64_t> countDist(1, 22000);

    const uint64_t copyCountVal = countDist(rng);

    const uint64_t passCountVal = countDist(rng);

    uint32_t result = 0;

    f(copyCountVal, passCountVal, &result);

    if (result != 0 || verbose) {

        llvm::errs() << "TEST: " << numElements << 'x' << fieldWidth << 'w' << patternLength <<
                        " copyCount = " << copyCountVal <<
                        " passCount = " << passCountVal <<
                        " -- ";
        if (result == 0) {
            llvm::errs() << "success";
        } else {
            llvm::errs() << "failed";
        }
        llvm::errs() << '\n';
    }


    return (result != 0);
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {});
    CPUDriver pxDriver("test");
    std::random_device rd;
    std::default_random_engine rng(rd());

    bool testResult = false;
    for (unsigned rounds = 0; rounds < 1; ++rounds) {
        testResult |= runRepeatingStreamSetTest(pxDriver, rng);
    }
    return testResult ? -1 : 0;
}
