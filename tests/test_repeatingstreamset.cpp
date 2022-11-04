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
#include <boost/integer/common_factor_rt.hpp>
#include <llvm/IR/Constant.h>
#include <testing/assert.h>
#include <random>

#include <llvm/IR/Instructions.h>

using namespace kernel;
using namespace llvm;
using namespace testing;
using namespace boost::integer;

constexpr auto REPETITION_LENGTH = 999983ULL;

// constexpr auto REPETITION_LENGTH = 200ULL;

class RepeatingSourceKernel final : public SegmentOrientedKernel {
public:
    RepeatingSourceKernel(BuilderRef b, std::vector<std::vector<uint64_t>> pattern, StreamSet * output, const unsigned fillSize = 1024);
protected:
    bool allocatesInternalStreamSets() const override { return true; }
    void generateAllocateSharedInternalStreamSetsMethod(BuilderRef b, Value * expectedNumOfStrides) override;
    void generateDoSegmentMethod(BuilderRef b) override;
    void generateFinalizeMethod(BuilderRef b) override;

    StringRef getSignature() const override { return Signature; }
    bool hasSignature() const override { return true; }
private:
    LLVM_READNONE static std::string makeSignature(const std::vector<std::vector<uint64_t>> & pattern, const StreamSet * const output, const unsigned fillSize);
private:
    const std::string Signature;
    const std::vector<std::vector<uint64_t>> Pattern;
};

std::string RepeatingSourceKernel::makeSignature(const std::vector<std::vector<uint64_t>> & pattern, const StreamSet * const output, const unsigned fillSize) {
    std::string tmp;
    tmp.reserve(200);
    raw_string_ostream out(tmp);
    out << "repeating" << output->getFieldWidth() << 'C' << fillSize << ":{";
    for (const auto & vec : pattern) {
        char joiner = '{';
        for (const auto c : vec) {
            out << joiner;
            out.write_hex(c);
            joiner = ',';
        }
        out << '}';
    }
    out << '}';
    out.flush();
    return tmp;
}

RepeatingSourceKernel::RepeatingSourceKernel(BuilderRef b, std::vector<std::vector<uint64_t>> pattern, StreamSet * output, const unsigned fillSize)
: SegmentOrientedKernel(b, getStringHash(makeSignature(pattern, output, fillSize)),
// input streams
{},
// output stream
{Binding{"output", output, BoundedRate(0, fillSize), { ManagedBuffer(), Linear() }}},
// input scalar
{},
{},
// internal scalar
{})
, Signature(makeSignature(pattern, output, fillSize))
, Pattern(std::move(pattern)) {
    addAttribute(MustExplicitlyTerminate());
    setStride(1);
    PointerType * const voidPtrTy = b->getVoidPtrTy();
    addInternalScalar(voidPtrTy, "buffer");
    addInternalScalar(voidPtrTy, "ancillaryBuffer");
    IntegerType * const sizeTy = b->getSizeTy();
    addInternalScalar(sizeTy, "effectiveCapacity");
}



void RepeatingSourceKernel::generateAllocateSharedInternalStreamSetsMethod(BuilderRef b, Value * const expectedNumOfStrides) {
    const auto output = b->getOutputStreamSet("output");
    const auto fw = output->getFieldWidth();
    const Binding & binding = b->getOutputStreamSetBinding("output");
    const ProcessingRate & rate = binding.getRate();
    const auto itemsPerStrideTimesTwo = 2 * getStride() * rate.getUpperBound();
    const Rational bytesPerItem{fw * output->getNumElements(), 8};
    Value * const initialCapacity = b->CreateMulRational(expectedNumOfStrides, itemsPerStrideTimesTwo);
    Value * const bufferBytes = b->CreateMulRational(expectedNumOfStrides, itemsPerStrideTimesTwo * bytesPerItem);
    Value * const buffer = b->CreatePageAlignedMalloc(bufferBytes);
    PointerType * const ptrTy = b->getOutputStreamSetBuffer("output")->getPointerType();
    b->setBaseAddress("output", b->CreatePointerCast(buffer, ptrTy));
    b->setCapacity("output", initialCapacity);
    b->setScalarField("buffer", buffer);
    b->setScalarField("ancillaryBuffer", ConstantPointerNull::get(b->getVoidPtrTy()));
    b->setScalarField("effectiveCapacity", initialCapacity);
}

void RepeatingSourceKernel::generateDoSegmentMethod(BuilderRef b) {

    BasicBlock * const checkBuffer = b->CreateBasicBlock("checkBuffer");
    BasicBlock * const moveData = b->CreateBasicBlock("moveData");
    BasicBlock * const copyBack = b->CreateBasicBlock("CopyBack");
    BasicBlock * const expandAndCopyBack = b->CreateBasicBlock("ExpandAndCopyBack");
    BasicBlock * const prepareBuffer = b->CreateBasicBlock("PrepareBuffer");
    BasicBlock * const generateData = b->CreateBasicBlock("generateData");
    BasicBlock * const exit = b->CreateBasicBlock("exit");


    // build our pattern array
    const auto output = b->getOutputStreamSet("output");
    const auto fieldWidth = output->getFieldWidth();
    const auto numElements = output->getNumElements();
    size_t maxPatternSize = 0;
    for (unsigned i = 0; i < numElements; ++i) {
        const auto & vec = Pattern[i];
        const auto l = vec.size();
        maxPatternSize = std::max(maxPatternSize, l);
    }
    const Binding & binding = b->getOutputStreamSetBinding("output");
    const ProcessingRate & rate = binding.getRate();
    const auto fs = getStride() * rate.getUpperBound();
    assert (fs.denominator() == 1);
    const auto maxFillSize = fs.numerator();
    if (maxPatternSize > maxFillSize) {
        report_fatal_error("output rate should at least be as large as the pattern length");
    }

    const auto bw = b->getBitBlockWidth();
    if (fieldWidth > bw) {
        report_fatal_error("does not support field width sizes above " + std::to_string(bw));
    }
    if ((maxFillSize % bw) != 0) {
        report_fatal_error("output rate should be a multiple of " + std::to_string(bw)
                           + " to ensure proper streamset construction");
    }

   // const auto patternLength = ((maxPatternSize + (bw - 1ULL)) / bw) * bw;

    const auto maxVal = (1ULL << static_cast<uint64_t>(fieldWidth)) - 1ULL;

    StreamSetBuffer * const outputBuffer = b->getOutputStreamSetBuffer("output");
    PointerType * const outputStreamSetPtrTy = outputBuffer->getPointerType();    
    Type * const outputStreamSetTy = outputStreamSetPtrTy->getPointerElementType();



    const auto patternLength = ((maxPatternSize + (bw - 1ULL)) / bw) * bw;

    ConstantInt * const sz_ZERO = b->getSize(0);

   // Constant * const sz_outputTypeSize = ConstantExpr::getSizeOf(outputBuffer->getType());

    std::vector<Constant *> vectors(numElements);

    if (fieldWidth < 8) {
        // if our fieldwidth size is less than a byte, we treat it as a bit vector

        constexpr auto LANE_SIZE = 64ULL;
        assert ((LANE_SIZE % fieldWidth) == 0);
        const auto lanes = bw / LANE_SIZE;

        IntegerType * const int64Ty = b->getInt64Ty();

        FixedVectorType * const vecTy = FixedVectorType::get(int64Ty, lanes);

        ArrayType * const elementTy = ArrayType::get(vecTy, fieldWidth);

        ArrayType * const patternTy = ArrayType::get(elementTy, maxPatternSize);

        SmallVector<Constant *, 16> tmp(lanes);
        SmallVector<Constant *, 16> tmp2(fieldWidth);
        std::vector<Constant *> tmp3(maxPatternSize);

        for (unsigned p = 0; p < numElements; ++p) {
            const auto & vec = Pattern[p];
            const auto L = vec.size();
            for (uint64_t s = 0; s < maxPatternSize; ++s) {
                for (uint64_t i = 0; i < fieldWidth; ++i) {
                    for (uint64_t j = 0; j < lanes; ++j) {
                        for (uint64_t k = 0; k < patternLength; ) {
                            uint64_t V = 0;
                            for (uint64_t t = 0; t < LANE_SIZE; t += fieldWidth) {
                                const auto v = vec[(s + k++) % L];
                                if (LLVM_UNLIKELY(v > maxVal)) {
                                    report_fatal_error(std::to_string(v) + " exceeds a " +
                                                       std::to_string(fieldWidth) + "-bit value");
                                }
                                V |= (v << t);
                            }
                            tmp[j] = ConstantInt::get(int64Ty, V, false);
                        }
                    }
                    tmp2[i] = ConstantVector::get(tmp);
                    assert (tmp2[i]->getType() == vecTy);
                }
                tmp3[s] = ConstantArray::get(elementTy, tmp2);
            }
            vectors[p] = ConstantArray::get(patternTy, tmp3);
        }
    } else {
        // convert these to vec types too?

        IntegerType * const intTy = b->getIntNTy(fieldWidth);
        const auto totalSize = patternLength * 2ULL - 1ULL;
        std::vector<Constant *> array(totalSize);
        for (unsigned i = 0; i < numElements; ++i) {
            const auto & vec = Pattern[i];
            const auto L = vec.size();
            for (unsigned j = 0; j < totalSize; ++j) {
                const auto v = vec[j % L];
                if (LLVM_UNLIKELY(v > maxVal)) {
                    report_fatal_error(std::to_string(v) + " exceeds a " +
                                       std::to_string(fieldWidth) + "-bit value");
                }
                array[j] = ConstantInt::get(intTy, v);
            }
            vectors[i] = ConstantArray::get(ArrayType::get(intTy, totalSize), array);
        }
    }

    ArrayType * const patTy = ArrayType::get(vectors[0]->getType(), numElements);
    Constant * const patternVec = ConstantArray::get(patTy, vectors);

    Module & mod = *b->getModule();
    GlobalVariable * const patternData = new GlobalVariable(mod, patTy, true, GlobalValue::PrivateLinkage, patternVec);
    // TODO: this isn't being aligned as I expect? using unaligned memcpy as a temporary measure
    patternData->setAlignment(MaybeAlign{bw /8});

    ConstantInt * const sz_BlockWidth = b->getSize(bw);

    ConstantInt * const sz_strideFillSize = b->getSize(maxFillSize);

    Value * const produced = b->getProducedItemCount("output");

    Value * const consumed = b->CreateRoundDown(b->getConsumedItemCount("output"), sz_BlockWidth);

    Value * const baseAddress = outputBuffer->getBaseAddress(b);

    Value * const remaining = b->CreateSub(produced, consumed);

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value * const valid = b->CreateIsNull(b->CreateURem(remaining, sz_BlockWidth));
        b->CreateAssert(valid, "remaining was not a multiple of block width");
    }

    Value * const fillSize = b->CreateMul(sz_strideFillSize, b->getNumOfStrides());
    Value * const total = b->CreateAdd(fillSize, consumed); // produced + (fillSize - (produced - consumed))
    Value * const totalStrides = b->CreateExactUDiv(total, sz_BlockWidth);

    Value * const mustFill = b->CreateICmpULT(remaining, fillSize);

    BasicBlock * const entry = b->GetInsertBlock();
    b->CreateLikelyCondBr(mustFill, checkBuffer, exit);

    b->SetInsertPoint(checkBuffer);
    // Can we append to our existing buffer without impacting any subsequent kernel?
    Value * const effectiveCapacity = b->getScalarField("effectiveCapacity");
    Value * const permitted = b->CreateICmpULT(total, effectiveCapacity);
    BasicBlock * const checkBufferExit = b->GetInsertBlock();
    b->CreateLikelyCondBr(permitted, generateData, moveData);

    b->SetInsertPoint(moveData);
    Value * const consumedIndex = b->CreateExactUDiv(consumed, sz_BlockWidth);

    Value * const unreadData = outputBuffer->getStreamBlockPtr(b.get(), baseAddress, sz_ZERO, consumedIndex);
    Value * const baseBuffer = b->CreatePointerCast(b->getScalarField("buffer"), outputStreamSetPtrTy);

    FixedArray<Value *, 2> baseIndex;
    baseIndex[0] = sz_ZERO;
    baseIndex[1] = totalStrides;

    Value * const toWrite = b->CreateGEP(outputStreamSetTy, baseBuffer, baseIndex);
    Value * const canCopy = b->CreateICmpULT(toWrite, unreadData);
    Value * const capacity = b->getCapacity("output");
    // Have we consumed enough data that we can safely copy back the unconsumed data and still
    // leave enough space for one segment without needing a temporary buffer?

    const Rational bytesPerItem{fieldWidth * numElements, 8};
   // Value * const remainingStrides = b->CreateExactUDiv(remaining, sz_BlockWidth);
    Value * const remainingBytes = b->CreateMulRational(remaining, bytesPerItem);

    b->CreateLikelyCondBr(canCopy, copyBack, expandAndCopyBack);

    // If so, just copy the data ...
    b->SetInsertPoint(copyBack);
    b->CreateMemCpy(baseBuffer, unreadData, remainingBytes, 1);

    // Since our consumed count cannot exceed the effective capacity, in order for (consumed % capacity)
    // to be less than (effective capacity % capacity), we must have fully read all the data past the
    // effective capacity of the buffer. Thus we can set the effective capacity to the buffer capacity.
    // If, however, (consumed % capacity) >= (effective capacity % capacity), then we still have some
    // unconsumed data at the end of the buffer. Here, we can set the reclaimed capacity position to
    // (consumed % capacity).

    Value * const consumedModCap = b->CreateURem(consumed, capacity);
    Value * const effectiveCapacityModCap = b->CreateURem(effectiveCapacity, capacity);
    Value * const reclaimCapacity = b->CreateICmpULT(consumedModCap, effectiveCapacityModCap);
    Value * const reclaimedCapacity = b->CreateSelect(reclaimCapacity, capacity, consumedModCap);

    Value * const updatedEffectiveCapacity = b->CreateAdd(consumed, reclaimedCapacity);
    b->setScalarField("effectiveCapacity", updatedEffectiveCapacity);
    BasicBlock * const copyBackExit = b->GetInsertBlock();
    b->CreateBr(prepareBuffer);

    // Otherwise, allocate a buffer with twice the capacity and copy the unconsumed data back into it
    b->SetInsertPoint(expandAndCopyBack);
    Value * const expandedCapacity = b->CreateShl(capacity, 1);
    Value * const expandedBytes = b->CreateMulRational(expandedCapacity, bytesPerItem);

    Value * expandedBuffer = b->CreatePageAlignedMalloc(expandedBytes);

    b->CreateMemCpy(expandedBuffer, unreadData, remainingBytes, 1);
    // Free the prior buffer if it exists
    Value * const ancillaryBuffer = b->getScalarField("ancillaryBuffer");
    b->setScalarField("ancillaryBuffer", b->CreatePointerCast(baseBuffer, b->getVoidPtrTy()));
    b->CreateFree(ancillaryBuffer);
    b->setScalarField("buffer", expandedBuffer);
    b->setCapacity("output", expandedCapacity);
    Value * const expandedEffectiveCapacity = b->CreateAdd(consumed, expandedCapacity);
    b->setScalarField("effectiveCapacity", expandedEffectiveCapacity);
    expandedBuffer = b->CreatePointerCast(expandedBuffer, outputStreamSetPtrTy);
    BasicBlock * const expandAndCopyBackExit = b->GetInsertBlock();
    b->CreateBr(prepareBuffer);

    b->SetInsertPoint(prepareBuffer);
    PHINode * const newBaseBuffer = b->CreatePHI(outputStreamSetPtrTy, 2);
    newBaseBuffer->addIncoming(baseBuffer, copyBackExit);
    newBaseBuffer->addIncoming(expandedBuffer, expandAndCopyBackExit);

    Value * const newBaseAddress = b->CreateGEP(outputStreamSetTy, newBaseBuffer, b->CreateNeg(consumedIndex));
    assert (newBaseAddress->getType() == baseAddress->getType());

    b->setBaseAddress("output", newBaseAddress);
    BasicBlock * const prepareBufferExit = b->GetInsertBlock();
    b->CreateBr(generateData);

    b->SetInsertPoint(generateData);
    PHINode * const pos = b->CreatePHI(b->getSizeTy(), 3);
    pos->addIncoming(produced, checkBufferExit);
    pos->addIncoming(produced, prepareBufferExit);
    PHINode * const ba = b->CreatePHI(baseAddress->getType(), 3);
    ba->addIncoming(baseAddress, checkBufferExit);
    ba->addIncoming(newBaseAddress, prepareBufferExit);

    FixedArray<Value *,3> offset;
    offset[0] = b->getInt32(0);

    Value * const currentIndex = b->CreateExactUDiv(pos, sz_BlockWidth);

    ConstantInt * const copySize = b->getSize((fieldWidth * bw) / 8);

    // b->CreateAssert(b->CreateICmpEQ(sizeOfOutputStreamSetTy, sz_StripLength), "unexpected data size?");

    for (unsigned i = 0; i < numElements; ++i) {
        ConstantInt * const elementIndex = b->getInt32(i);
        const auto & vec = Pattern[i];
        offset[1] = elementIndex;
        offset[2] = b->CreateURem(pos, b->getSize(vec.size()));
        Value * const src = b->CreateGEP(patternData, offset);
        Value * const dst = outputBuffer->getStreamBlockPtr(b.get(), ba, elementIndex, currentIndex);
        b->CreateMemCpy(dst, src, copySize, 1);
    }

    Value * const nextProduced = b->CreateAdd(pos, sz_BlockWidth);
    BasicBlock * const generateDataExit = b->GetInsertBlock();
    pos->addIncoming(nextProduced, generateDataExit);
    ba->addIncoming(ba, generateDataExit);
    b->CreateCondBr(b->CreateICmpNE(nextProduced, total), generateData, exit);

    b->SetInsertPoint(exit);
    PHINode * const finalProduced = b->CreatePHI(b->getSizeTy(), 2);
    finalProduced->addIncoming(produced, entry);
    finalProduced->addIncoming(total, generateData);
    b->setProducedItemCount("output", finalProduced);

    Value * const finishedGenerating = b->CreateICmpUGE(finalProduced, b->getSize(REPETITION_LENGTH));
    ConstantInt * const none = b->getSize(KernelBuilder::TerminationCode::None);
    ConstantInt * const term = b->getSize(KernelBuilder::TerminationCode::Terminated);
    Value * const done = b->CreateSelect(finishedGenerating, term, none);
    b->setTerminationSignal(done);

}

void RepeatingSourceKernel::generateFinalizeMethod(BuilderRef b) {
    b->CreateFree(b->getScalarField("ancillaryBuffer"));
    b->CreateFree(b->getScalarField("buffer"));
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
    const uint32_t ITEMS_PER_BLOCK = b->getBitBlockWidth() / FW;

    BasicBlock * const entryBlock = b->GetInsertBlock();
    BasicBlock * const loopBlock = b->CreateBasicBlock("loop");
    BasicBlock * const exitBlock = b->CreateBasicBlock("exit");
    Value * const hasMoreItems = b->CreateICmpNE(b->getAccessibleItemCount("lhs"), b->getSize(0));
    Value * initialOffset = nullptr;
    if (FW != 1) {
        initialOffset = b->CreateUDiv(b->getProcessedItemCount("lhs"), b->getSize(ITEMS_PER_BLOCK));
    }
    b->CreateLikelyCondBr(hasMoreItems, loopBlock, exitBlock);

    b->SetInsertPoint(loopBlock);
    PHINode * strideNo = b->CreatePHI(b->getSizeTy(), 2);
    strideNo->addIncoming(b->getSize(0), entryBlock);
    Value * blockOffset = nullptr;
    if (FW != 1) {
        blockOffset = b->CreateAdd(strideNo, initialOffset);
    }

    for (uint32_t i = 0; i < COUNT; ++i) {
        Value * lhs;
        Value * rhs;
        if (FW == 1) {
            lhs = b->loadInputStreamBlock("lhs", b->getInt32(i), strideNo);
            rhs = b->loadInputStreamBlock("rhs", b->getInt32(i), strideNo);
        } else {

            lhs = b->getInputStreamPackPtr("lhs", b->getInt32(i), blockOffset);
            rhs = b->getInputStreamPackPtr("rhs", b->getInt32(i), blockOffset);

            b->CallPrintInt("lhs", lhs);
            b->CallPrintInt("rhs", rhs);

            lhs = b->loadInputStreamPack("lhs", b->getInt32(i), blockOffset);
            rhs = b->loadInputStreamPack("rhs", b->getInt32(i), blockOffset);
        }
        // Perform vector comparison lhs != rhs.
        // Result will be a vector of all zeros if lhs == rhs
        Value * const vComp = b->CreateICmpNE(lhs, rhs);
        Value * const vCompAsInt = b->CreateBitCast(vComp, b->getIntNTy(cast<IDISA::FixedVectorType>(vComp->getType())->getNumElements()));
        // `comp` will be `true` iff lhs == rhs (i.e., `vComp` is a vector of all zeros)
        Value * const comp = b->CreateICmpEQ(vCompAsInt, Constant::getNullValue(vCompAsInt->getType()));
        // `and` `comp` into `accum` so that `accum` will be `true` iff lhs == rhs for all blocks in the two streams
        Value * const accum = b->getScalarField("accum");
        Value * const newAccum = b->CreateAnd(accum, comp);
        b->setScalarField("accum", newAccum);
    }
    Value * const nextStrideNo = b->CreateAdd(strideNo, b->getSize(1));
    strideNo->addIncoming(nextStrideNo, loopBlock);
    b->CreateCondBr(b->CreateICmpNE(nextStrideNo, numOfStrides), loopBlock, exitBlock);

    b->SetInsertPoint(exitBlock);
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

typedef void (*TestFunctionType)(uint32_t * output);


TestFunctionType buildRepeatingStreamSetTest(CPUDriver & pxDriver,
                                             const unsigned fieldWidth,
                                             const unsigned numOfElements,
                                             const unsigned patternLength) {

    auto & b = pxDriver.getBuilder();

    auto P = pxDriver.makePipeline({Binding{b->getInt32Ty()->getPointerTo(), "output"}},{});

    // bindings.push_back({b->getInt32Ty()->getPointerTo(), "output"});

    std::random_device rd;
    std::default_random_engine rng(rd());
    std::uniform_int_distribution<uint64_t> dist(0ULL, (1ULL << static_cast<uint64_t>(fieldWidth)) - 1ULL);

    std::vector<std::vector<uint64_t>> pattern(numOfElements);
    for (unsigned i = 0; i < numOfElements; ++i) {
        auto & vec = pattern[i];
        vec.resize(patternLength);
        for (unsigned j = 0; j < patternLength; ++j) {
            vec[j] = dist(rng);
        }
    }

    RepeatingStreamSet * const RepeatingStream = P->CreateRepeatingStreamSet(fieldWidth, pattern);

    StreamSet * const Output = P->CreateStreamSet(numOfElements, fieldWidth);

    P->CreateKernelCall<RepeatingSourceKernel>(pattern, Output);

    P->CreateKernelCall<StreamEq>(RepeatingStream, Output, P->getInputScalar("output"));

    return reinterpret_cast<TestFunctionType>(P->compile());
}


int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {});
    CPUDriver pxDriver("test");
    const auto f = buildRepeatingStreamSetTest(pxDriver, 8, 1, 73);
    uint32_t result = 0;
    f(&result);
    return result;
}
