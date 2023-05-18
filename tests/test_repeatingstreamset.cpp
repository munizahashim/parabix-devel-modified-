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

constexpr auto REPETITION_LENGTH = 5563ULL;

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
    BasicBlock * const finishedDataLoop = b->CreateBasicBlock("finishedDataLoop");
    BasicBlock * const zeroExtraneousBytes = b->CreateBasicBlock("zeroExtraneousBytes");
    BasicBlock * const exit = b->CreateBasicBlock("exit");

    // build our pattern array
    const auto output = b->getOutputStreamSet("output");
    const auto fieldWidth = output->getFieldWidth();
    const auto numElements = output->getNumElements();

    const auto blockWidth = b->getBitBlockWidth();

    const auto maxVal = (1ULL << static_cast<uint64_t>(fieldWidth)) - 1ULL;

    // size_t patternLength = blockWidth;
    size_t maxPatternSize = 0;
    for (unsigned i = 0; i < numElements; ++i) {
        const auto & vec = Pattern[i];
        for (auto v : vec) {
            if (LLVM_UNLIKELY(v > maxVal)) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream msg(tmp);
                msg << "Value " << v << " exceeds a " << fieldWidth << "-bit value";
                report_fatal_error(msg.str());
            }
        }
        maxPatternSize = std::max(maxPatternSize, vec.size());
    }
    const Binding & binding = b->getOutputStreamSetBinding("output");
    const ProcessingRate & rate = binding.getRate();
    const auto fs = getStride() * rate.getUpperBound();
    assert (fs.denominator() == 1);
    const auto maxFillSize = fs.numerator();
    if (maxPatternSize > maxFillSize) {
        report_fatal_error("output rate should at least be as large as the pattern length");
    }


    if (fieldWidth > blockWidth) {
        report_fatal_error("does not support field width sizes above " + std::to_string(blockWidth));
    }
    if ((maxFillSize % blockWidth) != 0) {
        report_fatal_error("output rate should be a multiple of " + std::to_string(blockWidth)
                           + " to ensure proper streamset construction");
    }

    StreamSetBuffer * const outputBuffer = b->getOutputStreamSetBuffer("output");
    PointerType * const outputStreamSetPtrTy = outputBuffer->getPointerType();    
    Type * const outputStreamSetTy = outputStreamSetPtrTy->getPointerElementType();


    ConstantInt * const sz_ZERO = b->getSize(0);

    FixedVectorType * const vecTy = b->getBitBlockType();
    IntegerType * const intTy = cast<IntegerType>(vecTy->getScalarType());
    const auto laneWidth = intTy->getIntegerBitWidth();
    const auto numLanes = blockWidth / laneWidth;
    ArrayType * const elementTy = ArrayType::get(vecTy, fieldWidth);

    SmallVector<Constant *, 16> laneVal(numLanes);
    SmallVector<Constant *, 16> packVal(fieldWidth);
    SmallVector<GlobalVariable *, 16> streamVal(numElements);

    Module & mod = *b->getModule();

    for (unsigned p = 0; p < numElements; ++p) {
        const auto & vec = Pattern[p];
        const auto L = vec.size();
        const auto patternLength = boost::lcm<size_t>(blockWidth, L);
        const auto runLength = (patternLength / blockWidth);

        std::vector<Constant *> dataVectorArray(runLength);

        uint64_t pos = 0;
        for (unsigned r = 0; r < runLength; ++r) {
            for (uint64_t i = 0; i < fieldWidth; ++i) {
                for (uint64_t j = 0; j < numLanes; ++j) {
                    uint64_t V = 0;
                    for (uint64_t k = 0; k != laneWidth; k += fieldWidth) {
                        const auto v = vec[pos % L];
                        V |= (v << k);
                        ++pos;
                    }
                    laneVal[j] = ConstantInt::get(intTy, V, false);
                }
                packVal[i] = ConstantVector::get(laneVal);
            }
            dataVectorArray[r] = ConstantArray::get(elementTy, packVal);
        }

        ArrayType * const streamTy = ArrayType::get(elementTy, runLength);
        Constant * const patternVal = ConstantArray::get(streamTy, dataVectorArray);
        GlobalVariable * const gv = new GlobalVariable(mod, streamTy, true, GlobalValue::PrivateLinkage, patternVal);
        gv->setAlignment(MaybeAlign{blockWidth /8});

        streamVal[p] = gv;
    }

    ConstantInt * const sz_BlockWidth = b->getSize(blockWidth);

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

    FixedArray<Value *,2> offset;
    offset[0] = sz_ZERO;

    Value * const currentIndex = b->CreateExactUDiv(pos, sz_BlockWidth);
    const auto length = (fieldWidth * blockWidth) / 8;
    ConstantInt * const elementSize = b->getSize(length);
    for (unsigned i = 0; i < numElements; ++i) {
        const auto patternLength = boost::lcm<size_t>(blockWidth, Pattern[i].size());
        const auto runLength = (patternLength / blockWidth);
        offset[1] = b->CreateURem(currentIndex, b->getSize(runLength));
        Value * const src = b->CreateGEP(streamVal[i], offset);
        Value * const dst = outputBuffer->getStreamBlockPtr(b.get(), ba, b->getInt32(i), currentIndex);
        b->CreateMemCpy(dst, src, elementSize, blockWidth / 8);
    }

    Value * const nextProduced = b->CreateAdd(pos, sz_BlockWidth);
    BasicBlock * const generateDataExit = b->GetInsertBlock();
    pos->addIncoming(nextProduced, generateDataExit);
    ba->addIncoming(ba, generateDataExit);
    b->CreateCondBr(b->CreateICmpNE(nextProduced, total), generateData, finishedDataLoop);

    b->SetInsertPoint(finishedDataLoop);
    Value * const finalProduced = nextProduced;
    b->setProducedItemCount("output", finalProduced);
    Constant * const MAX = b->getSize(REPETITION_LENGTH);
    Value * const finishedGenerating = b->CreateICmpUGE(finalProduced, MAX);
    b->CreateUnlikelyCondBr(finishedGenerating, zeroExtraneousBytes, exit);

    b->SetInsertPoint(zeroExtraneousBytes);
    b->setProducedItemCount("output", MAX);
    b->setTerminationSignal();

    Constant * const startIndex = ConstantExpr::getUDiv(MAX, sz_BlockWidth);

    ConstantInt * const sz_BlockMask = b->getSize(blockWidth - 1U);
    ConstantInt * const sz_FieldWidth = b->getSize(fieldWidth);

    Value * packIndex = nullptr;
    Value * maskOffset = b->CreateAnd(MAX, sz_BlockMask);
    if (fieldWidth > 1) {
        Value * const position = b->CreateMul(maskOffset, sz_FieldWidth);
        packIndex = b->CreateUDiv(position, sz_BlockWidth);
        maskOffset = b->CreateAnd(position, sz_BlockMask);
    }

    Value * const mask = b->CreateNot(b->bitblock_mask_from(maskOffset));
    for (unsigned i = 0; i < numElements; ++i) {
        Value * ptr = nullptr;
        if (fieldWidth == 1) {
            ptr = outputBuffer->getStreamBlockPtr(b.get(), ba, b->getInt32(i), startIndex);
        } else {
            ptr = outputBuffer->getStreamPackPtr(b.get(), ba, b->getInt32(i), startIndex, packIndex);
        }
        Value * const val = b->CreateBlockAlignedLoad(ptr);
        Value * const maskedVal = b->CreateAnd(val, mask);
        b->CreateBlockAlignedStore(maskedVal, ptr);
    }

    ConstantInt * const sz_ONE = b->getSize(1);

    if (fieldWidth > 1) {
        BasicBlock * const clearRemainingPacks = b->CreateBasicBlock("clearRemainingPacks", exit);
        BasicBlock * const clearRemainingPacksExit = b->CreateBasicBlock("clearRemainingPacksExit", exit);

        Constant * const vec_ZERO = ConstantVector::getNullValue(b->getBitBlockType());

        Value * const firstPackIndex = b->CreateAdd(packIndex, sz_ONE);

        b->CreateCondBr(b->CreateICmpULT(firstPackIndex, sz_FieldWidth), clearRemainingPacks, clearRemainingPacksExit);

        b->SetInsertPoint(clearRemainingPacks);
        PHINode * const packIndexPhi = b->CreatePHI(b->getSizeTy(), 2);
        packIndexPhi->addIncoming(firstPackIndex, zeroExtraneousBytes);
        for (unsigned i = 0; i < numElements; ++i) {
            Value * ptr = outputBuffer->getStreamPackPtr(b.get(), ba, b->getInt32(i), startIndex, packIndexPhi);
            b->CreateBlockAlignedStore(vec_ZERO, ptr);
        }
        Value * const nextPackIndex = b->CreateAdd(packIndexPhi, sz_ONE);
        packIndexPhi->addIncoming(nextPackIndex, clearRemainingPacks);
        b->CreateCondBr(b->CreateICmpULT(nextPackIndex, sz_FieldWidth), clearRemainingPacks, clearRemainingPacksExit);

        b->SetInsertPoint(clearRemainingPacksExit);
    }

    Value * const nextIndex = b->CreateAdd(startIndex, sz_ONE);
    Value * const endIndex = b->CreateAdd(b->CreateUDiv(total, sz_BlockWidth), sz_ONE);
    Value * const startPtr = outputBuffer->getStreamBlockPtr(b.get(), ba, sz_ZERO, nextIndex);
    Value * const endPtr = outputBuffer->getStreamBlockPtr(b.get(), ba, sz_ZERO, endIndex);
    DataLayout DL(b->getModule());
    Type * const intPtrTy = DL.getIntPtrType(startPtr->getType());
    Value * const startPtrInt = b->CreatePtrToInt(startPtr, intPtrTy);
    Value * const endPtrInt = b->CreatePtrToInt(endPtr, intPtrTy);
    Value * const numBytes = b->CreateSub(endPtrInt, startPtrInt);
    b->CreateMemZero(startPtr, numBytes, blockWidth / 8);

    b->CreateBr(exit);


    b->SetInsertPoint(exit);

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
    //    b->CallPrintRegister("lhs", lhs);
    //    b->CallPrintRegister("rhs", rhs);

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

typedef void (*TestFunctionType)(uint32_t * output);

const static bool verbose = false;

bool runRepeatingStreamSetTest(CPUDriver & pxDriver, std::default_random_engine & rng) {

    auto & b = pxDriver.getBuilder();

    auto P = pxDriver.makePipeline({Binding{b->getInt32Ty()->getPointerTo(), "output"}},{});

    std::uniform_int_distribution<uint64_t> fwDist(0, 5);

    const auto fieldWidth = 1ULL << fwDist(rng);

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

    RepeatingStreamSet * const RepeatingStream = P->CreateRepeatingStreamSet(fieldWidth, pattern);

    StreamSet * const Output = P->CreateStreamSet(numElements, fieldWidth);

    P->CreateKernelCall<RepeatingSourceKernel>(pattern, Output);

    Scalar * output = P->getInputScalar("output");

    P->CreateKernelCall<StreamEq>(RepeatingStream, Output, output);

    P->CreateKernelCall<StreamEq>(Output, RepeatingStream, output);

    const auto f = reinterpret_cast<TestFunctionType>(P->compile());

    uint32_t result = 0;
    f(&result);

    if (result != 0 || verbose) {

        llvm::errs() << "TEST: " << numElements << 'x' << fieldWidth << 'w' << patternLength << " : ";

        char joiner = '[';

        for (unsigned i = 0; i < numElements; ++i) {
            auto & vec = pattern[i];
            llvm::errs() << joiner;
            joiner = '{';
            for (unsigned j = 0; j < patternLength; ++j) {
                llvm::errs() << joiner << vec[j];
                joiner = ',';
            }
            llvm::errs() << '}';
        }

        llvm::errs() << "] -- ";
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
    //for (unsigned rounds = 0; rounds < 10; ++rounds) {
        testResult |= runRepeatingStreamSetTest(pxDriver, rng);
    //}
    return testResult ? -1 : 0;
}
