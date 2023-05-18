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

    Constant * const FIELD_WIDTH = b->getSize(fw);

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
            b->CallPrintInt("maskPos", maskPos);
            Value * const mask = b->CreateNot(b->bitblock_mask_from(maskPos));
            b->CallPrintRegister("mask", mask);
            Value * val = b->CreateAnd(b->CreateAlignedLoad(inputPtr, bw / 8), mask);
            b->CallPrintRegister("val", val);
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
    Value * const avail = b->getAvailableItemCount("input");
    b->CreateLikelyCondBr(b->CreateICmpULE(avail, upTo), segmentExit, termKernel);

    b->SetInsertPoint(termKernel);
    b->setProducedItemCount("output", upTo);
    b->setTerminationSignal();
    b->CreateBr(segmentExit);

    b->SetInsertPoint(segmentExit);
}

