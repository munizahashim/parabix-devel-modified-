/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <testing/assert.h>

#include <kernel/core/kernel_builder.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/pipeline/program_builder.h>

using namespace llvm;
using namespace kernel;

namespace kernel {

inline std::string KernelName(StreamEquivalenceKernel::Mode mode, StreamSet * x, StreamSet * y) {
    std::string backing;
    raw_string_ostream str(backing);
    str << "Stream"
        << (mode == StreamEquivalenceKernel::Mode::EQ ? "EQ" : "NE")
        << x->getNumElements() << 'x' << x->getFieldWidth();
    return str.str();
}

StreamEquivalenceKernel::StreamEquivalenceKernel(LLVMTypeSystemInterface & ts,
    Mode mode,
    StreamSet * lhs,
    StreamSet * rhs,
    Scalar * outPtr)
: MultiBlockKernel(ts, KernelName(mode, lhs, rhs),
    {{"lhs", lhs}, {"rhs", rhs}},
    {},
    {{"result_ptr", outPtr}},
    {},
    {InternalScalar(ts.getInt1Ty(), "anyNonMatch")})
, mMode(mode)
{
    assert(lhs->getFieldWidth() == rhs->getFieldWidth());
    assert(lhs->getNumElements() == rhs->getNumElements());
    addAttribute(SideEffecting());
}

void StreamEquivalenceKernel::generateInitializeMethod(KernelBuilder & b) {
    // b.setScalarField("anyNonMatch", b.getInt1(mMode == Mode::EQ));
}

void StreamEquivalenceKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    const StreamSet * const lhs = b.getInputStreamSet("lhs");
    const auto fieldWidth = lhs->getFieldWidth();
    const auto numElements = lhs->getNumElements();

    const StreamSet * const rhs = b.getInputStreamSet("lhs");
    if (rhs->getFieldWidth() != fieldWidth || rhs->getNumElements() != numElements) {
        report_fatal_error("StreamEquivalenceKernel error: lhs field size and element count does not match rhs");
    }

    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const loopBlock = b.CreateBasicBlock("loop");
    BasicBlock * const exitBlock = b.CreateBasicBlock("exit");

    Value * const initialAccum = b.getScalarField("anyNonMatch");
    Constant * const sz_ZERO = b.getSize(0);
    Constant * const sz_ONE = b.getSize(1);

    const auto m = std::max(std::max(fieldWidth, numElements), 2U) + 1;
    assert (m > 1);

    std::vector<ConstantInt *> IDX(m);

    for (unsigned i = 0; i < m; ++i) {
        IDX[i] = b.getInt32(i);
    }

    IntegerType * intVecTy = b.getIntNTy(cast<FixedVectorType>(b.getBitBlockType())->getNumElements());

    b.CreateBr(loopBlock);

    b.SetInsertPoint(loopBlock);
    PHINode * const strideNo = b.CreatePHI(b.getSizeTy(), 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode * const accumPhi = b.CreatePHI(b.getInt1Ty(), 2);
    accumPhi->addIncoming(initialAccum, entryBlock);
    Value * nextAccum = accumPhi;
    for (unsigned i = 0; i < numElements; ++i) {
        for (unsigned j = 0; j < fieldWidth; ++j) {
            Value * lhs;
            Value * rhs;
            if (fieldWidth == 1) {
                lhs = b.loadInputStreamBlock("lhs", IDX[i], strideNo);
                rhs = b.loadInputStreamBlock("rhs", IDX[i], strideNo);
            } else {
                lhs = b.loadInputStreamPack("lhs", IDX[i], IDX[j], strideNo);
                rhs = b.loadInputStreamPack("rhs", IDX[i], IDX[j], strideNo);
            }
            b.CallPrintRegister("lhs", lhs);
            b.CallPrintRegister("rhs", rhs);
            Value * const nonMatches = b.CreateICmpNE(lhs, rhs);
            assert (intVecTy->getIntegerBitWidth() == cast<FixedVectorType>(nonMatches->getType())->getNumElements());
            Value * anyNonMatch = b.CreateIsNotNull(b.CreateBitCast(nonMatches, intVecTy));
            b.CallPrintInt("anyNonMatch", anyNonMatch);
            nextAccum = b.CreateOr(nextAccum, anyNonMatch);
        }
    }

    Value * const nextStrideNo = b.CreateAdd(strideNo, sz_ONE);
    strideNo->addIncoming(nextStrideNo, loopBlock);
    accumPhi->addIncoming(nextAccum, loopBlock);
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), loopBlock, exitBlock);

    b.SetInsertPoint(exitBlock);
    b.setScalarField("anyNonMatch", nextAccum);
}

void StreamEquivalenceKernel::generateFinalizeMethod(KernelBuilder & b) {
    // a `result` value of `true` means the assertion passed
    Value * anyNonMatch = b.getScalarField("anyNonMatch");
    if (mMode == Mode::EQ) {
        anyNonMatch = b.CreateNot(anyNonMatch);
    }


    // A `ptrVal` value of `0` means that the test is currently passing and a
    // value of `1` means the test is failing. If the test is already failing,
    // then we don't need to update the test state.
    Value * resultPtr = b.getScalarField("result_ptr");


    Value * const ptrVal = b.CreateLoad(b.getInt32Ty(), resultPtr);
    Value * resultState = b.CreateSelect(anyNonMatch, ptrVal, b.getInt32(1));
    b.CreateStore(resultState, resultPtr);

//    if (mMode == Mode::EQ) {
//        // ptrVal is initially zero
//        resultState = b.CreateSelect(anyNonMatch, ptrVal, b.getInt32(1));
//    } else {
//        // To preserve commutativity of `NE` comparisons, two additional test
//        // states are needed. State `2` represents a partial passing `NE`
//        // comparison and state `3` represents a partial failing `NE` comparison.
//        // `AssertNE` first checks `A != B` putting the test into a parital
//        // state (`2` if the comparison returns `true` or `3` if it returns `false`).
//        // The second comparison `B != A` resolves the partial state. If the second
//        // comparison returns `false` and the first comparison did as well (i.e.,
//        // the test is in state `3`) then the test is put into a failing state.
//        // Otherwise, if either of the tests returned `true` the total assertion
//        // passed.
//        Value * const isParitalState = b.CreateOr(b.CreateICmpEQ(ptrVal, b.getInt32(2)), b.CreateICmpEQ(ptrVal, b.getInt32(3)));
//        Value * const resolveToFail = b.CreateAnd(b.CreateICmpEQ(ptrVal, b.getInt32(3)), b.CreateNot(anyNonMatch));
//        resultState = b.CreateSelect(
//            isParitalState,
//            b.CreateSelect(resolveToFail, b.getInt32(1), b.getInt32(0)),
//            b.CreateSelect(anyNonMatch, b.getInt32(2), b.getInt32(3))
//        );
//    }
//    Value * const newVal = b.CreateSelect(b.CreateICmpEQ(ptrVal, b.getInt32(1)), b.getInt32(1), resultState);
//    b.CreateStore(newVal, resultPtr);
}

} // namespace kernel

namespace testing {

void AssertEQ(kernel::PipelineBuilder & P, StreamSet * lhs, StreamSet * rhs) {
    auto ptr = P.getInputScalar("output");
    // given equal length inputs, both LHS and RHS are equivalent
    P.CreateKernelCall<StreamEquivalenceKernel>(StreamEquivalenceKernel::Mode::EQ, lhs, rhs, ptr);
//    P.CreateKernelCall<StreamEquivalenceKernel>(StreamEquivalenceKernel::Mode::EQ, rhs, lhs, ptr);
}

void AssertNE(kernel::PipelineBuilder & P, StreamSet * lhs, StreamSet * rhs) {
    auto ptr = P.getInputScalar("output");
    P.CreateKernelCall<StreamEquivalenceKernel>(StreamEquivalenceKernel::Mode::NE, lhs, rhs, ptr);
//    P.CreateKernelCall<StreamEquivalenceKernel>(StreamEquivalenceKernel::Mode::NE, rhs, lhs, ptr);
}

} // namespace testing
