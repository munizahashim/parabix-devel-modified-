/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include "editd_cpu_kernel.h"
#include <kernel/core/kernel_builder.h>
#include <llvm/IR/Module.h>

using namespace llvm;

namespace kernel {

void editdCPUKernel::bitblock_advance_ci_co(KernelBuilder & b,
                                            Value * val, unsigned shift, Value * strideCarryArr, unsigned carryIdx,
                                            std::vector<std::vector<Value *>> & adv,
                                            std::vector<std::vector<int>> & calculated, int i, int j) const {
    if (calculated[i][j] == 0) {
        FixedArray<Value *, 2> indices;
        indices[0] = b.getInt32(0);
        indices[1] = b.getInt32(carryIdx);
        Type * bbTy = b.getBitBlockType();
        Value * ptr = b.CreateGEP(bbTy, strideCarryArr, indices);
        Value * ci = b.CreateLoad(bbTy, ptr);
        std::pair<Value *, Value *> rslt = b.bitblock_advance(val, ci, shift);
        b.CreateStore(std::get<0>(rslt), ptr);
        adv[i][j] = std::get<1>(rslt);
        calculated[i][j] = 1;
    }
}

void editdCPUKernel::reset_to_zero(std::vector<std::vector<int>> & calculated){
    for (auto & sub : calculated) {
        std::fill(sub.begin(), sub.end(), 0);
    }
}

void editdCPUKernel::generateDoBlockMethod(KernelBuilder & b) {

    Type * const int32ty = b.getInt32Ty();
    Type * const int8ty = b.getInt8Ty();

    Value * pattStartPtr = b.getScalarField("pattStream");
    Value * strideCarryArr = b.getScalarField("strideCarry");

    unsigned carryIdx = 0;

    std::vector<std::vector<Value *>> e(mPatternLen + 1, std::vector<Value *>(mEditDistance + 1));
    std::vector<std::vector<Value *>> adv(mPatternLen, std::vector<Value *>(mEditDistance + 1));
    std::vector<std::vector<int>> calculated(mPatternLen, std::vector<int>(mEditDistance + 1, 0));
    Value * pattPos = b.getInt32(0);
    for(unsigned j = 0; j <= mEditDistance; j++){
        e[mPatternLen][j] = b.allZeroes();
    }

    for(unsigned j = 1; j <= mEditDistance; j++){
        e[0][j] = b.allOnes();
    }

    Type * bbTy = b.getBitBlockType();

    for(unsigned g = 0; g < mGroupSize; g++){
        Value * pattCh = b.CreateLoad(bbTy, b.CreateGEP(bbTy, pattStartPtr, pattPos));
        Value * pattIdx = b.CreateAnd(b.CreateLShr(pattCh, 1), ConstantInt::get(int8ty, 3));
        Value * pattStream = b.loadInputStreamBlock("CCStream", b.CreateZExt(pattIdx, int32ty));
        pattPos = b.CreateAdd(pattPos, ConstantInt::get(int32ty, 1));

        e[0][0] = pattStream;
        for(unsigned i = 1; i < mPatternLen; i++){
            pattCh = b.CreateLoad(bbTy, b.CreateGEP(bbTy, pattStartPtr, pattPos));
            pattIdx = b.CreateAnd(b.CreateLShr(pattCh, 1), ConstantInt::get(int8ty, 3));
            pattStream = b.loadInputStreamBlock("CCStream", b.CreateZExt(pattIdx, int32ty));
            pattPos = b.CreateAdd(pattPos, ConstantInt::get(int32ty, 1));
            bitblock_advance_ci_co(b, e[i-1][0], 1, strideCarryArr, carryIdx++, adv, calculated, i-1, 0);
            e[i][0] = b.CreateAnd(adv[i-1][0], pattStream);
            for(unsigned j = 1; j<= mEditDistance; j++){
                bitblock_advance_ci_co(b, e[i-1][j], 1, strideCarryArr, carryIdx++, adv, calculated, i-1, j);
                bitblock_advance_ci_co(b, e[i-1][j-1], 1, strideCarryArr, carryIdx++, adv, calculated, i-1, j-1);
                bitblock_advance_ci_co(b, e[i][j-1], 1, strideCarryArr, carryIdx++, adv, calculated, i, j-1);
                Value * tmp1 = b.CreateAnd(adv[i-1][j], pattStream);
                Value * tmp2 = b.CreateAnd(adv[i-1][j-1], b.CreateNot(pattStream));
                Value * tmp3 = b.CreateOr(adv[i][j-1], e[i-1][j-1]);
                e[i][j] = b.CreateOr(b.CreateOr(tmp1, tmp2), tmp3);
            }
        }
        e[mPatternLen][0] = b.CreateOr(e[mPatternLen][0], e[mPatternLen-1][0]);
        for(unsigned j = 1; j<= mEditDistance; j++){
            e[mPatternLen][j] = b.CreateOr(e[mPatternLen][j], b.CreateAnd(e[mPatternLen - 1][j], b.CreateNot(e[mPatternLen - 1][j - 1])));
        }
        reset_to_zero(calculated);
    }

    for(unsigned j = 0; j<= mEditDistance; j++){
        b.storeOutputStreamBlock("ResultStream", b.getInt32(j), e[mPatternLen][j]);
    }
}

void editdCPUKernel::generateFinalBlockMethod(KernelBuilder & b, Value * remainingBytes) {
    b.setScalarField("EOFmask", b.bitblock_mask_from(remainingBytes));
    RepeatDoBlockLogic(b);
}



editdCPUKernel::editdCPUKernel(KernelBuilder & b,
                               const unsigned editDistance,
                               const unsigned patternLen,
                               const unsigned groupSize,
                               Scalar * const pattStream,
                               StreamSet * const CCStream, StreamSet * const ResultStream)
: BlockOrientedKernel(b, "EditDistanceCPU" + std::to_string(patternLen) + ":" + std::to_string(groupSize),
// input stream
{Binding{"CCStream", CCStream}},
// output stream
{Binding{"ResultStream", ResultStream}},
// input scalar
{Binding{"pattStream", pattStream}},
// output scalar
{},
// internal scalars
{InternalScalar{ScalarType::NonPersistent, b.getBitBlockType(), "EOFmask"},
 InternalScalar{ScalarType::Internal, ArrayType::get(b.getBitBlockType(), (patternLen * groupSize * 4 * ResultStream->getNumElements())), "strideCarry"}})
, mEditDistance(editDistance)
, mPatternLen(patternLen)
, mGroupSize(groupSize) {

}

}
