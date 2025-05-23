/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_builder.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Module.h>
#include <llvm/ADT/APInt.h>
#include <llvm/Support/raw_ostream.h>
#include <toolchain/toolchain.h>
#include <unistd.h>
#include <boost/intrusive/detail/math.hpp>

using boost::intrusive::detail::floor_log2;

using namespace llvm;

namespace IDISA {

bool isStreamTy(const Type * const t) {
    return isa<FixedVectorType>(t) && (cast<FixedVectorType>(t)->getNumElements() == 0);
}

bool isStreamSetTy(const Type * const t) {
    return t->isArrayTy() && (isStreamTy(t->getArrayElementType()));
}

unsigned getNumOfStreams (const Type * const t) {
    if (isStreamTy(t)) return 1;
    assert(isStreamSetTy(t));
    return cast<ArrayType>(t)->getNumElements();
}

unsigned getStreamFieldWidth (const Type * const t) {
    if (isStreamTy(t)) return t->getScalarSizeInBits();
    assert(isStreamSetTy(t));
    return cast<ArrayType>(t)->getElementType()->getScalarSizeInBits();
}

unsigned getVectorBitWidth(Value * a) {
    Type * aTy = a->getType();
    if (isa<IntegerType>(aTy)) return aTy->getPrimitiveSizeInBits();
    return cast<FixedVectorType>(aTy)->getPrimitiveSizeInBits();
}

FixedVectorType * IDISA_Builder::fwVectorType(const unsigned fw) {
    return FixedVectorType::get(getIntNTy(fw), mBitBlockWidth / fw);
}

Value * IDISA_Builder::fwCast(const unsigned fw, Value * const a) {
    unsigned vecWidth = getVectorBitWidth(a);
    Type * vecTy = FixedVectorType::get(getIntNTy(fw), vecWidth / fw);
    if (a->getType() == vecTy) return a;
    return CreateBitCast(a, vecTy);
}

[[noreturn]] void IDISA_Builder::UnsupportedFieldWidthError(const unsigned fw, std::string op_name) {
    report_fatal_error(StringRef(op_name) + ": Unsupported field width: " +  std::to_string(fw));
}

CallInst * IDISA_Builder::CallPrintRegister(StringRef name, Value * const value, const STD_FD fd) {
    Module * const m = getModule();
    Function * printRegister = m->getFunction("print_register");
    if (LLVM_UNLIKELY(printRegister == nullptr)) {
        FunctionType *FT = FunctionType::get(getVoidTy(), { getInt32Ty(), getInt8PtrTy(0), getBitBlockType() }, false);
        Function * function = Function::Create(FT, Function::InternalLinkage, "print_register", m);
        auto arg = function->arg_begin();
        std::string tmp;
        raw_string_ostream out(tmp);
        out << "%-40s =";
        for(unsigned i = 0; i < (getBitBlockWidth() / 8); ++i) {
            out << " %02" PRIx32;
        }
        out << '\n';
        BasicBlock * entry = BasicBlock::Create(m->getContext(), "entry", function);
        IRBuilder<> builder(entry);
        Value * const fdInt = &*(arg++);
        Value * const name = &*(arg++);
        name->setName("name");
        Value * value = &*arg;
        value->setName("value");
        Type * const byteFixedVectorType = FixedVectorType::get(getInt8Ty(), (mBitBlockWidth / 8));
        value = builder.CreateBitCast(value, byteFixedVectorType);

        std::vector<Value *> args;
        args.push_back(fdInt);
        args.push_back(GetString(out.str()));
        args.push_back(name);
        for(unsigned i = (getBitBlockWidth() / 8); i != 0; --i) {
            args.push_back(builder.CreateZExt(builder.CreateExtractElement(value, builder.getInt32(i - 1)), builder.getInt32Ty()));
        }
        Function * Dprintf = GetDprintf();
        builder.CreateCall(Dprintf->getFunctionType(), Dprintf, args);
        builder.CreateRetVoid();
        printRegister = function;
    }
    return CreateCall(printRegister->getFunctionType(), printRegister, {getInt32(static_cast<uint32_t>(fd)), GetString(name), CreateBitCast(value, getBitBlockType())});
}
//pairwise

llvm::Value * IDISA_Builder::hsimd_pairwisesum(unsigned fw, llvm::Value * Val_a, llvm::Value * Val_b){
  
//    // Extract upper 16 bits
//    llvm::Value * highA = simd_srli(2*fw, Val_a, fw);  // fw to make it work for any power of 2**k = fw
//    llvm::Value * highB = simd_srli(2*fw, Val_b, fw);
//
//    // Mask with all elements set to 0xFFFF
//    llvm::Value * mask = simd_fill(2*fw, getIntN(2*fw, (1ULL<<fw) - 1ULL));  // 0xFFFF = 16 - 1s
//    
//  
//    // Mask lower 16 bits
//    llvm::Value * lowA = simd_and(Val_a, mask);
//    llvm::Value * lowB = simd_and(Val_b, mask);
//
//    // Sum the upper and lower 16-bit parts for pairwise sum
//    llvm::Value * sumA = simd_add(2*fw, lowA, highA);
//    llvm::Value * sumB = simd_add(2*fw, lowB, highB);

    // Truncate to 16-bit values
//        llvm::Value * shortA = b.CreateTrunc(sumA, getIntNTy(16));
//        llvm::Value * shortB = b.CreateTrunc(sumB, getIntNTy(16));

    //llvm::Value * result = hsimd_packl(2*fw, sumA, sumB);  // hsimd_packl - takes the lower half and packs them into a single vector
    llvm::Value * result = simd_add(fw, hsimd_packl(2*fw, Val_a, Val_b), hsimd_packh(2*fw, Val_a, Val_b));
    // Shuffle to concatenate results
//        llvm::Value * result = b.shufflevector(shortA, shortB, {0, 1, 2, 3, 4, 5, 6, 7});

    return result;
}


Constant *IDISA_Builder::getSplat(const unsigned fieldCount, Constant *Elt) {
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(12, 0, 0)
    return ConstantVector::getSplat(ElementCount::get(fieldCount, false), Elt);
#elif LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(11, 0, 0)
    return ConstantVector::getSplat({fieldCount, false}, Elt);
#else
    return ConstantVector::getSplat(fieldCount, Elt);
#endif
}

Constant * IDISA_Builder::simd_himask(unsigned fw) {
    return getSplat(mBitBlockWidth/fw, Constant::getIntegerValue(getIntNTy(fw), APInt::getHighBitsSet(fw, fw/2)));
}

Constant * IDISA_Builder::simd_lomask(unsigned fw) {
    return getSplat(mBitBlockWidth/fw, Constant::getIntegerValue(getIntNTy(fw), APInt::getLowBitsSet(fw, fw/2)));
}

Value * IDISA_Builder::simd_select_hi(unsigned fw, Value * a) {
    const unsigned vectorWidth = getVectorBitWidth(a);
    Constant * maskField = Constant::getIntegerValue(getIntNTy(fw), APInt::getHighBitsSet(fw, fw/2));
    return simd_and(a, getSplat(vectorWidth/fw, maskField));
}

Value * IDISA_Builder::simd_select_lo(unsigned fw, Value * a) {
    const unsigned vectorWidth = getVectorBitWidth(a);
    Constant * maskField = Constant::getIntegerValue(getIntNTy(fw), APInt::getLowBitsSet(fw, fw/2));
    return simd_and(a, getSplat(vectorWidth/fw, maskField));
}

Constant * IDISA_Builder::getConstantVectorSequence(unsigned fw, unsigned first, unsigned last, unsigned by) {
    const unsigned seqLgth = (last - first)/by + 1;
    assert(((first + (seqLgth - 1) * by) == last) && "invalid element sequence");
    Type * fwTy = getIntNTy(fw);
    SmallVector<Constant *, 16> elements(seqLgth);
    for (unsigned i = 0; i < seqLgth; i++) {
        elements[i] = ConstantInt::get(fwTy, i*by + first);
    }
    return ConstantVector::get(elements);
}

Constant * IDISA_Builder::getRepeatingConstantVectorSequence(unsigned fw, unsigned repeat, unsigned first, unsigned last, unsigned by) {
    const unsigned seqLgth = (last - first)/by + 1;
    assert(((first + (seqLgth - 1) * by) == last) && "invalid element sequence");
    Type * fwTy = getIntNTy(fw);
    SmallVector<Constant *, 16> elements(seqLgth * repeat);
    for (unsigned i = 0; i < seqLgth; i++) {
        Constant * c = ConstantInt::get(fwTy, i*by + first);
        for (unsigned j = 0; j < repeat; j++) {
            elements[i + j * seqLgth] = c;
        }
    }
    return ConstantVector::get(elements);
}

Value * IDISA_Builder::CreateHalfVectorHigh(Value * vec) {
    Value * v = fwCast(mLaneWidth, vec);
    const unsigned N = getVectorBitWidth(v)/mLaneWidth;
    return CreateShuffleVector(v, UndefValue::get(v->getType()), getConstantVectorSequence(32, N/2, N-1));
}

Value * IDISA_Builder::CreateHalfVectorLow(Value * vec) {
    Value * v = fwCast(mLaneWidth, vec);
    const unsigned N = getVectorBitWidth(v)/mLaneWidth;
    return CreateShuffleVector(v, UndefValue::get(v->getType()), getConstantVectorSequence(32, 0, N/2-1));
}

Value * IDISA_Builder::CreateDoubleVector(Value * lo, Value * hi) {
    const unsigned N = getVectorBitWidth(lo)/mLaneWidth;
    return CreateShuffleVector(fwCast(mLaneWidth, lo), fwCast(mLaneWidth, hi), getConstantVectorSequence(32, 0, 2*N-1));
}

Value * IDISA_Builder::simd_fill(unsigned fw, Value * a) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "simd_fill");
    const unsigned field_count = mBitBlockWidth/fw;
    Type * singleFieldVecTy = FixedVectorType::get(getIntNTy(fw), 1);
    Value * aVec = CreateBitCast(CreateZExtOrTrunc(a, getIntNTy(fw)), singleFieldVecTy);
    return CreateShuffleVector(aVec, UndefValue::get(singleFieldVecTy), Constant::getNullValue(FixedVectorType::get(getInt32Ty(), field_count)));
}

Value * IDISA_Builder::simd_add(unsigned fw, Value * a, Value * b) {
    const unsigned vectorWidth = getVectorBitWidth(a);
    if (fw == 1) {
        return fwCast(1, simd_xor(a, b));
    } else if (fw < 8) {
        Constant * hi_bit_mask = Constant::getIntegerValue(getIntNTy(vectorWidth),
                                                           APInt::getSplat(vectorWidth, APInt::getHighBitsSet(fw, 1)));
        Constant * lo_bit_mask = Constant::getIntegerValue(getIntNTy(vectorWidth),
                                                           APInt::getSplat(vectorWidth, APInt::getLowBitsSet(fw, fw-1)));
        Value * hi_xor = simd_xor(simd_and(a, hi_bit_mask), simd_and(b, hi_bit_mask));
        Value * part_sum = simd_add(32, simd_and(a, lo_bit_mask), simd_and(b, lo_bit_mask));
        return fwCast(fw, simd_xor(part_sum, hi_xor));
    }
    return CreateAdd(fwCast(fw, a), fwCast(fw, b));
}

Value * IDISA_Builder::simd_sub(unsigned fw, Value * a, Value * b) {
    if (fw == 1) {
        return fwCast(1, simd_xor(a, b));
    }
    if (fw < 8) UnsupportedFieldWidthError(fw, "sub");
    return CreateSub(fwCast(fw, a), fwCast(fw, b));
}

Value * IDISA_Builder::simd_mult(unsigned fw, Value * a, Value * b) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "mult");
    return CreateMul(fwCast(fw, a), fwCast(fw, b));
}

Value * IDISA_Builder::simd_eq(unsigned fw, Value * a, Value * b) {
    if (fw < 8) {
        Value * eq_bits = simd_not(simd_xor(a, b));
        if (fw == 1) return eq_bits;
        eq_bits = simd_or(simd_and(simd_srli(32, simd_select_hi(2, eq_bits), 1), eq_bits),
                          simd_and(simd_slli(32, simd_select_lo(2, eq_bits), 1), eq_bits));
        if (fw == 2) return eq_bits;
        eq_bits = simd_or(simd_and(simd_srli(32, simd_select_hi(4, eq_bits), 2), eq_bits),
                          simd_and(simd_slli(32, simd_select_lo(4, eq_bits), 2), eq_bits));
        return eq_bits;
    }
    Value * a1 = fwCast(fw, a);
    Value * b1 = fwCast(fw, b);
    return CreateSExt(CreateICmpEQ(a1, b1), a1->getType());
}

Value * IDISA_Builder::simd_any(unsigned fw, Value * a) {
    return CreateNot(simd_eq(fw, a, ConstantVector::getNullValue(a->getType())));
}

Value * IDISA_Builder::simd_ne(unsigned fw, Value * a, Value * b) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "ne");
    Value * a1 = fwCast(fw, a);
    Value * b1 = fwCast(fw, b);
    return CreateSExt(CreateICmpNE(a1, b1), a1->getType());
}

Value * IDISA_Builder::simd_gt(unsigned fw, Value * a, Value * b) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "gt");
    Value * a1 = fwCast(fw, a);
    Value * b1 = fwCast(fw, b);
    return CreateSExt(CreateICmpSGT(a1, b1), a1->getType());
}

Value * IDISA_Builder::simd_ge(unsigned fw, Value * a, Value * b) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "ge");
    Value * a1 = fwCast(fw, a);
    Value * b1 = fwCast(fw, b);
    return CreateSExt(CreateICmpSGE(a1, b1), a1->getType());
}

Value * IDISA_Builder::simd_ugt(unsigned fw, Value * a, Value * b) {
    if (fw == 1) return simd_and(a, simd_not(b));
    if (fw < 8) {
        Value * half_ugt = simd_ugt(fw/2, a, b);
        Value * half_eq = simd_eq(fw/2, a, b);
        Value * ugt_0 = simd_or(simd_srli(fw, half_ugt, fw/2), simd_and(half_ugt, simd_srli(fw, half_eq, fw/2)));
        return simd_or(ugt_0, simd_slli(32, ugt_0, fw/2));
    }
    Value * a1 = fwCast(fw, a);
    Value * b1 = fwCast(fw, b);
    return CreateSExt(CreateICmpUGT(a1, b1), a1->getType());
}

Value * IDISA_Builder::simd_lt(unsigned fw, Value * a, Value * b) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "lt");
    Value * a1 = fwCast(fw, a);
    Value * b1 = fwCast(fw, b);
    return CreateSExt(CreateICmpSLT(a1, b1), a1->getType());
}

Value * IDISA_Builder::simd_le(unsigned fw, Value * a, Value * b) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "le");
    Value * a1 = fwCast(fw, a);
    Value * b1 = fwCast(fw, b);
    return CreateSExt(CreateICmpSLE(a1, b1), a1->getType());
}

Value * IDISA_Builder::simd_ult(unsigned fw, Value * a, Value * b) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "ult");
    Value * a1 = fwCast(fw, a);
    Value * b1 = fwCast(fw, b);
    return CreateSExt(CreateICmpULT(a1, b1), a1->getType());
}

Value * IDISA_Builder::simd_ule(unsigned fw, Value * a, Value * b) {
    if (fw == 1) return simd_or(simd_not(a), b);
    if (fw < 8) {
        Value * hi_rslt = simd_select_hi(2*fw, simd_ule(2*fw, simd_select_hi(2*fw, a), b));
        Value * lo_rslt = simd_select_lo(2*fw, simd_ule(2*fw, simd_select_lo(2*fw, a), simd_select_lo(2*fw, b)));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value * a1 = fwCast(fw, a);
    Value * b1 = fwCast(fw, b);
    return CreateSExt(CreateICmpULE(a1, b1), a1->getType());
}

Value * IDISA_Builder::simd_uge(unsigned fw, Value * a, Value * b) {
    if (fw == 1) return simd_or(a, simd_not(b));
    if (fw < 8) {
        Value * hi_rslt = simd_select_hi(2*fw, simd_uge(2*fw, a, simd_select_hi(2*fw, b)));
        Value * lo_rslt = simd_select_lo(2*fw, simd_uge(2*fw, simd_select_lo(2*fw, a), simd_select_lo(2*fw, b)));
        return simd_or(hi_rslt, lo_rslt);
    }
    if (fw < 8) UnsupportedFieldWidthError(fw, "ult");
    Value * a1 = fwCast(fw, a);
    Value * b1 = fwCast(fw, b);
    return CreateSExt(CreateICmpUGE(a1, b1), a1->getType());
}

Value * IDISA_Builder::simd_max(unsigned fw, Value * a, Value * b) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "max");
    Value * aVec = fwCast(fw, a);
    Value * bVec = fwCast(fw, b);
    return CreateSelect(CreateICmpSGT(aVec, bVec), aVec, bVec);
}

Value * IDISA_Builder::simd_umax(unsigned fw, Value * a, Value * b) {
    if (fw == 1) return simd_or(a, b);
    if (fw < 8) {
        Value * hi_rslt = simd_select_hi(2*fw, simd_umax(2*fw, a, b));
        Value * lo_rslt = simd_umax(2*fw, simd_select_lo(2*fw, a), simd_select_lo(2*fw, b));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value * aVec = fwCast(fw, a);
    Value * bVec = fwCast(fw, b);
    return CreateSelect(CreateICmpUGT(aVec, bVec), aVec, bVec);
}

Value * IDISA_Builder::simd_min(unsigned fw, Value * a, Value * b) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "min");
    Value * aVec = fwCast(fw, a);
    Value * bVec = fwCast(fw, b);
    return CreateSelect(CreateICmpSLT(aVec, bVec), aVec, bVec);
}

Value * IDISA_Builder::simd_umin(unsigned fw, Value * a, Value * b) {
    if (fw == 1) return simd_and(a, b);
    if (fw < 8) {
        Value * hi_rslt = simd_select_hi(2*fw, simd_umin(2*fw, a, b));
        Value * lo_rslt = simd_umin(2*fw, simd_select_lo(2*fw, a), simd_select_lo(2*fw, b));
        return simd_or(hi_rslt, lo_rslt);
    }
    Value * aVec = fwCast(fw, a);
    Value * bVec = fwCast(fw, b);
    return CreateSelect(CreateICmpULT(aVec, bVec), aVec, bVec);
}

Value * IDISA_Builder::mvmd_sll(unsigned fw, Value * value, Value * shift, const bool safe) {
    FixedVectorType * const vecTy = fwVectorType(fw);
    IntegerType * const intTy = getIntNTy(mBitBlockWidth);
    Type * shiftTy = shift->getType();
    if (LLVM_UNLIKELY(!shiftTy->isIntegerTy())) {
        report_fatal_error("shift value type must be an integer");
    }
    // make sure the maximum bitwidth of the value can hold the multiplied value
    if (shiftTy->getIntegerBitWidth() < vecTy->getNumElements()) {
        shiftTy = vecTy->getElementType();
        shift = CreateZExt(shift, shiftTy);
    }

    Constant * const FIELD_WIDTH = ConstantInt::get(shiftTy, fw);
    shift = CreateMul(shift, FIELD_WIDTH);
    if (LLVM_UNLIKELY(safe && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value * const inbounds = CreateICmpULT(shift, ConstantInt::get(shiftTy, mBitBlockWidth));
        CreateAssert(inbounds, "poison shift value: >= vector width");
    }
    Value * result = nullptr;
    value = CreateBitCast(value, intTy);
//    if (safe) {
        shift = CreateZExtOrTrunc(shift, intTy);
        result = CreateShl(value, shift);
//    } else {
//        // TODO: check the ASM generated by this to see what the select generates
//        Value * const moddedShift = CreateURem(shift, BLOCK_WIDTH);
//        Value * const inbounds = CreateICmpEQ(moddedShift, shift);
//        shift = CreateZExtOrTrunc(moddedShift, intTy);
//        Constant * const ZEROES = Constant::getNullValue(intTy);
//        result = CreateShl(value, shift);
//        result = CreateSelect(inbounds, result, ZEROES);
//    }
    return CreateBitCast(result, vecTy);
}

Value * IDISA_Builder::mvmd_dsll(unsigned fw, Value * a, Value * b, Value * shift) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "mvmd_dsll");
    const auto field_count = mBitBlockWidth/fw;
    Type * fwTy = getIntNTy(fw);
    SmallVector<Constant *, 16> Idxs(field_count);
    for (unsigned i = 0; i < field_count; i++) {
        Idxs[i] = ConstantInt::get(fwTy, i + field_count);
    }
    Value * shuffle_indexes = simd_sub(fw, ConstantVector::get(Idxs), simd_fill(fw, shift));
    return mvmd_shuffle2(fw, fwCast(fw, b), fwCast(fw, a), shuffle_indexes);
}

Value * IDISA_Builder::mvmd_srl(unsigned fw, Value * value, Value * shift, const bool safe) {
    FixedVectorType * const vecTy = fwVectorType(fw);
    IntegerType * const intTy = getIntNTy(mBitBlockWidth);
    Type * shiftTy = shift->getType();
    if (LLVM_UNLIKELY(!shiftTy->isIntegerTy())) {
        report_fatal_error("shift value type must be an integer");
    }
    // make sure the maximum bitwidth of the value can hold the multiplied value
    if (shiftTy->getIntegerBitWidth() < vecTy->getNumElements()) {
        shiftTy = vecTy->getElementType();
        shift = CreateZExt(shift, shiftTy);
    }

    Constant * const FIELD_WIDTH = ConstantInt::get(shiftTy, fw);

    shift = CreateMul(shift, FIELD_WIDTH);
    if (LLVM_UNLIKELY(safe && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value * const inbounds = CreateICmpULT(shift, ConstantInt::get(shiftTy, mBitBlockWidth));
        CreateAssert(inbounds, "poison shift value: >= vector width");
    }
    Value * result = nullptr;
    value = CreateBitCast(value, intTy);
//    if (safe) {
        shift = CreateZExtOrTrunc(shift, intTy);
        result = CreateLShr(value, shift);
//    } else {
//        // TODO: check the ASM generated by this to see what the select generates
//        Value * const moddedShift = CreateURem(shift, BLOCK_WIDTH);
//        Value * const inbounds = CreateICmpEQ(moddedShift, shift);
//        shift = CreateZExtOrTrunc(moddedShift, intTy);
//        Constant * const ZEROES = Constant::getNullValue(intTy);
//        result = CreateLShr(value, shift);
//        result = CreateSelect(inbounds, result, ZEROES);
//    }
    return CreateBitCast(result, vecTy);
}

Value * IDISA_Builder::simd_slli(unsigned fw, Value * a, unsigned shift) {
    if (shift == 0) return a;
    const unsigned vectorWidth = getVectorBitWidth(a);
    if (fw > MAX_NATIVE_SIMD_SHIFT) {
        unsigned fullFieldShift = shift / MAX_NATIVE_SIMD_SHIFT;
        unsigned subFieldShift = shift % MAX_NATIVE_SIMD_SHIFT;
        Value * fieldShifted = a;
        if (fullFieldShift > 0) {
            fieldShifted = mvmd_slli(MAX_NATIVE_SIMD_SHIFT, a, fullFieldShift);
            unsigned remaining = fw - fullFieldShift * MAX_NATIVE_SIMD_SHIFT;
            Constant * mask = Constant::getIntegerValue(getIntNTy(fw),
                                                        APInt::getSplat(vectorWidth, APInt::getHighBitsSet(fw, remaining)));
            fieldShifted = simd_and(fieldShifted, mask);
        }
        if (subFieldShift == 0) return fieldShifted;
        Value * extendedShift = simd_slli(MAX_NATIVE_SIMD_SHIFT, fieldShifted, subFieldShift);
        if (fw - fullFieldShift < MAX_NATIVE_SIMD_SHIFT) {
            // No additional bits to combined
            return extendedShift;
        }
        Value * overShifted = mvmd_slli(MAX_NATIVE_SIMD_SHIFT, a, fullFieldShift + 1);
        unsigned backShift = MAX_NATIVE_SIMD_SHIFT - subFieldShift;
        Value * backShifted = simd_srli(MAX_NATIVE_SIMD_SHIFT, overShifted, backShift);
        return simd_or(extendedShift, backShifted);
    }
    if (fw < MIN_NATIVE_SIMD_SHIFT) {
        Constant * value_mask = Constant::getIntegerValue(getIntNTy(vectorWidth),
                                                          APInt::getSplat(vectorWidth, APInt::getLowBitsSet(fw, fw-shift)));
        return CreateShl(fwCast(MIN_NATIVE_SIMD_SHIFT, simd_and(a, value_mask)), shift);
    }
    return CreateShl(fwCast(fw, a), shift);
}

Value * IDISA_Builder::simd_srli(unsigned fw, Value * a, unsigned shift) {
    if (shift == 0) return a;
    const unsigned vectorWidth = getVectorBitWidth(a);
    if (fw > MAX_NATIVE_SIMD_SHIFT) {
        unsigned fullFieldShift = shift / MAX_NATIVE_SIMD_SHIFT;
        unsigned subFieldShift = shift % MAX_NATIVE_SIMD_SHIFT;
        Value * fieldShifted = a;
        if (fullFieldShift > 0) {
            fieldShifted = mvmd_srli(MAX_NATIVE_SIMD_SHIFT, a, fullFieldShift);
            unsigned remaining = fw - fullFieldShift * MAX_NATIVE_SIMD_SHIFT;
            Constant * mask = Constant::getIntegerValue(getIntNTy(fw),
                                                        APInt::getSplat(vectorWidth, APInt::getLowBitsSet(fw, remaining)));
            fieldShifted = simd_and(fieldShifted, mask);
        }
        if (subFieldShift == 0) return fieldShifted;
        Value * extendedShift = simd_srli(MAX_NATIVE_SIMD_SHIFT, fieldShifted, subFieldShift);
        if (fw - fullFieldShift < MAX_NATIVE_SIMD_SHIFT) {
            // No additional bits to combined
            return extendedShift;
        }
        Value * overShifted = mvmd_srli(MAX_NATIVE_SIMD_SHIFT, a, fullFieldShift + 1);
        unsigned backShift = MAX_NATIVE_SIMD_SHIFT - subFieldShift;
        Value * backShifted = simd_slli(MAX_NATIVE_SIMD_SHIFT, overShifted, backShift);
        return simd_or(extendedShift, backShifted);
    }
    if (fw < MIN_NATIVE_SIMD_SHIFT) {
        Constant * value_mask = Constant::getIntegerValue(getIntNTy(vectorWidth),
                                                          APInt::getSplat(vectorWidth, APInt::getHighBitsSet(fw, fw-shift)));
        return CreateLShr(fwCast(MIN_NATIVE_SIMD_SHIFT, simd_and(a, value_mask)), shift);
    }
    return CreateLShr(fwCast(fw, a), shift);
}

Value * IDISA_Builder::simd_srai(unsigned fw, Value * a, unsigned shift) {
    if (shift == 0) return a;
    const unsigned vectorWidth = getVectorBitWidth(a);
    if (fw < MIN_NATIVE_SIMD_SHIFT) {
        Constant * sign_mask = Constant::getIntegerValue(getIntNTy(vectorWidth),
                                                       APInt::getSplat(vectorWidth, APInt::getHighBitsSet(fw, 1)));
        Value * sign = simd_and(sign_mask, a);
        if (shift == 1) return simd_or(sign, simd_srli(MIN_NATIVE_SIMD_SHIFT, sign, 1));
        return simd_or(sign, simd_sub(MIN_NATIVE_SIMD_SHIFT, sign, simd_srli(MIN_NATIVE_SIMD_SHIFT, sign, shift)));
    }
    return CreateAShr(fwCast(fw, a), shift);
}

Value * IDISA_Builder::simd_sllv(unsigned fw, Value * v, Value * shifts) {
    if (fw >= 8) return CreateShl(fwCast(fw, v), fwCast(fw, shifts));
    auto vec_width = getVectorBitWidth(v);
    Value * vecZeroes = ConstantVector::getNullValue(v->getType());
    Value * w = v;
    IntegerType * const intTy = getIntNTy(vec_width);
    for (unsigned shft_amt = 1; shft_amt < fw; shft_amt *= 2) {
        APInt bit_in_field(fw, shft_amt);
        // To simulate shift within a fw, we need to mask off the high shft_amt bits of each element.
        Constant * value_mask = Constant::getIntegerValue(intTy,
                                                          APInt::getSplat(vec_width, APInt::getLowBitsSet(fw, fw-shft_amt)));
        Constant * bit_select = Constant::getIntegerValue(intTy,
                                                          APInt::getSplat(vec_width, bit_in_field));
        Value * unshifted_field_mask = simd_eq(fw, simd_and(bit_select, shifts), vecZeroes);
        Value * fieldsToShift = simd_and(w, simd_and(value_mask, simd_not(unshifted_field_mask)));
        w = simd_or(simd_and(w, unshifted_field_mask), simd_slli(32, fieldsToShift, shft_amt));
    }
    return w;
}

Value * IDISA_Builder::simd_srlv(unsigned fw, Value * v, Value * shifts) {
    if (fw >= 8) return CreateLShr(fwCast(fw, v), fwCast(fw, shifts));
    Value * vecZeroes = ConstantVector::getNullValue(v->getType());
    auto vec_width = getVectorBitWidth(v);
    Value * w = v;
    IntegerType * const intTy = getIntNTy(vec_width);
    for (unsigned shft_amt = 1; shft_amt < fw; shft_amt *= 2) {
        APInt bit_in_field(fw, shft_amt);
        // To simulate shift within a fw, we need to mask off the low shft_amt bits of each element.
        Constant * value_mask = Constant::getIntegerValue(intTy,
                                                          APInt::getSplat(vec_width, APInt::getHighBitsSet(fw, fw-shft_amt)));
        Constant * bit_select = Constant::getIntegerValue(intTy,
                                                          APInt::getSplat(vec_width, bit_in_field));
        Value * unshifted_field_mask = simd_eq(fw, simd_and(bit_select, shifts), vecZeroes);
        Value * fieldsToShift = simd_and(w, simd_and(value_mask, simd_not(unshifted_field_mask)));
        w = simd_or(simd_and(w, unshifted_field_mask), simd_srli(32, fieldsToShift, shft_amt));
    }
    return w;
}

Value * IDISA_Builder::simd_rotl(unsigned fw, Value * v, Value * rotates) {
    Type * fwTy = getIntNTy(fw);
    unsigned numFields = getVectorBitWidth(v)/fw;
    Constant * fw_mask =  getSplat(numFields, ConstantInt::get(fwTy, fw - 1));
    Constant * fw_splat =  getSplat(numFields, ConstantInt::get(fwTy, fw));
    Value * shft = simd_and(fw_mask, rotates);
    Value * fwd = simd_sllv(fw, v, shft);
    // Masking is necessary to avoid a srlv by fw (poison value) when the
    // rotate amount is 0.
    Value * back = simd_srlv(fw, v, simd_and(fw_mask, simd_sub(fw, fw_splat, shft)));
    return simd_or(fwd, back);
}

Value * IDISA_Builder::simd_rotr(unsigned fw, Value * v, Value * rotates) {
    Type * fwTy = getIntNTy(fw);
    unsigned numFields = getVectorBitWidth(v)/fw;
    Constant * fw_mask =  getSplat(numFields, ConstantInt::get(fwTy, fw - 1));
    Constant * fw_splat =  getSplat(numFields, ConstantInt::get(fwTy, fw));
    Value * shft = simd_and(fw_mask, rotates);
    // Masking is necessary to avoid a sllv by fw (poison value) when the
    // rotate amount is 0.
    Value * fwd = simd_sllv(fw, v, simd_and(fw_mask, simd_sub(fw, fw_splat, shft)));
    Value * back = simd_srlv(fw, v, shft);
    return simd_or(fwd, back);
}

std::vector<Value *> IDISA_Builder::simd_pext(unsigned fieldwidth, std::vector<Value *> v, Value * extract_mask) {
    Value * delcounts = CreateNot(extract_mask);  // initially deletion counts per 1-bit field
    std::vector<Value *> w(v.size());
    for (unsigned i = 0; i < v.size(); i++) {
        w[i] = simd_and(extract_mask, v[i]);
    }
    for (unsigned fw = 2; fw < fieldwidth; fw = fw * 2) {
        Value * shift_fwd_amts = simd_srli(fw, simd_select_lo(fw*2, delcounts), fw/2);
        Value * shift_back_amts = simd_select_lo(fw, simd_select_hi(fw*2, delcounts));
        for (unsigned i = 0; i < v.size(); i++) {
            w[i] = simd_or(simd_sllv(fw, simd_select_lo(fw*2, w[i]), shift_fwd_amts),
                           simd_srlv(fw, simd_select_hi(fw*2, w[i]), shift_back_amts));
        }
        delcounts = simd_add(fw, simd_select_lo(fw, delcounts), simd_srli(fw, delcounts, fw/2));
    }
    // Now shift back all fw fields.
    Value * shift_back_amts = simd_select_lo(fieldwidth, delcounts);
    for (unsigned i = 0; i < v.size(); i++) {
        w[i] = simd_srlv(fieldwidth, w[i], shift_back_amts);
    }
    return w;
}

Value * IDISA_Builder::simd_pext(unsigned fieldwidth, Value * v, Value * extract_mask) {
    return simd_pext(fieldwidth, std::vector<Value *>{v}, extract_mask)[0];
}

Value * IDISA_Builder::CreatePextract(Value * v, Value * mask, const Twine Name) {
    Type * Ty = v->getType();
    unsigned width = Ty->getPrimitiveSizeInBits();
    return CreateBitCast(IDISA_Builder::simd_pext(width, fwCast(width, v), fwCast(width, mask)), Ty);
}

Value * IDISA_Builder::simd_pdep(unsigned fieldwidth, Value * v, Value * deposit_mask) {
    // simd_pdep is implemented by reversing the process of simd_pext.
    // First determine the deletion counts necessary for each stage of the process.
    std::vector<Value *> delcounts;
    delcounts.push_back(simd_not(deposit_mask)); // initially deletion counts per 1-bit field
    for (unsigned fw = 2; fw < fieldwidth; fw = fw * 2) {
        delcounts.push_back(simd_add(fw, simd_select_lo(fw, delcounts.back()), simd_srli(fw, delcounts.back(), fw/2)));
    }
    //
    // Now reverse the pext process.  First reverse the final shift_back.
    Value * pext_shift_back_amts = simd_select_lo(fieldwidth, delcounts.back());
    Value * w = simd_sllv(fieldwidth, v, pext_shift_back_amts);
    //
    // No work through the smaller field widths.
    for (unsigned fw = fieldwidth/2; fw >= 2; fw = fw/2) {
        delcounts.pop_back();
        Value * pext_shift_fwd_amts = simd_srli(fw, simd_select_lo(fw * 2, delcounts.back()), fw/2);
        Value * pext_shift_back_amts = simd_select_lo(fw, simd_select_hi(fw*2, delcounts.back()));
        w = simd_or(simd_srlv(fw, simd_select_lo(fw * 2, w), pext_shift_fwd_amts),
                    simd_sllv(fw, simd_select_hi(fw * 2, w), pext_shift_back_amts));
    }
    return simd_and(w, deposit_mask);
}

Value * IDISA_Builder::CreatePdeposit(Value * v, Value * mask, const Twine Name) {
    Type * Ty = v->getType();
    unsigned width = Ty->getPrimitiveSizeInBits();
    return CreateBitCast(simd_pdep(width, fwCast(width, v), fwCast(width, mask)), Ty);
}

Value * IDISA_Builder::simd_popcount(unsigned fw, Value * a) {
    if (fw == 1) {
        return a;
    } else if (fw == 2) {
        // For each 2-bit field ab we can use the subtraction ab - 0a to generate
        // the popcount without carry/borrow from the neighbouring 2-bit field.
        // case 00:  ab - 0a = 00 - 00 = 00
        // case 01:  ab - 0a = 01 - 00 = 01
        // case 10:  ab - 0a = 10 - 01 = 01 (no borrow)
        // case 11:  ab - 0a = 11 - 01 = 10
        return simd_sub(64, a, simd_srli(64, simd_select_hi(2, a), 1));
    } else if (fw <= 64) {
        Value * c = simd_popcount(fw/2, a);
        c = simd_add(64, simd_select_lo(fw, c), simd_srli(fw, c, fw/2));
        return c;
    } else {
        return CreatePopcount(fwCast(fw, a));
    }
}

Value * IDISA_Builder::hsimd_partial_sum(unsigned fw, Value * a) {
    const unsigned vectorWidth = getVectorBitWidth(a);
    Value * partial_sum = fwCast(fw, a);
    const auto count = vectorWidth / fw;
    for (unsigned move = 1; move < count; move *= 2) {
        partial_sum = simd_add(fw, partial_sum, mvmd_slli(fw, partial_sum, move));
    }
    return partial_sum;
}

Value * IDISA_Builder::simd_cttz(unsigned fw, Value * a) {
    if (fw == 1) {
        return simd_not(a);
    } else {
        Value* v = simd_sub(fw, a, simd_fill(fw, getIntN(fw, 1)));
        v = simd_or(v, a);
        v = simd_xor(v, a);
        v = simd_popcount(fw, v);
        return v;
    }
}

Value * IDISA_Builder::simd_bitreverse(unsigned fw, Value * a) {
    /*  Pure sequential solution too slow!
     Function * func = Intrinsic::getDeclaration(getModule(), Intrinsic::bitreverse, fwVectorType(fw));
     return CreateCall(func->getFunctionType(), func, fwCast(fw, a));
     */
    if (fw > 8) {
        // Reverse the bits of each byte and then use a byte shuffle to complete the job.
        Value * bitrev8 = fwCast(8, simd_bitreverse(8, a));
        const auto bytes_per_field = fw/8;
        const unsigned vectorWidth = getVectorBitWidth(a);
        const auto byte_count = vectorWidth / 8;
        SmallVector<Constant *, 16> Idxs(byte_count);
        for (unsigned i = 0; i < byte_count; i += bytes_per_field) {
            for (unsigned j = 0; j < bytes_per_field; j++) {
                Idxs[i + j] = getInt32(i + bytes_per_field - j - 1);
            }
        }
        return CreateShuffleVector(bitrev8, UndefValue::get(bitrev8->getType()), ConstantVector::get(Idxs));
    }
    else {
        if (fw > 2) {
            a = simd_bitreverse(fw/2, a);
        }
        return simd_or(simd_srli(16, simd_select_hi(fw, a), fw/2), simd_slli(16, simd_select_lo(fw, a), fw/2));
    }
}

Value * IDISA_Builder::simd_if(unsigned fw, Value * cond, Value * a, Value * b) {
    if (fw == 1) {
        Value * a1 = bitCast(a);
        Value * b1 = bitCast(b);
        Value * c = bitCast(cond);
        return CreateOr(CreateAnd(a1, c), CreateAnd(CreateXor(c, b1), b1));
    } else {
        if (fw < 8) UnsupportedFieldWidthError(fw, "simd_if");
        Value * aVec = fwCast(fw, a);
        Value * bVec = fwCast(fw, b);
        return CreateSelect(CreateICmpSLT(fwCast(fw, cond), ConstantVector::getNullValue(aVec->getType())), aVec, bVec);
    }
}

//
// Return a logic expression in terms of bitwise And, Or and Not for an
// arbitrary two-operand binary function corresponding to a 4-bit truth table mask.
// The 4-bit mask xyzw specifies the two-operand function fn defined by
// the following table.
//  bit_1  bit_0   fn
//    0      0     w
//    0      1     z
//    1      0     y
//    1      1     x
Value * IDISA_Builder::simd_binary(unsigned char truth_table_mask, Value * bit_1, Value * bit_0) {
    assert (bit_1->getType() == bit_0->getType());
    switch(truth_table_mask) {
        case 0x00:
            return Constant::getNullValue(bit_1->getType());
        case 0x01:
            return CreateNot(CreateOr(bit_1, bit_0));
        case 0x02:
            return CreateAnd(CreateNot(bit_1), bit_0);
        case 0x03:
            return CreateNot(bit_1);
        case 0x04:
            return CreateAnd(bit_1, CreateNot(bit_0));
        case 0x05:
            return CreateNot(bit_0);
        case 0x06:
            return CreateXor(bit_1, bit_0);
        case 0x07:
            return CreateNot(CreateAnd(bit_1, bit_0));
        case 0x08:
            return CreateAnd(bit_1, bit_0);
        case 0x09:
            return CreateNot(CreateXor(bit_1, bit_0));
        case 0x0A:
            return bit_0;
        case 0x0B:
            return CreateOr(CreateNot(bit_1), bit_0);
        case 0x0C:
            return bit_1;
        case 0x0D:
            return CreateOr(bit_1, CreateNot(bit_0));
        case 0x0E:
            return CreateOr(bit_1, bit_0);
        case 0x0F:
            return Constant::getAllOnesValue(bit_1->getType());
        default: report_fatal_error("simd_binary mask is in wrong format!");
    }
}

Value * IDISA_Builder::simd_ternary(unsigned char mask, Value * a, Value * b, Value * c) {
    assert (a->getType() == b->getType());
    assert (b->getType() == c->getType());

    if (mask == 0) {
        return Constant::getNullValue(a->getType());
    }
    if (mask == 0xFF) {
        return Constant::getAllOnesValue(a->getType());
    }

    unsigned char not_a_mask = mask & 0x0F;
    unsigned char a_mask = (mask >> 4) & 0x0F;
    if (a_mask == not_a_mask) {
        return simd_binary(a_mask, b, c);
    }

    unsigned char b_mask = ((mask & 0xC0) >> 4) | ((mask & 0x0C) >> 2);
    unsigned char not_b_mask = ((mask & 0x30) >> 2) | (mask & 0x03);
    if (b_mask == not_b_mask) {
        return simd_binary(b_mask, a, c);
    }

    unsigned char c_mask = ((mask & 0x80) >> 4) | ((mask & 0x20) >> 3) | ((mask & 0x08) >> 2) | ((mask & 02) >> 1);
    unsigned char not_c_mask = ((mask & 0x40) >> 3) | ((mask & 0x10) >> 2) | ((mask & 0x04) >> 1) | (mask & 01);
    if (c_mask == not_c_mask) {
        return simd_binary(c_mask, a, b);
    }

    Value * bc_hi = simd_binary(a_mask, b, c);
    Value * bc_lo = simd_binary(not_a_mask, b, c);
    Value * a_bc = CreateAnd(a, bc_hi);
    Value * not_a_bc = CreateAnd(CreateNot(a), bc_lo);
    return CreateOr(a_bc, not_a_bc);
}

Value * IDISA_Builder::esimd_mergeh(unsigned fw, Value * a, Value * b) {
    if (fw < 8) {
        if (getVectorBitWidth(a) > mNativeBitBlockWidth) {
            Value * a_hi = CreateHalfVectorHigh(a);
            Value * b_hi = CreateHalfVectorHigh(b);
            return CreateDoubleVector(esimd_mergel(fw, a_hi, b_hi), esimd_mergeh(fw, a_hi, b_hi));
        }
        Value * abh = simd_or(simd_select_hi(fw*2, b), simd_srli(32, simd_select_hi(fw*2, a), fw));
        Value * abl = simd_or(simd_slli(32, simd_select_lo(fw*2, b), fw), simd_select_lo(fw*2, a));
        return esimd_mergeh(fw * 2, abl, abh);
    }
    const auto field_count = getVectorBitWidth(a) / fw;

    SmallVector<Constant *, 16> Idxs(field_count);
    for (unsigned i = 0; i < field_count / 2; i++) {
        Idxs[2 * i] = getInt32(i + field_count / 2); // selects elements from first reg.
        Idxs[2 * i + 1] = getInt32(i + field_count / 2 + field_count); // selects elements from second reg.
    }
    return CreateShuffleVector(fwCast(fw, a), fwCast(fw, b), ConstantVector::get(Idxs));
}

Value * IDISA_Builder::esimd_mergel(unsigned fw, Value * a, Value * b) {
    if (fw < 8) {
        if (getVectorBitWidth(a) > mNativeBitBlockWidth) {
            Value * a_lo = CreateHalfVectorLow(a);
            Value * b_lo = CreateHalfVectorLow(b);
            return CreateDoubleVector(esimd_mergel(fw, a_lo, b_lo), esimd_mergeh(fw, a_lo, b_lo));
        }
        Value * abh = simd_or(simd_select_hi(fw*2, b), simd_srli(32, simd_select_hi(fw*2, a), fw));
        Value * abl = simd_or(simd_slli(32, simd_select_lo(fw*2, b), fw), simd_select_lo(fw*2, a));
        return esimd_mergel(fw * 2, abl, abh);
    }
    const auto field_count = getVectorBitWidth(a) / fw;
    SmallVector<Constant *, 16> Idxs(field_count);
    for (unsigned i = 0; i < field_count / 2; i++) {
        Idxs[2 * i] = getInt32(i); // selects elements from first reg.
        Idxs[2 * i + 1] = getInt32(i + field_count); // selects elements from second reg.
    }
    return CreateShuffleVector(fwCast(fw, a), fwCast(fw, b), ConstantVector::get(Idxs));
}

Value * IDISA_Builder::esimd_bitspread(unsigned fw, Value * bitmask) {
    if (LLVM_UNLIKELY(fw < 8)) {
        UnsupportedFieldWidthError(fw, "esimd_bitspread");
    }
    const auto field_count = mBitBlockWidth / fw;
    IntegerType * field_type = getIntNTy(fw);
    Value * broadcast = nullptr;
    Constant * bitSelVec = nullptr;
    Constant * bitShiftVec = nullptr;
    if (field_count <= fw) {
        Value * spread_field = CreateBitCast(CreateZExtOrTrunc(bitmask, field_type), FixedVectorType::get(getIntNTy(fw), 1));
        Value * undefVec = UndefValue::get(FixedVectorType::get(getIntNTy(fw), 1));
        broadcast = CreateShuffleVector(spread_field, undefVec, Constant::getNullValue(FixedVectorType::get(getInt32Ty(), field_count)));
        SmallVector<Constant *, 16> bitSel(field_count);
        SmallVector<Constant *, 16> bitShift(field_count);
        for (unsigned i = 0; i < field_count; i++) {
            bitSel[i] = ConstantInt::get(field_type, 1ULL << i);
            bitShift[i] = ConstantInt::get(field_type, i);
        }
        bitSelVec = ConstantVector::get(bitSel);
        bitShiftVec = ConstantVector::get(bitShift);
    } else {
        assert ((field_count % fw) == 0);
        const auto m = field_count / fw;
        IntegerType * const intFieldCountTy = getIntNTy(field_count);
        VectorType * const intFieldCountVecTy = FixedVectorType::get(field_type, m);
        Value * spread_field = CreateBitCast(CreateZExtOrTrunc(bitmask, intFieldCountTy), intFieldCountVecTy);
        SmallVector<Constant *, 16> shuffle(field_count);
        for (unsigned i = 0; i < m; ++i) {
            ConstantInt * I = getInt32(i);
            for (unsigned j = 0; j < fw; ++j) {
                const auto k = i * fw + j;
                assert (k < field_count);
                shuffle[k] = I;
            }
        }
        Constant * shuffleVec = ConstantVector::get(shuffle);
        Constant * undefVec = UndefValue::get(intFieldCountVecTy);
        broadcast = CreateShuffleVector(spread_field, undefVec, shuffleVec);
        SmallVector<Constant *, 16> bitSel(field_count);
        SmallVector<Constant *, 16> bitShift(field_count);
        for (unsigned j = 0; j < fw; ++j) {
            ConstantInt * sel = ConstantInt::get(field_type, 1ULL << j);
            ConstantInt * shift = ConstantInt::get(field_type, j);
            for (unsigned i = 0; i < m; ++i) {
                const auto k = i * fw + j;
                assert (k < field_count);
                bitSel[k] = sel;
                bitShift[k] = shift;
            }
        }
        bitSelVec = ConstantVector::get(bitSel);
        bitShiftVec = ConstantVector::get(bitShift);
    }
    return CreateLShr(CreateAnd(bitSelVec, broadcast), bitShiftVec);
}

Value * IDISA_Builder::hsimd_packh(unsigned fw, Value * a, Value * b) {
    if (fw <= 8) {
        const unsigned fw_wkg = 32;
        Value * aLo = simd_srli(fw_wkg, a, fw/2);
        Value * bLo = simd_srli(fw_wkg, b, fw/2);
        return hsimd_packl(fw, aLo, bLo);
    }
    Value * aVec = fwCast(fw/2, a);
    Value * bVec = fwCast(fw/2, b);
    const auto field_count = 2 * mBitBlockWidth / fw;
    SmallVector<Constant *, 16> Idxs(field_count);
    for (unsigned i = 0; i < field_count; i++) {
        Idxs[i] = getInt32(2 * i + 1);
    }
    return CreateShuffleVector(aVec, bVec, ConstantVector::get(Idxs));
}

Value * IDISA_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {
    if (fw <= 8) {
        const unsigned fw_wkg = 32;
        Value * aLo = simd_srli(fw_wkg, a, fw/2);
        Value * bLo = simd_srli(fw_wkg, b, fw/2);
        return hsimd_packl(fw*2,
                           simd_or(simd_select_hi(fw, aLo), simd_select_lo(fw, a)),
                           simd_or(simd_select_hi(fw, bLo), simd_select_lo(fw, b)));
    }
    Value * aVec = fwCast(fw/2, a);
    Value * bVec = fwCast(fw/2, b);
    const auto field_count = 2 * mBitBlockWidth / fw;
    SmallVector<Constant *, 16> Idxs(field_count);
    for (unsigned i = 0; i < field_count; i++) {
        Idxs[i] = getInt32(2 * i);
    }
    return CreateShuffleVector(aVec, bVec, ConstantVector::get(Idxs));
}

Value * IDISA_Builder::hsimd_packss(unsigned fw, Value * a, Value * b) {
    Constant * top_bit = Constant::getIntegerValue(getIntNTy(mBitBlockWidth),
                                                  APInt::getSplat(mBitBlockWidth, APInt::getHighBitsSet(fw/2, 1)));
    Value * hi = hsimd_packh(fw, a, b);
    Value * lo = hsimd_packl(fw, a, b);
    Value * bits_that_must_match_sign = simd_if(1, top_bit, lo, hi);
    Value * sign_mask = simd_srai(fw/2, hi, fw/2 - 1);
    Value * safe = simd_eq(fw/2, sign_mask, bits_that_must_match_sign);
    return simd_if(fw/2, safe, lo, simd_eq(1, top_bit, sign_mask));
}

Value * IDISA_Builder::hsimd_packus(unsigned fw, Value * a, Value * b) {
    Value * hi = hsimd_packh(fw, a, b);
    Value * lo = hsimd_packl(fw, a, b);
    Value * high_mask = simd_gt(fw/2, hi, ConstantVector::getNullValue(a->getType()));
    Value * low_mask = simd_ge(fw/2, hi, ConstantVector::getNullValue(a->getType()));
    return simd_and(simd_or(high_mask, lo), low_mask);
}

Value * IDISA_Builder::hsimd_packh_in_lanes(unsigned lanes, unsigned fw, Value * a, Value * b) {
    if (fw < 16) UnsupportedFieldWidthError(fw, "packh_in_lanes");
    const unsigned fw_out = fw / 2;
    const unsigned fields_per_lane = mBitBlockWidth / (fw_out * lanes);
    const unsigned field_offset_for_b = mBitBlockWidth / fw_out;
    const unsigned field_count = mBitBlockWidth / fw_out;
    SmallVector<Constant *, 16> Idxs(field_count);
    for (unsigned lane = 0, j = 0; lane < lanes; lane++) {
        const unsigned first_field_in_lane = lane * fields_per_lane; // every second field
        for (unsigned i = 0; i < fields_per_lane / 2; i++) {
            Idxs[j++] = getInt32(first_field_in_lane + (2 * i) + 1);
        }
        for (unsigned i = 0; i < fields_per_lane / 2; i++) {
            Idxs[j++] = getInt32(field_offset_for_b + first_field_in_lane + (2 * i) + 1);
        }
    }
    return CreateShuffleVector(fwCast(fw_out, a), fwCast(fw_out, b), ConstantVector::get(Idxs));
}

Value * IDISA_Builder::hsimd_packl_in_lanes(unsigned lanes, unsigned fw, Value * a, Value * b) {
    if (fw < 16) UnsupportedFieldWidthError(fw, "packl_in_lanes");
    const unsigned fw_out = fw / 2;
    const unsigned fields_per_lane = mBitBlockWidth / (fw_out * lanes);
    const unsigned field_offset_for_b = mBitBlockWidth / fw_out;
    const unsigned field_count = mBitBlockWidth / fw_out;
    SmallVector<Constant *, 16> Idxs(field_count);
    for (unsigned lane = 0, j = 0; lane < lanes; lane++) {
        const unsigned first_field_in_lane = lane * fields_per_lane; // every second field
        for (unsigned i = 0; i < fields_per_lane / 2; i++) {
            Idxs[j++] = getInt32(first_field_in_lane + (2 * i));
        }
        for (unsigned i = 0; i < fields_per_lane / 2; i++) {
            Idxs[j++] = getInt32(field_offset_for_b + first_field_in_lane + (2 * i));
        }
    }
    return CreateShuffleVector(fwCast(fw_out, a), fwCast(fw_out, b), ConstantVector::get(Idxs));
}

Value * IDISA_Builder::hsimd_signmask(unsigned fw, Value * a) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "hsimd_signmask");
    Value * a1 = fwCast(fw, a);
    Value * mask = CreateICmpSLT(a1, ConstantAggregateZero::get(a1->getType()));
    mask = CreateBitCast(mask, getIntNTy(mBitBlockWidth/fw));
    if (mBitBlockWidth/fw < 32) return CreateZExt(mask, getInt32Ty());
    else return mask;
}

Value * IDISA_Builder::mvmd_extract(unsigned fw, Value * a, unsigned fieldIndex) {
    if (fw < 8) {
        unsigned byte_no = (fieldIndex * fw) / 8;
        unsigned intrabyte_shift = (fieldIndex * fw) % 8;
        Value * byte = CreateExtractElement(fwCast(8, a), getInt32(byte_no));
        return CreateTrunc(CreateLShr(byte, getInt8(intrabyte_shift)), getIntNTy(fw));
    }
    return CreateExtractElement(fwCast(fw, a), getInt32(fieldIndex));
}

Value * IDISA_Builder::mvmd_insert(unsigned fw, Value * a, Value * elt, unsigned fieldIndex) {
    if (fw < 8) {
        unsigned byte_no = (fieldIndex * fw) / 8;
        unsigned intrabyte_shift = (fieldIndex * fw) % 8;
        unsigned field_mask = ((1 << fw) - 1) << intrabyte_shift;
        Value * byte = CreateAnd(CreateExtractElement(fwCast(8, a), getInt32(byte_no)), getInt8(0xFF &~ field_mask));
        byte = CreateOr(byte, CreateShl(CreateZExtOrTrunc(elt, getInt8Ty()), getInt8(intrabyte_shift)));
        return CreateInsertElement(fwCast(8, a), byte, getInt32(byte_no));
    }
    return CreateInsertElement(fwCast(fw, a), elt, getInt32(fieldIndex));
}

Value * IDISA_Builder::mvmd_slli(unsigned fw, Value * a, unsigned shift) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "mvmd_slli");
    if (shift == 0) return a;
    Value * a1 = fwCast(fw, a);
    Value * r = mvmd_dslli(fw, a1, Constant::getNullValue(a1->getType()), shift);
    assert (r->getType() == a1->getType());
    return r;
}

Value * IDISA_Builder::mvmd_srli(unsigned fw, Value * a, unsigned shift) {
    if (fw < 8) UnsupportedFieldWidthError(fw, "mvmd_srli");
    if (shift == 0) return a;
    const auto field_count = getVectorBitWidth(a) / fw;
    Value * a1 = fwCast(fw, a);
    return mvmd_dslli(fw, Constant::getNullValue(a1->getType()), a1, field_count - shift);
}

Value * IDISA_Builder::mvmd_dslli(unsigned fw, Value * a, Value * b, unsigned shift) {
    if (shift == 0) return a;
    if (fw > 32) {
        return fwCast(fw, mvmd_dslli(32, a, b, shift * (fw/32)));
    } else if (((shift % 2) == 0) && (fw < 32)) {
        return fwCast(fw, mvmd_dslli(2 * fw, a, b, shift / 2));
    }
    if (fw >= 16) {
        const auto field_count = getVectorBitWidth(a) / fw;
        SmallVector<Constant *, 16> Idxs(field_count);
        for (unsigned i = 0; i < field_count; i++) {
            Idxs[i] = getInt32(i + field_count - shift);
        }
        return CreateShuffleVector(fwCast(fw, b), fwCast(fw, a), ConstantVector::get(Idxs));
    } else {
        unsigned field32_shift = (shift * fw) / 32;
        unsigned bit_shift = (shift * fw) % 32;
        Value * const L = simd_slli(32, mvmd_dslli(32, a, b, field32_shift), bit_shift);
        Value * const R = simd_srli(32, mvmd_dslli(32, a, b, field32_shift + 1), 32 - bit_shift);
        return fwCast(fw, CreateOr(L, R));
    }
}


Value * IDISA_Builder::mvmd_shuffle(unsigned fw, Value * table, Value * index_vector) {
    if (fw == mBitBlockWidth) {
        Value * isIndex0 = CreateIsNull(index_vector);
        return CreateSelect(isIndex0, table, ConstantInt::getNullValue(mBitBlockType));
    }
    UnsupportedFieldWidthError(fw, "mvmd_shuffle");
}

Value * IDISA_Builder::mvmd_shuffle2(unsigned fw, Value * table0, Value * table1, Value * index_vector) {
    //  Use two shuffles, with selection by the bit value within the shuffle_table.
    const auto field_count = mBitBlockWidth/fw;
    Constant * selectorSplat = getSplat(field_count, ConstantInt::get(getIntNTy(fw), field_count));
    Value * selectMask = simd_eq(fw, simd_and(index_vector, selectorSplat), selectorSplat);
    Value * idx = simd_and(index_vector, simd_not(selectorSplat));
    return simd_or(simd_and(mvmd_shuffle(fw, table0, idx), simd_not(selectMask)), simd_and(mvmd_shuffle(fw, table1, idx), selectMask));
}


Value * IDISA_Builder::mvmd_compress(unsigned fw, Value * v, Value * select_mask) {
    if (LLVM_UNLIKELY(fw < 8)) {
        UnsupportedFieldWidthError(fw, "mvmd_compress");
    } else {
        IntegerType *  const fieldTy = getIntNTy(fw);
        const auto fieldCount = mBitBlockWidth / fw;
        Type * maskTy = select_mask->getType();
        if (maskTy->isIntegerTy()) {
            if (fieldCount <= fw) {
                SmallVector<Constant *, 16> elements(fieldCount);
                for (unsigned i = 0; i < fieldCount; i++) {
                    elements[i] = ConstantInt::get(fieldTy, 1ULL << i);
                }
                Constant * seq = ConstantVector::get(elements);
                select_mask = simd_eq(fw, simd_and(simd_fill(fw, select_mask), seq), seq);
            } else {
                Value * const spread_mask = esimd_bitspread(fw, select_mask);
                select_mask = simd_any(fw, spread_mask);
            }
        }
        Value * selected = simd_and(v, select_mask);
        Constant * oneSplat = getSplat(fieldCount, ConstantInt::get(fieldTy, 1));
        Value * deletion_counts = simd_add(fw, oneSplat, select_mask);
        Value * deletion_totals = hsimd_partial_sum(fw, deletion_counts);
        unsigned shiftAmount = 1;
        while (shiftAmount < fieldCount) {
            Value * shift_splat = getSplat(fieldCount, ConstantInt::get(fieldTy, shiftAmount));
            Value * shift_select = simd_and(deletion_totals, shift_splat);
            Value * shift_mask = simd_eq(fw, shift_select, shift_splat);
            Value * to_shift = simd_and(shift_mask, selected);
            Value * shifted = mvmd_srli(fw, to_shift, shiftAmount);
            deletion_totals = simd_sub(fw, deletion_totals, shift_select);
            selected = simd_or(shifted, simd_xor(selected, to_shift));
            shiftAmount *= 2;
        }
        return selected;
    }
}

Value * IDISA_Builder::mvmd_expand(unsigned fw, Value * v, Value * select_mask) {
    if (LLVM_UNLIKELY(fw < 8)) {
        UnsupportedFieldWidthError(fw, "mvmd_compress");
    } else {

        IntegerType *  const fieldTy = getIntNTy(fw);
        const auto fieldCount = mBitBlockWidth / fw;
        Type * maskTy = select_mask->getType();
        if (maskTy->isIntegerTy()) {
            select_mask = esimd_bitspread(fw, select_mask);
        }

        Constant * oneSplat = getSplat(fieldCount, ConstantInt::get(fieldTy, 1));
        Value * movements_remaining = hsimd_partial_sum(fw, CreateXor(select_mask, oneSplat));
        assert (movements_remaining->getType() == oneSplat->getType());
        Value * result = nullptr;

        Value * pending = v;
        assert (v->getType() == oneSplat->getType());

        unsigned shiftAmount = fieldCount;
        while (shiftAmount > 0) {

            shiftAmount /= 2;

            Value * shift_splat = getSplat(fieldCount, ConstantInt::get(fieldTy, shiftAmount));
            assert (shift_splat->getType() == oneSplat->getType());
            Value * shift_select = CreateAnd(movements_remaining, shift_splat);
            assert (shift_select->getType() == oneSplat->getType());
            Value * shift_mask = simd_eq(fw, shift_select, shift_splat);
            assert (shift_mask->getType() == oneSplat->getType());
            Value * shifted = CreateAnd(mvmd_slli(fw, pending, shiftAmount), shift_mask);
            assert (shifted->getType() == oneSplat->getType());
            movements_remaining = CreateXor(movements_remaining, shift_select);
            assert (movements_remaining->getType() == oneSplat->getType());
            Value * keep_mask = simd_eq(fw, movements_remaining, shift_select);
            assert (keep_mask->getType() == oneSplat->getType());
            Value * newVals = CreateAnd(pending, keep_mask);
            assert (newVals->getType() == oneSplat->getType());
            if (result) {
                assert (result->getType() == newVals->getType());
                result = CreateOr(result, newVals);
            } else {
                result = newVals;
            }
            Value * A = CreateAnd(pending, CreateNot(shift_mask));
            assert (A->getType() == oneSplat->getType());
            pending = CreateOr(A, shifted);
            assert (pending->getType() == oneSplat->getType());
        }
        return CreateAnd(result, simd_any(fw, select_mask));
    }
}

Value * IDISA_Builder::bitblock_any(Value * a) {
    Type * aType = a->getType();
    if (aType->isIntegerTy()) {
        return CreateICmpNE(a, ConstantInt::getNullValue(aType));
    } else {
        Value * r = simd_ne(mLaneWidth, a,  ConstantInt::getNullValue(mBitBlockType));
        r = hsimd_signmask(mLaneWidth, r);
        return CreateICmpNE(r, ConstantInt::getNullValue(r->getType()), "bitblock_any");
    }
}

// full add producing {carryout, sum}
std::pair<Value *, Value *> IDISA_Builder::bitblock_add_with_carry(Value * a, Value * b, Value * carryin) {
    Type * const carryTy = carryin->getType();
    if (carryTy != mBitBlockType) {
        assert (carryTy->isIntegerTy());
        if (LLVM_LIKELY(carryTy->getIntegerBitWidth() < mLaneWidth)) {
            carryin = CreateZExt(carryin, getIntNTy(mLaneWidth));
        }
        carryin = CreateInsertElement(ConstantVector::getNullValue(a->getType()), carryin, getInt32(0));
    }
    Value * carrygen = simd_and(a, b);
    Value * carryprop = simd_or(a, b);
    Value * sum = simd_add(mBitBlockWidth, simd_add(mBitBlockWidth, a, b), carryin);
    Value * carryout = simd_or(carrygen, simd_and(carryprop, CreateNot(sum)));
    carryout = simd_srli(mBitBlockWidth, carryout, mBitBlockWidth - 1);
    carryout = CreateBitCast(carryout, mBitBlockType);
    if (carryout->getType() != carryTy) {
        carryout = CreateExtractElement(carryout, getInt32(0));
        carryout = CreateZExtOrTrunc(carryout, carryTy);
    }
    return std::pair<Value *, Value *>(carryout, bitCast(sum));
}

// full subtract producing {borrowOut, difference}
std::pair<Value *, Value *> IDISA_Builder::bitblock_subtract_with_borrow(Value * a, Value * b, Value * borrowIn) {
    Value * in = borrowIn;
    Type * borrowInTy = borrowIn->getType();
    if (borrowInTy != mBitBlockType) {
        assert (borrowInTy->isIntegerTy());
        in = CreateZExtOrTrunc(in, mBitBlockType->getElementType());
        in = CreateInsertElement(Constant::getNullValue(mBitBlockType), in, getInt32(0));
    }
    Value * partial = simd_sub(mBitBlockWidth, simd_sub(mBitBlockWidth, a, b), in);
    Value * borrowOut = simd_srli(mBitBlockWidth, partial, mBitBlockWidth - 1);

    borrowOut = CreateBitCast(borrowOut, mBitBlockType);
    if (borrowInTy != mBitBlockType) {
        borrowOut = CreateExtractElement(borrowOut, getInt32(0));
        borrowOut = CreateZExtOrTrunc(borrowOut, borrowInTy);
    }
    return std::make_pair(borrowOut, bitCast(partial));
}

// full shift producing {shiftout, shifted}
std::pair<Value *, Value *> IDISA_Builder::bitblock_advance(Value * const a, Value * shiftin, const unsigned shift) {
    Value * shifted = nullptr;
    Value * shiftout = nullptr;
    Type * shiftTy = shiftin->getType();
    assert (a->getType() == mBitBlockType);
    assert (mBitBlockType->getElementType()->getIntegerBitWidth() == mLaneWidth);
    if (LLVM_UNLIKELY(shift == 0)) {
        return std::pair<Value *, Value *>(Constant::getNullValue(shiftTy), a);
    }
    if (shiftTy != mBitBlockType) {
        assert (shiftTy->isIntegerTy());
        if (LLVM_LIKELY(shiftTy->getIntegerBitWidth() < mLaneWidth)) {
            shiftin = CreateZExt(shiftin, getIntNTy(mLaneWidth));
        }
        shiftin = CreateInsertElement(ConstantVector::getNullValue(a->getType()), shiftin, getInt32(0));
    }
    assert (shiftin->getType() == mBitBlockType);

    auto getShiftout = [&](Value * v) {
        if (v->getType() != shiftTy) {
            v = CreateExtractElement(v, getInt32(0));
            if (LLVM_LIKELY(shiftTy->getIntegerBitWidth() < mLaneWidth)) {
                v = CreateTrunc(v, shiftTy);
            }
        }
        return v;
    };

    if (LLVM_UNLIKELY(shift == mBitBlockWidth)) {
        return std::pair<Value *, Value *>(getShiftout(a), shiftin);
    }
#ifndef LEAVE_CARRY_UNNORMALIZED
    if (LLVM_UNLIKELY((shift % 8) == 0)) { // Use a single whole-byte shift, if possible.
        shifted = CreateOr(bitCast(mvmd_slli(8, a, shift / 8)), shiftin);
        shiftout = bitCast(mvmd_srli(8, a, (mBitBlockWidth - shift) / 8));
    } else {
        Value * shiftback = simd_srli(mLaneWidth, a, mLaneWidth - (shift % mLaneWidth));
        Value * shiftfwd = simd_slli(mLaneWidth, a, shift % mLaneWidth);
        if (LLVM_LIKELY(shift < mLaneWidth)) {
            shiftout = mvmd_srli(mLaneWidth, shiftback, mBitBlockWidth/mLaneWidth - 1);
            shifted = CreateOr(CreateOr(shiftfwd, shiftin), mvmd_slli(mLaneWidth, shiftback, 1));
        } else {
            shiftout = CreateOr(shiftback, mvmd_srli(mLaneWidth, shiftfwd, 1));
            shifted = CreateOr(shiftin, mvmd_slli(mLaneWidth, shiftfwd, (mBitBlockWidth - shift) / mLaneWidth));
            if ((shift + mLaneWidth) < mBitBlockWidth) {
                shiftout = mvmd_srli(mLaneWidth, shiftout, (mBitBlockWidth - shift) / mLaneWidth);
                shifted = CreateOr(shifted, mvmd_slli(mLaneWidth, shiftback, shift/mLaneWidth + 1));
            }
        }
    }
#else
    shiftout = a;
    if (LLVM_UNLIKELY((shift % 8) == 0)) { // Use a single whole-byte shift, if possible.
        shifted = mvmd_dslli(8, a, shiftin, (mBitBlockWidth - shift) / 8);
    }
    else if (LLVM_LIKELY(shift < mLaneWidth)) {
        Value * ahead = mvmd_dslli(mLaneWidth, a, shiftin, mBitBlockWidth / mLaneWidth - 1);
        shifted = simd_or(simd_srli(mLaneWidth, ahead, mLaneWidth - shift), simd_slli(mLaneWidth, a, shift));
    }
    else {
        throw std::runtime_error("Unsupported shift.");
    }
#endif
    assert (shifted->getType() == mBitBlockType);
    assert (shiftout->getType() == mBitBlockType);
    return std::pair<Value *, Value *>(getShiftout(shiftout), shifted);
}

// full shift producing {shiftout, shifted}
std::pair<Value *, Value *> IDISA_Builder::bitblock_indexed_advance(Value * strm, Value * index_strm, Value * shiftIn, unsigned shiftAmount) {
    const unsigned bitWidth = getSizeTy()->getBitWidth();
    Type * const iBitBlock = getIntNTy(getBitBlockWidth());
    Value * const shiftVal = getSize(shiftAmount);
    Value * const extracted_bits = simd_pext(bitWidth, strm, index_strm);
    Value * const ix_popcounts = simd_popcount(bitWidth, index_strm);
    const auto n = getBitBlockWidth() / bitWidth;
    FixedVectorType * const vecTy = FixedVectorType::get(getSizeTy(), n);

    Value * carryOut = nullptr;
    Value * result = UndefValue::get(vecTy);
    if (LLVM_LIKELY(shiftAmount < bitWidth)) {
        Value * carry = mvmd_extract(bitWidth, shiftIn, 0);
        for (unsigned i = 0; i < n; i++) {
            Value * ix_popcnt = mvmd_extract(bitWidth, ix_popcounts, i);
            Value * bits = mvmd_extract(bitWidth, extracted_bits, i);
            Value * adv = CreateOr(CreateShl(bits, shiftVal), carry);
            // We have two cases depending on whether the popcount of the index pack is < shiftAmount or not.
            Value * popcount_small = CreateICmpULT(ix_popcnt, shiftVal);
            Value * carry_if_popcount_small =
                CreateOr(CreateShl(bits, CreateSub(shiftVal, ix_popcnt)),
                            CreateLShr(carry, ix_popcnt));
            Value * carry_if_popcount_large = CreateLShr(bits, CreateSub(ix_popcnt, shiftVal));
            carry = CreateSelect(popcount_small, carry_if_popcount_small, carry_if_popcount_large);
            result = mvmd_insert(bitWidth, result, adv, i);
        }
        carryOut = mvmd_insert(bitWidth, allZeroes(), carry, 0);
    }
    else if (shiftAmount <= mBitBlockWidth) {
        // The shift amount is always greater than the popcount of the individual
        // elements that we deal with.   This simplifies some of the logic.
        Value * carry = CreateBitCast(shiftIn, iBitBlock);
        for (unsigned i = 0; i < n; i++) {
            Value * ix_popcnt = mvmd_extract(bitWidth, ix_popcounts, i);
            Value * bits = mvmd_extract(bitWidth, extracted_bits, i);  // All these bits are shifted out (appended to carry).
            result = mvmd_insert(bitWidth, result, mvmd_extract(bitWidth, carry, 0), i);
            carry = CreateLShr(carry, CreateZExt(ix_popcnt, iBitBlock)); // Remove the carry bits consumed, make room for new bits.
            carry = CreateOr(carry, CreateShl(CreateZExt(bits, iBitBlock), CreateZExt(CreateSub(shiftVal, ix_popcnt), iBitBlock)));
        }
        carryOut = carry;
    }
    else {
        // The shift amount is greater than the total popcount.   We will consume popcount
        // bits from the shiftIn value only, and produce a carry out value of the selected bits.
        Value * carry = CreateBitCast(shiftIn, iBitBlock);
        carryOut = ConstantInt::getNullValue(iBitBlock);
        Value * generated = getSize(0);
        for (unsigned i = 0; i < n; i++) {
            Value * ix_popcnt = mvmd_extract(bitWidth, ix_popcounts, i);
            Value * bits = mvmd_extract(bitWidth, extracted_bits, i);  // All these bits are shifted out (appended to carry).
            result = mvmd_insert(bitWidth, result, mvmd_extract(bitWidth, carry, 0), i);
            carry = CreateLShr(carry, CreateZExt(ix_popcnt, iBitBlock)); // Remove the carry bits consumed.
            carryOut = CreateOr(carryOut, CreateShl(CreateZExt(bits, iBitBlock), CreateZExt(generated, iBitBlock)));
            generated = CreateAdd(generated, ix_popcnt);
        }
    }
    return std::pair<Value *, Value *>{bitCast(carryOut), simd_pdep(bitWidth, result, index_strm)};
}

Value * IDISA_Builder::bitblock_mask_from(Value * const position, const bool safe) {
    Value * const originalPos = CreateZExtOrTrunc(position, getSizeTy());
    if (LLVM_UNLIKELY(safe && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Constant * const BLOCK_WIDTH = getSize(mBitBlockWidth);
        CreateAssert(CreateICmpULT(originalPos, BLOCK_WIDTH), "position exceeds block width");
    }
    Value * const pos = safe ? position : CreateAnd(originalPos, getSize(mBitBlockWidth - 1));
    const unsigned fieldWidth = getSizeTy()->getBitWidth();
    const auto fieldCount = mBitBlockWidth / fieldWidth;
    SmallVector<Constant *, 16> posBase(fieldCount);
    for (unsigned i = 0; i < fieldCount; i++) {
        posBase[i] = ConstantInt::get(getSizeTy(), fieldWidth * i);
    }
    Value * const posBaseVec = ConstantVector::get(posBase);
    Value * const positionVec = simd_fill(fieldWidth, pos);
    Value * const fullFieldWidthMasks = CreateSExt(CreateICmpUGT(posBaseVec, positionVec), fwVectorType(fieldWidth));
    Constant * const FIELD_ONES = ConstantInt::getAllOnesValue(getSizeTy());
    Value * const bitField = CreateShl(FIELD_ONES, CreateAnd(pos, getSize(fieldWidth - 1)));
    Value * const fieldNo = CreateLShr(pos, getSize(floor_log2(fieldWidth)));
    Value * result = CreateInsertElement(fullFieldWidthMasks, bitField, fieldNo);
    if (!safe) { // if the originalPos doesn't match the moddedPos then the originalPos must exceed the block width.
        Constant * const VECTOR_ZEROES = Constant::getNullValue(fwVectorType(fieldWidth));
        result = CreateSelect(CreateICmpEQ(originalPos, pos), result, VECTOR_ZEROES);
    }
    return bitCast(result);
}

Value * IDISA_Builder::bitblock_mask_to(Value * const position, const bool safe) {
    Value * const originalPos = CreateZExtOrTrunc(position, getSizeTy());
    if (LLVM_UNLIKELY(safe && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Constant * const BLOCK_WIDTH = getSize(mBitBlockWidth);
        CreateAssert(CreateICmpULT(originalPos, BLOCK_WIDTH), "position exceeds block width");
    }
    Value * const pos = safe ? position : CreateAnd(originalPos, getSize(mBitBlockWidth - 1));
    const unsigned fieldWidth = getSizeTy()->getBitWidth();
    const auto fieldCount = mBitBlockWidth / fieldWidth;
    SmallVector<Constant *, 16> posBase(fieldCount);
    for (unsigned i = 0; i < fieldCount; i++) {
        posBase[i] = ConstantInt::get(getSizeTy(), fieldWidth * i);
    }
    Value * const posBaseVec = ConstantVector::get(posBase);
    Value * const positionVec = simd_fill(fieldWidth, pos);
    Value * const fullFieldWidthMasks = CreateSExt(CreateICmpULT(posBaseVec, positionVec), fwVectorType(fieldWidth));
    Constant * const FIELD_ONES = ConstantInt::getAllOnesValue(getSizeTy());
    Value * const bitField = CreateLShr(FIELD_ONES, CreateAnd(getSize(fieldWidth - 1), CreateNot(pos)));
    Value * const fieldNo = CreateLShr(pos, getSize(floor_log2(fieldWidth)));
    Value * result = CreateInsertElement(fullFieldWidthMasks, bitField, fieldNo);
    if (!safe) { // if the originalPos doesn't match the moddedPos then the originalPos must exceed the block width.
        Constant * const VECTOR_ONES = Constant::getAllOnesValue(fwVectorType(fieldWidth));
        result = CreateSelect(CreateICmpEQ(originalPos, pos), result, VECTOR_ONES);
    }
    return bitCast(result);
}

Value * IDISA_Builder::bitblock_set_bit(Value * const position, const bool safe) {
    Value * const originalPos = CreateZExtOrTrunc(position, getSizeTy());
    if (LLVM_UNLIKELY(safe && codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Constant * const BLOCK_WIDTH = getSize(mBitBlockWidth);
        CreateAssert(CreateICmpULT(originalPos, BLOCK_WIDTH), "position exceeds block width");
    }
    const unsigned fieldWidth = getSizeTy()->getBitWidth();
    Value * const bitField = CreateShl(getSize(1), CreateAnd(originalPos, getSize(fieldWidth - 1)));
    Value * const pos = safe ? position : CreateAnd(originalPos, getSize(mBitBlockWidth - 1));
    Value * const fieldNo = CreateLShr(pos, getSize(floor_log2(fieldWidth)));
    Constant * const VECTOR_ZEROES = Constant::getNullValue(fwVectorType(fieldWidth));
    Value * result = CreateInsertElement(VECTOR_ZEROES, bitField, fieldNo);
    if (!safe) { // If the originalPos doesn't match the moddedPos then the originalPos must exceed the block width.
        result = CreateSelect(CreateICmpEQ(originalPos, pos), result, VECTOR_ZEROES);
    }
    return bitCast(result);
}

Value * IDISA_Builder::bitblock_popcount(Value * const to_count) {
    const auto fieldWidth = getSizeTy()->getBitWidth();
    auto fields = (getBitBlockWidth() / fieldWidth);
    Value * fieldCounts = simd_popcount(fieldWidth, to_count);
    while (fields > 1) {
        fields /= 2;
        fieldCounts = CreateAdd(fieldCounts, mvmd_srli(fieldWidth, fieldCounts, fields));
    }
    return mvmd_extract(fieldWidth, fieldCounts, 0);
}

Value * IDISA_Builder::simd_and(Value * a, Value * b, StringRef s) {
    return a->getType() == b->getType() ? CreateAnd(a, b, s) : CreateAnd(bitCast(a), bitCast(b), s);
}

Value * IDISA_Builder::simd_or(Value * a, Value * b, StringRef s) {
    return a->getType() == b->getType() ? CreateOr(a, b, s) : CreateOr(bitCast(a), bitCast(b), s);
}

Value * IDISA_Builder::simd_xor(Value * a, Value * b, StringRef s) {
    return a->getType() == b->getType() ? CreateXor(a, b, s) : CreateXor(bitCast(a), bitCast(b), s);
}

Value * IDISA_Builder::simd_not(Value * a, StringRef s) {
    return simd_xor(a, Constant::getAllOnesValue(a->getType()), s);
}

Constant * IDISA_Builder::bit_interleave_byteshuffle_table(unsigned fw) {
    const unsigned fieldCount = mNativeBitBlockWidth/8;
    if (fw > 2) report_fatal_error("bit_interleave_byteshuffle_table requires fw == 1 or fw == 2");
    // Bit interleave using shuffle.
    // Make a shuffle table that translates the lower 4 bits of each byte in
    // order to spread out the bits: xxxxdcba => .d.c.b.a (fw = 1)
    SmallVector<Constant *, 64> bit_interleave(fieldCount);
    for (unsigned i = 0; i < fieldCount; i++) {
        if (fw == 1)
            bit_interleave[i] = getInt8((i & 1) | ((i & 2) << 1) | ((i & 4) << 2) | ((i & 8) << 3));
        else bit_interleave[i] = getInt8((i & 3) | ((i & 0x0C) << 2));
    }
    return ConstantVector::get(bit_interleave);
}

IDISA_Builder::IDISA_Builder(LLVMContext & C, const FeatureSet &featureSet, unsigned nativeVectorWidth,
                             unsigned vectorWidth, unsigned laneWidth, unsigned maxShiftFw, unsigned minShiftFw)
: CBuilder(C)
, mNativeBitBlockWidth(nativeVectorWidth)
, mBitBlockWidth(vectorWidth)
, mLaneWidth(laneWidth)
, MAX_NATIVE_SIMD_SHIFT(maxShiftFw)
, MIN_NATIVE_SIMD_SHIFT(minShiftFw)
, mBitBlockType(FixedVectorType::get(IntegerType::get(C, mLaneWidth), vectorWidth / mLaneWidth))
, mZeroInitializer(Constant::getNullValue(mBitBlockType))
, mOneInitializer(Constant::getAllOnesValue(mBitBlockType))
, mPrintRegisterFunction(nullptr)
, mFeatureSet(featureSet) {

}

IDISA_Builder::~IDISA_Builder() {

}

}
