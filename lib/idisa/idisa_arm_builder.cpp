#include <idisa/idisa_arm_builder.h>

#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicsAArch64.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Host.h>

using namespace llvm;

namespace IDISA {

std::string IDISA_ARM_Builder::getBuilderUniqueName() { return mBitBlockWidth != 128 ? "ARM_" + std::to_string(mBitBlockWidth) : "ARM";}

Value * IDISA_ARM_Builder::hsimd_signmask(unsigned fw, Value * a) {
  if (getVectorBitWidth(a) == ARM_width) {
    SmallVector<Constant *, 16> shuffle_amount;
    switch(fw) {
      case 64:
        for (int i = 0; i < 2; i++) {
          shuffle_amount.push_back(ConstantInt::get(getInt64Ty(), i));
        }
        break;
      case 32:
        for (int i = 0; i < 4; i++) {
          shuffle_amount.push_back(ConstantInt::get(getInt32Ty(), i));
        }
        break;
      case 8:
        for (int i = 0; i < 16; i++) {
          shuffle_amount.push_back(ConstantInt::get(getInt8Ty(), i));
        }
        break;
      default:
        return IDISA_Builder::hsimd_signmask(fw, a);
    }
    Value * shift = ConstantVector::get(shuffle_amount);
    Value * temp = CreateLShr(fwCast(fw, a), fw-1);
    Value * shift_left_a = CreateShl(fwCast(fw, temp), shift);
    return CreateAddReduce(fwCast(fw, shift_left_a));    
  }
  return IDISA_Builder::hsimd_signmask(fw, a);
}

Value * IDISA_ARM_Builder::mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector) {
  if (mBitBlockWidth == 128 && fw > 8) {
    // Create a table for shuffling with smaller field widths.
    const unsigned fieldCount = mBitBlockWidth/fw;
    Constant * idxMask = getSplat(fieldCount, ConstantInt::get(getIntNTy(fw), fieldCount-1));
    Value * idx = simd_and(index_vector, idxMask);
    unsigned half_fw = fw/2;
    unsigned field_count = mBitBlockWidth/half_fw;
    // Build a ConstantVector of alternating 0 and 1 values.
    SmallVector<Constant *, 16> Idxs(field_count);
    for (unsigned int i = 0; i < field_count; i++) {
      Idxs[i] = ConstantInt::get(getIntNTy(fw/2), i & 1);
    }
    Constant * splat01 = ConstantVector::get(Idxs);
    
    Value * half_fw_indexes = simd_or(idx, mvmd_slli(half_fw, idx, 1));
    half_fw_indexes = simd_add(fw, simd_add(fw, half_fw_indexes, half_fw_indexes), splat01);
    Value * rslt = mvmd_shuffle(half_fw, data_table, half_fw_indexes);
    return rslt;
  }
  if (mBitBlockWidth == 128 && fw == 8) {
    Function * shuf8Func = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_neon_tbl1, FixedVectorType::get(getInt8Ty(), 16));
    return fwCast(8, CreateCall(shuf8Func->getFunctionType(), shuf8Func, {fwCast(8, data_table), fwCast(8, simd_select_lo(fw, index_vector))}));
  }
  return IDISA_Builder::mvmd_shuffle(fw, data_table, index_vector);
}

Value * IDISA_ARM_Builder::mvmd_compress(unsigned fw, Value * a, Value * selector) {
  if ((mBitBlockWidth == 128) && (fw == 64)) {
    Constant * keep[2] = {ConstantInt::get(getInt64Ty(), 1), ConstantInt::get(getInt64Ty(), 3)};
    Constant * keep_mask = ConstantVector::get({keep, 2});
    Constant * shift[2] = {ConstantInt::get(getInt64Ty(), 2), ConstantInt::get(getInt64Ty(), 0)};
    Constant * shifted_mask = ConstantVector::get({shift, 2});
    Value * a_srli1 = mvmd_srli(64, a, 1);
    Value * bdcst = simd_fill(64, CreateZExt(selector, getInt64Ty()));
    Value * kept = simd_and(simd_eq(64, simd_and(keep_mask, bdcst), keep_mask), a);
    Value * shifted = simd_and(a_srli1, simd_eq(64, shifted_mask, bdcst));
    return simd_or(kept, shifted);
    }
    if ((mBitBlockWidth == 128) && (fw == 32)) {
      Value * bdcst = simd_fill(32, CreateZExtOrTrunc(selector, getInt32Ty()));
      Constant * fieldBit[4] =
      {ConstantInt::get(getInt32Ty(), 1), ConstantInt::get(getInt32Ty(), 2),
        ConstantInt::get(getInt32Ty(), 4), ConstantInt::get(getInt32Ty(), 8)};
      Constant * fieldMask = ConstantVector::get({fieldBit, 4});
      Value * a_selected = simd_and(simd_eq(32, fieldMask, simd_and(fieldMask, bdcst)), a);
      Constant * rotateInwards[4] =
      {ConstantInt::get(getInt32Ty(), 1), ConstantInt::get(getInt32Ty(), 0),
        ConstantInt::get(getInt32Ty(), 3), ConstantInt::get(getInt32Ty(), 2)};
      Constant * rotateVector = ConstantVector::get({rotateInwards, 4});
      Value * rotated = CreateShuffleVector(fwCast(32, a_selected), UndefValue::get(fwVectorType(fw)), rotateVector);
      Constant * rotate_bit[2] = {ConstantInt::get(getInt64Ty(), 2), ConstantInt::get(getInt64Ty(), 4)};
      Constant * rotate_mask = ConstantVector::get({rotate_bit, 2});
      Value * rotateControl = simd_eq(64, fwCast(64, simd_and(bdcst, rotate_mask)), allZeroes());
      Value * centralResult = simd_if(1, rotateControl, rotated, a_selected);
      Value * delete_marks_lo = CreateAnd(CreateNot(selector), ConstantInt::get(selector->getType(), 3));
      Value * delCount_lo = CreateSub(delete_marks_lo, CreateLShr(delete_marks_lo, 1));
      return mvmd_srl(32, centralResult, delCount_lo, true);
    }
    return IDISA_Builder::mvmd_compress(fw, a, selector);
}

Value * IDISA_ARM_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {
  if ((fw == 16) && (getVectorBitWidth(a) == ARM_width)) {
    Value * mask = simd_lomask(16);
    return hsimd_packus(fw, fwCast(16, simd_and(a, mask)), fwCast(16, simd_and(b, mask)));
  }
  // Otherwise use default logic.
  return IDISA_Builder::hsimd_packl(fw, a, b);
}

Value * IDISA_ARM_Builder::hsimd_packh(unsigned fw, Value * a, Value * b) {
  if ((fw == 16) && (getVectorBitWidth(a) == ARM_width)) {
    Function * vqmovun_s16_func = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_neon_uqxtn, FixedVectorType::get(getInt8Ty(), 8));
    Value * sat_a = CreateCall(vqmovun_s16_func->getFunctionType(), vqmovun_s16_func, simd_srli(16, a, 8));
    Value * sat_b = CreateCall(vqmovun_s16_func->getFunctionType(), vqmovun_s16_func, simd_srli(16, b, 8));
    return fwCast(8, CreateDoubleVector(sat_a, sat_b));
  }
  // Otherwise use default logic.
  return IDISA_Builder::hsimd_packh(fw, a, b);
}

Value * IDISA_ARM_Builder::hsimd_packus(unsigned fw, Value * a, Value * b) {
  if ((fw == 16) && (getVectorBitWidth(a) == ARM_width)) {
    Function * vqmovun_s16_func = Intrinsic::getDeclaration(getModule(), Intrinsic::aarch64_neon_uqxtn, FixedVectorType::get(getInt8Ty(), 8));
    Value * sat_a = CreateCall(vqmovun_s16_func->getFunctionType(), vqmovun_s16_func, fwCast(16, a));
    Value * sat_b = CreateCall(vqmovun_s16_func->getFunctionType(), vqmovun_s16_func, fwCast(16, b));
    return fwCast(8, CreateDoubleVector(sat_a, sat_b));
  }
  // Otherwise use default logic.
  return IDISA_Builder::hsimd_packus(fw, a, b);
}

#define SHIFT_FIELDWIDTH 64

#define CAST_SHIFT_OUT(shiftout) \
  shiftTy == mBitBlockType ? bitCast(shiftout) : CreateTrunc(CreateBitCast(shiftout, getIntNTy(mBitBlockWidth)), shiftTy)

std::pair<Value *, Value *> IDISA_ARM_Builder::bitblock_advance(Value * a, Value * shiftin, unsigned shift) {
  Value * shifted = nullptr;
  Value * shiftout = nullptr;
  Type * shiftTy = shiftin->getType();
  if (LLVM_UNLIKELY(shift == 0)) {
    return std::pair<Value *, Value *>(Constant::getNullValue(shiftTy), a);
  }
  Value * si = shiftin;
  if (shiftTy != mBitBlockType) {
    si = bitCast(CreateZExt(shiftin, getIntNTy(mBitBlockWidth)));
  }
  if (LLVM_UNLIKELY(shift == mBitBlockWidth)) {
    return std::pair<Value *, Value *>(CreateBitCast(a, shiftTy), si);
  }
#ifndef LEAVE_CARRY_UNNORMALIZED
  if (LLVM_UNLIKELY((shift % 8) == 0)) { // Use a single whole-byte shift, if possible.
    shifted = bitCast(simd_or(mvmd_slli(8, a, shift / 8), si));
    shiftout = bitCast(mvmd_srli(8, a, (mBitBlockWidth - shift) / 8));
    return std::pair<Value *, Value *>(CAST_SHIFT_OUT(shiftout), shifted);
  }
  Value * shiftback = simd_srli(SHIFT_FIELDWIDTH, a, SHIFT_FIELDWIDTH - (shift % SHIFT_FIELDWIDTH));
  Value * shiftfwd = simd_slli(SHIFT_FIELDWIDTH, a, shift % SHIFT_FIELDWIDTH);
  if (LLVM_LIKELY(shift < SHIFT_FIELDWIDTH)) {
    shiftout = mvmd_srli(SHIFT_FIELDWIDTH, shiftback, mBitBlockWidth/SHIFT_FIELDWIDTH - 1);
    shifted = simd_or(simd_or(shiftfwd, si), mvmd_slli(SHIFT_FIELDWIDTH, shiftback, 1));
  }
  else {
    shiftout = simd_or(shiftback, mvmd_srli(SHIFT_FIELDWIDTH, shiftfwd, 1));
    shifted = simd_or(si, mvmd_slli(SHIFT_FIELDWIDTH, shiftfwd, (mBitBlockWidth - shift) / SHIFT_FIELDWIDTH));
    if (shift < mBitBlockWidth - SHIFT_FIELDWIDTH) {
      shiftout = mvmd_srli(SHIFT_FIELDWIDTH, shiftout, (mBitBlockWidth - shift) / SHIFT_FIELDWIDTH);
      shifted = simd_or(shifted, mvmd_slli(SHIFT_FIELDWIDTH, shiftback, shift/SHIFT_FIELDWIDTH + 1));
    }
  }
#endif
#ifdef LEAVE_CARRY_UNNORMALIZED
  shiftout = a;
  if (LLVM_UNLIKELY((shift % 8) == 0)) { // Use a single whole-byte shift, if possible.
    shifted = mvmd_dslli(8, a, shiftin, (mBitBlockWidth - shift) / 8);
  }
  else if (LLVM_LIKELY(shift < SHIFT_FIELDWIDTH)) {
    Value * ahead = mvmd_dslli(SHIFT_FIELDWIDTH, a, shiftin, mBitBlockWidth / SHIFT_FIELDWIDTH - 1);
    shifted = simd_or(simd_srli(SHIFT_FIELDWIDTH, ahead, SHIFT_FIELDWIDTH - shift), simd_slli(SHIFT_FIELDWIDTH, a, shift));
  }
  else {
    throw std::runtime_error("Unsupported shift.");
  }
#endif
  return std::pair<Value *, Value *>(CAST_SHIFT_OUT(shiftout), shifted);
}

/* merge_h arm */
Value * IDISA_ARM_Builder::esimd_mergeh(unsigned fw, Value * a, Value * b) {
  if ((fw == 1) || (fw == 2)) {
    Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
    // Merge the bytes.
    Value * byte_merge = esimd_mergeh(8, a, b);
    Value * low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8))));
    Value * high_bits = simd_slli(16, mvmd_shuffle(8, interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))), fw);
    // For each 16-bit field, interleave the low bits of the two bytes.
    low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8-fw));
    // For each 16-bit field, interleave the high bits of the two bytes.
    high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8-fw));
    return simd_or(low_bits, high_bits);
  }
  // Otherwise use default logic.
  return IDISA_Builder::esimd_mergeh(fw, a, b);
}

/* merge_l arm */
Value * IDISA_ARM_Builder::esimd_mergel(unsigned fw, Value * a, Value * b) {
  if ((fw == 1) || (fw == 2)) {
    Constant * interleave_table = bit_interleave_byteshuffle_table(fw);
    // Merge the bytes.
    Value * byte_merge = esimd_mergel(8, a, b);
    Value * low_bits = mvmd_shuffle(8, interleave_table, fwCast(8, simd_and(byte_merge, simd_lomask(8))));
    Value * high_bits = simd_slli(16, mvmd_shuffle(8, interleave_table, fwCast(8, simd_srli(8, byte_merge, 4))), fw);
    // For each 16-bit field, interleave the low bits of the two bytes.
    low_bits = simd_or(simd_select_lo(16, low_bits), simd_srli(16, low_bits, 8-fw));
    // For each 16-bit field, interleave the high bits of the two bytes.
    high_bits = simd_or(simd_select_hi(16, high_bits), simd_slli(16, high_bits, 8-fw));
    return simd_or(low_bits, high_bits);
  }
  // Otherwise use default logic.
  return IDISA_Builder::esimd_mergel(fw, a, b);
}

}
