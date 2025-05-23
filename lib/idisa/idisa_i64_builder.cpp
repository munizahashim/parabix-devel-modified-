/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <idisa/idisa_i64_builder.h>

using namespace llvm;

namespace IDISA {
    
std::string IDISA_I64_Builder::getBuilderUniqueName() { return mBitBlockWidth != 64 ? "C" + std::to_string(mBitBlockWidth) : "C";}

Value * IDISA_I64_Builder::hsimd_packh(unsigned fw, Value * a, Value * b) {
    Value * a_ = a;
    Value * b_ = b;
    for (unsigned w = fw; w < mBitBlockWidth; w *= 2) {
        Value * himask_odd = simd_and(simd_himask(w), simd_himask(2*w));  // high half of odd fields
        Value * himask_even = simd_and(simd_himask(w), simd_lomask(2*w));  // high half of even fields
        b_ = simd_or(simd_and(b_, himask_odd), simd_slli(mBitBlockWidth, simd_and(b_, himask_even), w/2));
        a_ = simd_or(simd_and(a_, himask_odd), simd_slli(mBitBlockWidth, simd_and(a_, himask_even), w/2));
    }
    return simd_or(b_, simd_srli(mBitBlockWidth, a_, mBitBlockWidth/2));
}

Value * IDISA_I64_Builder::hsimd_packl(unsigned fw, Value * a, Value * b) {
    Value * a_ = a;
    Value * b_ = b;
    for (unsigned w = fw; w < mBitBlockWidth; w *= 2) {
        Value * lomask_odd = simd_and(simd_lomask(w), simd_himask(2*w));  // high half of odd fields
        Value * lomask_even = simd_and(simd_lomask(w), simd_lomask(2*w));  // high half of even fields
        b_ = simd_or(simd_and(b_, lomask_even), simd_srli(mBitBlockWidth, simd_and(b_, lomask_odd), w/2));
        a_ = simd_or(simd_and(a_, lomask_even), simd_srli(mBitBlockWidth, simd_and(a_, lomask_odd), w/2));
    }
    return simd_or(simd_slli(mBitBlockWidth, b_, mBitBlockWidth/2), a_);
}

}
