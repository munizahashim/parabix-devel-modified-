/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/adt/re_range.h>

#include <llvm/Support/raw_ostream.h>
#include <re/adt/adt.h>
#include <llvm/ADT/Twine.h>

using namespace llvm;

namespace re {

RE * makeRange(RE * lo, RE * hi) {
    if (isa<CC>(lo) && isa<CC>(hi)) {
        CC * const cc_lo = cast<CC>(lo);
        CC * const cc_hi = cast<CC>(hi);
        if (LLVM_LIKELY((cc_lo->count() == 1) && (cc_hi->count() == 1))) {
            const auto lo_val = cc_lo->at(0);
            const auto hi_val = cc_hi->at(0);
            if (LLVM_LIKELY(lo_val <= hi_val)) {
                return makeCC(lo_val, hi_val, dyn_cast<CC>(hi)->getAlphabet());
            }
        }
        std::string tmp;
        raw_string_ostream out(tmp);
        cc_lo->print(out);
        out << " to ";
        cc_hi->print(out);
        out << " are invalid range operands";
        report_fatal_error(Twine(out.str()));
    }
    else if (isa<Name>(lo) && (cast<Name>(lo)->getDefinition() != nullptr)) {
        return makeRange(cast<Name>(lo)->getDefinition(), hi);
    }
    else if (isa<Name>(hi) && (cast<Name>(hi)->getDefinition() != nullptr)) {
        return makeRange(lo, cast<Name>(hi)->getDefinition());
    }
    else if (lo == hi) { // TODO: general check for equality, not just instance equality
        return lo;
    }
    return Range::Create(lo, hi);
}
    
}
