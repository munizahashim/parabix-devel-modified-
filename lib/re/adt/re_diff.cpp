/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/adt/adt.h>

using namespace llvm;

namespace re {

RE * makeDiff(RE * lh, RE * rh) {
    if (lh == rh) return makeEmptySet();
    if (LLVM_UNLIKELY(isEmptySet(rh))) {
        return lh;
    } else if (isEmptySeq(lh)) {
        if (isEmptySeq(rh)) return makeEmptySet();
        if (isa<Rep>(rh) && (cast<Rep>(rh)->getLB() == 0)) return makeEmptySet();
        if (isa<CC>(rh) || isa<Seq>(rh)) return lh;
    } else if (LLVM_UNLIKELY(isEmptySet(rh))) {
        return lh;
    } else if (LLVM_UNLIKELY(isEmptySet(lh))) {
        return lh;
    } else if (Diff * d = dyn_cast<Diff>(lh)) {
        return makeDiff(d->getLH(), makeAlt({d->getRH(), rh}));
    }
    return Diff::Create(lh, rh);
}

}
