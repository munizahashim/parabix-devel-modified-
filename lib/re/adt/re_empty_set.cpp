/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/adt/re_empty_set.h>
#include <re/adt/re_alt.h>

namespace re {

bool isEmptySet(RE * const re) {
    return llvm::isa<Alt>(re) && llvm::cast<Alt>(re)->empty();
}

RE * makeEmptySet() {
    return makeAlt();
}

}
