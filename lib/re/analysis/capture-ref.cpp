/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <re/analysis/capture-ref.h>
#include <re/adt/adt.h>
#include <re/analysis/re_analysis.h>
#include <re/alphabet/alphabet.h>
#include <llvm/Support/raw_ostream.h>
#include <limits.h>

using namespace llvm;

namespace re {

using CapturePostfixMap = std::map<const Capture *, std::vector<const RE *>>;

void updateCaptures(const RE * re, CapturePostfixMap & cm) {
    CapturePostfixMap cm1 = cm;
    for (auto & mapping : cm1) {
        mapping.second.push_back(re);
    }
    if (const Capture * c = dyn_cast<Capture>(re)) {
        cm1.emplace(c, std::vector<const RE *>{});
    }
}

void updateReferenceMap(const RE * re, CapturePostfixMap cm, ReferenceMap & rm) {
    if (const Name * name = dyn_cast<Name>(re)) {
        if (LLVM_LIKELY(name->getDefinition() != nullptr)) {
            return updateReferenceMap(name->getDefinition(), cm, rm);
        } else {
            UndefinedNameError(name);
        }
    } else if (const Seq * seq = dyn_cast<Seq>(re)) {
        if (!seq->empty()) {
            RE * r1 = seq->front();
            updateReferenceMap(r1, cm, rm);
            CapturePostfixMap cm1 = cm;  // copy
            for (auto it = seq->begin()+1; it != seq->end(); it++)  {
                updateCaptures(r1, cm1);
                r1 = *it;
                updateReferenceMap(r1, cm1, rm);
            }
        }
    } else if (const Alt * alt = dyn_cast<Alt>(re)) {
        for (auto & ai : *alt) {
            updateReferenceMap(ai, cm, rm);
        }
    } else if (const Rep * rep = dyn_cast<Rep>(re)) {
        updateReferenceMap(rep->getRE(), cm, rm);
    } else if (const Diff * diff = dyn_cast<Diff>(re)) {
        updateReferenceMap(diff->getLH(), cm, rm);
        updateReferenceMap(diff->getRH(), cm, rm);
    } else if (const Intersect * ix = dyn_cast<Intersect>(re)) {
        updateReferenceMap(ix->getLH(), cm, rm);
        updateReferenceMap(ix->getRH(), cm, rm);
    } else if (const Reference * ref = dyn_cast<Reference>(re)) {
        auto f = cm.find(ref->getCapture());
        if (f != cm.end()) {
            rm.emplace(ref, f->second);
        } else {
            // reference without a defined capture
            //rm.emplace(ref, nullptr);
        }
    } else if (const PropertyExpression * pe = dyn_cast<PropertyExpression>(re)) {
        // Future extension:  property reference such as \p{Sc=\1} (same script as
        // the capture char.
    }
}

ReferenceMap buildReferenceMap(const RE * re) {
    ReferenceMap rm;
    CapturePostfixMap cm;
    updateReferenceMap(re, cm, rm);
    for (auto & mapping : rm) {
        llvm::errs() << Printer_RE::PrintRE(mapping.first) << ":\n";
        for (auto & r : mapping.second) {
            llvm::errs() << Printer_RE::PrintRE(r);
        }
        llvm::errs() << "\n_______\n";
    }
    return rm;
}

static std::pair<int, int> getLengthRange(std::vector<const RE *> v) {
    int lo_len = 0;
    int hi_len = 0;
    for (auto & r: v) {
        auto rg = getLengthRange(r, &cc::Unicode);
        lo_len += rg.first;
        if (rg.second == INT_MAX) hi_len = INT_MAX;
        if (hi_len != INT_MAX) hi_len += rg.second;
    }
    return std::make_pair(lo_len, hi_len);
}

std::set<unsigned> referenceDistances(ReferenceMap rm) {
    std::set<unsigned> distances;
    for (auto & mapping : rm) {
        auto rg1 = getLengthRange(mapping.first->getCapture(), &cc::Unicode);
        auto rg2 = getLengthRange(mapping.second);
        if ((rg1.first == rg1.second) && (rg2.first == rg2.second)) {
            distances.insert(rg1.first + rg2.first);
        }
    }
    return distances;
}

}
