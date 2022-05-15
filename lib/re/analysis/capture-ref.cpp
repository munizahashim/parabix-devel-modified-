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

void updateReferenceInfo(const RE * re, CapturePostfixMap cm, ReferenceInfo & info) {
    if (const Name * name = dyn_cast<Name>(re)) {
        if (LLVM_LIKELY(name->getDefinition() != nullptr)) {
            return updateReferenceInfo(name->getDefinition(), cm, info);
        } else {
            UndefinedNameError(name);
        }
    } else if (const Capture * c = dyn_cast<Capture>(re)) {
        info.captureRefs.emplace(c, std::vector<const Reference *>{});
        updateReferenceInfo(c->getCapturedRE(), cm, info);
    } else if (const Seq * seq = dyn_cast<Seq>(re)) {
        if (!seq->empty()) {
            RE * r1 = seq->front();
            updateReferenceInfo(r1, cm, info);
            CapturePostfixMap cm1 = cm;  // copy
            for (auto it = seq->begin()+1; it != seq->end(); it++)  {
                updateCaptures(r1, cm1);
                r1 = *it;
                updateReferenceInfo(r1, cm1, info);
            }
        }
    } else if (const Alt * alt = dyn_cast<Alt>(re)) {
        for (auto & ai : *alt) {
            updateReferenceInfo(ai, cm, info);
        }
    } else if (const Rep * rep = dyn_cast<Rep>(re)) {
        updateReferenceInfo(rep->getRE(), cm, info);
    } else if (const Diff * diff = dyn_cast<Diff>(re)) {
        updateReferenceInfo(diff->getLH(), cm, info);
        updateReferenceInfo(diff->getRH(), cm, info);
    } else if (const Intersect * ix = dyn_cast<Intersect>(re)) {
        updateReferenceInfo(ix->getLH(), cm, info);
        updateReferenceInfo(ix->getRH(), cm, info);
    } else if (const Reference * ref = dyn_cast<Reference>(re)) {
        const Capture * c = ref->getCapture();
        auto f = cm.find(c);
        if (f != cm.end()) {
            info.twixtREs.emplace(ref, f->second);
            auto rl = info.captureRefs.find(c);
            if (rl == info.captureRefs.end()) {
                llvm::report_fatal_error("reference analysis: out of scope reference");
            }
            rl->second.push_back(ref);
        } else {
            llvm::report_fatal_error("reference without capture");
        }
    } else if (const PropertyExpression * pe = dyn_cast<PropertyExpression>(re)) {
        // Future extension:  property reference such as \p{Sc=\1} (same script as
        // the capture char.
    }
}

ReferenceInfo buildReferenceMap(const RE * re) {
    ReferenceInfo info;
    CapturePostfixMap cm;
    updateReferenceInfo(re, cm, info);
    for (auto & mapping : info.twixtREs) {
        llvm::errs() << Printer_RE::PrintRE(mapping.first) << ":\n";
        for (auto & r : mapping.second) {
            llvm::errs() << Printer_RE::PrintRE(r);
        }
        llvm::errs() << "\n_______\n";
    }
    return info;
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

std::set<unsigned> referenceDistances(ReferenceInfo info) {
    std::set<unsigned> distances;
    for (auto & mapping : info.twixtREs) {
        auto rg1 = getLengthRange(mapping.first->getCapture(), &cc::Unicode);
        auto rg2 = getLengthRange(mapping.second);
        if ((rg1.first == rg1.second) && (rg2.first == rg2.second)) {
            distances.insert(rg1.first + rg2.first);
        }
    }
    return distances;
}

}
