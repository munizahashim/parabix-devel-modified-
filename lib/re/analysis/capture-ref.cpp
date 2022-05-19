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

using CapturePostfixMap = std::map<Capture *, std::vector<RE *>>;

void updateCaptures(RE * re, CapturePostfixMap & cm) {
    for (auto & mapping : cm) {
        mapping.second.push_back(re);
    }
    if (Capture * c = dyn_cast<Capture>(re)) {
        cm.emplace(c, std::vector<RE *>{});
    }
}

void updateReferenceInfo(RE * re, CapturePostfixMap & cm, ReferenceInfo & info) {
    if (Capture * c = dyn_cast<Capture>(re)) {
        updateReferenceInfo(c->getCapturedRE(), cm, info);
        info.captureRefs.emplace(c, std::vector<std::string>{});
    } else if (Seq * seq = dyn_cast<Seq>(re)) {
        if (!seq->empty()) {
            CapturePostfixMap cm1 = cm;  // copy
            RE * r1 = seq->front();
            updateReferenceInfo(r1, cm1, info);
            for (auto it = seq->begin()+1; it != seq->end(); it++)  {
                updateCaptures(r1, cm1);
                r1 = *it;
                updateReferenceInfo(r1, cm1, info);
            }
        }
    } else if (Alt * alt = dyn_cast<Alt>(re)) {
        for (auto & ai : *alt) {
            updateReferenceInfo(ai, cm, info);
        }
    } else if (Rep * rep = dyn_cast<Rep>(re)) {
        updateReferenceInfo(rep->getRE(), cm, info);
    } else if (Diff * diff = dyn_cast<Diff>(re)) {
        updateReferenceInfo(diff->getLH(), cm, info);
        updateReferenceInfo(diff->getRH(), cm, info);
    } else if (Intersect * ix = dyn_cast<Intersect>(re)) {
        updateReferenceInfo(ix->getLH(), cm, info);
        updateReferenceInfo(ix->getRH(), cm, info);
    } else if (Reference * ref = dyn_cast<Reference>(re)) {
        std::string refName = ref->getName();
        Capture * c = ref->getCapture();
        auto f = cm.find(c);
        if (f != cm.end()) {
            RE * twixt = makeSeq(f->second.begin(), f->second.end());
            info.twixtREs.emplace(refName, twixt);
            auto rl = info.captureRefs.find(c);
            if (rl == info.captureRefs.end()) {
                llvm::report_fatal_error("reference analysis: out of scope reference");
            }
            rl->second.push_back(refName);
        } else {
            llvm::report_fatal_error("reference without capture");
        }
    } else if (PropertyExpression * pe = dyn_cast<PropertyExpression>(re)) {
        // Future extension:  property reference such as \p{Sc=\1} (same script as
        // the capture char.
    }
}

ReferenceInfo buildReferenceInfo(RE * re) {
    ReferenceInfo info;
    CapturePostfixMap cm;
    updateReferenceInfo(re, cm, info);
    return info;
}
}
