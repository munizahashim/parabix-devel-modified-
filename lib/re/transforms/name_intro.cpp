/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <re/transforms/name_intro.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <re/adt/adt.h>
#include <re/adt/re_alt.h>
#include <re/alphabet/alphabet.h>
#include <re/analysis/re_analysis.h>
#include <re/transforms/re_transformer.h>
#include <unicode/utf/utf_encoder.h>
#include <map>
#include <memory>

using namespace llvm;

namespace re {

Name * NameIntroduction::createName(std::string name, RE * defn) {
    auto f = mNameMap.find(name);
    if (f == mNameMap.end()) {
        mNameMap.emplace(name, defn);
        return makeName(name, defn);
    } else {
        return makeName(name, f->second);
    }
}

void NameIntroduction::showProcessing() {
    for (auto m: mNameMap) {
        llvm::errs() << "Name " << m.first << " ==> " << Printer_RE::PrintRE(m.second) << "\n";
    }
}

class VariableLengthCCNamer final : public NameIntroduction {
public:
    VariableLengthCCNamer(unsigned UTF_bits) : NameIntroduction("VariableLengthCCNamer") {
        mEncoder.setCodeUnitBits(UTF_bits);
    }
    RE * transformCC (CC * cc) override {
        bool variable_length = false;
        variable_length = mEncoder.encoded_length(lo_codepoint(cc->front())) < mEncoder.encoded_length(hi_codepoint(cc->back()));
        if (variable_length) {
            return createName(cc->canonicalName(), cc);
        }
        return cc;
    }
private:
    UTF_Encoder mEncoder;
};

RE * name_variable_length_CCs(RE * r, unsigned UTF_bits) {
    return VariableLengthCCNamer(UTF_bits).transformRE(r);
}

class FixedLengthAltNamer final : public NameIntroduction {
public:
    FixedLengthAltNamer(const cc::Alphabet * a, std::string lengthPrefix) : NameIntroduction("FixedLengthAltNamer"), mAlphabet(a), mLgthPrefix(lengthPrefix) {}
    RE * transformAlt (Alt * a) override;
private:
    const cc::Alphabet * mAlphabet;
    std::string mLgthPrefix;
};

RE * FixedLengthAltNamer::transformAlt(Alt * alt) {
    if (mInitialRE != alt) return alt;
    std::vector<RE *> newAlts;
    std::map<int, std::vector<RE *>> fixedLengthAlts;
    for (auto e : *alt) {
        auto rg = getLengthRange(e, mAlphabet);
        if (rg.first == 0) return alt;  //  zero-length REs cause problems
        if (rg.first == rg.second) {
            auto f = fixedLengthAlts.find(rg.first);
            if (f == fixedLengthAlts.end()) {
                fixedLengthAlts.emplace(rg.first, std::vector<RE *>{e});
            } else {
                f->second.push_back(e);
            }
        } else {
            newAlts.push_back(e);
        }
    }
    if (fixedLengthAlts.empty()) return alt;
    for (auto grp : fixedLengthAlts) {
        RE * defn;
        if (grp.second.size() == 1) {
            defn = grp.second[0];
        } else {
            defn = makeAlt(grp.second.begin(), grp.second.end());
        }
        Name * n = createName(mLgthPrefix + std::to_string(grp.first), defn);
        newAlts.push_back(n);
    }
    if (newAlts.size() == 1) return newAlts[0];
    return makeAlt(newAlts.begin(), newAlts.end());
}

RE * name_fixed_length_alts(RE * r, const cc::Alphabet * a, std::string lengthPrefix) {
    return FixedLengthAltNamer(a, lengthPrefix).transformRE(r);
}

RE * name_start_anchored_alts(RE * r) {
    if (Seq * s = dyn_cast<Seq>(r)) {
        if (isa<Start>(s->front())) {
            Name * anchored = makeName("StartAnchored");
            anchored->setDefinition(r);
            return anchored;
        }
        return r;
    }
    if (Alt * alt = dyn_cast<Alt>(r)) {
        std::vector<RE *> nonAnchoredAlts;
        std::vector<RE *> anchoredAlts;
        for (auto & e : *alt) {
            if (Seq * s = dyn_cast<Seq>(e)) {
                if (isa<Start>(s->front())) {
                    anchoredAlts.push_back(e);
                    continue;
                }
            }
            nonAnchoredAlts.push_back(e);
        }
        if (anchoredAlts.empty()) return r;
        Name * anchored = makeName("StartAnchored");
        if (anchoredAlts.size() == 1) {
            anchored->setDefinition(anchoredAlts[0]);
        } else {
            anchored->setDefinition(makeAlt(anchoredAlts.begin(), anchoredAlts.end()));
        }
        if (nonAnchoredAlts.empty()) return anchored;
        nonAnchoredAlts.push_back(anchored);
        return makeAlt(nonAnchoredAlts.begin(), nonAnchoredAlts.end());
    }
    return r;
}

}

