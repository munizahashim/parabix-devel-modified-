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

VariableLengthCCNamer::VariableLengthCCNamer(unsigned UTF_bits) : NameIntroduction("VariableLengthCCNamer") {
        mEncoder.setCodeUnitBits(UTF_bits);
    }

RE * VariableLengthCCNamer::transformCC (CC * cc) {
    bool variable_length = false;
    variable_length = mEncoder.encoded_length(lo_codepoint(cc->front())) < mEncoder.encoded_length(hi_codepoint(cc->back()));
    if (variable_length) {
        return createName(cc->canonicalName(), cc);
    }
    return cc;
}

FixedSpanNamer::FixedSpanNamer(const cc::Alphabet * a) : NameIntroduction("FixedSpanNamer"), mAlphabet(a), mLgthPrefix("len") {}

RE * FixedSpanNamer::transform(RE * r) {
    auto rg = getLengthRange(r, mAlphabet);
    if ((rg.first == rg.second) && (rg.first > 0)) {
        Name * n = createName(mLgthPrefix + std::to_string(rg.first), r);
        return n;
    }
    if (Alt * alt = dyn_cast<Alt>(r)) {
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
    return r;
}

UniquePrefixNamer::UniquePrefixNamer() : NameIntroduction("UniquePrefixNamer") {}

RE * UniquePrefixNamer::transform(RE * r) {
    if (Alt * alt = dyn_cast<Alt>(r)) {
        std::vector<RE *> alts;
        bool fixedPrefixFound = false;
        for (auto e : *alt) {
            RE * prefix, * suffix;
            std::tie(prefix, suffix) = ParseUniquePrefix(e);
            if (isEmptySeq(prefix) || isEmptySeq(suffix)) {
                alts.push_back(e);
            } else {
                fixedPrefixFound = true;
                std::string prefixName = Printer_RE::PrintRE(prefix);
                Name * pfx = makeName(prefixName, prefix);
                std::string altName = Printer_RE::PrintRE(e);
                Name * n = createName(altName, makeSeq({pfx, suffix}));
                alts.push_back(n);
            }
        }
        if (fixedPrefixFound) {
            return makeAlt(alts.begin(), alts.end());
        }
        return r;
    }
    RE * prefix, * suffix;
    std::tie(prefix, suffix) = ParseUniquePrefix(r);
    if (isEmptySeq(prefix) || isEmptySeq(suffix)) {
        return r;
    }
    std::string prefixName = Printer_RE::PrintRE(prefix);
    Name * pfx = makeName(prefixName, prefix);
    std::string rName = Printer_RE::PrintRE(r);
    return createName(rName, makeSeq({pfx, suffix}));
}


class Canonical_External_Names : public RE_Transformer {
public:
    Canonical_External_Names(std::vector<std::string> & external_names);
protected:
    RE * transformName (Name * n) override;
private:
    std::map<std::string, Name *>  mExternalMap;
};

Canonical_External_Names::Canonical_External_Names(std::vector<std::string> & external_names)
: RE_Transformer("Canonical_External_Names") {
    for (unsigned i = 0; i < external_names.size(); i++) {
        mExternalMap.emplace(external_names[i], makeName("@" + std::to_string(i)));
    }
}

RE * Canonical_External_Names::transformName(Name * name) {
    auto f = mExternalMap.find(name->getFullName());
    if (f == mExternalMap.end()) return name;
    Name * canon_name = f->second;
    if (canon_name->getDefinition() == nullptr) {
        canon_name->setDefinition(name->getDefinition());
    }
    return canon_name;
}

RE * canonicalizeExternals(RE * r, std::vector<std::string> & external_names) {
    return Canonical_External_Names(external_names).transformRE(r);
}

}

