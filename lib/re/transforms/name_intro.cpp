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

class VariableLengthCCNamer final : public RE_Transformer {
public:
    VariableLengthCCNamer(unsigned UTF_bits) : RE_Transformer("VariableLengthCCNamer") {
        mEncoder.setCodeUnitBits(UTF_bits);
    }
    RE * transformCC (CC * cc) override {
        bool variable_length = false;
        variable_length = mEncoder.encoded_length(lo_codepoint(cc->front())) < mEncoder.encoded_length(hi_codepoint(cc->back()));
        if (variable_length) {
            return makeName(cc->canonicalName(), cc);
        }
        return cc;
    }
private:
    UTF_Encoder mEncoder;
};

RE * name_variable_length_CCs(RE * r, unsigned UTF_bits) {
    return VariableLengthCCNamer(UTF_bits).transformRE(r);
}

RE * name_min_length_alts(RE * r, std::string minLengthPrefix) {
    if (Alt * alt = dyn_cast<Alt>(r)) {
        std::vector<RE *> namedAlts;
        std::map<int, std::vector<RE *>> minLengthAlts;
        for (auto & e : *alt) {
            auto rg = getLengthRange(e, &cc::Unicode);
            auto f = minLengthAlts.find(rg.first);
            if (f == minLengthAlts.end()) {
                minLengthAlts.emplace(rg.first, std::vector<RE *>{e});
            } else {
                f->second.push_back(e);
            }
        }
        for (auto & grp : minLengthAlts) {
            Name * n = makeName(minLengthPrefix + std::to_string(grp.first));
            if (grp.second.size() == 1) {
                n->setDefinition(grp.second[0]);
            } else {
                n->setDefinition(makeAlt(grp.second.begin(), grp.second.end()));
            }
            namedAlts.push_back(n);
        }
        return makeAlt(namedAlts.begin(), namedAlts.end());
    } else return r;
}

}

