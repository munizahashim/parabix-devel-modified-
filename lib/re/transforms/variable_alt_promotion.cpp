#include <re/adt/adt.h>
#include <re/alphabet/alphabet.h>
#include <re/analysis/re_analysis.h>
#include <re/transforms/re_transformer.h>
#include <re/transforms/variable_alt_promotion.h>
#include <map>
#include <vector>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace re {

std::vector<RE *> LengthFactorAlts(Alt * alt, const cc::Alphabet * lengthAlphabet) {
    std::vector<RE *> newAlts;
    std::map<int, std::vector<RE *>> fixedLengthAlts;
    for (auto e : *alt) {
        auto rg = getLengthRange(e, lengthAlphabet);
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
    for (auto grp : fixedLengthAlts) {
        RE * fixedAlt;
        if (grp.second.size() == 1) {
            fixedAlt = grp.second[0];
        } else {
            fixedAlt = makeAlt(grp.second.begin(), grp.second.end());
        }
        newAlts.push_back(fixedAlt);
    }
    return newAlts;
}

RE * SeqOfAlt2AltOfSeq(std::vector<std::vector<RE *>> SofA, int from) {
    if (SofA.size() <= from) {
        return makeSeq();
    }
    std::vector<RE *> a1 = SofA[from];
    RE * AoS_tail = SeqOfAlt2AltOfSeq(SofA, from + 1);
    std::vector<RE *> alts;
    if (Alt * alt_tail = dyn_cast<Alt>(AoS_tail)) {
        for (unsigned i = 0; i < a1.size(); i++) {
            for (auto t : *alt_tail) {
                alts.push_back(makeSeq({a1[i], t}));
            }
        }
    } else {
        for (unsigned i = 0; i < a1.size(); i++) {
            alts.push_back(makeSeq({a1[i], AoS_tail}));
        }
    }
    if (alts.size() == 1) return alts[0];
    return makeAlt(alts.begin(), alts.end());
}

class VariableAltPromotor : public RE_Transformer {
public:
    VariableAltPromotor(const cc::Alphabet * lengthAlphabet) : RE_Transformer("VariableAltPromotor"),
        mAlphabet(lengthAlphabet) {}
protected:
    RE * transformSeq(Seq * s) override;
private:
    const cc::Alphabet * mAlphabet;
};

RE * VariableAltPromotor::transformSeq(Seq * s) {
    std::vector<std::vector<RE *>> altGroups;
    for (auto e : *s) {
        if (Alt * a = dyn_cast<Alt>(e)) {
            altGroups.push_back(LengthFactorAlts(a, mAlphabet));
        } else {
            altGroups.push_back(std::vector<RE *>{e});
        }
    }
    return SeqOfAlt2AltOfSeq(altGroups, 0);
}

RE * variableAltPromotion(RE * r, const cc::Alphabet * lengthAlphabet) {
    return VariableAltPromotor(lengthAlphabet).transformRE(r);
}

}