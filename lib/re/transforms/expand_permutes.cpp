#include <re/adt/adt.h>
#include <re/transforms/re_transformer.h>
#include <re/transforms/expand_permutes.h>
#include <re/printer/re_printer.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace re {



class ExpandPermutes final : public RE_Transformer {
public:
    ExpandPermutes() : RE_Transformer("ExpandPermutes") {}
    RE * transformPermute(Permute * p) override;
};

RE * ExpandPermutes::transformPermute(Permute * p) {
    unsigned perm_size = p->size();
    std::vector<RE *> alts;
    for (auto perm : *p) {
        alts.push_back(perm);
    }
    std::vector<RE *> elems;
    RE * anyAlt = makeAlt(alts.begin(), alts.end());
    elems.push_back(makeRep(anyAlt, perm_size, perm_size));
    for (auto perm : *p) {
        RE * negated = makeDiff(anyAlt, perm);
        RE * r = makeSeq({perm, makeRep(negated, 0, perm_size - 1)});
        elems.push_back(makeLookBehindAssertion(r));
    }
    return makeSeq(elems.begin(), elems.end());
}

RE * expandPermutes(RE * r) {
    return ExpandPermutes().transformRE(r);
}

}
