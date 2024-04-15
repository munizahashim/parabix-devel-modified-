/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/transforms/exclude_CC.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <re/adt/adt.h>
#include <re/transforms/re_transformer.h>

using namespace llvm;

namespace re {
 
struct CC_Remover : public RE_Transformer {
    CC_Remover(CC * toExclude, bool processAsserted) : RE_Transformer("Exclude"),
       mExcludedCC(toExclude), mProcessAsserted(processAsserted) {}
    RE * transformCC (CC * cc) override {
        if (cc->getAlphabet() != mExcludedCC->getAlphabet()) {
            return cc;
        }
        if (subset(cc, mExcludedCC)) return makeAlt();
        if (intersects(mExcludedCC, cc)) return subtractCC(cc, mExcludedCC);
        else return cc;
    }
    RE * transformAny (Any * a) override {
        return makeDiff(a, mExcludedCC);
    }
    RE * transformName (Name * name) override {
        RE * defn = name->getDefinition();
        if (!defn || isa<Any>(defn)) return makeDiff(name, mExcludedCC);
        RE * d = transform(defn);
        if (d == defn) return name;
        return d;
    }
    RE * transformPropertyExpression (PropertyExpression * pe) override {
        RE * defn = pe->getResolvedRE();
        if (!defn) return pe;
        RE * d = transform(defn);
        if (d == defn) return pe;
        return d;
    }
    RE * transformAssertion (Assertion * a) override {
        if (!mProcessAsserted) return a;
        RE * a0 = a->getAsserted();
        RE * a1 = transform(a0);
        if (a0 == a1) return a;
        return makeAssertion(a1, a->getKind(), a->getSense());
    }
    CC * mExcludedCC;
    bool mProcessAsserted;
};
    
RE * exclude_CC(RE * re, CC * cc, bool processAsserted) {
    return CC_Remover(cc, processAsserted).transformRE(re);
}
}
