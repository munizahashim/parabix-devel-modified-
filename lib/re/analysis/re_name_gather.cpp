#include <re/analysis/re_name_gather.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <re/adt/adt.h>
#include <re/analysis/re_analysis.h>
#include <re/analysis/re_inspector.h>

using namespace llvm;
namespace re {
    
struct NameCollector final : public RE_Inspector {

    NameCollector(std::set<Name *> & nameSet)
    : RE_Inspector()
    , mNameSet(nameSet) {

    }

    void inspectName(Name * n) final {
        mNameSet.insert(n);
    }

private:
    std::set<Name *> & mNameSet;
};

void gatherNames(RE * const re, std::set<Name *> & nameSet) {
    NameCollector collector(nameSet);
    collector.inspectRE(re);
}

struct ExternalCollector final : public RE_Inspector {

    ExternalCollector()
    : RE_Inspector() {}

    void inspectName(Name * n) final {
        ExternalSet.insert(n->getFullName());
    }

    void inspectCC(CC * cc) final {
        auto alpha = cc->getAlphabet();
        if ((alpha == &cc::Unicode) || (alpha == &cc::UTF8)) {
            ExternalSet.insert("basis");
        } else {
            ExternalSet.insert(alpha->getName() + "_basis");
        }
    }

    std::set<std::string> ExternalSet;
};

std::vector<std::string> gatherExternals(RE * const re) {
    ExternalCollector collector;
    collector.inspectRE(re);
    std::vector<std::string> externals;
    for (auto & e : collector.ExternalSet) {
        externals.emplace_back(std::move(e));
    }
    return externals;
}
}
