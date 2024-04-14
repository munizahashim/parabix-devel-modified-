#pragma once

#include <string>
#include <set>
#include <vector>

namespace re {

    class RE; class Name;

    void gatherNames(RE * const re, std::set<Name *> & mNameSet);

    std::vector<std::string> gatherExternals(RE * const re);

}
