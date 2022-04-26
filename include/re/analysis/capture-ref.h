#pragma once
#include <vector>
#include <map>

namespace re {
class RE; class Reference;

using ReferenceMap = std::map<const Reference *, std::vector<const RE *>>;

ReferenceMap buildReferenceMap(const RE * re);

}
