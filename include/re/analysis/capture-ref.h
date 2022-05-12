#pragma once
#include <vector>
#include <map>
#include <set>

namespace re {
class RE; class Reference;

using ReferenceMap = std::map<const Reference *, std::vector<const RE *>>;

ReferenceMap buildReferenceMap(const RE * re);

std::set<unsigned> referenceDistances(ReferenceMap rm);
}
