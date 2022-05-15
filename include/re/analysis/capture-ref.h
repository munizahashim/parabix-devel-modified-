#pragma once
#include <vector>
#include <map>
#include <set>

namespace re {
class RE; class Capture; class Reference;


// Mapping from captures to all references to the capture.
using RefMap = std::map<const Capture *, std::vector<const Reference *>>;

// Mapping from references to twixt expressions between
// the defining capture and the reference.
using TwixtMap = std::map<const Reference *, std::vector<const RE *>>;

struct ReferenceInfo {
    RefMap captureRefs;
    TwixtMap twixtREs;
};

ReferenceInfo buildReferenceInfo(const RE * re);

std::set<unsigned> referenceDistances(ReferenceInfo info);
}
