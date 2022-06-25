#pragma once
#include <vector>
#include <map>
#include <set>
#include <string>

namespace re {
class RE; class Capture; class Reference; class Name;


// Mapping from capture names to all references to the capture.
using RefMap = std::map<std::string, std::vector<std::string>>;

// Mapping from references to twixt expressions between
// the defining capture and the reference.
using TwixtMap = std::map<std::string, RE *>;

struct ReferenceInfo {
    RefMap captureRefs;
    TwixtMap twixtREs;
};

ReferenceInfo buildReferenceInfo(RE * re);

}
