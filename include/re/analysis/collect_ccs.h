#ifndef COLLECT_CCS_H
#define COLLECT_CCS_H

#include <vector>
#include <set>
#include <re/analysis/re_inspector.h>

namespace cc { class Alphabet; }
namespace re {

class RE;
class CC;

using CC_Set = std::vector<CC *>;

CC_Set collectCCs(RE * const re, const cc::Alphabet & a,
                  re::NameProcessingMode m = re::NameProcessingMode::None);


using Alphabet_Set = std::set<const cc::Alphabet *>;

void collectAlphabets(RE * const re, Alphabet_Set & s,
                      re::NameProcessingMode m = re::NameProcessingMode::None);

}
#endif
