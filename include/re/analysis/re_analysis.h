#ifndef RE_ANALYSIS_H
#define RE_ANALYSIS_H

#include <utility>
namespace re { class RE; class Name; class CC; class Capture; class Reference;}
namespace cc { class Alphabet;}

namespace re {

bool isUnicodeUnitLength(const RE * re);

std::pair<int, int> getLengthRange(const RE * re, const cc::Alphabet * indexingAlphabet);

// Attempt to parse a regular expression into a prefix-suffix pair
// such that any match to the prefix cannot be matched at any
// other position within the RE.   If no such parse is found,
// return a pair consisting of the empty Sequence and the original RE.
std::pair<RE *, RE *> ParseUniquePrefix(RE * r);

bool isFixedLength(const RE * re);

int minMatchLength(const RE * re);

/* Validate that the given RE can be compiled in UTF-8 mode
   without variable advances. */
bool validateFixedUTF8(const RE * r);

bool isTypeForLocal(const RE * re);
    
bool hasAssertion(const RE * re);
    
bool byteTestsWithinLimit(RE * re, unsigned limit);
    
bool hasStartAnchor(const RE * r);

bool hasEndAnchor(const RE * r);
    
bool DefiniteLengthBackReferencesOnly(const RE * re);
    
unsigned grepOffset(const RE * re);
}

#endif // RE_ANALYSIS_H
