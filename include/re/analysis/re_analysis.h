#ifndef RE_ANALYSIS_H
#define RE_ANALYSIS_H

#include <utility>
namespace re { class RE; class Name; class CC; class Capture; class Reference;}
namespace cc { class Alphabet;}

namespace re {

bool isUnicodeUnitLength(const RE * re);

std::pair<int, int> getLengthRange(const RE * re, const cc::Alphabet * indexingAlphabet);

std::pair<RE *, RE *> ExtractFixedLengthPrefix(RE * r, const cc::Alphabet * a);

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
