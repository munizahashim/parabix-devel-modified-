#pragma once

#include <unicode/core/unicode_set.h>

namespace UCD {
const UnicodeSet * getRadicalSet(codepoint_t radical_cp);
}

