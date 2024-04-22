/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#include <unicode/utf/utf_encoder.h>

using codepoint_t = UCD::codepoint_t;

UTF_Encoder::UTF_Encoder(unsigned bits) : mCodeUnitBits(bits) {}

unsigned UTF_Encoder::encoded_length(codepoint_t cp) {
    if (mCodeUnitBits == 8) {
        if (cp <= 0x7F) return 1;
        else if (cp <= 0x7FF) return 2;
        else if (cp <= 0xFFFF) return 3;
        else return 4;
    } else if (mCodeUnitBits == 16) {
        if (cp <= 0xFFFF) return 1;
        else return 2;
    } else return 1;
}

codepoint_t UTF_Encoder::max_codepoint_of_length(unsigned length) {
    if (mCodeUnitBits == 8) {
        if (length == 1) return 0x7F;
        else if (length == 2) return 0x7FF;
        else if (length == 3) return 0xFFFF;
        else {
            assert(length == 4);
            return 0x10FFFF;
        }
    } else if (mCodeUnitBits == 16) {
        if (length == 1) return 0xFFFF;
        else {
            assert(length == 2);
            return 0x10FFFF;
        }
    } else return 0x10FFFF;
}

bool UTF_Encoder::isLowCodePointAfterNthCodeUnit(codepoint_t cp, unsigned n) {
    if (cp == 0) return true;
    else if (encoded_length(cp - 1) < encoded_length(cp)) return true;
    else return nthCodeUnit(cp - 1, n) != nthCodeUnit(cp, n);
}

bool UTF_Encoder::isHighCodePointAfterNthCodeUnit(codepoint_t cp, unsigned n) {
    if (cp == 0x10FFFF) return true;
    else if (encoded_length(cp + 1) > encoded_length(cp)) return true;
    else return nthCodeUnit(cp + 1, n) != nthCodeUnit(cp, n);
}

unsigned UTF_Encoder::nthCodeUnit(codepoint_t cp, unsigned n) {
    const auto length = encoded_length(cp);
    if (mCodeUnitBits == 8) {
        if (n == 1) {
            switch (length) {
                case 1: return static_cast<unsigned>(cp);
                case 2: return static_cast<unsigned>(0xC0 | (cp >> 6));
                case 3: return static_cast<unsigned>(0xE0 | (cp >> 12));
                case 4: return static_cast<unsigned>(0xF0 | (cp >> 18));
            }
        }
        return static_cast<unsigned>(0x80 | ((cp >> (6 * (length - n))) & 0x3F));
    } else if (mCodeUnitBits == 16) {
        if (length == 1) {
            assert(n == 1);
            return static_cast<unsigned>(cp);
        }
        else if (n == 1)
            return static_cast<unsigned>(0xD800 | ((cp - 0x10000) >> 10));
        else
            return static_cast<unsigned>(0xDC00 | ((cp - 0x10000) & 0x3FF));
    } else {
        assert(n == 1);
        return static_cast<unsigned>(cp);
    }
}

codepoint_t UTF_Encoder::minCodePointWithCommonCodeUnits(codepoint_t cp, unsigned common) {
    const auto length = UTF_Encoder::encoded_length(cp);
    if (mCodeUnitBits == 8) {
        const auto mask = (static_cast<codepoint_t>(1) << (length - common) * 6) - 1;
        const auto lo_cp = cp &~ mask;
        return (lo_cp == 0) ? mask + 1 : lo_cp;
    } else if (mCodeUnitBits == 16) {
        if (length == 1) return 0;
        else if (common == 1) {
            return cp & 0x1FFC00;
        }
        else return cp;
    } else return 0;
}

codepoint_t UTF_Encoder::maxCodePointWithCommonCodeUnits(codepoint_t cp, unsigned common) {
    const auto length = UTF_Encoder::encoded_length(cp);
    if (mCodeUnitBits == 8) {
        const auto mask = (static_cast<codepoint_t>(1) << (length - common) * 6) - 1;
        return cp | mask;
    } else if (mCodeUnitBits == 16) {
        if (length == 1) return 0xFFFF;
        else if (common == 1) {
            return 0x10FFFF;
        }
        else return cp;
    } else return 0x10FFFF;
}
