/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once
#include <unicode/core/unicode_set.h>

using codepoint_t = UCD::codepoint_t;

class UTF_Encoder {
public:
    UTF_Encoder(unsigned bits = 8);
    unsigned encoded_length(codepoint_t cp);
    codepoint_t max_codepoint_of_length(unsigned lgth);
    bool isLowCodePointAfterNthCodeUnit(codepoint_t cp, unsigned n);
    bool isHighCodePointAfterNthCodeUnit(codepoint_t cp, unsigned n);
    codepoint_t minCodePointWithCommonCodeUnits(codepoint_t cp, unsigned common);
    codepoint_t maxCodePointWithCommonCodeUnits(codepoint_t cp, unsigned common);
    unsigned nthCodeUnit(codepoint_t cp, unsigned n);
    void setCodeUnitBits(unsigned bits) {mCodeUnitBits = bits;}

private:
    unsigned mCodeUnitBits;
};
