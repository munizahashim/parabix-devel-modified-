/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <re/parse/parser.h>

namespace re {
    class FixedStringParser : public RE_Parser {
    public:
        FixedStringParser(const std::string & regular_expression) : RE_Parser(regular_expression) {
            mReSyntax = RE_Syntax::FixedStrings;
        }
        RE * parse_alt () override;
        RE * parse_seq () override;
    };
}

