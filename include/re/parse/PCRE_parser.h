/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <re/parse/parser.h>

namespace re {
    class PCRE_Parser : public RE_Parser {
    public:
        PCRE_Parser(const std::string & regular_expression) : RE_Parser(regular_expression) {
            mReSyntax = RE_Syntax::PCRE;
        }
    protected:
        virtual bool isSetEscapeChar(char c) override;
    };
}

