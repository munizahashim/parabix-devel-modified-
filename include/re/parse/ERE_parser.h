/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <re/parse/parser.h>

namespace re {
    class ERE_Parser : public RE_Parser  {
    public:
        ERE_Parser(const std::string & regular_expression) : RE_Parser(regular_expression) {
            mReSyntax = RE_Syntax::ERE;
        }

    protected:
       virtual RE * parse_next_item() override;

       virtual RE * parse_group() override;

       virtual RE * parse_escaped() override;
       
       RE * parse_bracket_expr();

       
    };
}

