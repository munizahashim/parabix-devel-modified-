/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <re/parse/parser.h>
#include <re/parse/ERE_parser.h>

namespace re {
    class BRE_Parser : public ERE_Parser  {
    public:
        BRE_Parser(const std::string & regular_expression) : ERE_Parser(regular_expression) {
            mReSyntax = RE_Syntax::BRE;
        }

    protected:
        RE * parse_alt() override;
        RE * parse_seq() override;
        RE * parse_next_item() override ;
        RE * extend_item(RE * re) override;
        RE * parse_group() override;
        std::pair<int, int> parse_range_bound() override;
    };
}

