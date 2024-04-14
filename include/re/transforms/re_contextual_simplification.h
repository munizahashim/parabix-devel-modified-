/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <re/transforms/re_transformer.h>

namespace re {

RE * simplifyAssertions(RE * r);

class RE_ContextSimplifier : public RE_Transformer {
public:
    inline RE_ContextSimplifier() : RE_Transformer("ContextSimplification") {}
    RE * transformSeq(Seq * s) override;
};

}

