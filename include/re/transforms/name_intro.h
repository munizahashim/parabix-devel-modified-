/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#pragma once

#include <string>
#include <map>
#include <re/transforms/re_transformer.h>

namespace cc {class Alphabet;}

namespace re {
class RE; class Name;

class NameIntroduction : public RE_Transformer {
public:
    NameIntroduction(std::string xfrmName) : RE_Transformer(xfrmName) {}
    std::map<std::string, RE *> mNameMap;
protected:
    Name * createName(std::string, RE * defn);
    void showProcessing() override;
};

RE * name_variable_length_CCs(RE * r, unsigned UTF_bits = 8);

RE * name_fixed_length_alts(RE * r, const cc::Alphabet * a, std::string pfx = "lgth");

RE * name_start_anchored_alts(RE * r);
}

