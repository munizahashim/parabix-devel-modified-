/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#pragma once

#include <string>
#include <map>
#include <re/transforms/re_transformer.h>
#include <unicode/utf/utf_encoder.h>

namespace cc {class Alphabet;}

namespace re {
class RE; class Name; class Alt; class Seq;

class NameIntroduction : public RE_Transformer {
public:
    NameIntroduction(std::string xfrmName) : RE_Transformer(xfrmName) {}
    std::map<std::string, RE *> mNameMap;
protected:
    Name * createName(std::string, RE * defn);
    void showProcessing() override;
};

class VariableLengthCCNamer final : public NameIntroduction {
public:
    VariableLengthCCNamer(unsigned UTF_bits = 8);
protected:
    RE * transformCC (CC * cc) override;
private:
    UTF_Encoder mEncoder;
};

class FixedLengthAltNamer final : public NameIntroduction {
public:
    FixedLengthAltNamer(const cc::Alphabet * a, std::string lgthPfx = "lgth");
protected:
    RE * transformAlt (Alt * a) override;
private:
    const cc::Alphabet * mAlphabet;
    std::string mLgthPrefix;
};

class StartAnchoredAltNamer final : public NameIntroduction {
public:
    StartAnchoredAltNamer();
protected:
    RE * transformAlt (Alt * a) override;
    RE * transformSeq (Seq * s) override;
};
}
