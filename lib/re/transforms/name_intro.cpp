/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <re/transforms/name_intro.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/ErrorHandling.h>
#include <re/adt/adt.h>
#include <re/transforms/re_transformer.h>
#include <unicode/utf/utf_encoder.h>
#include <memory>

using namespace llvm;

namespace re {

class VariableLengthCCNamer final : public RE_Transformer {
public:
    VariableLengthCCNamer(unsigned UTF_bits) : RE_Transformer("VariableLengthCCNamer") {
        mEncoder.setCodeUnitBits(UTF_bits);
    }
    RE * transformCC (CC * cc) override {
        bool variable_length = false;
        variable_length = mEncoder.encoded_length(lo_codepoint(cc->front())) < mEncoder.encoded_length(hi_codepoint(cc->back()));
        if (variable_length) {
            return makeName(cc->canonicalName(), Name::Type::Unicode, cc);
        }
        return cc;
    }
private:
    UTF_Encoder mEncoder;
};

RE * name_variable_length_CCs(RE * r, unsigned UTF_bits) {
    return VariableLengthCCNamer(UTF_bits).transformRE(r);
}

}

