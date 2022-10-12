/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#pragma once

#include <re/transforms/re_transformer.h>
#include <re/transforms/name_intro.h>


namespace re {

struct ReferenceInfo; class RE;

struct FixedReferenceTransformer : public NameIntroduction {
public:
    FixedReferenceTransformer(const ReferenceInfo & info) :
        NameIntroduction("FixedReferenceTransformer"), mRefInfo(info) {}
    RE * transformReference(Reference * r) override;
private:
    const ReferenceInfo & mRefInfo;
};

}
