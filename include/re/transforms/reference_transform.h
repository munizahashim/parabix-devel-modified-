/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#pragma once

namespace re {

struct ReferenceInfo; class RE;

RE * fixedReferenceTransform(const ReferenceInfo & info, RE * r);

}
