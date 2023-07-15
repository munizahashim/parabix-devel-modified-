/*
 *  Copyright (c) 2023 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#pragma once
#include <re/transforms/re_transformer.h>

namespace re {

class RE;

RE * removeNullablePrefix(RE * re);

RE * removeNullableSuffix(RE * re);

RE * zeroBoundElimination(RE * re,
                          NameTransformationMode m = NameTransformationMode::None);
}
