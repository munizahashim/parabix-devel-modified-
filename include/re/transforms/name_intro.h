/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#pragma once

#include <string>

namespace cc {class Alphabet;}

namespace re {
class RE;

RE * name_variable_length_CCs(RE * r, unsigned UTF_bits = 8);

RE * name_fixed_length_alts(RE * r, const cc::Alphabet * a, std::string pfx = "lgth");

RE * name_start_anchored_alts(RE * r);
}

