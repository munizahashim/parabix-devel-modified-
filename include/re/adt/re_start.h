/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <re/adt/re_re.h>

namespace re {

class Start : public RE {
public:
    static Start * Create() {return new Start();}
    RE_SUBTYPE(Start)
private:
    Start() : RE(ClassTypeId::Start) {}
};

inline Start * makeStart() {return Start::Create();}

}

