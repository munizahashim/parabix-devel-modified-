/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <re/adt/re_re.h>

namespace re {

class End : public RE {
public:
    static End * Create() {return new End();}
    RE_SUBTYPE(End)
private:
    End() : RE(ClassTypeId::End) {}
};

inline End * makeEnd() {return End::Create();}

}

