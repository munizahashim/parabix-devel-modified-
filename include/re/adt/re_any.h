/*
 *  Copyright (c) 2017 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#ifndef ANY_H
#define ANY_H

#include <re/alphabet/alphabet.h>
#include <re/adt/re_re.h>

namespace re {

class Any : public RE {
public:
    static Any * Create(const cc::Alphabet * a) {return new Any(a);}
    RE_SUBTYPE(Any)
    const cc::Alphabet * getAlphabet() const {return mAlphabet;}
private:
    Any(const cc::Alphabet * a) : RE(ClassTypeId::Any), mAlphabet(a) {}
    const cc::Alphabet * mAlphabet;
};

inline Any * makeAny(const cc::Alphabet * a = &cc::Unicode) {return Any::Create(a);}

}

#endif // ANY_H
