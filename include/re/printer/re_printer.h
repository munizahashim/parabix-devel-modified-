/*
 *  Copyright (c) 2014 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#ifndef PRINTER_RE_H
#define PRINTER_RE_H

#include <string>
#include <set>

namespace re {
    class RE;
}

class Printer_RE
{
public:
    static const std::string PrintRE(const re::RE *re, std::set<std::string> externals);
    static const std::string PrintRE(const re::RE *re);
private:
    std::set<std::string> mExternals;
};

#endif // PRINTER_RE_H
