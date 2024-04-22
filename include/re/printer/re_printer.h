/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

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

