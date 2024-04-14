/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <string>
#include <vector>
namespace pablo { class PabloAST; }
namespace pablo { class PabloBuilder; }
namespace pablo { class PabloKernel; }

namespace re {

class Pattern_Compiler {
public:

    Pattern_Compiler(pablo::PabloKernel & kernel);

    void compile(const std::vector<std::string> & patterns, pablo::PabloBuilder & pb, pablo::PabloAST *basisBits[], int dist, unsigned optPosition, int stepSize);

private:

    pablo::PabloKernel & mKernel;
};

}

