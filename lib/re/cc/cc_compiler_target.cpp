/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <re/cc/cc_compiler_target.h>
#include <toolchain/toolchain.h>

using namespace std;
using namespace llvm;
using namespace codegen;
using namespace pablo;
using namespace re;

namespace cc {

Parabix_CC_Compiler_Builder::Parabix_CC_Compiler_Builder(pablo::PabloBlock * scope, std::vector<pablo::PabloAST *> basisBitSet)
: CC_Compiler(scope) {
    bool use_binary = codegen::CCCOption.compare("binary") == 0;
    // Workaround because Parabix_Ternary_CC_Compiler only works with UTF-8.
    use_binary |= basisBitSet.size() != 8;
    if (use_binary) {
        ccc = std::make_unique<Parabix_CC_Compiler>(scope, basisBitSet);
    } else {
        ccc = std::make_unique<Parabix_Ternary_CC_Compiler>(scope, basisBitSet);
    }
}

PabloAST * Parabix_CC_Compiler_Builder::compileCC(const std::string & canonicalName, const CC *cc, PabloBlock & block) {
    return ccc->compileCC(canonicalName, cc, block);
}

PabloAST * Parabix_CC_Compiler_Builder::compileCC(const std::string & canonicalName, const CC *cc, PabloBuilder & builder) {
    return ccc->compileCC(canonicalName, cc, builder);
}

}
