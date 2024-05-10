/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/unicode/charclasses.h>

#include <re/toolchain/toolchain.h>
#include <kernel/core/kernel_builder.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/adt/re_name.h>
#include <unicode/utf/utf_compiler.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>

using NameMap = UTF::UTF_Compiler::NameMap;

using namespace cc;
using namespace kernel;
using namespace pablo;
using namespace re;
using namespace llvm;
using namespace UTF;

std::string makeSignature(const StreamSet * const basis, const std::vector<re::CC *> & ccs) {
    std::string tmp;
    raw_string_ostream out(tmp);
    out << basis->getNumElements() << 'x' << basis->getFieldWidth();
    if (LLVM_LIKELY(!ccs.empty())) {
        char joiner = '[';
        for (const auto & set : ccs) {
            out << joiner;
            set->print(out);
            joiner = ',';
        }
        out << ']';
    }
    out.flush();
    return tmp;
}

CharClassesKernel::CharClassesKernel(KernelBuilder & b, std::vector<CC *> ccs, StreamSet * BasisBits, StreamSet * CharClasses)
: CharClassesKernel(b, makeSignature(BasisBits, ccs), std::move(ccs), BasisBits, CharClasses) {

}

CharClassesKernel::CharClassesKernel(KernelBuilder & b, std::string signature, std::vector<CC *> && ccs, StreamSet * BasisBits, StreamSet * CharClasses)
: PabloKernel(b, "cc_" + getStringHash(signature)
, {Binding{"basis", BasisBits}}, {Binding{"charclasses", CharClasses}})
, mCCs(ccs)
, mSignature(signature) {

}

llvm::StringRef CharClassesKernel::getSignature() const {
    return mSignature;
}

void CharClassesKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    unsigned n = mCCs.size();

    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
    std::vector<Var *> mpx;
    for (unsigned i = 0; i < n; i++) {
        mpx.push_back(pb.createVar("mpx_basis" + std::to_string(i), pb.createZeroes()));
        unicodeCompiler.addTarget(mpx[i], mCCs[i]);
    }
    if (LLVM_UNLIKELY(AlgorithmOptionIsSet(DisableIfHierarchy))) {
        unicodeCompiler.compile(UTF_Compiler::IfHierarchy::None);
    } else {
        unicodeCompiler.compile();
    }
    for (unsigned i = 0; i < mpx.size(); i++) {
        Extract * const r = pb.createExtract(getOutput(0), pb.getInteger(i));
        pb.createAssign(r, pb.createInFile(mpx[i]));
    }
}

ByteClassesKernel::ByteClassesKernel(KernelBuilder & b,
                                     std::vector<re::CC *> ccs,
                                     StreamSet * inputStream,
                                     StreamSet * CharClasses)
: ByteClassesKernel(b, makeSignature(inputStream, ccs), std::move(ccs), inputStream, CharClasses) {

}

ByteClassesKernel::ByteClassesKernel(KernelBuilder & b,
                                     std::string signature,
                                     std::vector<re::CC *> && ccs,
                                     StreamSet * inputStream,
                                     StreamSet * CharClasses)
: PabloKernel(b, "bcc_" + getStringHash(signature)
, {Binding{"basis", inputStream}}, {Binding{"charclasses", CharClasses}})
, mCCs(ccs)
, mSignature(signature) {

}

StringRef ByteClassesKernel::getSignature() const {
    return mSignature;
}

void ByteClassesKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::unique_ptr<cc::CC_Compiler> ccc;

    auto basisBits = getInputStreamSet("basis");
    if (basisBits.size() == 1) {
        ccc = std::make_unique<cc::Direct_CC_Compiler>(getEntryScope(), basisBits[0]);
    } else {
        ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(getEntryScope(), basisBits);
    }
    unsigned n = mCCs.size();

    NameMap nameMap;
    std::vector<Name *> names;
    for (unsigned i = 0; i < n; i++) {
        Name * name = re::makeName("mpx_basis" + std::to_string(i), mCCs[i]);

        nameMap.emplace(name, ccc->compileCC(mCCs[i]));
        names.push_back(name);

    }
    for (unsigned i = 0; i < names.size(); i++) {
        auto t = nameMap.find(names[i]);
        if (t != nameMap.end()) {
            Extract * const r = pb.createExtract(getOutput(0), pb.getInteger(i));
            pb.createAssign(r, pb.createInFile(t->second));
        } else {
            llvm::report_fatal_error("Can't compile character classes.");
        }
    }
}
