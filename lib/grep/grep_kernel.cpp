/*
 *  Copyright (c) 2018 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include <grep/grep_kernel.h>

#include <grep/grep_engine.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/codegenstate.h>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>          // for Ones
#include <pablo/pe_var.h>           // for Var
#include <pablo/pe_zeroes.h>        // for Zeroes
#include <pablo/pe_infile.h>
#include <pablo/pe_advance.h>
#include <pablo/boolean.h>
#include <pablo/pe_count.h>
#include <pablo/pe_matchstar.h>
#include <pablo/pe_pack.h>
#include <pablo/pe_debugprint.h>
#include <re/printer/re_printer.h>
#include <re/adt/re_cc.h>
#include <re/adt/re_name.h>
#include <re/alphabet/alphabet.h>
#include <re/analysis/re_analysis.h>
#include <re/toolchain/toolchain.h>
#include <re/transforms/re_reverse.h>
#include <re/transforms/re_transformer.h>
#include <re/analysis/collect_ccs.h>
#include <re/transforms/exclude_CC.h>
#include <re/transforms/re_multiplex.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/streams_merge.h>
#include <kernel/unicode/boundary_kernels.h>
#include <kernel/unicode/utf8_decoder.h>
#include <kernel/unicode/UCD_property_kernel.h>
#include <re/analysis/re_name_gather.h>
#include <re/unicode/boundaries.h>
#include <re/unicode/re_name_resolve.h>
#include <re/unicode/resolve_properties.h>
#include <kernel/unicode/charclasses.h>
#include <re/cc/cc_compiler.h>         // for CC_Compiler
#include <re/cc/cc_compiler_target.h>
#include <re/cc/cc_kernel.h>
#include <re/alphabet/multiplex_CCs.h>
#include <re/compile/re_compiler.h>
#include <unicode/data/PropertyAliases.h>
#include <unicode/data/PropertyObjectTable.h>

using namespace kernel;
using namespace pablo;
using namespace re;
using namespace llvm;

namespace kernel {
ExternalStreamObject::Allocator ExternalStreamObject::mAllocator;
}


StreamIndexCode ExternalStreamTable::declareStreamIndex(std::string name, StreamIndexCode base, std::string indexStreamName) {
    StreamIndexCode newCode = mStreamIndices.size();
    mStreamIndices.push_back({name, base, indexStreamName});
    mExternalMap.resize(mStreamIndices.size());
    return newCode;
}

StreamIndexCode ExternalStreamTable::getStreamIndex(std::string indexName) {
    for (unsigned i = 0; i < mStreamIndices.size(); i++) {
        if (mStreamIndices[i].name == indexName) return i;
    }
    llvm::report_fatal_error("Undeclared stream index" + indexName);
}

void ExternalStreamTable::declareExternal(StreamIndexCode c, std::string externalName, ExternalStreamObject * ext) {
    auto f = mExternalMap[c].find(externalName);
    if (f != mExternalMap[c].end()) {
        //llvm::errs() << "declareExternal: " << mStreamIndices[c].name << "_" << externalName << " redeclared!\n";
        f->second = ext;
    }
    mExternalMap[c].emplace(externalName, ext);
}

ExternalStreamObject * ExternalStreamTable::lookup(StreamIndexCode c, std::string ssname) {
    auto f = mExternalMap[c].find(ssname);
    if (f == mExternalMap[c].end()) {
        report_fatal_error("Cannot get external stream object " +
                           mStreamIndices[c].name + "_" + ssname);
    }
    return f->second;
}

bool ExternalStreamTable::isDeclared(StreamIndexCode c, std::string ssname) {
    return mExternalMap[c].find(ssname) != mExternalMap[c].end();
}

bool ExternalStreamTable::hasReferenceTo(StreamIndexCode c, std::string ssname) {
    for (auto & entry : mExternalMap[c]) {
        auto params = entry.second->getParameters();
        for (auto p : params) {
            if (p == ssname) return true;
        }
    }
    return false;
}

StreamSet * ExternalStreamTable::getStreamSet(ProgBuilderRef b, StreamIndexCode c, std::string ssname) {
    ExternalStreamObject * ext = lookup(c, ssname);
    if (!ext->isResolved()) {
        std::vector<std::string> paramNames = ext->getParameters();
        StreamIndexCode code = isa<FilterByMaskExternal>(ext) ? mStreamIndices[c].base : c;
        bool all_found = true;
        for (auto & p : paramNames) {
            if ((code == c) && (p == ssname)) {
                llvm::report_fatal_error("Recursion in external resolution: " + ssname);
            }
            auto f = mExternalMap[code].find(p);
            if (f == mExternalMap[code].end()) {
                all_found = false;
            }
        }
        if (all_found) {
            std::vector<StreamSet *> paramStreams;
            for (auto pName : paramNames) {
                paramStreams.push_back(getStreamSet(b, code, pName));
            }
            ext->resolveStreamSet(b, paramStreams);
        } else {
            auto base = mStreamIndices[c].base;
            if (base == c) {
                llvm::report_fatal_error("Cannot resolve " + mStreamIndices[c].name + "_" + ssname);
            }
            mExternalMap[base].emplace(ssname, ext);
            StreamSet * baseSet = getStreamSet(b, base, ssname);
            StreamSet * mask = getStreamSet(b, base, mStreamIndices[c].indexStreamName);
            StreamSet * filtered = b->CreateStreamSet(baseSet->getNumElements());
            FilterByMask(b, mask, baseSet, filtered);
            ext->installStreamSet(filtered);
        }
        StreamSet * s = ext->getStreamSet();
        if (mIllustrator) {
            if (s->getNumElements() == 1) {
                mIllustrator->captureBitstream(b, mStreamIndices[c].name + "_" + ssname, s);
            } else {
                mIllustrator->captureBixNum(b, mStreamIndices[c].name + "_" + ssname, s);
            }
        }
    }
    return ext->getStreamSet();
}

void ExternalStreamTable::resetExternals() {
    for (unsigned i = 0; i < mExternalMap.size(); i++) {
        for (auto & entry : mExternalMap[i]) {
            entry.second->mStreamSet = nullptr;
        }
    }
}

void ExternalStreamTable::resolveExternals(ProgBuilderRef b) {
    for (unsigned i = 0; i < mExternalMap.size(); i++) {
        for (auto & entry : mExternalMap[i]) {
            if (!entry.second->isResolved()) {
                getStreamSet(b, i, entry.first);
            }
        }
    }
}

void ExternalStreamObject::installStreamSet(StreamSet * s) {
    mStreamSet = s;
}

void U21_External::resolveStreamSet(ProgBuilderRef P, std::vector<StreamSet *> inputs) {
    StreamSet * U21 = P->CreateStreamSet(21, 1);
    P->CreateKernelCall<UTF8_Decoder>(inputs[0], U21);
    installStreamSet(U21);
}

void PropertyExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * pStrm  = b->CreateStreamSet(1);
    b->CreateKernelCall<UnicodePropertyKernelBuilder>(mName, inputs[0], pStrm);
    installStreamSet(pStrm);
}

std::vector<std::string> PropertyBoundaryExternal::getParameters() {
    std::string basis_name = UCD::getPropertyFullName(mProperty) + "_basis";
    return std::vector<std::string>{basis_name, "u8index"};
}

void PropertyBoundaryExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * basis = inputs[0];
    StreamSet * index = inputs[1];
    StreamSet * bStrm  = b->CreateStreamSet(1);
    b->CreateKernelCall<BoundaryKernel>(basis, index, bStrm);
    installStreamSet(bStrm);
}

void CC_External::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * ccStrm = b->CreateStreamSet(1);
    std::vector<re::CC *> ccs = {mCharClass};
    b->CreateKernelCall<CharClassesKernel>(ccs, inputs[0], ccStrm);
    installStreamSet(ccStrm);
}
std::pair<int, int> RE_External::getLengthRange() {
    return re::getLengthRange(mRE, mIndexAlphabet);
}

void RE_External::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * reStrm  = b->CreateStreamSet(1);
    auto offset = mGrepEngine->RunGrep(b, mRE, inputs[0], reStrm);
    assert(offset == mOffset);
    installStreamSet(reStrm);
}

std::pair<int, int> StartAnchoredExternal::getLengthRange() {
    return re::getLengthRange(mRE, mIndexAlphabet);
}

void StartAnchoredExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * reStrm  = b->CreateStreamSet(1);
    mOffset = mGrepEngine->RunGrep(b, mRE, inputs[0], reStrm);
    installStreamSet(reStrm);
}

void PropertyDistanceExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * propertyBasis = inputs[0];
    StreamSet * distStrm = b->CreateStreamSet(1);
    b->CreateKernelCall<FixedDistanceMatchesKernel>(mDistance, propertyBasis, distStrm);
    installStreamSet(distStrm);
}

std::vector<std::string> PropertyDistanceExternal::getParameters() {
    if (mProperty == UCD::identity) return {"basis"};
    return {UCD::getPropertyFullName(mProperty) + "_basis"};
}

void PropertyBasisExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    if (mProperty == UCD::identity) {
        StreamSet * u21 = b->CreateStreamSet(21);
        b->CreateKernelCall<UTF8_Decoder>(inputs[0], u21);
        installStreamSet(u21);
    } else {
        UCD::PropertyObject * propObj = UCD::getPropertyObject(mProperty);
        if (auto * obj = dyn_cast<UCD::EnumeratedPropertyObject>(propObj)) {
            std::vector<UCD::UnicodeSet> & bases = obj->GetEnumerationBasisSets();
            std::vector<re::CC *> ccs;
            for (auto & b : bases) ccs.push_back(makeCC(b, &cc::Unicode));
            StreamSet * basis = b->CreateStreamSet(ccs.size());
            b->CreateKernelCall<CharClassesKernel>(ccs, inputs[0], basis);
            installStreamSet(basis);
        }
    }
}

void MultiplexedExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    auto mpx_basis = mAlphabet->getMultiplexedCCs();
    StreamSet * const u8CharClasses = b->CreateStreamSet(mpx_basis.size());
    b->CreateKernelCall<CharClassesKernel>(mpx_basis, inputs[0], u8CharClasses);
    installStreamSet(u8CharClasses);
}

std::vector<std::string> GraphemeClusterBreak::getParameters() {
    return std::vector<std::string>{"UCD:" + getPropertyFullName(UCD::GCB) + "_basis", "Extended_Pictographic"};
}

void GraphemeClusterBreak::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * GCBstream = b->CreateStreamSet(1);
    re::RE * GCB_RE = re::generateGraphemeClusterBoundaryRule();
    GCB_RE = UCD::enumeratedPropertiesToCCs(std::set<UCD::property_t>{UCD::GCB}, GCB_RE);
    GCB_RE = UCD::externalizeProperties(GCB_RE);
    mGrepEngine->RunGrep(b, GCB_RE, nullptr, GCBstream);
    //GraphemeClusterLogic(b, mUTF8_transformer, inputs[0], inputs[1], GCBstream);
    installStreamSet(GCBstream);
}

std::vector<std::string> WordBoundaryExternal::getParameters() {
    return std::vector<std::string>{"basis", "u8index"};
}

void WordBoundaryExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * wb = b->CreateStreamSet(1);
    WordBoundaryLogic(b, inputs[0], inputs[1], wb);
    installStreamSet(wb);
}

std::vector<std::string> FilterByMaskExternal::getParameters() {
    return mParamNames;
}

void FilterByMaskExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * mask = inputs[0];
    StreamSet * toFilter = inputs[1];
    StreamSet * filtered = b->CreateStreamSet(toFilter->getNumElements());
    FilterByMask(b, mask, toFilter, filtered);
    installStreamSet(filtered);
}

std::vector<std::string> FixedSpanExternal::getParameters() {
    return std::vector<std::string>{mMatchMarks};
}

void FixedSpanExternal::resolveStreamSet(ProgBuilderRef b, std::vector<StreamSet *> inputs) {
    StreamSet * matchMarks = inputs[0];
    StreamSet * spans = b->CreateStreamSet(1, 1);
    b->CreateKernelCall<FixedMatchSpansKernel>(mLength, mOffset, matchMarks, spans);
    installStreamSet(spans);
}

void UTF8_index::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::unique_ptr<cc::CC_Compiler> ccc;
    bool useDirectCC = getInput(0)->getType()->getArrayNumElements() == 1;
    if (useDirectCC) {
        ccc = std::make_unique<cc::Direct_CC_Compiler>(getEntryScope(), pb.createExtract(getInput(0), pb.getInteger(0)));
    } else {
        ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(getEntryScope(), getInputStreamSet("source"));
    }

    Zeroes * const ZEROES = pb.createZeroes();
    PabloAST * const u8pfx = ccc->compileCC(makeByte(0xC0, 0xFF));


    Var * const nonFinal = pb.createVar("nonFinal", u8pfx);
    Var * const u8invalid = pb.createVar("u8invalid", ZEROES);
    Var * const valid_pfx = pb.createVar("valid_pfx", u8pfx);

    auto it = pb.createScope();
    pb.createIf(u8pfx, it);
    PabloAST * const u8pfx2 = ccc->compileCC(makeByte(0xC2, 0xDF), it);
    PabloAST * const u8pfx3 = ccc->compileCC(makeByte(0xE0, 0xEF), it);
    PabloAST * const u8pfx4 = ccc->compileCC(makeByte(0xF0, 0xF4), it);

    //
    // Two-byte sequences
    Var * const anyscope = it.createVar("anyscope", ZEROES);
    auto it2 = it.createScope();
    it.createIf(u8pfx2, it2);
    it2.createAssign(anyscope, it2.createAdvance(u8pfx2, 1));


    //
    // Three-byte sequences
    Var * const EF_invalid = it.createVar("EF_invalid", ZEROES);
    auto it3 = it.createScope();
    it.createIf(u8pfx3, it3);
    PabloAST * const u8scope32 = it3.createAdvance(u8pfx3, 1);
    it3.createAssign(nonFinal, it3.createOr(nonFinal, u8scope32));
    PabloAST * const u8scope33 = it3.createAdvance(u8pfx3, 2);
    PabloAST * const u8scope3X = it3.createOr(u8scope32, u8scope33);
    it3.createAssign(anyscope, it3.createOr(anyscope, u8scope3X));

    PabloAST * const advE0 = it3.createAdvance(ccc->compileCC(makeByte(0xE0), it3), 1, "advEO");
    PabloAST * const range80_9F = ccc->compileCC(makeByte(0x80, 0x9F), it3);
    PabloAST * const E0_invalid = it3.createAnd(advE0, range80_9F, "E0_invalid");

    PabloAST * const advED = it3.createAdvance(ccc->compileCC(makeByte(0xED), it3), 1, "advED");
    PabloAST * const rangeA0_BF = ccc->compileCC(makeByte(0xA0, 0xBF), it3);
    PabloAST * const ED_invalid = it3.createAnd(advED, rangeA0_BF, "ED_invalid");

    PabloAST * const EX_invalid = it3.createOr(E0_invalid, ED_invalid);
    it3.createAssign(EF_invalid, EX_invalid);

    //
    // Four-byte sequences
    auto it4 = it.createScope();
    it.createIf(u8pfx4, it4);
    PabloAST * const u8scope42 = it4.createAdvance(u8pfx4, 1, "u8scope42");
    PabloAST * const u8scope43 = it4.createAdvance(u8scope42, 1, "u8scope43");
    PabloAST * const u8scope44 = it4.createAdvance(u8scope43, 1, "u8scope44");
    PabloAST * const u8scope4nonfinal = it4.createOr(u8scope42, u8scope43);
    it4.createAssign(nonFinal, it4.createOr(nonFinal, u8scope4nonfinal));
    PabloAST * const u8scope4X = it4.createOr(u8scope4nonfinal, u8scope44);
    it4.createAssign(anyscope, it4.createOr(anyscope, u8scope4X));
    PabloAST * const F0_invalid = it4.createAnd(it4.createAdvance(ccc->compileCC(makeByte(0xF0), it4), 1), ccc->compileCC(makeByte(0x80, 0x8F), it4));
    PabloAST * const F4_invalid = it4.createAnd(it4.createAdvance(ccc->compileCC(makeByte(0xF4), it4), 1), ccc->compileCC(makeByte(0x90, 0xBF), it4));
    PabloAST * const FX_invalid = it4.createOr(F0_invalid, F4_invalid);
    it4.createAssign(EF_invalid, it4.createOr(EF_invalid, FX_invalid));

    //
    // Invalid cases
    PabloAST * const legalpfx = it.createOr(it.createOr(u8pfx2, u8pfx3), u8pfx4);
    //  Any scope that does not have a suffix byte, and any suffix byte that is not in
    //  a scope is a mismatch, i.e., invalid UTF-8.
    PabloAST * const u8suffix = ccc->compileCC("u8suffix", makeByte(0x80, 0xBF), it);
    PabloAST * const mismatch = it.createXor(anyscope, u8suffix);
    //
    PabloAST * const pfx_invalid = it.createXor(valid_pfx, legalpfx);
    it.createAssign(u8invalid, it.createOr(pfx_invalid, it.createOr(mismatch, EF_invalid)));
    PabloAST * const u8valid = it.createNot(u8invalid, "u8valid");
    //
    //
    it.createAssign(nonFinal, it.createAnd(nonFinal, u8valid));
    //pb.createAssign(nonFinal, pb.createOr(nonFinal, CRLF));
    //PabloAST * unterminatedLineAtEOF = pb.createAtEOF(pb.createAdvance(pb.createNot(LineBreak), 1), "unterminatedLineAtEOF");

    Var * const u8index = getOutputStreamVar("u8index");
    PabloAST * u8final = pb.createInFile(pb.createNot(nonFinal));
    pb.createAssign(pb.createExtract(u8index, pb.getInteger(0)), u8final);
}

UTF8_index::UTF8_index(BuilderRef kb, StreamSet * Source, StreamSet * u8index)
: PabloKernel(kb, "UTF8_index_" + std::to_string(Source->getNumElements()) + "x" + std::to_string(Source->getFieldWidth()),
// input
{Binding{"source", Source}},
// output
{Binding{"u8index", u8index}}) {

}

void GrepKernelOptions::setIndexing(StreamSet * idx) {
    mIndexStream = idx;
}

void GrepKernelOptions::setRE(RE * e) {mRE = e;}
void GrepKernelOptions::setSource(StreamSet * s) {mSource = s;}
void GrepKernelOptions::setCombiningStream(GrepCombiningType t, StreamSet * toCombine){
    mCombiningType = t;
    mCombiningStream = toCombine;
}
void GrepKernelOptions::setResults(StreamSet * r) {mResults = r;}

void GrepKernelOptions::addAlphabet(const cc::Alphabet * a, StreamSet * basis) {
    mAlphabets.emplace_back(a, basis);
}

unsigned round_up_to_blocksize(unsigned offset) {
    unsigned lookahead_blocks = (codegen::BlockSize - 1 + offset)/codegen::BlockSize;
    return lookahead_blocks * codegen::BlockSize;
}

void GrepKernelOptions::addExternal(std::string name, StreamSet * strm, unsigned offset, std::pair<int, int> lengthRange) {
    if (offset == 0) {
        if (mSource) {
            mExternalBindings.emplace_back(name, strm, FixedRate(), ZeroExtended());
        } else {
            mExternalBindings.emplace_back(name, strm);
        }
    } else {
        unsigned ahead = round_up_to_blocksize(offset);
        if (mSource) {
            std::initializer_list<Attribute> attrs{ZeroExtended(), LookAhead(ahead)};
            mExternalBindings.emplace_back(name, strm, FixedRate(), attrs);
        } else {
            mExternalBindings.emplace_back(name, strm, FixedRate(), LookAhead(ahead));
        }
    }
    mExternalOffsets.push_back(offset);
    mExternalLengths.push_back(lengthRange);
}

Bindings GrepKernelOptions::streamSetInputBindings() {
    Bindings inputs;
    if (mSource) {
        inputs.emplace_back(mCodeUnitAlphabet->getName() + "_basis", mSource);
    }
    for (const auto & a : mAlphabets) {
        inputs.emplace_back(a.first->getName() + "_basis", a.second);
    }
    for (const auto & a : mExternalBindings) {
        inputs.emplace_back(a);
    }
    if (mIndexStream) {
        inputs.emplace_back("mIndexing", mIndexStream);
    }
    if (mCombiningType != GrepCombiningType::None) {
        inputs.emplace_back("toCombine", mCombiningStream, FixedRate(), Add1());
    }
    return inputs;
}

Bindings GrepKernelOptions::streamSetOutputBindings() {
    return {Binding{"matches", mResults, FixedRate(), Add1()}};
}

Bindings GrepKernelOptions::scalarInputBindings() {
    return {};
}

Bindings GrepKernelOptions::scalarOutputBindings() {
    return {};
}

GrepKernelOptions::GrepKernelOptions(const cc::Alphabet * codeUnitAlphabet)
: mCodeUnitAlphabet(codeUnitAlphabet) {

}

std::string GrepKernelOptions::makeSignature() {
    std::string tmp;
    raw_string_ostream sig(tmp);
    if (mSource) {
        sig << mSource->getNumElements() << 'x' << mSource->getFieldWidth();
        sig << '/' << mCodeUnitAlphabet->getName();
    }
    for (const auto & e : mExternalBindings) {
        sig << '_' << e.getName();
    }
    for (const auto & a: mAlphabets) {
        sig << '_' << a.first->getName();
    }
    if (mCombiningType == GrepCombiningType::Exclude) {
        sig << "&~";
    } else if (mCombiningType == GrepCombiningType::Include) {
        sig << "|=";
    }
    sig << ':' << Printer_RE::PrintRE(mRE);
    sig.flush();
    return tmp;
}

ICGrepKernel::ICGrepKernel(BuilderRef b, std::unique_ptr<GrepKernelOptions> && options)
: PabloKernel(b, AnnotateWithREflags("ic") + getStringHash(options->makeSignature()),
options->streamSetInputBindings(),
options->streamSetOutputBindings(),
options->scalarInputBindings(),
options->scalarOutputBindings()),
mOptions(std::move(options)),
mSignature(mOptions->makeSignature()) {
    addAttribute(InfrequentlyUsed());
    mOffset = grepOffset(mOptions->mRE);
}

StringRef ICGrepKernel::getSignature() const {
    return mSignature;
}

void ICGrepKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    RE_Compiler re_compiler(getEntryScope(), mOptions->mCodeUnitAlphabet);
    if (mOptions->mSource) {
        std::vector<pablo::PabloAST *> basis_set = getInputStreamSet(mOptions->mCodeUnitAlphabet->getName() + "_basis");
        re_compiler.addAlphabet(mOptions->mCodeUnitAlphabet, basis_set);
    }
    for (unsigned i = 0; i < mOptions->mAlphabets.size(); i++) {
        auto & alpha = mOptions->mAlphabets[i].first;
        auto basis = getInputStreamSet(alpha->getName() + "_basis");
        re_compiler.addAlphabet(alpha, basis);
    }
    if (mOptions->mIndexStream) {
        PabloAST * idxStrm = pb.createExtract(getInputStreamVar("mIndexing"), pb.getInteger(0));
        re_compiler.setIndexing(&cc::Unicode, idxStrm);
    }
    for (unsigned i = 0; i < mOptions->mExternalBindings.size(); i++) {
        auto extName = mOptions->mExternalBindings[i].getName();
        PabloAST * extStrm = pb.createExtract(getInputStreamVar(extName), pb.getInteger(0));
        unsigned offset = mOptions->mExternalOffsets[i];
        std::pair<int, int> lgthRange = mOptions->mExternalLengths[i];
        re_compiler.addPrecompiled(extName, RE_Compiler::ExternalStream(RE_Compiler::Marker(extStrm, offset), lgthRange));
    }
    Var * const final_matches = pb.createVar("final_matches", pb.createZeroes());
    RE_Compiler::Marker matches = re_compiler.compileRE(mOptions->mRE);
    PabloAST * matchResult = matches.stream();
    if (matches.offset() != mOffset) {
        //llvm::errs() << Printer_RE::PrintRE(mOptions->mRE) <<"\n mOffset = " << mOffset << "\n";
        //llvm::report_fatal_error("matches.offset() != mOffset");
    }
    pb.createAssign(final_matches, matchResult);
    Var * const output = pb.createExtract(getOutputStreamVar("matches"), pb.getInteger(0));
    PabloAST * value = nullptr;
    if (mOptions->mCombiningType == GrepCombiningType::None) {
        value = final_matches;
    } else {
        PabloAST * toCombine = pb.createExtract(getInputStreamVar("toCombine"), pb.getInteger(0));
        if (mOptions->mCombiningType == GrepCombiningType::Exclude) {
            value = pb.createAnd(toCombine, pb.createNot(final_matches), "toCombine");
        } else {
            value = pb.createOr(toCombine, final_matches, "toCombine");
        }
    }
    pb.createAssign(output, value);
}

void MatchedLinesKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    auto matchResults = getInputStreamSet("matchResults");
    PabloAST * lineBreaks = pb.createExtract(getInputStreamVar("lineBreaks"), pb.getInteger(0));
    PabloAST * notLB = pb.createNot(lineBreaks);
    PabloAST * match_follow = pb.createMatchStar(matchResults.back(), notLB);
    Var * const matchedLines = getOutputStreamVar("matchedLines");
    pb.createAssign(pb.createExtract(matchedLines, pb.getInteger(0)), pb.createAnd(match_follow, lineBreaks, "matchedLines"));
}

MatchedLinesKernel::MatchedLinesKernel (BuilderRef iBuilder, StreamSet * Matches, StreamSet * LineBreakStream, StreamSet * MatchedLines)
: PabloKernel(iBuilder, "MatchedLines" + std::to_string(Matches->getNumElements()),
// inputs
{Binding{"matchResults", Matches}
,Binding{"lineBreaks", LineBreakStream, FixedRate()}},
// output
{Binding{"matchedLines", MatchedLines}}) {

}

void InvertMatchesKernel::generateDoBlockMethod(BuilderRef iBuilder) {
    Value * input = iBuilder->loadInputStreamBlock("matchedLines", iBuilder->getInt32(0));
    Value * lbs = iBuilder->loadInputStreamBlock("lineBreaks", iBuilder->getInt32(0));
    Value * inverted = iBuilder->CreateAnd(iBuilder->CreateNot(input), lbs, "inverted");
    iBuilder->storeOutputStreamBlock("nonMatches", iBuilder->getInt32(0), inverted);
}

InvertMatchesKernel::InvertMatchesKernel(BuilderRef b, StreamSet * Matches, StreamSet * LineBreakStream, StreamSet * InvertedMatches)
: BlockOrientedKernel(b, "Invert" + std::to_string(Matches->getNumElements()),
// Inputs
{Binding{"matchedLines", Matches},
 Binding{"lineBreaks", LineBreakStream}},
// Outputs
{Binding{"nonMatches", InvertedMatches}},
// Input/Output Scalars and internal state
{}, {}, {}) {

}

FixedMatchSpansKernel::FixedMatchSpansKernel(BuilderRef b, unsigned length, unsigned offset, StreamSet * MatchMarks, StreamSet * MatchSpans)
: PabloKernel(b, "FixedMatchSpansKernel" + std::to_string(MatchMarks->getNumElements()) + "x1_by" + std::to_string(length) + '@' + std::to_string(offset),
{Binding{"MatchMarks", MatchMarks, FixedRate(1), LookAhead(round_up_to_blocksize(length))}}, {Binding{"MatchSpans", MatchSpans}}),
mMatchLength(length), mOffset(offset) {
}

void FixedMatchSpansKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * marks = pb.createExtract(getInputStreamVar("MatchMarks"), pb.getInteger(0));
    Var * matchSpansVar = getOutputStreamVar("MatchSpans");
    // starts of all the matches
    PabloAST * starts = pb.createLookahead(marks, mMatchLength + mOffset - 1);
    // now find all consecutive positions within mMatchLength of any start.
    unsigned consecutiveCount = 1;
    PabloAST * consecutive = starts;
    for (unsigned i = 1; i <= mMatchLength/2; i *= 2) {
        consecutiveCount += i;
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, i),
                                  "consecutive" + std::to_string(consecutiveCount));
    }
    if (consecutiveCount < mMatchLength) {
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, mMatchLength - consecutiveCount),
                                  "consecutive" + std::to_string(mMatchLength));
    }
    pb.createAssign(pb.createExtract(matchSpansVar, 0), consecutive);
}

SpansToMarksKernel::SpansToMarksKernel(BuilderRef b, StreamSet * Spans, StreamSet * Marks)
: PabloKernel(b, "SpansToMarksKernel",
{Binding{"Spans", Spans}}, {Binding{"Marks", Marks}}) {}

void SpansToMarksKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * spans = getInputStreamSet("Spans")[0];
    Var * matchEndsVar = getOutputStreamVar("Marks");
    PabloAST * starts = pb.createAnd(spans, pb.createNot(pb.createAdvance(spans, 1)), "starts");
    PabloAST * follows = pb.createAnd(pb.createAdvance(spans, 1), pb.createNot(spans), "follows");
    pb.createAssign(pb.createExtract(matchEndsVar, 0), starts);
    pb.createAssign(pb.createExtract(matchEndsVar, 1), follows);
}

U8Spans::U8Spans(BuilderRef b, StreamSet * marks, StreamSet * u8index, StreamSet * spans)
: PabloKernel(b, "U8Spans",
{Binding{"marks", marks}, Binding{"u8index", u8index}}, {Binding{"spans", spans}}) {}

void U8Spans::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * marks = getInputStreamSet("marks")[0];
    PabloAST * u8index = getInputStreamSet("u8index")[0];
    PabloAST * spans = pb.createMatchStar(marks, pb.createNot(u8index));
    Var * spansVar = getOutputStreamVar("spans");
    pb.createAssign(pb.createExtract(spansVar, 0), spans);
}

void PopcountKernel::generatePabloMethod() {
    auto pb = getEntryScope();
    const auto toCount = pb->createExtract(getInputStreamVar("toCount"), pb->getInteger(0));
    pablo::Var * countResult = getOutputScalarVar("countResult");

    pb->createAssign(countResult, pb->createCount(pb->createInFile(toCount)));
}

PopcountKernel::PopcountKernel (BuilderRef iBuilder, StreamSet * const toCount, Scalar * countResult)
: PabloKernel(iBuilder, "Popcount",
{Binding{"toCount", toCount}},
{},
{},
{Binding{"countResult", countResult}}) {

}

PabloAST * matchDistanceCheck(PabloBuilder & b, unsigned distance, std::vector<PabloAST *> basis) {
    PabloAST * differ = b.createZeroes();
    for (unsigned i = 0; i < basis.size(); i++) {
        PabloAST * basis_bits_i = basis[i];
        PabloAST * advanced = b.createAdvance(basis_bits_i, distance);
        differ = b.createOr(differ, b.createXor(basis_bits_i, advanced));
    }
    return differ;
}

void FixedDistanceMatchesKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    auto Basis = getInputStreamSet("Basis");
    Var * mismatch = pb.createVar("mismatch", pb.createZeroes());
    if (mHasCheckStream) {
        auto ToCheck = getInputStreamSet("ToCheck")[0];
        auto it = pb.createScope();
        pb.createIf(ToCheck, it);
        PabloAST * differ = matchDistanceCheck(it, mMatchDistance, Basis);
        it.createAssign(mismatch, it.createAnd(ToCheck, differ));
    } else {
        pb.createAssign(mismatch, matchDistanceCheck(pb, mMatchDistance, Basis));
    }
    Var * const MatchVar = getOutputStreamVar("Matches");
    pb.createAssign(pb.createExtract(MatchVar, pb.getInteger(0)), pb.createNot(mismatch, "Matches"));
}

FixedDistanceMatchesKernel::FixedDistanceMatchesKernel (BuilderRef b, unsigned distance, StreamSet * Basis, StreamSet * Matches, StreamSet * ToCheck)
: PabloKernel(b, "Distance_" + std::to_string(distance) + "_Matches_" + std::to_string(Basis->getNumElements()) + "x1" + (ToCheck == nullptr ? "" : "_withCheck"),
// inputs
{Binding{"Basis", Basis}},
// output
{Binding{"Matches", Matches}}), mMatchDistance(distance), mHasCheckStream(ToCheck != nullptr) {
    if (mHasCheckStream) {
        mInputStreamSets.push_back({"ToCheck", ToCheck});
    }
}

void AbortOnNull::generateMultiBlockLogic(BuilderRef b, llvm::Value * const numOfStrides) {
    Module * const m = b->getModule();
    DataLayout DL(m);
    IntegerType * const intPtrTy = DL.getIntPtrType(m->getContext());
    Type * voidPtrTy = b->getVoidPtrTy();
    Type * blockTy = b->getBitBlockType();
    const auto blocksPerStride = getStride() / b->getBitBlockWidth();
    Constant * const BLOCKS_PER_STRIDE = b->getSize(blocksPerStride);
    BasicBlock * const entry = b->GetInsertBlock();
    BasicBlock * const strideLoop = b->CreateBasicBlock("strideLoop");
    BasicBlock * const stridesDone = b->CreateBasicBlock("stridesDone");
    BasicBlock * const nullByteDetection = b->CreateBasicBlock("nullByteDetection");
    BasicBlock * const nullByteFound = b->CreateBasicBlock("nullByteFound");
    BasicBlock * const finalStride = b->CreateBasicBlock("finalStride");
    BasicBlock * const segmentDone = b->CreateBasicBlock("segmentDone");

    Value * const numOfBlocks = b->CreateMul(numOfStrides, BLOCKS_PER_STRIDE);
    Value * itemsToDo = b->getAccessibleItemCount("byteData");
    //
    // Fast loop to prove that there are no null bytes in a multiblock region.
    // We repeatedly combine byte packs using a SIMD unsigned min operation
    // (implemented as a Select/ICmpULT combination).
    //
    Value * byteStreamBasePtr = b->getInputStreamBlockPtr("byteData", b->getSize(0), b->getSize(0));
    Value * outputStreamBasePtr = b->getOutputStreamBlockPtr("untilNull", b->getSize(0), b->getSize(0));

    //
    // We set up a a set of eight accumulators to accumulate the minimum byte
    // values seen at each position in a block.   The initial min value at
    // each position is 0xFF (all ones).
    Value * blockMin[8];
    for (unsigned i = 0; i < 8; i++) {
        blockMin[i] = b->fwCast(8, b->allOnes());
    }
    // If we're in the final block bypass the fast loop.
    b->CreateCondBr(b->isFinal(), finalStride, strideLoop);

    b->SetInsertPoint(strideLoop);
    PHINode * const baseBlockIndex = b->CreatePHI(b->getSizeTy(), 2);
    baseBlockIndex->addIncoming(ConstantInt::get(baseBlockIndex->getType(), 0), entry);
    PHINode * const blocksRemaining = b->CreatePHI(b->getSizeTy(), 2);
    blocksRemaining->addIncoming(numOfBlocks, entry);
    for (unsigned i = 0; i < 8; i++) {
        Value * next = b->CreateBlockAlignedLoad(b->CreateGEP(blockTy, byteStreamBasePtr, {baseBlockIndex, b->getSize(i)}));
        b->CreateBlockAlignedStore(next, b->CreateGEP(blockTy, outputStreamBasePtr, {baseBlockIndex, b->getSize(i)}));
        next = b->fwCast(8, next);
        blockMin[i] = b->CreateSelect(b->CreateICmpULT(next, blockMin[i]), next, blockMin[i]);
    }
    Value * nextBlockIndex = b->CreateAdd(baseBlockIndex, ConstantInt::get(baseBlockIndex->getType(), 1));
    Value * nextRemaining = b->CreateSub(blocksRemaining, ConstantInt::get(blocksRemaining->getType(), 1));
    baseBlockIndex->addIncoming(nextBlockIndex, strideLoop);
    blocksRemaining->addIncoming(nextRemaining, strideLoop);
    b->CreateCondBr(b->CreateICmpUGT(nextRemaining, ConstantInt::getNullValue(blocksRemaining->getType())), strideLoop, stridesDone);

    b->SetInsertPoint(stridesDone);
    // Combine the 8 blockMin values.
    for (unsigned i = 0; i < 4; i++) {
        blockMin[i] = b->CreateSelect(b->CreateICmpULT(blockMin[i], blockMin[i+4]), blockMin[i], blockMin[i+4]);
    }
    for (unsigned i = 0; i < 2; i++) {
        blockMin[i] = b->CreateSelect(b->CreateICmpULT(blockMin[i], blockMin[i+4]), blockMin[i], blockMin[i+2]);
    }
    blockMin[0] = b->CreateSelect(b->CreateICmpULT(blockMin[0], blockMin[1]), blockMin[0], blockMin[1]);
    Value * anyNull = b->bitblock_any(b->simd_eq(8, blockMin[0], b->allZeroes()));

    b->CreateCondBr(anyNull, nullByteDetection, segmentDone);


    b->SetInsertPoint(finalStride);
    b->CreateMemCpy(b->CreatePointerCast(outputStreamBasePtr, voidPtrTy), b->CreatePointerCast(byteStreamBasePtr, voidPtrTy), itemsToDo, 1);
    b->CreateBr(nullByteDetection);

    b->SetInsertPoint(nullByteDetection);
    //  Find the exact location using memchr, which should be fast enough.
    //
    Value * ptrToNull = b->CreateMemChr(b->CreatePointerCast(byteStreamBasePtr, voidPtrTy), b->getInt32(0), itemsToDo);
    Value * ptrAddr = b->CreatePtrToInt(ptrToNull, intPtrTy);
    b->CreateCondBr(b->CreateICmpEQ(ptrAddr, ConstantInt::getNullValue(intPtrTy)), segmentDone, nullByteFound);

    // A null byte has been located; set the termination code and call the signal handler.
    b->SetInsertPoint(nullByteFound);
    Value * nullPosn = b->CreateSub(b->CreatePtrToInt(ptrToNull, intPtrTy), b->CreatePtrToInt(byteStreamBasePtr, intPtrTy));
    b->setFatalTerminationSignal();
    Function * const dispatcher = m->getFunction("signal_dispatcher"); assert (dispatcher);
    Value * handler = b->getScalarField("handler_address");
    b->CreateCall(dispatcher, {handler, ConstantInt::get(b->getInt32Ty(), static_cast<unsigned>(grep::GrepSignal::BinaryFile))});
    b->CreateBr(segmentDone);

    b->SetInsertPoint(segmentDone);
    PHINode * const produced = b->CreatePHI(b->getSizeTy(), 3);
    produced->addIncoming(nullPosn, nullByteFound);
    produced->addIncoming(itemsToDo, stridesDone);
    produced->addIncoming(itemsToDo, nullByteDetection);
    Value * producedCount = b->getProducedItemCount("untilNull");
    producedCount = b->CreateAdd(producedCount, produced);
    b->setProducedItemCount("untilNull", producedCount);
}

AbortOnNull::AbortOnNull(BuilderRef b, StreamSet * const InputStream, StreamSet * const OutputStream, Scalar * callbackObject)
: MultiBlockKernel(b, "AbortOnNull",
// inputs
{Binding{"byteData", InputStream, FixedRate(), Principal()}},
// outputs
{Binding{ "untilNull", OutputStream, FixedRate(), Deferred()}},
// input scalars
{Binding{"handler_address", callbackObject}},
{}, {}) {
    addAttribute(CanTerminateEarly());
    addAttribute(MayFatallyTerminate());
}

ContextSpan::ContextSpan(BuilderRef b, StreamSet * const markerStream, StreamSet * const contextStream, unsigned before, unsigned after)
: PabloKernel(b, "ContextSpan-" + std::to_string(before) + "+" + std::to_string(after),
              // input
{Binding{"markerStream", markerStream, FixedRate(1), LookAhead(before)}},
              // output
{Binding{"contextStream", contextStream}}),
mBeforeContext(before), mAfterContext(after) {
}

void ContextSpan::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    Var * markerStream = pb.createExtract(getInputStreamVar("markerStream"), pb.getInteger(0));
    PabloAST * contextStart = pb.createLookahead(markerStream, pb.getInteger(mBeforeContext));
    unsigned lgth = mBeforeContext + 1 + mAfterContext;
    PabloAST * consecutive = contextStart;
    unsigned consecutiveCount = 1;
    for (unsigned i = 1; i <= lgth/2; i *= 2) {
        consecutiveCount += i;
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, i),
                                  "consecutive" + std::to_string(consecutiveCount));
    }
    if (consecutiveCount < lgth) {
        consecutive = pb.createOr(consecutive,
                                  pb.createAdvance(consecutive, lgth - consecutiveCount),
                                  "consecutive" + std::to_string(lgth));
    }
    pb.createAssign(pb.createExtract(getOutputStreamVar("contextStream"), pb.getInteger(0)), pb.createInFile(consecutive));
}

void kernel::GraphemeClusterLogic(ProgBuilderRef P,
                                  StreamSet * Source, StreamSet * U8index, StreamSet * GCBstream) {
    
    re::RE * GCB = re::generateGraphemeClusterBoundaryRule();
    const auto GCB_Sets = re::collectCCs(GCB, cc::Unicode, re::NameProcessingMode::ProcessDefinition);
    auto GCB_mpx = cc::makeMultiplexedAlphabet("GCB_mpx", GCB_Sets);
    GCB = transformCCs(GCB_mpx, GCB, re::NameTransformationMode::TransformDefinition);
    auto GCB_basis = GCB_mpx->getMultiplexedCCs();
    StreamSet * const GCB_Classes = P->CreateStreamSet(GCB_basis.size());
    P->CreateKernelCall<CharClassesKernel>(GCB_basis, Source, GCB_Classes);
    std::unique_ptr<GrepKernelOptions> options = std::make_unique<GrepKernelOptions>();
    options->setIndexing(U8index);
    options->setRE(GCB);
    options->setSource(GCB_Classes);
    options->addAlphabet(GCB_mpx, GCB_Classes);
    options->setResults(GCBstream);
    options->addExternal("UTF8_index", U8index);
    P->CreateKernelCall<ICGrepKernel>(std::move(options));
}

void kernel::WordBoundaryLogic(ProgBuilderRef P,
                                  StreamSet * Source, StreamSet * U8index, StreamSet * wordBoundary_stream) {
    
    re::RE * wordProp = re::makePropertyExpression(PropertyExpression::Kind::Codepoint, "word");
    wordProp = UCD::linkAndResolve(wordProp);
    re::Name * word = re::makeName("word");
    word->setDefinition(wordProp);
    StreamSet * WordStream = P->CreateStreamSet(1);
    P->CreateKernelCall<UnicodePropertyKernelBuilder>(word, Source, WordStream);
    P->CreateKernelCall<BoundaryKernel>(WordStream, U8index, wordBoundary_stream);
}

LongestMatchMarks::LongestMatchMarks(BuilderRef b, StreamSet * start_ends, StreamSet * marks)
: PabloKernel(b, "LongestMatchMarks",
              {Binding{"start_ends", start_ends, FixedRate(1), LookAhead(1)}},
              {Binding{"marks", marks}}) {}

void LongestMatchMarks::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * starts_ends = pb.createExtract(getInputStreamVar("start_ends"), pb.getInteger(0));
    PabloAST * end_follow = pb.createLookahead(starts_ends, 1);
    PabloAST * span_starts = pb.createAnd(pb.createNot(starts_ends), end_follow, "span_starts");
    PabloAST * span_ends = pb.createAnd(starts_ends, end_follow, "span_ends");
    Var * marksVar = getOutputStreamVar("marks");
    pb.createAssign(pb.createExtract(marksVar, pb.getInteger(0)), span_starts);
    pb.createAssign(pb.createExtract(marksVar, pb.getInteger(1)), span_ends);
}

InclusiveSpans::InclusiveSpans(BuilderRef b, StreamSet * marks, StreamSet * spans, unsigned start_offset = 0)
: PabloKernel(b, "InclusiveSpans@" + std::to_string(start_offset),
              {Binding{"marks", marks, FixedRate(1), LookAhead(round_up_to_blocksize(start_offset))}},
              {Binding{"spans", spans}}),
    mOffset(start_offset) {
}

void InclusiveSpans::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    Var * marksVar = getInputStreamVar("marks");
    PabloAST * starts = pb.createExtract(marksVar, pb.getInteger(0));
    if (mOffset > 0) {
        starts = pb.createLookahead(starts, mOffset);
    }
    PabloAST * ends = pb.createExtract(marksVar, pb.getInteger(1));
    PabloAST * spans = pb.createIntrinsicCall(pablo::Intrinsic::InclusiveSpan, {starts, ends});
    pb.createAssign(pb.createExtract(getOutputStreamVar("spans"), pb.getInteger(0)), spans);
}

void kernel::PrefixSuffixSpan(ProgBuilderRef P,
                      StreamSet * Prefix, StreamSet * Suffix, StreamSet * Spans, unsigned offset) {
    std::vector<StreamSet *> marks = {Prefix, Suffix};
    StreamSet * mask = P->CreateStreamSet();
    P->CreateKernelCall<StreamsMerge>(marks, mask);
    StreamSet * filteredSuffix = P->CreateStreamSet();
    FilterByMask(P, mask, Suffix, filteredSuffix);
    StreamSet * matchMarks = P->CreateStreamSet(2);
    P->CreateKernelCall<LongestMatchMarks>(filteredSuffix, matchMarks);
    StreamSet * spanMarks = P->CreateStreamSet(2);
    SpreadByMask(P, mask, matchMarks, spanMarks);
    P->CreateKernelCall<InclusiveSpans>(spanMarks, Spans, offset);
}

