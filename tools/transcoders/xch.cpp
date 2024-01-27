/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */


#include <cstdio>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <pablo/codegenstate.h>
#include <pablo/pe_zeroes.h>        // for Zeroes
#include <grep/grep_kernel.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/util/debug_display.h>
#include <kernel/unicode/charclasses.h>
#include <kernel/unicode/utf8gen.h>
#include <kernel/unicode/utf8_decoder.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <string>
#include <toolchain/toolchain.h>
#include <pablo/pablo_toolchain.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include <unicode/data/PropertyAliases.h>
#include <unicode/data/PropertyObjects.h>
#include <unicode/data/PropertyObjectTable.h>
#include <unicode/utf/utf_compiler.h>
#include <re/toolchain/toolchain.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory Xch_Options("Character transformation Options", "Character transformation Options.");
static cl::opt<std::string> XfrmProperty(cl::Positional, cl::desc("transformation kind (Unicode property)"), cl::Required, cl::cat(Xch_Options));
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(Xch_Options));

#define SHOW_STREAM(name) if (illustratorAddr) illustrator.captureBitstream(P, #name, name)
#define SHOW_BIXNUM(name) if (illustratorAddr) illustrator.captureBixNum(P, #name, name)
#define SHOW_BYTES(name) if (illustratorAddr) illustrator.captureByteData(P, #name, name)

class AdjustU8bixnum : public pablo::PabloKernel {
public:
    AdjustU8bixnum(BuilderRef b,
                   StreamSet * Basis, StreamSet * InsertBixNum,
                   StreamSet * AdjustedBixNum);
protected:
    void generatePabloMethod() override;
private:
    unsigned mBixBits;
};

AdjustU8bixnum::AdjustU8bixnum (BuilderRef b,
                                StreamSet * Basis, StreamSet * InsertBixNum,
                                StreamSet * AdjustedBixNum)
: PabloKernel(b, "adjust_bixnum" + std::to_string(InsertBixNum->getNumElements()) + "x1",
// inputs
{Binding{"basis", Basis}, Binding{"insert_bixnum", InsertBixNum, FixedRate(1), LookAhead(2)}},
// output
{Binding{"adjusted_bixnum", AdjustedBixNum}}),
    mBixBits(InsertBixNum->getNumElements()) {
}

void AdjustU8bixnum::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    Var * insert_bixnum = getInputStreamVar("insert_bixnum");
    Var * insert_bit0 = pb.createExtract(insert_bixnum, pb.getInteger(0));
    std::vector<PabloAST *> adjusted_bixnum;
    // Insertion positions for a three byte UTF-8 sequence are
    // marked at the third byte.   The max insertion is 1, for
    // conversion to a four-byte sequence.  The insertion position
    // must be adjusted two positions forward.
    PabloAST * prefix = pb.createAnd(basis[7], basis[6]);
    PabloAST * prefix34 = pb.createAnd(prefix, basis[5]);
    PabloAST * prefix3 = pb.createAnd(prefix34, pb.createNot(basis[4]));
    PabloAST * p3_adjust = pb.createAnd(prefix3, pb.createLookahead(insert_bit0, 2));
    p3_adjust = pb.createOr(p3_adjust, pb.createAdvance(p3_adjust, 2));
    adjusted_bixnum.push_back(pb.createXor(p3_adjust, insert_bit0));
    // Insertion positions for a two byte UTF-8 sequence are
    // marked at the second byte.   The insertion is 1 or 2, for
    // conversion to a three- or four-byte sequence.  Adjust one
    // position forward.
    PabloAST * prefix2 = pb.createAnd(prefix, pb.createNot(basis[5]));
    PabloAST * p2_adjust_0 = pb.createAnd(prefix2, pb.createLookahead(insert_bit0, 1));
    p2_adjust_0 = pb.createOr(p2_adjust_0, pb.createAdvance(p2_adjust_0, 1));
    adjusted_bixnum[0] = pb.createXor(p2_adjust_0, adjusted_bixnum[0]);
    if (mBixBits == 2) {
        Var * insert_bit1 = pb.createExtract(insert_bixnum, pb.getInteger(1));
        PabloAST * p2_adjust_1 = pb.createAnd(prefix2, pb.createLookahead(insert_bit1, 1));
        p2_adjust_1 = pb.createOr(p2_adjust_1, pb.createAdvance(p2_adjust_1, 1));
        adjusted_bixnum.push_back(pb.createXor(p2_adjust_1, insert_bit1));
    }
    Var * const adjusted = getOutputStreamVar("adjusted_bixnum");
    for (unsigned i = 0; i < mBixBits; i++) {
        pb.createAssign(pb.createExtract(adjusted, pb.getInteger(i)), adjusted_bixnum[i]);
    }
}

class CreateU8delMask : public pablo::PabloKernel {
public:
    CreateU8delMask(BuilderRef b,
                    StreamSet * Basis, StreamSet * DeletionBixNum,
                    StreamSet * DelMask);
protected:
    void generatePabloMethod() override;
private:
    unsigned mBixBits;
};

CreateU8delMask::CreateU8delMask (BuilderRef b,
                                StreamSet * Basis, StreamSet * DeletionBixNum,
                                StreamSet * DelMask)
: PabloKernel(b, "u8_delmask" + std::to_string(DeletionBixNum->getNumElements()) + "x1",
// inputs
{Binding{"basis", Basis}, Binding{"deletion_bixnum", DeletionBixNum, FixedRate(1), LookAhead(3)}},
// output
{Binding{"selection_mask", DelMask}}),
    mBixBits(DeletionBixNum->getNumElements()) {
}

void CreateU8delMask::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    Var * deletion_bixnum = getInputStreamVar("deletion_bixnum");
    Var * del_bixnum0 = pb.createExtract(deletion_bixnum, pb.getInteger(0));
    // Deletion bixnums are calculated at the final position of a UTF-8
    // sequence.   If the deletion amount is nonzero, then the
    // prefix position is marked for deletion.
    PabloAST * prefix = pb.createAnd(basis[7], basis[6]);
    PabloAST * prefix2 = pb.createAnd(prefix, pb.createNot(basis[5]));
    PabloAST * prefix34 = pb.createAnd(prefix, basis[5]);
    PabloAST * prefix3 = pb.createAnd(prefix34, pb.createNot(basis[4]));
    PabloAST * prefix4 = pb.createAnd(prefix34, basis[4]);
    PabloAST * del_mask = pb.createAnd(prefix2, pb.createLookahead(del_bixnum0, 1));
    del_mask = pb.createOr(del_mask, pb.createAnd(prefix3, pb.createLookahead(del_bixnum0, 2)));
    del_mask = pb.createOr(del_mask, pb.createAnd(prefix4, pb.createLookahead(del_bixnum0, 3)));
    if (mBixBits == 2) {
        Var * del_bixnum1 = pb.createExtract(deletion_bixnum, pb.getInteger(1));
        del_mask = pb.createOr(del_mask, pb.createAnd(prefix3, pb.createLookahead(del_bixnum1, 2)));
        del_mask = pb.createOr(del_mask, pb.createAnd(prefix4, pb.createLookahead(del_bixnum1, 3)));
        // The second byte of a three-byte sequence is deleted if the deletion amount is 2.
        PabloAST * scope32 = pb.createAdvance(prefix3, 1);
        del_mask = pb.createOr(del_mask, pb.createAnd(scope32, pb.createLookahead(del_bixnum1, 1)));
        // The second byte of a four-byte sequence is deleted if the deletion amount is 2 or 3.
        PabloAST * scope42 = pb.createAdvance(prefix4, 1);
        del_mask = pb.createOr(del_mask, pb.createAnd(scope42, pb.createLookahead(del_bixnum1, 2)));
        // The third byte of a four-byte sequence is deleted if the deletion amount is 3.
        PabloAST * scope43 = pb.createAdvance(prefix4, 2);
        PabloAST * del3 = pb.createAnd(pb.createLookahead(del_bixnum0, 1), pb.createLookahead(del_bixnum1, 1));
        del_mask = pb.createOr(del_mask, pb.createAnd(scope43, del3));
    }
    PabloAST * selected = pb.createInFile(pb.createNot(del_mask));
    Var * const selection_mask = getOutputStreamVar("selection_mask");
    pb.createAssign(pb.createExtract(selection_mask, pb.getInteger(0)), selected);
}

class UTF8_CharacterTranslator : public pablo::PabloKernel {
public:
    UTF8_CharacterTranslator
        (BuilderRef b,
         UCD::CodePointPropertyObject * p,
         StreamSet * Basis, StreamSet * SpreadMask, StreamSet * FilterMask,
         StreamSet * Output);
protected:
    void generatePabloMethod() override;
private:
    UCD::CodePointPropertyObject * mPropertyObject;
};

UTF8_CharacterTranslator::UTF8_CharacterTranslator
    (BuilderRef b,
     UCD::CodePointPropertyObject * p,
     StreamSet * Basis, StreamSet * SpreadMask, StreamSet * FilterMask,
     StreamSet * Output)
: PabloKernel(b, "u8_" + getPropertyEnumName(p->getPropertyCode()) + "_transformer",
{Binding{"basis", Basis, FixedRate(1), LookAhead(2)},
 Binding{"spreadmask", SpreadMask, FixedRate(1), LookAhead(2)},
 Binding{"filtermask", FilterMask}},
// output
{Binding{"output_basis", Output}}),
    mPropertyObject(p) {
}

void UTF8_CharacterTranslator::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<UCD::UnicodeSet> & xfrms = mPropertyObject->GetBitTransformSets();
    std::vector<re::CC *> xfrm_ccs;
    for (auto & b : xfrms) {
        xfrm_ccs.push_back(re::makeCC(b, &cc::Unicode));
    }
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
    std::vector<Var *> xfrm_vars;
    for (unsigned i = 0; i < xfrm_ccs.size(); i++) {
        xfrm_vars.push_back(pb.createVar("xfrm_cc_" + std::to_string(i), pb.createZeroes()));
        unicodeCompiler.addTarget(xfrm_vars[i], xfrm_ccs[i]);
    }
    if (LLVM_UNLIKELY(re::AlgorithmOptionIsSet(re::DisableIfHierarchy))) {
        unicodeCompiler.compile(UTF::UTF_Compiler::IfHierarchy::None);
    } else {
        unicodeCompiler.compile();
    }
    Var * basisVar = getInputStreamVar("basis");
    std::vector<PabloAST *> basis(8);
    for (unsigned i = 0; i < 8; i++) {
        basis[i] = pb.createExtract(basisVar, pb.getInteger(i));
    }
    Var * spreadmask = pb.createExtract(getInputStreamVar("spreadmask"), pb.getInteger(0));
    Var * filtermask = pb.createExtract(getInputStreamVar("filtermask"), pb.getInteger(0));

    // Classify the input data based on source information.
    PabloAST * src_prefix = pb.createAnd(basis[7], basis[6]);
    PabloAST * src_prefix2 = pb.createAnd(src_prefix, pb.createNot(basis[5]));
    PabloAST * src_prefix34 = pb.createAnd(src_prefix, basis[5]);
    PabloAST * src_prefix3 = pb.createAnd(src_prefix34, pb.createNot(basis[4]));
    PabloAST * src_prefix4 = pb.createAnd(src_prefix34, basis[4]);
    PabloAST * src_scope22 = pb.createAdvance(src_prefix2, 1);
    PabloAST * src_scope32 = pb.createAdvance(src_prefix3, 1);
    PabloAST * src_scope33 = pb.createAdvance(src_scope32, 1);
    PabloAST * src_scope42 = pb.createAdvance(src_prefix4, 1);
    PabloAST * src_scope43 = pb.createAdvance(src_scope42, 1);
    PabloAST * src_scope44 = pb.createAdvance(src_scope43, 1);
    PabloAST * src_ASCII = pb.createNot(basis[7]);

    // Bit transformations can be unified based on counting from the
    // end of the UTF-8 sequence.
    PabloAST * src_u8last = pb.createOr(src_ASCII, pb.createOr3(src_scope44, src_scope33, src_scope22));
    PabloAST * src_secondlast = pb.createOr3(src_scope43, src_scope32, src_prefix2);
    PabloAST * src_thirdlast = pb.createOr(src_scope42, src_prefix3);

    // Now compute the target classifications.
    //
    // After any byte position to be deleted, the following byte must be
    // converted from a UTF-8 suffix byte to the initial byte of a
    // a UTF-8 sequence (either an ASCII byte or a prefix byte).
    PabloAST * afterDeletion = pb.createNot(pb.createAdvance(filtermask, 1));
    PabloAST * suffix_to_ASCII = pb.createAnd(afterDeletion, src_u8last);
    PabloAST * suffix_to_prefix2 = pb.createAnd(afterDeletion, src_secondlast);
    PabloAST * suffix_to_prefix3 = pb.createAnd(afterDeletion, src_thirdlast);
    //
    // After any position to be inserted, the following byte must
    // be converted from the initial byte of a UTF-8 sequence to a suffix.
    PabloAST * inserted = pb.createNot(spreadmask);
    PabloAST * afterInsertion = pb.createAdvance(inserted, 1);
    PabloAST * ASCII_to_suffix = pb.createAnd(afterInsertion, src_ASCII);
    PabloAST * prefix2_to_suffix = pb.createAnd(afterInsertion, src_prefix2);
    PabloAST * prefix3_to_suffix = pb.createAnd(afterInsertion, src_prefix3);
    PabloAST * newSuffix = pb.createOr3(ASCII_to_suffix, prefix2_to_suffix, prefix3_to_suffix);
    //
    // Newly inserted bytes must also be classified.
    //
    PabloAST * inserted_prefix = pb.createAnd(inserted, pb.createNot(afterInsertion));
    PabloAST * inserted_suffix = pb.createAnd(inserted, afterInsertion);
    PabloAST * ins_ahead1 = pb.createNot(pb.createLookahead(spreadmask, 1));
    PabloAST * ins_ahead2 = pb.createNot(pb.createLookahead(spreadmask, 2));
    PabloAST * last_insertion = pb.createAnd(inserted, pb.createNot(ins_ahead1));
    PabloAST * secondlast_insertion = pb.createAnd3(inserted, ins_ahead1, pb.createNot(ins_ahead2));
    PabloAST * bit7_ahead1 = pb.createLookahead(basis[7], 1);
    PabloAST * bit7_ahead2 = pb.createLookahead(basis[7], 2);
    PabloAST * bit6_ahead1 = pb.createLookahead(basis[6], 1);
    PabloAST * bit5_ahead1 = pb.createLookahead(basis[5], 1);
    PabloAST * ASCII_next = pb.createAnd(last_insertion, pb.createNot(bit7_ahead1));
    PabloAST * ASCII_ahead2 = pb.createAnd(secondlast_insertion, pb.createNot(bit7_ahead2));
    PabloAST * inserted_prefix2 = pb.createAnd(inserted_prefix, ASCII_next);
    PabloAST * prefix_next = pb.createAnd(bit7_ahead1, bit6_ahead1);
    PabloAST * prefix2_next = pb.createAnd(prefix_next, pb.createNot(bit5_ahead1));
    PabloAST * inserted_prefix3 = pb.createAnd(inserted_prefix, pb.createOr(ASCII_ahead2, prefix2_next));
    PabloAST * inserted_prefix4 = pb.createAnd(inserted_prefix, pb.createNot(pb.createOr(inserted_prefix2, inserted_prefix3)));

    //
    //  Initial ASCII bit movement:
    basis[6] = pb.createSel(suffix_to_ASCII, pb.createAdvance(basis[0], 1), basis[6]);
    basis[0] = pb.createSel(inserted_prefix2, bit6_ahead1, basis[0]);

    //  Now proceed to set up the correct UTF-8 bits encoding ASCII, prefixes
    //  and suffixes appropriately.
    //  Clear bit 7 of new ASCII bytes
    basis[7] = pb.createAnd(basis[7], pb.createNot(suffix_to_ASCII));
    //  Set bit 7 of newly inserted suffix and prefix bytes.
    basis[7] = pb.createOr3(basis[7], inserted, ASCII_to_suffix);
    //pb.createIntrinsicCall(pablo::Intrinsic::PrintRegister, {basis[7]});
    //  Clear bit 6 of newly converted suffix bytes.
    basis[6] = pb.createAnd(basis[6], pb.createNot(newSuffix));
    //  Set bit 6 of new prefix bytes.
    basis[6] = pb.createOr(basis[6], pb.createOr3(suffix_to_prefix2, suffix_to_prefix3, inserted_prefix));
    //  Clear bit 5 of prefix3 bytes converted to suffix bytes.
    basis[5] = pb.createAnd(basis[5], pb.createNot(prefix3_to_suffix));
    //  Set bit 5 of new prefix3/4 bytes bytes.
    basis[5] = pb.createOr(basis[5], pb.createOr3(suffix_to_prefix3, inserted_prefix3, inserted_prefix4));
    //  Set bit 4 of new prefix4 bytes bytes.
    basis[4] = pb.createOr(basis[4], inserted_prefix4);
    //
    // Now prepare for character translation
    PabloAST * new_prefix2 = pb.createOr(inserted_prefix2, suffix_to_prefix2);
    PabloAST * new_prefix3 = pb.createOr(inserted_prefix3, suffix_to_prefix3);
    PabloAST * new_scope32 = pb.createAdvance(new_prefix3, 1);
    PabloAST * new_scope42 = pb.createAdvance(inserted_prefix4, 1);
    PabloAST * new_scope43 = pb.createAdvance(inserted_prefix4, 2);

    PabloAST * tgt_ASCII = pb.createOr(suffix_to_ASCII, pb.createAnd(src_ASCII, pb.createNot(ASCII_to_suffix)));
    PabloAST * tgt_u8last = src_u8last;
    PabloAST * new_secondlast = pb.createOr3(new_prefix2, new_scope32, new_scope43);
    PabloAST * tgt_secondlast = pb.createOr(src_secondlast, new_secondlast);
    PabloAST * tgt_thirdlast = pb.createOr3(src_thirdlast, new_prefix3, new_scope42);
    PabloAST * tgt_prefix4 = pb.createOr(inserted_prefix4, src_prefix4);

    //  Translate Unicode bits 0 through 5 at the u8last position.
    for (unsigned U_bit = 0; U_bit < 6; U_bit++) {
        if (U_bit < xfrm_vars.size()) {
            unsigned u8_bit = U_bit;
            basis[u8_bit] = pb.createSel(tgt_u8last,
                                         pb.createXor(xfrm_vars[U_bit], basis[u8_bit]), basis[u8_bit]);
        }
    }
    // Translate bit 6 at ASCII positions.
    if (xfrm_vars.size() > 6) {
        basis[6] = pb.createSel(tgt_ASCII, pb.createXor(xfrm_vars[6], basis[6]), basis[6]);
    }
    //  Translate Unicode bits 6 through 11 at the second last UTF-8 byte position.
    for (unsigned U_bit = 6; U_bit < 11; U_bit++) {
        if (U_bit < xfrm_vars.size()) {
            unsigned u8_bit = U_bit - 6;
            basis[u8_bit] = pb.createSel(tgt_secondlast,
                                         pb.createXor(xfrm_vars[U_bit], basis[u8_bit]), basis[u8_bit]);
        }
    }
    //  Translate Unicode bits 12 through 17 at the third last UTF-8 byte position.
    for (unsigned U_bit = 12; U_bit < 17; U_bit++) {
        if (U_bit < xfrm_vars.size()) {
            unsigned u8_bit = U_bit - 12;
            basis[u8_bit] = pb.createSel(tgt_thirdlast,
                                         pb.createXor(xfrm_vars[U_bit], basis[u8_bit]), basis[u8_bit]);
        }
    }
    //  Translate Unicode bits 18 through 20 at the UTF-8 prefix4 byte position.
    for (unsigned U_bit = 18; U_bit < 20; U_bit++) {
        if (U_bit < xfrm_vars.size()) {
            unsigned u8_bit = U_bit - 18;
            basis[u8_bit] = pb.createSel(tgt_prefix4,
                                         pb.createXor(xfrm_vars[U_bit], basis[u8_bit]), basis[u8_bit]);
        }
    }
    Var * translatedVar = getOutputStreamVar("output_basis");
    for (unsigned i = 0; i < 8; i++) {
        pb.createAssign(pb.createExtract(translatedVar, pb.getInteger(i)), basis[i]);
    }
}

class ApplyTransform : public pablo::PabloKernel {
public:
    ApplyTransform(BuilderRef b,
                   UCD::CodePointPropertyObject * p,
                   StreamSet * Basis, StreamSet * Output);
protected:
    void generatePabloMethod() override;
private:
    UCD::CodePointPropertyObject * mPropertyObject;
};

ApplyTransform::ApplyTransform (BuilderRef b,
                                UCD::CodePointPropertyObject * p,
                                StreamSet * Basis, StreamSet * Output)
: PabloKernel(b, getPropertyEnumName(p->getPropertyCode()) + "_transformer_" + std::to_string(Basis->getNumElements()) + "x1",
// inputs
{Binding{"basis", Basis}},
// output
{Binding{"output_basis", Output}}),
    mPropertyObject(p) {
}

void ApplyTransform::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    //const UCD::UnicodeSet & nullSet = mPropertyObject->GetNullSet();
    std::vector<UCD::UnicodeSet> & xfrms = mPropertyObject->GetBitTransformSets();
    std::vector<re::CC *> xfrm_ccs;
    for (auto & b : xfrms) {
        xfrm_ccs.push_back(re::makeCC(b, &cc::Unicode));
    }
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
    std::vector<Var *> xfrm_vars;
    for (unsigned i = 0; i < xfrm_ccs.size(); i++) {
        xfrm_vars.push_back(pb.createVar("xfrm_cc_" + std::to_string(i), pb.createZeroes()));
        unicodeCompiler.addTarget(xfrm_vars[i], xfrm_ccs[i]);
    }
    if (LLVM_UNLIKELY(re::AlgorithmOptionIsSet(re::DisableIfHierarchy))) {
        unicodeCompiler.compile(UTF::UTF_Compiler::IfHierarchy::None);
    } else {
        unicodeCompiler.compile();
    }
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::vector<PabloAST *> transformed(basis.size());
    Var * const out = getOutputStreamVar("output_basis");
    for (unsigned i = 0; i < basis.size(); i++) {
        if (i < xfrm_vars.size()) {
            transformed[i] = pb.createXor(xfrm_vars[i], basis[i]);
        } else {
            transformed[i] = basis[i];
        }
        pb.createAssign(pb.createExtract(out, pb.getInteger(i)), transformed[i]);
    }
}

typedef void (*XfrmFunctionType)(uint32_t fd, ParabixIllustrator * illustrator);

XfrmFunctionType generatePipeline(CPUDriver & pxDriver,
                                  UCD::CodePointPropertyObject * p,
                                  ParabixIllustrator & illustrator) {
    // A Parabix program is build as a set of kernel calls called a pipeline.
    // A pipeline is construction using a Parabix driver object.
    auto & b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline({Binding{b->getInt32Ty(), "inputFileDecriptor"},
                                    Binding{b->getIntAddrTy(), "illustratorAddr"}}, {});
    //  The program will use a file descriptor as an input.
    Scalar * fileDescriptor = P->getInputScalar("inputFileDecriptor");
    //   If the --illustrator-width= parameter is specified, bitstream
    //   data is to be displayed.
    Scalar * illustratorAddr = nullptr;
    if (codegen::IllustratorDisplay > 0) {
        illustratorAddr = P->getInputScalar("illustratorAddr");
        illustrator.registerIllustrator(illustratorAddr);
    }
    // File data from mmap
    StreamSet * ByteStream = P->CreateStreamSet(1, 8);
    //  MMapSourceKernel is a Parabix Kernel that produces a stream of bytes
    //  from a file descriptor.
    P->CreateKernelCall<MMapSourceKernel>(fileDescriptor, ByteStream);
    SHOW_BYTES(ByteStream);

    //  The Parabix basis bits representation is created by the Parabix S2P kernel.
    StreamSet * BasisBits = P->CreateStreamSet(8, 1);
    P->CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);

    std::vector<UCD::UnicodeSet> & insertion_bixnum = p->GetUTF8insertionBixNum();
    unsigned bix_bits = insertion_bixnum.size();

    StreamSet * ExpandedBasis = BasisBits;
    StreamSet * SpreadMask = nullptr;
    if (bix_bits > 0) {
        std::vector<re::CC *> insertion_ccs;
        for (auto & b : insertion_bixnum) {
            insertion_ccs.push_back(re::makeCC(b, &cc::Unicode));
        }
        StreamSet * InsertBixNum = P->CreateStreamSet(bix_bits);
        P->CreateKernelCall<CharClassesKernel>(insertion_ccs, BasisBits, InsertBixNum);
        SHOW_BIXNUM(InsertBixNum);

        StreamSet * AdjustedBixNum = P->CreateStreamSet(bix_bits);
        P->CreateKernelCall<AdjustU8bixnum>(BasisBits, InsertBixNum, AdjustedBixNum);
        SHOW_BIXNUM(AdjustedBixNum);

        SpreadMask = InsertionSpreadMask(P, AdjustedBixNum, InsertPosition::Before);
        SHOW_STREAM(SpreadMask);

        ExpandedBasis = P->CreateStreamSet(8, 1);
        SpreadByMask(P, SpreadMask, BasisBits, ExpandedBasis);
        SHOW_BIXNUM(ExpandedBasis);
    } else {
        llvm::errs() << "bit_bits = 0\n";
    }
    StreamSet * SelectionMask = nullptr;
    std::vector<UCD::UnicodeSet> & deletion_bixnum = p->GetUTF8deletionBixNum();
    unsigned del_bix_bits = deletion_bixnum.size();
    if (del_bix_bits > 0) {
        std::vector<re::CC *> deletion_ccs;
        for (auto & b : deletion_bixnum) {
            deletion_ccs.push_back(re::makeCC(b, &cc::Unicode));
        }

        StreamSet * DeletionBixNum = P->CreateStreamSet(del_bix_bits);
        P->CreateKernelCall<CharClassesKernel>(deletion_ccs, BasisBits, DeletionBixNum);
        SHOW_BIXNUM(DeletionBixNum);

        SelectionMask = P->CreateStreamSet(1);
        P->CreateKernelCall<CreateU8delMask>(BasisBits, DeletionBixNum, SelectionMask);
        SHOW_STREAM(SelectionMask);

        StreamSet * ExpandedSelectionMask = P->CreateStreamSet(1);
        SpreadByMask(P, SpreadMask, SelectionMask, ExpandedSelectionMask);
        SHOW_STREAM(ExpandedSelectionMask);
        SelectionMask = ExpandedSelectionMask;
    } else {
        llvm::errs() << "del_bit_bits = 0\n";
    }
    StreamSet * Translated = P->CreateStreamSet(8);
    P->CreateKernelCall<UTF8_CharacterTranslator>(p, ExpandedBasis, SpreadMask, SelectionMask, Translated);
    SHOW_BIXNUM(Translated);

    StreamSet * OutputBasis = P->CreateStreamSet(8);
    FilterByMask(P, SelectionMask, Translated, OutputBasis);
    SHOW_BIXNUM(OutputBasis);

    StreamSet * OutputBytes = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<P2SKernel>(Translated, OutputBytes);
    //SHOW_BYTES(OutputBytes);

    P->CreateKernelCall<StdOutKernel>(OutputBytes);
    return reinterpret_cast<XfrmFunctionType>(P->compile());
}

int main(int argc, char *argv[]) {
    //  ParseCommandLineOptions uses the LLVM CommandLine processor, but we also add
    //  standard Parabix command line options such as -help, -ShowPablo and many others.
    codegen::ParseCommandLineOptions(argc, argv, {&Xch_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    ParabixIllustrator illustrator(codegen::IllustratorDisplay);

    UCD::property_t prop = UCD::getPropertyCode(XfrmProperty);
    UCD::PropertyObject * propObj = UCD::getPropertyObject(prop);
    if (UCD::CodePointPropertyObject * p = dyn_cast<UCD::CodePointPropertyObject>(propObj)) {
        CPUDriver driver("xfrm_function");
        //  Build and compile the Parabix pipeline by calling the Pipeline function above.
        XfrmFunctionType fn = generatePipeline(driver, p, illustrator);
        //  The compile function "fn"  can now be used.   It takes a file
        //  descriptor as an input, which is specified by the filename given by
        //  the inputFile command line option.]

        const int fd = open(inputFile.c_str(), O_RDONLY);
        if (LLVM_UNLIKELY(fd == -1)) {
            llvm::errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
        } else {
            fn(fd, &illustrator);
            close(fd);
            if (codegen::IllustratorDisplay > 0) {
                illustrator.displayAllCapturedData();
            }
        }
    } else {
        llvm::report_fatal_error("Expecting codepoint property");
    }
    return 0;
}
