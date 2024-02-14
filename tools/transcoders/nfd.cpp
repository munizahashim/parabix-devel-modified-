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
#include <pablo/bixnum/bixnum.h>
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
#include <unicode/utf/transchar.h>
#include <codecvt>
#include <re/toolchain/toolchain.h>

using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory NFD_Options("Decompositon Options", "Decompositon Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(NFD_Options));

#define SHOW_STREAM(name) if (illustratorAddr) illustrator.captureBitstream(P, #name, name)
#define SHOW_BIXNUM(name) if (illustratorAddr) illustrator.captureBixNum(P, #name, name)
#define SHOW_BYTES(name) if (illustratorAddr) illustrator.captureByteData(P, #name, name)

const UCD::codepoint_t Hangul_SBase = 0xAC00;
const UCD::codepoint_t Hangul_LBase = 0x1100;
const UCD::codepoint_t Hangul_VBase = 0x1161;
const UCD::codepoint_t Hangul_TBase = 0x11A7;
const unsigned Hangul_LCount = 19;
const unsigned Hangul_VCount = 21;
const unsigned Hangul_TCount = 28;
const unsigned Hangul_NCount = 588;
const unsigned Hangul_SCount = 11172;


std::vector<re::CC *> HangulInsertionBixNumCCs() {
    UCD::codepoint_t Max_Hangul_Precomposed = Hangul_SBase + Hangul_SCount - 1;
    UCD::UnicodeSet Hangul_Precomposed_LV;
    for (UCD::codepoint_t cp = Hangul_SBase; cp <= Max_Hangul_Precomposed; cp += Hangul_TCount) {
        Hangul_Precomposed_LV.insert(cp);
    }
    UCD::UnicodeSet Hangul_Precomposed(Hangul_SBase, Max_Hangul_Precomposed) ;
    UCD::UnicodeSet Hangul_Precomposed_LVT = Hangul_Precomposed - Hangul_Precomposed_LV;
    return {re::makeCC(Hangul_Precomposed_LV, &cc::Unicode),
            re::makeCC(Hangul_Precomposed_LVT, &cc::Unicode)};
}

//
// A set of five Unicode CCs can be used to determine a 5-bit
// BixNum representing the L_index value of Hangul precomposed
// characters.
std::vector<re::CC *> LIndexBixNumCCs() {
    std::vector<re::CC *> L_CC(5, re::makeCC(&cc::Unicode));
    UCD::codepoint_t low_cp = Hangul_SBase;
    for (unsigned L_index = 0; L_index < Hangul_LCount; L_index++) {
        UCD::codepoint_t next_base = low_cp + Hangul_NCount;
        UCD::codepoint_t hi_cp = next_base - 1;
        for (unsigned bit = 0; bit < 5; bit++) {
            unsigned bit_val = (L_index >> bit) & 1;
            if (bit_val == 1) {
                 L_CC[bit] = re::makeCC(re::makeCC(low_cp, hi_cp, &cc::Unicode), L_CC[bit]);
            }
        }
        low_cp = next_base;
    }
    return L_CC;
}

//
// A set of five CCs can be used to determine a 5-bit BixNum
// reprsenting the V_index value of Hangul precomposed characters
// from a 9-bit VT index value in the range 0 .. Hangul_NCount - 1.
std::vector<re::CC *> VIndexBixNumCCs() {
    std::vector<re::CC *> V_CC(5, re::makeCC(&cc::Unicode));
    unsigned low = 0;
    for (unsigned V_index = 0; V_index < Hangul_VCount; V_index++) {
        unsigned next_base = low + Hangul_TCount;
        unsigned hi = next_base - 1;
        for (unsigned bit = 0; bit < 5; bit++) {
            unsigned bit_val = (V_index >> bit) & 1;
            if (bit_val == 1) {
                V_CC[bit] = re::makeCC(re::makeCC(low, hi), V_CC[bit]);
            }
        }
        low = next_base;
    }
    return V_CC;
}

class Hangul_VT_Indices : public pablo::PabloKernel {
public:
    Hangul_VT_Indices(BuilderRef b,
                      StreamSet * Basis, StreamSet * LV_LVT, StreamSet * L_index,
                      StreamSet * V_index, StreamSet * T_index);
protected:
    void generatePabloMethod() override;
};

Hangul_VT_Indices::Hangul_VT_Indices (BuilderRef b,
                                      StreamSet * Basis, StreamSet * LV_LVT, StreamSet * L_index,
                                      StreamSet * V_index, StreamSet * T_index)
: PabloKernel(b, "Hangul_VT_indices_" + std::to_string(Basis->getNumElements()) + "x1",
// inputs
{Binding{"basis", Basis},
    Binding{"LV_LVT", LV_LVT},
    Binding{"L_index", L_index}},
// output
{Binding{"V_index", V_index}, Binding{"T_index", T_index}}) {
}

void Hangul_VT_Indices::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::vector<PabloAST *> LV_LVT = getInputStreamSet("LV_LVT");
    std::vector<PabloAST *> L_index = getInputStreamSet("L_index");
    PabloAST * precomposed = pb.createOr(LV_LVT[0], LV_LVT[1]);
    // Set up Vars to receive the V_index and T_index values.
    std::vector<Var *> V_indexVar(5);
    for (unsigned i = 0; i < 5; i++) {
        V_indexVar[i] = pb.createVar("V_index" + std::to_string(i), pb.createZeroes());
    }
    std::vector<Var *> T_indexVar(5);
    for (unsigned i = 0; i < 5; i++) {
        T_indexVar[i] = pb.createVar("T_index" + std::to_string(i), pb.createZeroes());
    }
    auto nested = pb.createScope();
    //pb.createIf(precomposed, nested);
    //BixNumCompiler bnc(nested);
    BixNumCompiler bnc(pb);
    //
    // For each distinct Hangul L prefix, there is a block of
    // Hangul_NCount entries.  The relative offset of the block
    // (offset from Hangul_SBase) is L_index * Hangul_NCount.
    // The offset will have up to 14 significant bits.
    //
    BixNum rel_offset = bnc.ZeroExtend(L_index, 14);
    rel_offset = bnc.MulModular(L_index, Hangul_NCount);
    //
    // Compute the VT index as the index within the block of
    // Hangul_NCount entries.
    BixNum basis_offset = bnc.SubModular(basis, Hangul_SBase);
    BixNum VT_index = bnc.SubModular(basis_offset, rel_offset);
    VT_index = bnc.Truncate(VT_index, 10);  // Only 10 bits needed.
    //
    // Given the VT_index value as a basis, we can compute
    // the V_index from a set of five CCs.
    std::vector<re::CC *> V_CCs = VIndexBixNumCCs();
    cc::Parabix_CC_Compiler ccc(getEntryScope(), VT_index);
    //cc::Parabix_CC_Compiler ccc(nested.getPabloBlock(), VT_index);
    std::vector<PabloAST *> V_index(5);
    for (unsigned i = 0; i < 5; i++) {
        V_index[i] = ccc.compileCC(V_CCs[i]);
        pb.createAssign(V_indexVar[i], V_index[i]);
    }
    BixNum V_offset = bnc.ZeroExtend(V_index, 10);
    V_offset = bnc.MulModular(V_index, Hangul_TCount);
    BixNum T_index = bnc.SubModular(VT_index, V_offset);
    // Only 5 significant bits
    for (unsigned i = 0; i < 5; i++) {
        pb.createAssign(T_indexVar[i], T_index[i]);
    }
    //
    Var * V_out = getOutputStreamVar("V_index");
    for (unsigned i = 0; i < 5; i++) {
        pb.createAssign(pb.createExtract(V_out, pb.getInteger(i)), V_indexVar[i]);
    }
    Var * T_out = getOutputStreamVar("T_index");
    for (unsigned i = 0; i < 5; i++) {
        pb.createAssign(pb.createExtract(T_out, pb.getInteger(i)), T_indexVar[i]);
    }
}

class Hangul_NFD : public pablo::PabloKernel {
public:
    Hangul_NFD(BuilderRef b,
               StreamSet * Basis, StreamSet * LV_LVT, StreamSet * L_index,
               StreamSet * V_index, StreamSet * T_index, StreamSet * NFD_Basis);
protected:
    void generatePabloMethod() override;
};

Hangul_NFD::Hangul_NFD (BuilderRef b,
                        StreamSet * Basis, StreamSet * LV_LVT, StreamSet * L_index,
                        StreamSet * V_index, StreamSet * T_index, StreamSet * NFD_Basis)
: PabloKernel(b, "Hangul_NFD_" + std::to_string(Basis->getNumElements()) + "x1",
// inputs
{Binding{"basis", Basis},
    Binding{"L_index", L_index},
    Binding{"LV_LVT", LV_LVT},
    Binding{"V_index", V_index},
    Binding{"T_index", T_index}},
// output
{Binding{"NFD_Basis", NFD_Basis}}) {
}

void Hangul_NFD::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::vector<PabloAST *> LV_LVT = getInputStreamSet("LV_LVT");
    std::vector<PabloAST *> L_index = getInputStreamSet("L_index");
    std::vector<PabloAST *> V_index = getInputStreamSet("V_index");
    std::vector<PabloAST *> T_index = getInputStreamSet("T_index");
    PabloAST * precomposed = pb.createOr(LV_LVT[0], LV_LVT[1]);
    // Set up Vars to receive the generated basis values.
    std::vector<Var *> basisVar(21);
    for (unsigned i = 0; i < 21; i++) {
        basisVar[i] = pb.createVar("basisVar" + std::to_string(i), pb.createZeroes());
    }
    auto nested = pb.createScope();
    pb.createIf(precomposed, nested);
    BixNumCompiler bnc(nested);
    BixNum LPart = bnc.ZeroExtend(L_index, 21);
    // The LPart will be encoded at the original precomposed position.
    LPart = bnc.AddModular(LPart, Hangul_LBase);
    // The V Part, is one positions after the opening L Part.
    PabloAST * V_position = nested.createAdvance(precomposed, 1);
    for (unsigned i = 0; i < 5; i++) {
        V_index[i] = nested.createAdvance(V_index[i], 1);
    }
    BixNum VPart = bnc.ZeroExtend(V_index, 21);
    VPart = bnc.AddModular(VPart, Hangul_VBase);
    // The T Part, if it exists is two positions after the opening
    // L Part.  A T Part only exists if the T_index is nonzero.
    PabloAST * T_position = nested.createAdvance(LV_LVT[1], 2);
    for (unsigned i = 0; i < 5; i++) {
        T_index[i] = nested.createAdvance(T_index[i], 2);
        //T_position = nested.createOr(T_index[i], T_position);
    }
    //T_position = nested.createAnd(T_position, nested.createAdvance(V_position, 1));
    BixNum TPart = bnc.ZeroExtend(T_index, 21);
    TPart = bnc.AddModular(TPart, Hangul_TBase);
    for (unsigned i = 0; i < 21; i++) {
        PabloAST * bit = nested.createSel(precomposed, LPart[i], basis[i]);
        bit = nested.createSel(V_position, VPart[i], bit);
        bit = nested.createSel(T_position, TPart[i], bit);
        nested.createAssign(basisVar[i], bit);
    }
    Var * NFD_Basis_Var = getOutputStreamVar("NFD_Basis");
    for (unsigned i = 0; i < 21; i++) {
        pb.createAssign(pb.createExtract(NFD_Basis_Var, pb.getInteger(i)), basisVar[i]);
    }
}

typedef void (*XfrmFunctionType)(uint32_t fd, ParabixIllustrator * illustrator);

XfrmFunctionType generate_pipeline(CPUDriver & pxDriver,
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

    StreamSet * BasisBits = P->CreateStreamSet(8, 1);
    P->CreateKernelCall<S2PKernel>(ByteStream, BasisBits);
    SHOW_BIXNUM(BasisBits);

    StreamSet * u8index = P->CreateStreamSet(1, 1);
    P->CreateKernelCall<UTF8_index>(BasisBits, u8index);
    SHOW_STREAM(u8index);

    StreamSet * U21_u8indexed = P->CreateStreamSet(21, 1);
    P->CreateKernelCall<UTF8_Decoder>(BasisBits, U21_u8indexed);

    StreamSet * U21 = P->CreateStreamSet(21, 1);
    FilterByMask(P, u8index, U21_u8indexed, U21);
    SHOW_BIXNUM(U21);

    auto insert_ccs = HangulInsertionBixNumCCs();
    StreamSet * Hangul_Insertion_BixNum = P->CreateStreamSet(insert_ccs.size());
    P->CreateKernelCall<CharClassesKernel>(insert_ccs, U21, Hangul_Insertion_BixNum);
    SHOW_BIXNUM(Hangul_Insertion_BixNum);

    StreamSet * SpreadMask = InsertionSpreadMask(P, Hangul_Insertion_BixNum, InsertPosition::After);
    SHOW_STREAM(SpreadMask);

    StreamSet * ExpandedBasis = P->CreateStreamSet(21, 1);
    SpreadByMask(P, SpreadMask, U21, ExpandedBasis);
    SHOW_BIXNUM(ExpandedBasis);

    StreamSet * LV_LVT =  P->CreateStreamSet(insert_ccs.size());
    P->CreateKernelCall<CharClassesKernel>(insert_ccs, ExpandedBasis, LV_LVT);
    SHOW_BIXNUM(LV_LVT);

    auto Lindex_ccs = LIndexBixNumCCs();
    StreamSet * LIndexBixNum = P->CreateStreamSet(Lindex_ccs.size());
    P->CreateKernelCall<CharClassesKernel>(Lindex_ccs, ExpandedBasis, LIndexBixNum);
    SHOW_BIXNUM(LIndexBixNum);

    StreamSet * VIndexBixNum = P->CreateStreamSet(5);
    StreamSet * TIndexBixNum = P->CreateStreamSet(5);
    P->CreateKernelCall<Hangul_VT_Indices>(ExpandedBasis, LV_LVT, LIndexBixNum, VIndexBixNum, TIndexBixNum);
    SHOW_BIXNUM(VIndexBixNum);
    SHOW_BIXNUM(TIndexBixNum);

    StreamSet * NFD_Basis = P->CreateStreamSet(21, 1);
    P->CreateKernelCall<Hangul_NFD>(ExpandedBasis, LV_LVT, LIndexBixNum, VIndexBixNum, TIndexBixNum, NFD_Basis);
    SHOW_BIXNUM(NFD_Basis);

    StreamSet * const OutputBasis = P->CreateStreamSet(8);
    U21_to_UTF8(P, NFD_Basis, OutputBasis);

    SHOW_BIXNUM(OutputBasis);

    StreamSet * OutputBytes = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<P2SKernel>(OutputBasis, OutputBytes);
    P->CreateKernelCall<StdOutKernel>(OutputBytes);

    return reinterpret_cast<XfrmFunctionType>(P->compile());
}


int main(int argc, char *argv[]) {
    //  ParseCommandLineOptions uses the LLVM CommandLine processor, but we also add
    //  standard Parabix command line options such as -help, -ShowPablo and many others.
    codegen::ParseCommandLineOptions(argc, argv, {&NFD_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    ParabixIllustrator illustrator(codegen::IllustratorDisplay);
    CPUDriver driver("NFD_function");
    //  Build and compile the Parabix pipeline by calling the Pipeline function above.
    XfrmFunctionType fn;
    fn = generate_pipeline(driver, illustrator);
    //
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::errs() << "Error: cannot open " << inputFile << " for processing.\n";
    } else {
        fn(fd, &illustrator);
        close(fd);
        if (codegen::IllustratorDisplay > 0) {
            illustrator.displayAllCapturedData();
        }
    }
    return 0;
}
