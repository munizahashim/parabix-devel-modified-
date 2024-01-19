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

class ApplyTransform : public pablo::PabloKernel {
public:
    ApplyTransform(BuilderRef b, UCD::property_t prop, StreamSet * Basis, StreamSet * Output);
protected:
    void generatePabloMethod() override;
private:
    UCD::property_t mProperty;
};



ApplyTransform::ApplyTransform (BuilderRef b, UCD::property_t prop, StreamSet * Basis, StreamSet * Output)
: PabloKernel(b, getPropertyEnumName(prop) + "_transformer_" + std::to_string(Basis->getNumElements()) + "x1",
// inputs
{Binding{"basis", Basis}},
// output
{Binding{"output_basis", Output}}),
    mProperty(prop) {
}

void ApplyTransform::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UCD::PropertyObject * propObj = UCD::getPropertyObject(mProperty);
    if (UCD::CodePointPropertyObject * p = dyn_cast<UCD::CodePointPropertyObject>(propObj)) {
        //const UCD::UnicodeSet & nullSet = p->GetNullSet();
        std::vector<UCD::UnicodeSet> & xfrms = p->GetBitTransformSets();
        std::vector<re::CC *> xfrm_ccs;
        for (auto & b : xfrms) xfrm_ccs.push_back(re::makeCC(b, &cc::Unicode));
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
    } else {
        llvm::report_fatal_error("Expecting codepoint property");
    }
}

typedef void (*XfrmFunctionType)(uint32_t fd, ParabixIllustrator * illustrator);

XfrmFunctionType generatePipeline(CPUDriver & pxDriver, UCD::property_t prop, ParabixIllustrator & illustrator) {
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
    //  S2P stands for serial-to-parallel.
    StreamSet * BasisBits = P->CreateStreamSet(8);
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

    StreamSet * u32basis = P->CreateStreamSet(21, 1);
    P->CreateKernelCall<ApplyTransform>(prop, U21, u32basis);
    SHOW_BIXNUM(u32basis);

    // Buffers for calculated deposit masks.
    StreamSet * const u8fieldMask = P->CreateStreamSet();
    StreamSet * const u8final = P->CreateStreamSet();
    StreamSet * const u8initial = P->CreateStreamSet();
    StreamSet * const u8mask12_17 = P->CreateStreamSet();
    StreamSet * const u8mask6_11 = P->CreateStreamSet();

    // Intermediate buffers for deposited bits
    StreamSet * const deposit18_20 = P->CreateStreamSet(3);
    StreamSet * const deposit12_17 = P->CreateStreamSet(6);
    StreamSet * const deposit6_11 = P->CreateStreamSet(6);
    StreamSet * const deposit0_5 = P->CreateStreamSet(6);

    // Calculate the u8final deposit mask.
    StreamSet * const extractionMask = P->CreateStreamSet();
    P->CreateKernelCall<UTF8fieldDepositMask>(u32basis, u8fieldMask, extractionMask);
    P->CreateKernelCall<StreamCompressKernel>(extractionMask, u8fieldMask, u8final);

    P->CreateKernelCall<UTF8_DepositMasks>(u8final, u8initial, u8mask12_17, u8mask6_11);

    SpreadByMask(P, u8initial, u32basis, deposit18_20, /* inputOffset = */ 18);
    SpreadByMask(P, u8mask12_17, u32basis, deposit12_17, /* inputOffset = */ 12);
    SpreadByMask(P, u8mask6_11, u32basis, deposit6_11, /* inputOffset = */ 6);
    SpreadByMask(P, u8final, u32basis, deposit0_5, /* inputOffset = */ 0);

    // Final buffers for computed UTF-8 basis bits and byte stream.
    StreamSet * const OutputBasis = P->CreateStreamSet(8);

    P->CreateKernelCall<UTF8assembly>(deposit18_20, deposit12_17, deposit6_11, deposit0_5,
                                      u8initial, u8final, u8mask6_11, u8mask12_17,
                                      OutputBasis);
    SHOW_BIXNUM(OutputBasis);

    StreamSet * OutputBytes = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<P2SKernel>(OutputBasis, OutputBytes);
    SHOW_BYTES(OutputBytes);

    P->CreateKernelCall<StdOutKernel>(OutputBytes);
    return reinterpret_cast<XfrmFunctionType>(P->compile());
}

int main(int argc, char *argv[]) {
    //  ParseCommandLineOptions uses the LLVM CommandLine processor, but we also add
    //  standard Parabix command line options such as -help, -ShowPablo and many others.
    codegen::ParseCommandLineOptions(argc, argv, {&Xch_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    ParabixIllustrator illustrator(codegen::IllustratorDisplay);

    UCD::property_t prop = UCD::getPropertyCode(XfrmProperty);
    
    CPUDriver driver("xfrm_function");
    //  Build and compile the Parabix pipeline by calling the Pipeline function above.
    XfrmFunctionType fn = generatePipeline(driver, prop, illustrator);
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
    return 0;
}
