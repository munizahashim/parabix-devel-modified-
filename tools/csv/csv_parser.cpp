/*
 *  Copyright (c) 2020 International Characters.
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
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/util/debug_display.h>
#include <kernel/scan/scanmatchgen.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <string>
#include <toolchain/toolchain.h>
#include <toolchain/pablo_toolchain.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include "csv_util.hpp"


using namespace kernel;
using namespace llvm;
using namespace pablo;

//  These declarations are for command line processing.
//  See the LLVM CommandLine Library Manual https://llvm.org/docs/CommandLine.html
static cl::OptionCategory CSV_Options("CSV Processing Options", "CSV Processing Options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(CSV_Options));
static cl::opt<bool> HeaderSpecNamesFile("f", cl::desc("Interpret headers parameter as file name with header line"), cl::init(false), cl::cat(CSV_Options));
static cl::opt<std::string> HeaderSpec("headers", cl::desc("CSV column headers (explicit string or filename"), cl::init(""), cl::cat(CSV_Options));


class CSVparser : public PabloKernel {
public:
    CSVparser(BuilderRef kb, StreamSet * csvMarks, StreamSet * recordSeparators, StreamSet * fieldSeparators, StreamSet * quoteEscape, StreamSet * toKeep)
        : PabloKernel(kb, "CSVparser",
                      {Binding{"csvMarks", csvMarks, FixedRate(), LookAhead(1)}},
                      {Binding{"recordSeparators", recordSeparators},
                       Binding{"fieldSeparators", fieldSeparators},
                       Binding{"quoteEscape", quoteEscape},
                       Binding{"toKeep", toKeep}}) {}
protected:
    void generatePabloMethod() override;
};

enum {markLF, markCR, markDQ, markComma};

void CSVparser::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvMarks");
    PabloAST * dquote = csvMarks[markDQ];
    PabloAST * dquote_odd = pb.createEveryNth(dquote, pb.getInteger(2));
    PabloAST * dquote_even = pb.createXor(dquote, dquote_odd);
    PabloAST * quote_escape = pb.createAnd(dquote_even, pb.createLookahead(dquote, 1));
    PabloAST * escaped_quote = pb.createAdvance(quote_escape, 1);
    PabloAST * start_dquote = pb.createXor(dquote_odd, escaped_quote);
    PabloAST * end_dquote = pb.createXor(dquote_even, quote_escape);
    PabloAST * quoted_data = pb.createIntrinsicCall(pablo::Intrinsic::InclusiveSpan, {start_dquote, end_dquote});
    PabloAST * unquoted = pb.createNot(quoted_data);
    PabloAST * recordMarks = pb.createAnd(csvMarks[markLF], unquoted);
    PabloAST * fieldMarks = pb.createOr(pb.createAnd(csvMarks[markComma], unquoted), recordMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("recordSeparators"), pb.getInteger(0)), recordMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("fieldSeparators"), pb.getInteger(0)), fieldMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("quoteEscape"), pb.getInteger(0)), quote_escape);
    PabloAST * CRbeforeLF = pb.createAnd(csvMarks[markCR], pb.createLookahead(csvMarks[markLF], 1));
    PabloAST * toDelete = pb.createOr3(CRbeforeLF, start_dquote, end_dquote);
    PabloAST * toKeep = pb.createInFile(pb.createNot(toDelete));
    pb.createAssign(pb.createExtract(getOutputStreamVar("toKeep"), pb.getInteger(0)), toKeep);
}

typedef void (*CSVFunctionType)(uint32_t fd);

CSVFunctionType generatePipeline(CPUDriver & pxDriver, std::vector<std::string> templateStrs) {
    // A Parabix program is build as a set of kernel calls called a pipeline.
    // A pipeline is construction using a Parabix driver object.
    auto & b = pxDriver.getBuilder();
    auto P = pxDriver.makePipeline({Binding{b->getInt32Ty(), "inputFileDecriptor"}}, {});
    //  The program will use a file descriptor as an input.
    Scalar * fileDescriptor = P->getInputScalar("inputFileDecriptor");
    // File data from mmap
    StreamSet * ByteStream = P->CreateStreamSet(1, 8);
    //  MMapSourceKernel is a Parabix Kernel that produces a stream of bytes
    //  from a file descriptor.
    P->CreateKernelCall<MMapSourceKernel>(fileDescriptor, ByteStream);

    //  The Parabix basis bits representation is created by the Parabix S2P kernel.
    //  S2P stands for serial-to-parallel.
    StreamSet * BasisBits = P->CreateStreamSet(8);
    P->CreateKernelCall<S2PKernel>(ByteStream, BasisBits);

    //  We need to know which input positions are dquotes and which are not.
    StreamSet * csvCCs = P->CreateStreamSet(4);
    char charLF = 0xA;
    char charCR = 0xD;
    char charDQ = 0x22;
    char charComma = 0x2C;
    std::vector<re::CC *> csvSpecial =
      {re::makeByte(charLF), re::makeByte(charCR), re::makeByte(charDQ), re::makeByte(charComma)};
    P->CreateKernelCall<CharacterClassKernelBuilder>(csvSpecial, BasisBits, csvCCs);

    StreamSet * recordSeparators = P->CreateStreamSet(1);
    StreamSet * fieldSeparators = P->CreateStreamSet(1);
    StreamSet * quoteEscape = P->CreateStreamSet(1);
    StreamSet * toKeep = P->CreateStreamSet(1);
    P->CreateKernelCall<CSVparser>(csvCCs, recordSeparators, fieldSeparators, quoteEscape, toKeep);

    //P->CreateKernelCall<DebugDisplayKernel>("CSV marks", csvMarks);

    StreamSet * translatedBasis = P->CreateStreamSet(8);
    P->CreateKernelCall<CSV_Char_Replacement>(fieldSeparators, quoteEscape, BasisBits, translatedBasis);

//    StreamSet * filteredBasis = P->CreateStreamSet(8);
//    FilterByMask(P, toKeep, translatedBasis, filteredBasis);

//    StreamSet * filteredMarks = P->CreateStreamSet(3);
//    FilterByMask(P, toKeep, csvMarks, filteredMarks);
    //P->CreateKernelCall<DebugDisplayKernel>("filtered marks", filteredMarks);
    P->CreateKernelCall<DebugDisplayKernel>("fieldSeparators", fieldSeparators);
    P->CreateKernelCall<DebugDisplayKernel>("recordSeparators", recordSeparators);

    StreamSet * recordsByField = P->CreateStreamSet(1);
    FilterByMask(P, fieldSeparators, recordSeparators, recordsByField);
    
    const unsigned fieldCount = templateStrs.size();
    const unsigned fieldCountBits = ceil_log2(fieldCount + 1);  // 1-based numbering
    StreamSet * compressedSepNum = P->CreateStreamSet(fieldCountBits);

    P->CreateKernelCall<RunIndex>(recordsByField, compressedSepNum, nullptr, /*invert = */ true);
    P->CreateKernelCall<DebugDisplayKernel>("compressedSepNum", compressedSepNum);
    
    StreamSet * compressedFieldNum = P->CreateStreamSet(fieldCountBits);
    P->CreateKernelCall<FieldNumberingKernel>(compressedSepNum, recordsByField, compressedFieldNum);
    P->CreateKernelCall<DebugDisplayKernel>("compressedFieldNum", compressedFieldNum);

    StreamSet * fieldNum = P->CreateStreamSet(fieldCountBits);
    SpreadByMask(P, fieldSeparators, compressedFieldNum, fieldNum);
    
    P->CreateKernelCall<DebugDisplayKernel>("fieldNum", fieldNum);

    std::vector<unsigned> insertionAmts;
    unsigned maxInsertAmt = 0;
    for (auto & s : templateStrs) {
        unsigned insertAmt = s.size() - 1;
        insertionAmts.push_back(insertAmt);
        if (insertAmt > maxInsertAmt) maxInsertAmt = insertAmt;
    }
    llvm::errs() << "maxInsertamt = " << maxInsertAmt << "\n";
    
    const unsigned insertLengthBits = ceil_log2(maxInsertAmt+1);
    llvm::errs() << "insertLengthBits = " << insertLengthBits << "\n";

    StreamSet * InsertBixNum = P->CreateStreamSet(fieldCountBits);
    P->CreateKernelCall<StringInsertBixNum>(insertionAmts, fieldNum, InsertBixNum);
    P->CreateKernelCall<DebugDisplayKernel>("InsertBixNum", InsertBixNum);
    StreamSet * const SpreadMask = InsertionSpreadMask(P, InsertBixNum, InsertPosition::Before);

    // Baais bit streams expanded with 0 bits for each string to be inserted.
    StreamSet * ExpandedBasis = P->CreateStreamSet(8);
    SpreadByMask(P, SpreadMask, translatedBasis, ExpandedBasis);
    P->CreateKernelCall<DebugDisplayKernel>("ExpandedBasis", ExpandedBasis);
    
    // For each run of 0s marking insert positions, create a parallel
    // bixnum sequentially numbering the string insert positions.
    StreamSet * const InsertIndex = P->CreateStreamSet(insertLengthBits);
    P->CreateKernelCall<RunIndex>(SpreadMask, InsertIndex, nullptr, /*invert = */ true);
    P->CreateKernelCall<DebugDisplayKernel>("InsertIndex", InsertIndex);

    StreamSet * expandedFieldNum = P->CreateStreamSet(fieldCountBits);
    SpreadByMask(P, SpreadMask, fieldNum, expandedFieldNum);
    P->CreateKernelCall<DebugDisplayKernel>("expandedFieldNum", expandedFieldNum);

    StreamSet * InstantiatedBasis = P->CreateStreamSet(8);
    P->CreateKernelCall<StringReplaceKernel>(templateStrs, ExpandedBasis, SpreadMask, expandedFieldNum, InsertIndex, InstantiatedBasis, /* offset = */ -2);


    // The computed output can be converted back to byte stream form by the
    // P2S kernel (parallel-to-serial).
    StreamSet * Instantiated = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<P2SKernel>(InstantiatedBasis, Instantiated);

    //  The StdOut kernel writes a byte stream to standard output.
    P->CreateKernelCall<StdOutKernel>(Instantiated);

    return reinterpret_cast<CSVFunctionType>(P->compile());
}

int main(int argc, char *argv[]) {
    //  ParseCommandLineOptions uses the LLVM CommandLine processor, but we also add
    //  standard Parabix command line options such as -help, -ShowPablo and many others.
    codegen::ParseCommandLineOptions(argc, argv, {&CSV_Options, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});
    
    std::vector<std::string> headers;
    if (HeaderSpec == "") {
        headers = get_CSV_headers(inputFile);
    } else if (HeaderSpecNamesFile) {
        headers = get_CSV_headers(HeaderSpec);
    } else {
        headers = parse_CSV_headers(HeaderSpec);
    }
    std::vector<std::string> templateStrs = createJSONtemplateStrings(headers);

    //  A CPU driver is capable of compiling and running Parabix programs on the CPU.
    CPUDriver driver("csv_function");
    //  Build and compile the Parabix pipeline by calling the Pipeline function above.
    CSVFunctionType fn = generatePipeline(driver, templateStrs);
    //  The compile function "fn"  can now be used.   It takes a file
    //  descriptor as an input, which is specified by the filename given by
    //  the inputFile command line option.]

    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        llvm::errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    } else {
        //  Run the pipeline.
        fn(fd);
        close(fd);
    }
    return 0;
}
