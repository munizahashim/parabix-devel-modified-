/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include <kernel/io/source_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/basis/s2p_kernel.h>               // for S2PKernel
#include <kernel/io/stdout_kernel.h>               // for StdOutKernel
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/collapse.h>
#include <kernel/streamutils/multiplex.h>
#include <kernel/scan/scan.h>
#include <kernel/scan/reader.h>
#include <kernel/util/linebreak_kernel.h>
#include <kernel/util/debug_display.h>
#include <kernel/util/nesting.h>
#include <llvm/IR/Function.h>                      // for Function, Function...
#include <llvm/IR/Module.h>                        // for Module
#include <llvm/Support/CommandLine.h>              // for ParseCommandLineOp...
#include <llvm/Support/Debug.h>                    // for dbgs
#include <pablo/pablo_kernel.h>                    // for PabloKernel
#include <toolchain/pablo_toolchain.h>
#include <pablo/parse/pablo_source_kernel.h>
#include <pablo/parse/pablo_parser.h>
#include <pablo/parse/simple_lexer.h>
#include <pablo/parse/rd_parser.h>
#include <pablo/bixnum/bixnum.h>
#include <kernel/core/kernel_builder.h>
#include <pablo/pe_zeroes.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/core/streamset.h>
#include <kernel/streamutils/streams_merge.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/builder.hpp>
#include <fcntl.h>
#include <iostream>
#include <iomanip>
#include <kernel/pipeline/pipeline_builder.h>
#include "json-kernel.h"
#include "postprocess/json-simple.h"
#include "postprocess/json-detail.h"
#include "postprocess/json-parens.h"
#include "postprocess/json2csv.h"

namespace su = kernel::streamutils;

using namespace pablo;
using namespace pablo::parse;
using namespace kernel;
using namespace llvm;
using namespace codegen;

static cl::OptionCategory jsonOptions("json Options", "json options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(jsonOptions));
bool ToCSVFlag;
static cl::opt<bool, true> ToCSVOption("to-csv", cl::location(ToCSVFlag), cl::desc("Print equivalent CSV"), cl::cat(jsonOptions));
bool ShowLinesFlag;
static cl::opt<bool, true> ShowLinesOption("show-lines", cl::location(ShowLinesFlag), cl::desc("Display line number on error"), cl::cat(jsonOptions));
bool ShowStreamsFlag;
static cl::opt<bool, true> ShowStreamsOption("show-streams", cl::location(ShowStreamsFlag), cl::desc("Show streams with Parabix illustrator."), cl::cat(jsonOptions));
unsigned MaxDepth;
static cl::opt<unsigned, true> MaxDepthOption("max-depth", cl::location(MaxDepth), cl::desc("Max nesting depth for JSON."), cl::cat(jsonOptions), cl::init(15));
int OnlyDepth;
static cl::opt<int, true> OnlyDepthOption("only-depth", cl::location(OnlyDepth), cl::desc("Only generate code for depth n of JSON."), cl::cat(jsonOptions), cl::init(-1));

typedef void (*jsonFunctionType)(uint32_t fd);

ParabixIllustrator illustrator(64);

jsonFunctionType json_parsing_gen(CPUDriver & driver, std::shared_ptr<PabloParser> parser, std::shared_ptr<SourceFile> jsonPabloSrc) {

    auto & b = driver.getBuilder();
    Type * const int32Ty = b->getInt32Ty();
    auto P = driver.makePipeline({Binding{int32Ty, "fd"}});

    Scalar * const fileDescriptor = P->getInputScalar("fd");

    // Source data
    StreamSet * const codeUnitStream = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<MMapSourceKernel>(fileDescriptor, codeUnitStream);

    illustrator.captureByteData(P, "codeUnitStream", codeUnitStream);
    StreamSet * const u8basis = P->CreateStreamSet(8);
    P->CreateKernelCall<S2PKernel>(codeUnitStream, u8basis);

    // 1. Lexical analysis on basis stream
    StreamSet * const lexStream = P->CreateStreamSet(14);
    P->CreateKernelCall<PabloSourceKernel>(
        parser,
        jsonPabloSrc,
        "ClassifyBytes",
        Bindings { // Input Stream Bindings
            Binding {"basis", u8basis}
        },
        Bindings { // Output Stream Bindings
            Binding {"lex", lexStream}
        }
    );

    // 2. Find string marker (without backslashes)
    // 3. and make string span
    StreamSet * const stringMarker = P->CreateStreamSet(1);
    StreamSet * const stringSpan = P->CreateStreamSet(1);
    P->CreateKernelCall<JSONStringMarker>(
        lexStream,
        stringMarker,
        stringSpan
    );
    illustrator.captureBitstream(P, "stringSpan", stringSpan);

    // 4. Mark end of keywords (true, false, null)
    // Note: We mark the words later when we sanitize the input because
    // lookahead only works on input streams
    StreamSet * const keywordEndMarkers = P->CreateStreamSet(3);
    P->CreateKernelCall<JSONKeywordEndMarker>(
        u8basis,
        lexStream,
        stringSpan,
        keywordEndMarkers
    );

    // 5. Validate numbers
    StreamSet * const numberLex = P->CreateStreamSet(1);
    StreamSet * const numberSpan = P->CreateStreamSet(1);
    StreamSet * const numberErr = P->CreateStreamSet(1);
    P->CreateKernelCall<JSONNumberSpan>(
        u8basis,
        lexStream,
        stringSpan,
        numberLex,
        numberSpan,
        numberErr
    );
    illustrator.captureBitstream(P, "numberSpan", numberSpan);

    // 6. Validate strings
    StreamSet * const utf8Err = P->CreateStreamSet(1);
    P->CreateKernelCall<PabloSourceKernel>(
        parser,
        jsonPabloSrc,
        "ValidateUTF8",
        Bindings { // Input Stream Bindings
            Binding {"basis", u8basis},
            Binding {"lex", lexStream}
        },
        Bindings { // Output Stream Bindings
            Binding {"utf8Err", utf8Err}
        }
    );

    // 7. Clean lexers (in case there's special chars inside string)
    // 8. Validate rest of the output (check for extraneous chars)
    // We also take the opportunity to create the keyword marker
    StreamSet * const combinedLexers = P->CreateStreamSet(4);
    StreamSet * const extraErr = P->CreateStreamSet(1);
    P->CreateKernelCall<JSONFindKwAndExtraneousChars>(
        lexStream,
        stringSpan,
        numberSpan,
        keywordEndMarkers,
        combinedLexers,
        extraErr
    );

    // 9.1 Prepare and validate StreamSets
    if (!ToCSVFlag && !ShowLinesFlag) {
        StreamSet * const brackets = su::Select(P, combinedLexers, su::Range(1, 3));
        StreamSet * const depthErr = P->CreateStreamSet(1);
        StreamSet * const syntaxErr = P->CreateStreamSet(1);
        StreamSet * const encDepth = P->CreateStreamSet(std::ceil(std::log2(MaxDepth+1)));
        P->CreateKernelCall<NestingDepth>(
            brackets,
            encDepth,
            depthErr,
            MaxDepth
        );
        illustrator.captureBixNum(P, "encDepth", encDepth);
        P->CreateKernelCall<JSONParser>(
            lexStream,
            combinedLexers,
            encDepth,
            syntaxErr,
            MaxDepth,
            OnlyDepth
        );

        illustrator.captureBitstream(P, "syntaxErr", syntaxErr);

        StreamSet * const Errors = P->CreateStreamSet(5, 1);
        P->CreateKernelCall<StreamsMerge>(
            std::vector<StreamSet *>{extraErr, utf8Err, numberErr, depthErr, syntaxErr},
            Errors
        );

        StreamSet * const Errs = su::Collapse(P, Errors);
        StreamSet * const ErrIndices = scan::ToIndices(P, Errs);

        // 10. Output error in case JSON is not valid
        auto simpleErrFn = SCAN_CALLBACK(postproc_parensError);
        scan::Reader(P, driver, simpleErrFn, codeUnitStream, { ErrIndices });
    } else {
        StreamSet * collapsedLex;
        StreamSet * const symbols = su::Select(P, combinedLexers, 0);
        if (ToCSVFlag) {
            StreamSet * allLex = P->CreateStreamSet(5, 1);
            P->CreateKernelCall<StreamsMerge>(
                    std::vector<StreamSet *>{symbols, stringMarker, keywordEndMarkers, numberLex, stringSpan},
                    allLex
            );
            collapsedLex = su::Collapse(P, allLex);
        } else {
            StreamSet * allLex = P->CreateStreamSet(4, 1);
            P->CreateKernelCall<StreamsMerge>(
                std::vector<StreamSet *>{symbols, stringMarker, keywordEndMarkers, numberLex},
                allLex
            );
            collapsedLex = su::Collapse(P, allLex);
        }

        // 9.1.1 Prepare StreamSets to show lines on error
        //    If flag -c is provided, parse for CSV
        auto normalJsonFn = SCAN_CALLBACK(postproc_validateObjectsAndArrays);
        auto normalCsv2JsonFn = SCAN_CALLBACK(json2csv_validateObjectsAndArrays);
        auto doneJsonFn = SCAN_CALLBACK(postproc_doneCallback);
        auto doneCsv2JsonFn = SCAN_CALLBACK(json2csv_doneCallback);
        auto normalErrFn = SCAN_CALLBACK(postproc_errorStreamsCallback);

        auto const LineBreaks = P->CreateStreamSet(1);
        P->CreateKernelCall<UnixLinesKernelBuilder>(codeUnitStream, LineBreaks, UnterminatedLineAtEOF::Add1);
        StreamSet * const LineNumbers = scan::LineNumbers(P, collapsedLex, LineBreaks);
        StreamSet * const LineSpans = scan::LineSpans(P, LineBreaks);
        StreamSet * const Spans = scan::FilterLineSpans(P, LineNumbers, LineSpans);
        StreamSet * const Indices = scan::ToIndices(P, collapsedLex);

        // 9.1.2 Validate objects and arrays
        auto fn = ToCSVFlag ? normalCsv2JsonFn : normalJsonFn;
        auto doneFn = ToCSVFlag ? doneCsv2JsonFn : doneJsonFn;
        scan::Reader(P, driver, fn, doneFn, codeUnitStream, { Indices, Spans }, { LineNumbers, Indices });

        StreamSet * const Errors = P->CreateStreamSet(3, 1);
        P->CreateKernelCall<StreamsMerge>(
            std::vector<StreamSet *>{extraErr, utf8Err, numberErr},
            Errors
        );

        // 9.1.3 Prepare error StreamSets
        StreamSet * const Errs = su::Collapse(P, Errors);
        StreamSet * const ErrIndices = scan::ToIndices(P, Errs);
        StreamSet * const Codes = su::Multiplex(P, Errs);

        // 10. Output error in case JSON is not valid
        scan::Reader(P, driver, normalErrFn, codeUnitStream, { ErrIndices, Spans }, { LineNumbers, Codes });
    }

    return reinterpret_cast<jsonFunctionType>(P->compile());
}

int main(int argc, char ** argv) {
    codegen::ParseCommandLineOptions(argc, argv, {&jsonOptions, pablo::pablo_toolchain_flags(), codegen::codegen_flags()});

    CPUDriver pxDriver("json");
    auto em = ErrorManager::Create();
    auto parser = RecursiveParser::Create(SimpleLexer::Create(em), em);
    auto jsonSource = SourceFile::Relative("json.pablo");
    if (jsonSource == nullptr) {
        std::cerr << "pablo-parser: error loading pablo source file: json.pablo\n";
    }
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    } else {
        auto jsonParsingFunction = json_parsing_gen(pxDriver, parser, jsonSource);
        jsonParsingFunction(fd);
        close(fd);
        if (ShowStreamsFlag) illustrator.displayAllCapturedData();
    }
    return 0;
}
