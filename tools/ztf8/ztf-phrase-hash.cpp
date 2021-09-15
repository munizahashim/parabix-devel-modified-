/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <kernel/streamutils/deletion.h>                      // for DeletionKernel
#include <kernel/io/source_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/basis/s2p_kernel.h>                    // for S2PKernel
#include <kernel/io/stdout_kernel.h>                 // for StdOutKernel_
#include <kernel/streamutils/pdep_kernel.h>
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
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <grep/grep_kernel.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <re/ucd/ucd_compiler.hpp>
#include <re/unicode/resolve_properties.h>
#include <re/unicode/re_name_resolve.h>
#include <pablo/bixnum/bixnum.h>
#include <kernel/core/kernel_builder.h>
#include <pablo/pe_zeroes.h>
#include <toolchain/toolchain.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/core/streamset.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/streamutils/streams_merge.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/util/bixhash.h>
#include <kernel/util/debug_display.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <fcntl.h>
#include <iostream>
#include <iomanip>
#include <kernel/pipeline/pipeline_builder.h>
#include "ztf-logic.h"
#include "ztf-scan.h"
#include "ztf-phrase-scan.h"
#include "ztf-phrase-logic.h"

using namespace pablo;
using namespace parse;
using namespace kernel;
using namespace llvm;
using namespace codegen;

static cl::OptionCategory ztfHashOptions("ztfHash Options", "ZTF-Hash options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(ztfHashOptions));
static cl::opt<bool> Decompression("d", cl::desc("Decompress from ZTF-Runs to UTF-8."), cl::cat(ztfHashOptions), cl::init(false));
static cl::alias DecompressionAlias("decompress", cl::desc("Alias for -d"), cl::aliasopt(Decompression));
static cl::opt<int> SymCount("length", cl::desc("Length of words."), cl::init(2));

typedef void (*ztfHashFunctionType)(uint32_t fd);

EncodingInfo encodingScheme1(8,
                             {{3, 3, 2, 0xC0, 8, 0}, //minLen, maxLen, hashBytes, pfxBase, hashBits, length_extension_bits
                              {4, 4, 2, 0xC8, 8, 0},
                              {5, 8, 2, 0xD0, 8, 0},
                              {9, 16, 3, 0xE0, 8, 0},
                              {17, 32, 4, 0xF0, 8, 0},
                             });

ztfHashFunctionType ztfHash_compression_gen (CPUDriver & driver) {

    auto & b = driver.getBuilder();
    Type * const int32Ty = b->getInt32Ty();
    auto P = driver.makePipeline({Binding{int32Ty, "fd"}});

    Scalar * const fileDescriptor = P->getInputScalar("fd");

    // Source data
    StreamSet * const codeUnitStream = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<MMapSourceKernel>(fileDescriptor, codeUnitStream);

    StreamSet * u8basis = P->CreateStreamSet(8);
    P->CreateKernelCall<S2PKernel>(codeUnitStream, u8basis);

    StreamSet * WordChars = P->CreateStreamSet(1);
    P->CreateKernelCall<WordMarkKernel>(u8basis, WordChars);

    StreamSet * const phraseRuns = P->CreateStreamSet(1);
    StreamSet * const cwRuns = P->CreateStreamSet(1);
    P->CreateKernelCall<ZTF_Phrases>(u8basis, WordChars, phraseRuns, cwRuns);
    //P->CreateKernelCall<DebugDisplayKernel>("phraseRuns", phraseRuns);

    std::vector<StreamSet *> phraseLenBixnum(SymCount);
    StreamSet * const runIndex = P->CreateStreamSet(5);
    StreamSet * const overflow = P->CreateStreamSet(1);
    P->CreateKernelCall<RunIndex>(phraseRuns, runIndex, overflow);
    phraseLenBixnum[0] = runIndex;

    StreamSet * const phraseRunsFinal = P->CreateStreamSet(1);
    P->CreateKernelCall<ShiftBack>(phraseRuns, phraseRunsFinal, 1);
    //P->CreateKernelCall<DebugDisplayKernel>("phraseRunsFinal", phraseRunsFinal);
    //P->CreateKernelCall<DebugDisplayKernel>("phraseRuns", phraseRuns);
    for(unsigned i = 1; i < SymCount; i++) {
        StreamSet * const accumRunIndex = P->CreateStreamSet(5);
        StreamSet * const accumOverflow = P->CreateStreamSet(1);
        P->CreateKernelCall<AccumRunIndex>(phraseRunsFinal, runIndex, overflow, accumRunIndex, accumOverflow);
        phraseLenBixnum[i]= accumRunIndex;
        //P->CreateKernelCall<DebugDisplayKernel>("accumRunIndex", accumRunIndex);
    }

    std::vector<StreamSet *> bixHashes(SymCount);
    std::vector<StreamSet *> allHashValues(SymCount);
    StreamSet * basisStart = u8basis;
    for(unsigned i = 0; i < SymCount; i++) {
        StreamSet * const bixHash = P->CreateStreamSet(encodingScheme1.MAX_HASH_BITS);
        P->CreateKernelCall<BixHash>(basisStart, phraseRuns, bixHash, i);
        bixHashes[i] = bixHash;
        //P->CreateKernelCall<DebugDisplayKernel>("bixHashes"+std::to_string(i), bixHashes[i]);
        basisStart = bixHash;
        // only 1-symbol hashes have cumulative runLength appened to every byte of hash value
        // for 2-symbol phrase onwards, final cumulative runLength is appended only to the last byte of the symbol
        std::vector<StreamSet *> combinedHashData = {bixHashes[i], phraseLenBixnum[i]};
        StreamSet * const hashValues = P->CreateStreamSet(1, 16);
        P->CreateKernelCall<P2S16Kernel>(combinedHashData, hashValues);
        allHashValues[i] = hashValues;
        //P->CreateKernelCall<DebugDisplayKernel>("allHashValues["+std::to_string(i)+"]", allHashValues[i]);
    }

    StreamSet * u8bytes = codeUnitStream;
    std::vector<StreamSet *> extractionMasks;
    std::vector<StreamSet *> allHashMarks;

    std::vector<StreamSet *> bixnumMarks(4);
    for (unsigned sym = 0; sym < SymCount; sym++) {
        StreamSet * lgHashMarks = P->CreateStreamSet(1);
        for (unsigned i = 0; i < encodingScheme1.byLength.size(); i++) {
            StreamSet * groupMarks = P->CreateStreamSet(1);
            P->CreateKernelCall<LengthGroupSelector>(encodingScheme1, i, phraseRuns, phraseLenBixnum[sym], overflow, groupMarks);
            //P->CreateKernelCall<DebugDisplayKernel>("groupMarks", groupMarks);
            StreamSet * const hashMarks = P->CreateStreamSet(1);
            P->CreateKernelCall<MarkRepeatedHashvalue>(encodingScheme1, i, groupMarks, allHashValues[sym], hashMarks);
            //P->CreateKernelCall<DebugDisplayKernel>("hashMarks1", hashMarks);
            if (i > 0) {
                StreamSet * selectedHashMarks = P->CreateStreamSet(1);
                P->CreateKernelCall<InverseStream>(hashMarks, lgHashMarks, i, selectedHashMarks);
                //P->CreateKernelCall<DebugDisplayKernel>("selectedHashMarks", selectedHashMarks);
                lgHashMarks = selectedHashMarks;
            }
            else {
                lgHashMarks = hashMarks;
            }
            // gather all the lengthGroup bixnum positions
            if (sym > 0 && i > 0) { // no k-symbol phrases of length 3; start from len 4 phrases
                LengthGroupInfo groupInfo = encodingScheme1.byLength[i];
                unsigned lo = groupInfo.lo;
                unsigned hi = groupInfo.hi;
                unsigned groupSize = hi - lo + 1;
                StreamSet * const bixnumLenMarks = P->CreateStreamSet(groupSize);
                P->CreateKernelCall<LengthSelector>(encodingScheme1, i, phraseLenBixnum[sym], hashMarks, bixnumLenMarks);
                bixnumMarks[i-1] = bixnumLenMarks;
            }
        }
        // identify compressible phrases across length groups
        if (sym > 0) {
            StreamSet * lgSelectedUntilNow = P->CreateStreamSet(1);
            // lgHashMarks -> end pos of all the k-sym phrases with repeated hash codes
            //P->CreateKernelCall<DebugDisplayKernel>("lgHashMarks-before", lgHashMarks);
            // selectedBixnum -> 29 x i1 streamset of length-wise accumulated bixnumMarks
            StreamSet * const selectedBixnum = P->CreateStreamSet(29); // 1+4+8+16
            P->CreateKernelCall<StreamSelect>(selectedBixnum, Select( { {bixnumMarks[0]}, {bixnumMarks[1]}, {bixnumMarks[2]}, {bixnumMarks[3]} } ));
            //P->CreateKernelCall<DebugDisplayKernel>("selectedBixnum", selectedBixnum);
            // contains 29 x i1 stream indicating the hashMarks of each length in the range 4-32
            phraseLenBixnum[sym] = selectedBixnum;
            for(int i = 28; i >= 0; i--) {
                if (i == 28) {
                    lgSelectedUntilNow = lgHashMarks;
                }
                StreamSet * const selectedStep1 = P->CreateStreamSet(1);
                // 1. select max number of non-overlapping phrases of length i+4 (currLen)
                // 2. eliminate all the currLen phrases preceeded by longer length phrases
                P->CreateKernelCall<OverlappingLengthGroupMarker>(i, selectedBixnum, lgHashMarks, lgSelectedUntilNow, selectedStep1);
                // 3. eliminate all the curLen phrases preceeding longer length phrases
                StreamSet * const selectedStep2 = P->CreateStreamSet(1);
                P->CreateKernelCall<OverlappingLookaheadMarker>(i, selectedBixnum, lgSelectedUntilNow, selectedStep1, selectedStep2);
                lgSelectedUntilNow = selectedStep2;
            }
            lgHashMarks = lgSelectedUntilNow;
        }
        //P->CreateKernelCall<DebugDisplayKernel>("lgHashMarks-after", lgHashMarks);
        allHashMarks.push_back(lgHashMarks);
    }

    for(int i = SymCount-1; i > 0; i--) {
        StreamSet * hashMarksFinal = P->CreateStreamSet(1);
        // hashMark positions divided across min through max len values
        StreamSet * hashMarksBixNum = P->CreateStreamSet(29);
        P->CreateKernelCall<BixnumHashMarks>(phraseLenBixnum[i], allHashMarks[i], hashMarksBixNum);
        P->CreateKernelCall<PhraseSelection>(allHashMarks[i], hashMarksBixNum, allHashMarks[i-1], i, hashMarksFinal);
        allHashMarks[i-1] = hashMarksFinal;
    }

    for (int sym = SymCount-1; sym >= 0; sym--) {
        StreamSet * extractionMask = P->CreateStreamSet(1);
        StreamSet * input_bytes = u8bytes;
        StreamSet * output_bytes = P->CreateStreamSet(1, 8);
        //P->CreateKernelCall<DebugDisplayKernel>("allHashMarks["+std::to_string(sym)+"]", allHashMarks[sym]);
        P->CreateKernelCall<SymbolGroupCompression>(encodingScheme1, sym+1, allHashMarks[sym], allHashValues[sym], input_bytes, extractionMask, output_bytes);
        //P->CreateKernelCall<DebugDisplayKernel>("extractionMask", extractionMask);
        extractionMasks.push_back(extractionMask);
        u8bytes = output_bytes;
    }

    StreamSet * const combinedMask = P->CreateStreamSet(1);
    P->CreateKernelCall<StreamsIntersect>(extractionMasks, combinedMask);
    StreamSet * const encoded = P->CreateStreamSet(8);
    P->CreateKernelCall<S2PKernel>(u8bytes, encoded);

    StreamSet * const ZTF_basis = P->CreateStreamSet(8);
    FilterByMask(P, combinedMask, encoded, ZTF_basis);

    StreamSet * const ZTF_bytes = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<P2SKernel>(ZTF_basis, ZTF_bytes);
    P->CreateKernelCall<StdOutKernel>(ZTF_bytes);
    return reinterpret_cast<ztfHashFunctionType>(P->compile());
}


ztfHashFunctionType ztfHash_decompression_gen (CPUDriver & driver) {
    auto & b = driver.getBuilder();
    Type * const int32Ty = b->getInt32Ty();
    auto P = driver.makePipeline({Binding{int32Ty, "fd"}});
    Scalar * const fileDescriptor = P->getInputScalar("fd");

    // Source data
    StreamSet * const source = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<MMapSourceKernel>(fileDescriptor, source);
    StreamSet * const ztfHashBasis = P->CreateStreamSet(8);
    P->CreateKernelCall<S2PKernel>(source, ztfHashBasis);

    return reinterpret_cast<ztfHashFunctionType>(P->compile());
}

int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {&ztfHashOptions, pablo_toolchain_flags(), codegen::codegen_flags()});

    CPUDriver pxDriver("ztfPhraseHash");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    if (LLVM_UNLIKELY(fd == -1)) {
        errs() << "Error: cannot open " << inputFile << " for processing. Skipped.\n";
    } else {
        if (Decompression) {
            errs() << "Coming soon!" << "\n";
            //auto ztfHashDecompressionFunction = ztfHash_decompression_gen(pxDriver);
            //ztfHashDecompressionFunction(fd);
        } else {
            auto ztfHashCompressionFunction = ztfHash_compression_gen(pxDriver);
            ztfHashCompressionFunction(fd);
        }
        close(fd);
    }
    return 0;
}
