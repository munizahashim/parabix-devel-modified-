#ifndef GREP_COLOURIZATION_PIPELINE_HPP
#define GREP_COLOURIZATION_PIPELINE_HPP

#include <kernel/pipeline/pipeline_kernel.h>
#include <grep/grep_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <kernel/util/linebreak_kernel.h>
#include <kernel/scan/scanmatchgen.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/core/kernel_builder.h>

namespace kernel {


//class GrepColourizationPipeline : public PipelineKernel {
//public:
//    GrepColourizationPipeline(BuilderRef b,
//                              StreamSet * SourceCoords,
//                              StreamSet * MatchSpans,
//                              StreamSet * Basis)
//        : PipelineKernel(b
//                         // signature
//                         , "GrepColourization"
//                         // num of threads
//                         , 1
//                         // kernel list
//                         , {}
//                         // called functions
//                         , {}
//                         // stream inputs
//                         , {Bind("SourceCoords", SourceCoords, Deferred()),
//                            Bind("MatchSpans", MatchSpans, Deferred()),
//                            Bind("Basis", Basis, Deferred())}
//                         // stream outputs
//                         , {}
//                         // scalars
//                         , {}, {}
//                         // length assertions
//                         , {}) {
//        addAttribute(InternallySynchronized());
//        // NOTE: the 8x is to accommodate FilterByMask minimum I/O.
//        setStride(8 * b->getBitBlockWidth());
//    }

//    void instantiateNestedPipeline(const std::unique_ptr<PipelineBuilder> & E) final {
//        const std::string ESC = "\x1B";
//        const std::vector<std::string> colorEscapes = {ESC + "[01;31m" + ESC + "[K", ESC + "[m"};
//        const  unsigned insertLengthBits = 4;
//        std::vector<unsigned> insertAmts;
//        for (auto & s : colorEscapes) {insertAmts.push_back(s.size());}

//        StreamSet * const InsertMarks = E->CreateStreamSet(2, 1);
//        StreamSet * const MatchSpans = getInputStreamSet(1);
//        E->CreateKernelCall<SpansToMarksKernel>(MatchSpans, InsertMarks);

//        StreamSet * const InsertBixNum = E->CreateStreamSet(insertLengthBits, 1);
//        E->CreateKernelCall<ZeroInsertBixNum>(insertAmts, InsertMarks, InsertBixNum);
//        //E->CreateKernelCall<DebugDisplayKernel>("InsertBixNum", InsertBixNum);
//        StreamSet * const SpreadMask = InsertionSpreadMask(E, InsertBixNum, InsertPosition::Before);
//        //E->CreateKernelCall<DebugDisplayKernel>("SpreadMask", SpreadMask);

//        // For each run of 0s marking insert positions, create a parallel
//        // bixnum sequentially numbering the string insert positions.
//        StreamSet * const InsertIndex = E->CreateStreamSet(insertLengthBits);
//        E->CreateKernelCall<RunIndex>(SpreadMask, InsertIndex, nullptr, RunIndex::Kind::RunOf0);
//        //E->CreateKernelCall<DebugDisplayKernel>("InsertIndex", InsertIndex);
//        // Baais bit streams expanded with 0 bits for each string to be inserted.

//        StreamSet * const ExpandedBasis = E->CreateStreamSet(8);
//        StreamSet * const Basis = getInputStreamSet(2);
//        SpreadByMask(E, SpreadMask, Basis, ExpandedBasis);
//        //E->CreateKernelCall<DebugDisplayKernel>("ExpandedBasis", ExpandedBasis);

//        // Map the match start/end marks to their positions in the expanded basis.
//        StreamSet * const ExpandedMarks = E->CreateStreamSet(2);
//        SpreadByMask(E, SpreadMask, InsertMarks, ExpandedMarks);

//        StreamSet * ColorizedBasis = E->CreateStreamSet(8);
//        E->CreateKernelCall<StringReplaceKernel>(colorEscapes, ExpandedBasis, SpreadMask, ExpandedMarks, InsertIndex, ColorizedBasis, -1);


//        StreamSet * ColorizedBytes  = E->CreateStreamSet(1, 8);
//        E->CreateKernelCall<P2SKernel>(ColorizedBasis, ColorizedBytes);

//        StreamSet * ColorizedBreaks = E->CreateStreamSet(1);
//        E->CreateKernelCall<UnixLinesKernelBuilder>(ColorizedBasis, ColorizedBreaks, UnterminatedLineAtEOF::Add1);

//        StreamSet * ColorizedCoords = E->CreateStreamSet(3, sizeof(size_t) * 8);
//        E->CreateKernelCall<MatchCoordinatesKernel>(ColorizedBreaks, ColorizedBreaks, ColorizedCoords, 1);

//        // TODO: source coords >= colorized coords until the final stride?
//        // E->AssertEqualLength(SourceCoords, ColorizedCoords);


//        Scalar * const callbackObject = E->getInputScalar("callbackObject");

//        StreamSet * const SourceCoords = getInputStreamSet(0);
//        Kernel * const matchK = E->CreateKernelCall<ColorizedReporter>(ColorizedBytes, SourceCoords, ColorizedCoords, callbackObject);
//        matchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
//        matchK->link("finalize_match_wrapper", finalize_match_wrapper);
//    }

//};

}

#endif // GREP_COLOURIZATION_PIPELINE_HPP
