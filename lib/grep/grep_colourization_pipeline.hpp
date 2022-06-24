#ifndef GREP_COLOURIZATION_PIPELINE_HPP
#define GREP_COLOURIZATION_PIPELINE_HPP

#include <kernel/pipeline/pipeline_kernel.h>
#include <grep/grep_kernel.h>
#include <kernel/streamutils/string_insert.h>
#include <kernel/streamutils/run_index.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/pipeline/pipeline_builder.h>

namespace kernel {


class GrepColourizationPipeline : public PipelineKernel {
public:
    GrepColourizationPipeline(BuilderRef b,
                              StreamSet * MatchSpans,
                              StreamSet * Basis,
                              StreamSet * ColorizedBasis)
        : PipelineKernel(b
                         // signature
                         , "GrepColourization"
                         // num of threads
                         , 1
                         // kernel list
                         , {}
                         // called functions
                         , {}
                         // stream inputs
                         , {{"MatchSpans", MatchSpans}, {"Basis", Basis}}
                         // stream outputs
                         , {{"ColorizedBasis", ColorizedBasis}} // not fixed!
                         // scalars
                         , {}, {}
                         // length assertions
                         , {}) {
        addAttribute(InternallySynchronized());
    }

    void instantiateNestedPipeline(const std::unique_ptr<PipelineBuilder> & E) final {
        const std::string ESC = "\x1B";
        const std::vector<std::string> colorEscapes = {ESC + "[01;31m" + ESC + "[K", ESC + "[m"};
        const  unsigned insertLengthBits = 4;
        std::vector<unsigned> insertAmts;
        for (auto & s : colorEscapes) {insertAmts.push_back(s.size());}

        StreamSet * const InsertMarks = E->CreateStreamSet(2, 1);
        StreamSet * const MatchSpans = getInputStreamSet(0);
        E->CreateKernelCall<SpansToMarksKernel>(MatchSpans, InsertMarks);

        StreamSet * const InsertBixNum = E->CreateStreamSet(insertLengthBits, 1);
        E->CreateKernelCall<ZeroInsertBixNum>(insertAmts, InsertMarks, InsertBixNum);
        //E->CreateKernelCall<DebugDisplayKernel>("InsertBixNum", InsertBixNum);
        StreamSet * const SpreadMask = InsertionSpreadMask(E, InsertBixNum, InsertPosition::Before);
        //E->CreateKernelCall<DebugDisplayKernel>("SpreadMask", SpreadMask);

        // For each run of 0s marking insert positions, create a parallel
        // bixnum sequentially numbering the string insert positions.
        StreamSet * const InsertIndex = E->CreateStreamSet(insertLengthBits);
        E->CreateKernelCall<RunIndex>(SpreadMask, InsertIndex, nullptr, RunIndex::Kind::RunOf0);
        //E->CreateKernelCall<DebugDisplayKernel>("InsertIndex", InsertIndex);
        // Baais bit streams expanded with 0 bits for each string to be inserted.

        StreamSet * const ExpandedBasis = E->CreateStreamSet(8);
        StreamSet * const Basis = getInputStreamSet(1);
        SpreadByMask(E, SpreadMask, Basis, ExpandedBasis);
        //E->CreateKernelCall<DebugDisplayKernel>("ExpandedBasis", ExpandedBasis);

        // Map the match start/end marks to their positions in the expanded basis.
        StreamSet * const ExpandedMarks = E->CreateStreamSet(2);
        SpreadByMask(E, SpreadMask, InsertMarks, ExpandedMarks);
        StreamSet * const ColorizedBasis = getOutputStreamSet(0);
        E->CreateKernelCall<StringReplaceKernel>(colorEscapes, ExpandedBasis, SpreadMask, ExpandedMarks, InsertIndex, ColorizedBasis, -1);

    }

};

}

#endif // GREP_COLOURIZATION_PIPELINE_HPP
