/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/streamutils/sorting.h>
#include <kernel/streamutils/run_index.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_var.h>
#include <pablo/bixnum/bixnum.h>
#include <boost/intrusive/detail/math.hpp>

using boost::intrusive::detail::ceil_log2;
using namespace kernel;
using namespace pablo;

#define SHOW_STREAM(name) if (codegen::EnableIllustrator) P.captureBitstream(#name, name)
#define SHOW_BIXNUM(name) if (codegen::EnableIllustrator) P.captureBixNum(#name, name)
#define SHOW_BYTES(name) if (codegen::EnableIllustrator) P.captureByteData(#name, name)

BitonicCompareStep::BitonicCompareStep(LLVMTypeSystemInterface & ts, unsigned distance, unsigned region_size,
                                       StreamSet * SeqIndex, StreamSet * Basis, StreamSet * SwapMarks)
: PabloKernel(ts, "BitonicCompareStep<" + std::to_string(region_size) + "," + std::to_string(distance) + ">" +
              SeqIndex->shapeString() + "_" + Basis->shapeString(),
// inputs
{Binding{"SeqIndex", SeqIndex}, Binding{"Basis", Basis}},
// output
{Binding{"SwapMarks", SwapMarks}}), mCompareDistance(distance), mRegionSize(region_size) {
}

void BitonicCompareStep::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNum SeqIndex = getInputStreamSet("SeqIndex");
    BixNum Basis = getInputStreamSet("Basis");
    Var * SwapVar = pb.createVar("SwapVar", pb.createZeroes());
    BixNumCompiler bnc0(pb);
    PabloAST * DistN = bnc0.UGE(SeqIndex, mCompareDistance);
    // If no instance has sequential index reaching the comparison distance,
    // there will be nothing to compare.
    auto nested = pb.createScope();
    pb.createIf(DistN, nested);
    BixNumCompiler bnc(nested);
    BixNum Forward_Basis(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        Forward_Basis[i] = nested.createAdvance(Basis[i], pb.getInteger(mCompareDistance));
    }
    // Identify the separate regions.
    BixNum RegionNum;
    BixNum RegionIndex;
    bnc.Div(SeqIndex, mRegionSize, RegionNum, RegionIndex);
    // Alternate between ascending and descending regions based on
    // the region number (examing the low bit of RegionNum.
    PabloAST * descending_regions = RegionNum[0];
    PabloAST * compare = bnc.UGT(Forward_Basis, Basis);
    compare = nested.createXor(compare, descending_regions);
    // Identify the high element of each comparison for a potential swap mark.
    BixNum ComparisonGroup;
    BixNum GroupIndex;
    bnc.Div(RegionIndex, mCompareDistance * 2, ComparisonGroup, GroupIndex);
    PabloAST * hi_elements_in_comparisons = bnc.UGE(GroupIndex, mCompareDistance);
    PabloAST * swap_mark = nested.createAnd(compare, hi_elements_in_comparisons);
    nested.createAssign(SwapVar, swap_mark);
    pb.createAssign(pb.createExtract(getOutputStreamVar("SwapMarks"), pb.getInteger(0)), SwapVar);
}

SwapBack_N::SwapBack_N(LLVMTypeSystemInterface & ts, unsigned n, StreamSet * SwapMarks, StreamSet * Source, StreamSet * Swapped)
: PabloKernel(ts, "SwapBack" + std::to_string(n) + "_" + Source->shapeString(),
// inputs
{Binding{"SwapMarks", SwapMarks, FixedRate(1), LookAhead(n)},
 Binding{"Source", Source, FixedRate(1), LookAhead(n)}},
// output
{Binding{"Swapped", Swapped}}), mN(n) {
}

void SwapBack_N::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * SwapMarks = getInputStreamSet("SwapMarks")[0];
    PabloAST * PriorMark = pb.createLookahead(SwapMarks, mN);
    std::vector<PabloAST *> SourceSet = getInputStreamSet("Source");
    std::vector<Var *> SwappedVar(SourceSet.size());
    for (unsigned i = 0; i < SourceSet.size(); i++) {
        SwappedVar[i] = pb.createVar("SwapVar" + std::to_string(i), SourceSet[i]);
    }
    auto nested = pb.createScope();
    pb.createIf(pb.createOr(PriorMark, SwapMarks), nested);
    for (unsigned i = 0; i < SourceSet.size(); i++) {
        PabloAST * compare = nested.createXor(nested.createLookahead(SourceSet[i], mN), SourceSet[i]);
        compare = nested.createAnd(compare, PriorMark);
        PabloAST * flip = nested.createOr(compare, nested.createAdvance(compare, pb.getInteger(mN)));
        nested.createAssign(SwappedVar[i], nested.createXor(SourceSet[i], flip));
    }
    writeOutputStreamSet("Swapped", SwappedVar);
}

class RunTails : public PabloKernel {
public:
    RunTails(LLVMTypeSystemInterface & ts, unsigned lgth, StreamSet * Runs, StreamSet * SeqIndex, StreamSet * Tails);
protected:
    void generatePabloMethod() override;
private:
    unsigned mLgth;
};

RunTails::RunTails(LLVMTypeSystemInterface & ts, unsigned lgth, StreamSet * Runs, StreamSet * SeqIndex, StreamSet * Tails)
: PabloKernel(ts, "RunTail" + std::to_string(lgth) + "_" + SeqIndex->shapeString(),
// inputs
{Binding{"Runs", Runs, FixedRate(1), LookAhead(lgth+1)},
 Binding{"SeqIndex", SeqIndex, FixedRate(1), LookAhead(lgth)}},
// output
{Binding{"Tails", Tails}}), mLgth(lgth) {
}

void RunTails::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * Runs = getInputStreamSet("Runs")[0];
    std::vector<PabloAST *> SeqIndex = getInputStreamSet("SeqIndex");
    Var * tailVar = pb.createVar("Tails", pb.createZeroes());
    BixNumCompiler bnc0(pb);
    // First determine a potential tail candidate start position.
    PabloAST * ahead = pb.createLookahead(Runs, mLgth);
    PabloAST * tail_prior = pb.createAnd(Runs, ahead);
    auto nested = pb.createScope();
    pb.createIf(tail_prior, nested);
    BixNumCompiler bnc(nested);
    std::vector<PabloAST *> SeqIndexAhead(SeqIndex.size());
    for (unsigned i = 0; i < SeqIndex.size(); i++) {
        SeqIndexAhead[i] = nested.createLookahead(SeqIndex[i], mLgth);
    }
    PabloAST * confirm = bnc.EQ(bnc.AddModular(SeqIndex, mLgth), SeqIndexAhead);
    PabloAST * tail1 = nested.createAdvance(nested.createAnd(tail_prior, confirm), 1);
    tail1 = nested.createAnd(tail1, nested.createNot(ahead));
    PabloAST * tails = nested.createAnd(nested.createMatchStar(tail1, Runs), Runs);
    nested.createAssign(tailVar, tails);
    pb.createAssign(pb.createExtract(getOutputStreamVar("Tails"), pb.getInteger(0)), tailVar);
}

StreamSets  BitonicSortRuns(PipelineBuilder & P, unsigned instance_size, StreamSet * Runs, StreamSets & ToSort) {
    unsigned steps = ceil_log2(instance_size);
    StreamSet * SeqIndex = P.CreateStreamSet(steps);
    P.CreateKernelCall<RunIndex>(Runs, SeqIndex);
    SHOW_BIXNUM(SeqIndex);
    return BitonicSort(P, instance_size, Runs, SeqIndex, ToSort);
}

StreamSets BitonicSort(PipelineBuilder & P, unsigned instance_size, StreamSet * Runs, StreamSet * SeqIndex, StreamSets & ToSort) {
    unsigned region_size = instance_size/2;
    unsigned compare_distance = region_size/2;

    StreamSets PartiallySorted;
    if (compare_distance > 1) {
        PartiallySorted = BitonicSort(P, region_size, Runs, SeqIndex, ToSort);
    } else {
        PartiallySorted = ToSort;
    }

    StreamSet * SwapMarks = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<BitonicCompareStep>(compare_distance, region_size, SeqIndex, PartiallySorted[0], SwapMarks);
    SHOW_STREAM(SwapMarks);

    StreamSets Sorted(ToSort.size());
    for (unsigned i = 0; i < ToSort.size(); i++) {
        Sorted[i] = P.CreateStreamSet(ToSort[i]->getNumElements(), 1);
        P.CreateKernelCall<SwapBack_N>(compare_distance, SwapMarks, PartiallySorted[i], Sorted[i]);
        SHOW_BIXNUM(Sorted[i]);
    }

    StreamSet * Tails = P.CreateStreamSet(1, 1);
    StreamSet * TailIndex = P.CreateStreamSet(ceil_log2(region_size), 1);
    P.CreateKernelCall<RunTails>(region_size, Runs, SeqIndex, Tails);
    P.CreateKernelCall<RunIndex>(Tails, TailIndex);
    SHOW_STREAM(Tails);
    SHOW_BIXNUM(TailIndex);

    StreamSet * TailSwapMarks = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<BitonicCompareStep>(compare_distance, region_size, TailIndex, Sorted[0], TailSwapMarks);
    SHOW_STREAM(TailSwapMarks);

    StreamSets FullySorted(ToSort.size());
    for (unsigned i = 0; i < ToSort.size(); i++) {
        FullySorted[i] = P.CreateStreamSet(ToSort[i]->getNumElements(), 1);
        P.CreateKernelCall<SwapBack_N>(compare_distance, TailSwapMarks, Sorted[i], FullySorted[i]);
        SHOW_BIXNUM(FullySorted[i]);
    }

    if (region_size <=2 ) {
        return FullySorted;
    } else {
        return BitonicMerge(P, region_size, instance_size, SeqIndex, FullySorted);
    }
}

StreamSets BitonicMerge(PipelineBuilder & P, unsigned region_size, unsigned instance_size, StreamSet * RunIndex, StreamSets & ToMerge) {
    
    StreamSet * MergeSwapMarks = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<BitonicCompareStep>(region_size, instance_size, RunIndex, ToMerge[0], MergeSwapMarks);
    SHOW_STREAM(MergeSwapMarks);
    
    StreamSets Merged(ToMerge.size());
    for (unsigned i = 0; i < ToMerge.size(); i++) {
        Merged[i] = P.CreateStreamSet(ToMerge[i]->getNumElements(), 1);
        P.CreateKernelCall<SwapBack_N>(region_size, MergeSwapMarks, ToMerge[i], Merged[i]);
        SHOW_BIXNUM(Merged[i]);
    }
    if (region_size <= 4) {
        return Merged;
    } else {
        return BitonicMerge(P, region_size/2, instance_size, RunIndex, Merged);
    }
}
