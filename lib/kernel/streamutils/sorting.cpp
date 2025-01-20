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

BitonicCompareStep::BitonicCompareStep(LLVMTypeSystemInterface & ts, unsigned distance, Kind k,
                                       StreamSet * SeqIndex, StreamSet * Basis, StreamSet * SwapMarks)
: PabloKernel(ts, "BitonicCompareStep_" + std::to_string(distance) + kindString(k) +
              SeqIndex->shapeString() + "_" + Basis->shapeString(),
// inputs
{Binding{"SeqIndex", SeqIndex}, Binding{"Basis", Basis}},
// output
{Binding{"SwapMarks", SwapMarks}}), mDistance(distance), mCompareKind(k) {
}

void BitonicCompareStep::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNum SeqIndex = getInputStreamSet("SeqIndex");
    BixNum Basis = getInputStreamSet("Basis");
    Var * SwapVar = pb.createVar("SwapVar", pb.createZeroes());
    //
    // Bitonic swapping:
    // At step N (N = 0, 1, ...):
    // Comparison distance is mDistance = 1 << N.
    // Input is divided into groups of size 1 << (N + 2) (group size 4, for N = 0).
    // Each input group is split into 2 subgroups of size (1 << (N + 1)) (size 2 for N = 0)
    unsigned step = ceil_log2(mDistance);
    BixNumCompiler bnc0(pb);
    PabloAST * DistN = bnc0.UGE(SeqIndex, mDistance);
    auto nested = pb.createScope();
    pb.createIf(DistN, nested);
    BixNumCompiler bnc(nested);
    BixNum Forward_Basis(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        Forward_Basis[i] = nested.createAdvance(Basis[i], pb.getInteger(mDistance));
    }
    // Now we can identify the elements of subgroups by bit numbers.
    unsigned bit_identifying_hi_subgroup = step + 1;
    SeqIndex = bnc.ZeroExtend(SeqIndex, bit_identifying_hi_subgroup + 1);
    unsigned bit_identifying_subgroup_hi_elements = step;
    PabloAST * hi_elements_in_subgroups = SeqIndex[bit_identifying_subgroup_hi_elements];
    // Perform the comparisons
    PabloAST * compare = bnc.UGT(Forward_Basis, Basis);
    if (mCompareKind == Kind::BitonicSort) {
        PabloAST * hi_subgroups = SeqIndex[bit_identifying_hi_subgroup];
        // Reverse the comparisons for hi subgroups
        compare = nested.createXor(compare, hi_subgroups);
        // Bit flipping of > compare gives <= comparison, exclude the = cases.
        compare = nested.createAnd(compare, bnc.NEQ(Forward_Basis, Basis));
    }
    // Identify swaps at the high half of each subgroup only.
    PabloAST * swap_mark = nested.createAnd(compare, hi_elements_in_subgroups);
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

StreamSets  BitonicSortRuns(PipelineBuilder & P, unsigned runlgth, StreamSet * Runs, StreamSets & ToSort) {
    unsigned steps = ceil_log2(runlgth);
    StreamSet * runindex = P.CreateStreamSet(steps);
    P.CreateKernelCall<RunIndex>(Runs, runindex);
    return BitonicSort(P, runlgth, runindex, ToSort);
}

StreamSets BitonicSort(PipelineBuilder & P, unsigned runlgth, StreamSet * RunIndex, StreamSets & ToSort) {
    unsigned compare_distance = runlgth/2;
    
    StreamSets PartiallySorted;
    if (compare_distance > 1) {
        PartiallySorted = BitonicSort(P, compare_distance, RunIndex, ToSort);
    } else {
        PartiallySorted = ToSort;
    }

    StreamSet * SwapMarks = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<BitonicCompareStep>(compare_distance, BitonicCompareStep::Kind::BitonicSort, RunIndex, PartiallySorted[0], SwapMarks);
    SHOW_STREAM(SwapMarks);

    StreamSets FullySorted(PartiallySorted.size());
    for (unsigned i = 0; i < PartiallySorted.size(); i++) {
        FullySorted[i] = P.CreateStreamSet(PartiallySorted[i]->getNumElements(), 1);
        P.CreateKernelCall<SwapBack_N>(compare_distance, SwapMarks, PartiallySorted[i], FullySorted[i]);
        SHOW_BIXNUM(FullySorted[i]);
    }
    
    if (runlgth <=2 ) {
        return FullySorted;
    } else {
        return BitonicMerge(P, runlgth, RunIndex, FullySorted);
    }
}

StreamSets BitonicMerge(PipelineBuilder & P, unsigned runlgth, StreamSet * RunIndex, StreamSets & ToMerge) {
    unsigned half_run = runlgth/2;
    
    StreamSet * MergeSwapMarks = P.CreateStreamSet(1, 1);
    P.CreateKernelCall<BitonicCompareStep>(half_run, BitonicCompareStep::Kind::Merge, RunIndex, ToMerge[0], MergeSwapMarks);
    SHOW_STREAM(MergeSwapMarks);
    
    StreamSets Merged(ToMerge.size());
    for (unsigned i = 0; i < ToMerge.size(); i++) {
        Merged[i] = P.CreateStreamSet(ToMerge[i]->getNumElements(), 1);
        P.CreateKernelCall<SwapBack_N>(half_run, MergeSwapMarks, ToMerge[i], Merged[i]);
        SHOW_BIXNUM(Merged[i]);
    }
    if (runlgth <= 4) {
        return Merged;
    } else {
        return BitonicMerge(P, half_run, RunIndex, Merged);
    }
}
