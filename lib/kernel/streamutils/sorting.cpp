/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/streamutils/sorting.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_var.h>
#include <pablo/bixnum/bixnum.h>

using namespace kernel;
using namespace pablo;

BitonicCompareStep::BitonicCompareStep(LLVMTypeSystemInterface & ts, unsigned step, Kind k,
                                       StreamSet * Basis, StreamSet * SeqIndex, StreamSet * SwapMarks)
: PabloKernel(ts, "BitonicCompareStep_" + std::to_string(step) + kindString(k) +
                  Basis->shapeString() + "_" + SeqIndex->shapeString(),
// inputs
{Binding{"Basis", Basis}, Binding{"SeqIndex", SeqIndex}},
// output
{Binding{"SwapMarks", SwapMarks}}), mStep(step), mCompareKind(k) {
}

void BitonicCompareStep::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNum Basis = getInputStreamSet("Basis");
    BixNum SeqIndex = getInputStreamSet("SeqIndex");
    Var * SwapVar = pb.createVar("SwapVar", pb.createZeroes());
    //
    // Bitonic swapping:
    // At step N (N = 0, 1, ...):
    // Input is divided into groups of size 1 << (N + 2) (group size 4, for N = 0).
    // Each input group is split into 2 subgroups of size (1 << (N + 1)) (size 2 for N = 0)
    // Within each subgroup, comparisons are made between elements distant (1 << N) apart.
    unsigned compare_distance = 1 << mStep;
    BixNumCompiler bnc0(pb);
    PabloAST * DistN = bnc0.UGE(SeqIndex, compare_distance);
    auto nested = pb.createScope();
    pb.createIf(DistN, nested);
    BixNumCompiler bnc(nested);
    BixNum Forward_Basis(Basis.size());
    for (unsigned i = 0; i < Basis.size(); i++) {
        Forward_Basis[i] = nested.createAdvance(Basis[i], pb.getInteger(compare_distance));
    }
    // Now we can identify the elements of subgroups by bit numbers.
    unsigned bit_identifying_hi_subgroup = mStep + 1;
    SeqIndex = bnc.ZeroExtend(SeqIndex, bit_identifying_hi_subgroup + 1);
    unsigned bit_identifying_subgroup_hi_elements = mStep;
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
