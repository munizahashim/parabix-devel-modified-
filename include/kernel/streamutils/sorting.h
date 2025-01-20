/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/streamutils/sorting.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <pablo/builder.hpp>

using namespace kernel;
//
//   BitonicCompareStep implements one comparison step for multi-instance bitonic sorting/merging.
//   Inputs:
//     distance: the distance between compared items
//     k:  the comparison kind, either BitonicSort or Merge
//     SeqIndex:  a bixnum sequentially numbering items in each instance to be sorted
//     Basis: a bixnum defining the sort order, i.e., the values to be compared.
//   Output:
//     SwapMarks:  a bitstream indicating positions for swapping, i.e. positions i such that
//     the values to be sorted at positions i - distance and i are to be exchanged.
//
class BitonicCompareStep : public pablo::PabloKernel {
public:
    enum class Kind {BitonicSort, Merge};
    BitonicCompareStep(LLVMTypeSystemInterface & ts, unsigned distance, Kind k,
                       StreamSet * SeqIndex, StreamSet * Basis, StreamSet * SwapMarks);
protected:
    std::string kindString(Kind k) {return k == Kind::BitonicSort ? "S_" : "M_";}
    void generatePabloMethod() override;
private:
    unsigned mDistance;
    Kind mCompareKind;
};

class SwapBack_N : public pablo::PabloKernel {
public:
    SwapBack_N(LLVMTypeSystemInterface & ts, unsigned n, StreamSet * SwapMarks, StreamSet * Source, StreamSet * Swapped);
protected:
    void generatePabloMethod() override;
private:
    unsigned mN;
};

using StreamSets = std::vector<StreamSet *>;

StreamSets BitonicSortRuns(PipelineBuilder & P, unsigned runlgth, StreamSet * Runs, StreamSets & ToSort);

StreamSets BitonicSort(PipelineBuilder & P, unsigned runlgth, StreamSet * RunIndex, StreamSets & ToSort);

StreamSets BitonicMerge(PipelineBuilder & P, unsigned runlgth, StreamSet * RunIndex, StreamSets & ToMerge);
