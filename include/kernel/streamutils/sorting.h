/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel_builder.h>
#include <pablo/builder.hpp>
#include <kernel/streamutils/sorting.h>

using namespace kernel;
//
//   BitonicCompareStep implements one comparison step for multi-instance bitonic sorting/merging.
//   Inputs:
//     step: the step number counting from 0.  Items distance 1<<step apart will be compared.
//     k:  the comparison kind, either BitonicSort or Merge
//     Basis: a bixnum defining the sort order, i.e., the values to be compared.
//     SeqIndex:  a bixnum sequentially numbering items in each instance to be sorted
//   Output:
//     SwapMarks:  a bitstream indicating positions for swapping, i.e. positions i such that
//     the values to be sorted at positions i - (1 << step) and i are to be exchanged.
//
class BitonicCompareStep : public pablo::PabloKernel {
public:
    enum class Kind {BitonicSort, Merge};
    BitonicCompareStep(LLVMTypeSystemInterface & ts, unsigned step, Kind k,
                       StreamSet * Basis, StreamSet * SeqIndex, StreamSet * SwapMarks);
protected:
    std::string kindString(Kind k) {return k == Kind::BitonicSort ? "S_" : "M_";}
    void generatePabloMethod() override;
private:
    unsigned mStep;
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
