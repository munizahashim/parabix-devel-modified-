/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>
#include <vector>

namespace IDISA { class IDISA_Builder; }

namespace kernel {

class StreamsMerge : public BlockOrientedKernel {
public:
    StreamsMerge(LLVMTypeSystemInterface & ts, const std::vector<StreamSet *> & inputs, StreamSet * output);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
};

class StreamsCombineKernel : public BlockOrientedKernel {
public:
    StreamsCombineKernel(LLVMTypeSystemInterface & ts, std::vector<unsigned> streamsNumOfSets);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
private:
    const std::vector<unsigned> mStreamsNumOfSets;
};

class StreamsSplitKernel : public BlockOrientedKernel {
public:
    StreamsSplitKernel(LLVMTypeSystemInterface & ts, std::vector<unsigned> streamsNumOfSets);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
private:
    const std::vector<unsigned> mStreamsNumOfSets;
};

class StreamsIntersect : public BlockOrientedKernel {
public:
    StreamsIntersect(LLVMTypeSystemInterface & ts, const std::vector<StreamSet *> & inputs, StreamSet * output);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
};

}

