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
    StreamsMerge(VirtualDriver & driver, const std::vector<StreamSet *> & inputs, StreamSet * output);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
};

class StreamsCombineKernel : public BlockOrientedKernel {
public:
    StreamsCombineKernel(VirtualDriver & driver, std::vector<unsigned> streamsNumOfSets);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
private:
    const std::vector<unsigned> mStreamsNumOfSets;
};

class StreamsSplitKernel : public BlockOrientedKernel {
public:
    StreamsSplitKernel(VirtualDriver & driver, std::vector<unsigned> streamsNumOfSets);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
private:
    const std::vector<unsigned> mStreamsNumOfSets;
};

class StreamsIntersect : public BlockOrientedKernel {
public:
    StreamsIntersect(VirtualDriver & driver, const std::vector<StreamSet *> & inputs, StreamSet * output);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
};

}

