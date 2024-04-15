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
    StreamsMerge(BuilderRef b, const std::vector<StreamSet *> & inputs, StreamSet * output);
protected:
    void generateDoBlockMethod(BuilderRef b) override;
};

class StreamsCombineKernel : public BlockOrientedKernel {
public:
    StreamsCombineKernel(BuilderRef b, std::vector<unsigned> streamsNumOfSets);
protected:
    void generateDoBlockMethod(BuilderRef b) override;
private:
    const std::vector<unsigned> mStreamsNumOfSets;
};

class StreamsSplitKernel : public BlockOrientedKernel {
public:
    StreamsSplitKernel(BuilderRef b, std::vector<unsigned> streamsNumOfSets);
protected:
    void generateDoBlockMethod(BuilderRef b) override;
private:
    const std::vector<unsigned> mStreamsNumOfSets;
};

class StreamsIntersect : public BlockOrientedKernel {
public:
    StreamsIntersect(BuilderRef b, const std::vector<StreamSet *> & inputs, StreamSet * output);
protected:
    void generateDoBlockMethod(BuilderRef b) override;
};

}

