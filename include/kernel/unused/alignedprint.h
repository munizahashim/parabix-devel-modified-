/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>
namespace IDISA { class IDISA_Builder; }  // lines 14-14
namespace llvm { class Value; }

namespace kernel {

class PrintableBits final : public BlockOrientedKernel {
public:
    PrintableBits(KernelBuilder & builder);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
};

class SelectStream final : public BlockOrientedKernel {
public:
    SelectStream(KernelBuilder & builder, unsigned sizeInputStreamSet, unsigned streamIndex);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
    unsigned mSizeInputStreamSet;
    unsigned mStreamIndex;
};

class ExpandOrSelectStreams final : public BlockOrientedKernel {
public:
    ExpandOrSelectStreams(KernelBuilder & builder, unsigned sizeInputStreamSet, unsigned sizeOutputStreamSet);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
    unsigned mSizeInputStreamSet;
    unsigned mSizeOutputStreamSet;
};

class PrintStreamSet final : public BlockOrientedKernel {
public:
    PrintStreamSet(KernelBuilder & builder, std::vector<std::string> && names, const unsigned minWidth = 16);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
private:
    const std::vector<std::string> mNames;
    unsigned mNameWidth;
};

}
