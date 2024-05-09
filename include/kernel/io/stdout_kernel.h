/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>

namespace IDISA { class IDISA_Builder; }

namespace kernel {

class StdOutKernel final : public SegmentOrientedKernel {
public:
    StdOutKernel(KernelBuilder & b, StreamSet * codeUnitBuffer);
private:
    void linkExternalMethods(KernelBuilder & b) override;
    void generateDoSegmentMethod(KernelBuilder & b) override;
private:
    const unsigned mCodeUnitWidth;

};

class FileSink final : public SegmentOrientedKernel {
public:
    FileSink(KernelBuilder & b, Scalar * outputFileName, StreamSet * codeUnitBuffer);
protected:
    void linkExternalMethods(KernelBuilder & b) override;
    void generateInitializeMethod(KernelBuilder & b) override;
    void generateDoSegmentMethod(KernelBuilder & b) override;
    void generateFinalizeMethod(KernelBuilder & b) override;
private:
    const unsigned mCodeUnitWidth;

};
}



