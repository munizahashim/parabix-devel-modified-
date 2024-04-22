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
    StdOutKernel(BuilderRef iBuilder, StreamSet * codeUnitBuffer);
private:
    void linkExternalMethods(BuilderRef b) override;
    void generateDoSegmentMethod(BuilderRef b) override;
private:
    const unsigned mCodeUnitWidth;

};

class FileSink final : public SegmentOrientedKernel {
public:
    FileSink(BuilderRef iBuilder, Scalar * outputFileName, StreamSet * codeUnitBuffer);
protected:
    void linkExternalMethods(BuilderRef b) override;
    void generateInitializeMethod(BuilderRef b) override;
    void generateDoSegmentMethod(BuilderRef b) override;
    void generateFinalizeMethod(BuilderRef b) override;
private:
    const unsigned mCodeUnitWidth;

};
}



