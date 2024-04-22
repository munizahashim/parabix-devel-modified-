/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>
namespace IDISA { class IDISA_Builder; }
namespace llvm { class Function; }
namespace llvm { class Module; }

namespace kernel {


class ScanMatchKernel : public MultiBlockKernel {
public:
    ScanMatchKernel(BuilderRef b, StreamSet * const Matches, StreamSet * const LineBreakStream, StreamSet * const ByteStream, Scalar * const callbackObject, unsigned strideBlocks = 1);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;
};

class ScanBatchKernel : public MultiBlockKernel {
public:
    ScanBatchKernel(BuilderRef b, StreamSet * const Matches, StreamSet * const LineBreakStream, StreamSet * const ByteStream, Scalar * const callbackObject, unsigned strideBlocks = 1);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;
};

class MatchCoordinatesKernel : public MultiBlockKernel {
public:
    MatchCoordinatesKernel(BuilderRef b,
                           StreamSet * const Matches, StreamSet * const LineBreakStream,
                           StreamSet * const Coordinates, unsigned strideBlocks = 1);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;
};

class BatchCoordinatesKernel : public MultiBlockKernel {
public:
    BatchCoordinatesKernel(BuilderRef b,
                           StreamSet * const Matches, StreamSet * const LineBreakStream,
                           StreamSet * const Coordinates, Scalar * const callbackObject, unsigned strideBlocks = 1);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;
};

class MatchReporter : public SegmentOrientedKernel {
public:
    MatchReporter(BuilderRef b,
                  StreamSet * ByteStream, StreamSet * const Coordinates, Scalar * const callbackObject);
private:
    void generateDoSegmentMethod(BuilderRef iBuilder) override;
};

class MatchFilterKernel : public MultiBlockKernel {
public:
    MatchFilterKernel(BuilderRef b, StreamSet * const MatchStarts, StreamSet * const LineBreaks,
                      StreamSet * const ByteStream, StreamSet * Output, unsigned strideBlocks = 1);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;
};

class ColorizedReporter : public SegmentOrientedKernel {
public:
    ColorizedReporter(BuilderRef b,
                       StreamSet * ByteStream, StreamSet * const SourceCoords, StreamSet * const ColorizedCoords,
                       Scalar * const callbackObject);
private:
    void generateDoSegmentMethod(BuilderRef iBuilder) override;
    unsigned mColorizedLineNumberIndex;
};

}
