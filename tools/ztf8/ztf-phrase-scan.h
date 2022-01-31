/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */
#ifndef ZTF_PHRASE_SCAN_H
#define ZTF_PHRASE_SCAN_H

#include <pablo/pablo_kernel.h>
#include <kernel/core/kernel_builder.h>
#include "ztf-logic.h"

namespace kernel {

class MarkRepeatedHashvalue final : public MultiBlockKernel {
public:
    MarkRepeatedHashvalue(BuilderRef b,
                           EncodingInfo encodingScheme,
                           unsigned groupNo,
                           unsigned numSyms,
                           StreamSet * symbolMarks,
                           StreamSet * const hashValues,
                           StreamSet * const byteData,
                           StreamSet * hashMarks,
                           StreamSet * dictMask,
                           StreamSet * dictPhraseMask,
                           unsigned strideBlocks = 8);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;

    const EncodingInfo mEncodingScheme;
    const unsigned mGroupNo;
    const unsigned mNumSym;
};

class SymbolGroupCompression final : public MultiBlockKernel {
public:
    SymbolGroupCompression(BuilderRef b,
                           EncodingInfo encodingScheme,
                           unsigned groupNo,
                           unsigned numSyms,
                           StreamSet * symbolMarks,
                           StreamSet * hashValues,
                           StreamSet * const byteData,
                           StreamSet * compressionMask,
                           StreamSet * encodedBytes,
                           StreamSet * codewordMask,
                           StreamSet * dictBoundaryMask,
                           unsigned strideBlocks = 8);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;

    const EncodingInfo mEncodingScheme;
    const unsigned mGroupNo;
    const unsigned mNumSym;
};

class WriteDictionary final : public MultiBlockKernel {
public:
    WriteDictionary(BuilderRef b,
                    EncodingInfo encodingScheme,
                    unsigned numSyms,
                    StreamSet * byteData,
                    StreamSet * codedBytes,
                    StreamSet * extractionMask,
                    StreamSet * phraseMask,
                    StreamSet * allLenHashValues,
                    StreamSet * dictBytes,
                    StreamSet * dictMask,
                    unsigned strideBlocks = 8);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;
    const unsigned mNumSym;
    const unsigned mSubStride;
};

class InterleaveCompressionSegment final : public MultiBlockKernel {
public:
    InterleaveCompressionSegment(BuilderRef b,
                           StreamSet * byteData,
                           StreamSet * codedBytes,
                           StreamSet * extractionMask,
                           StreamSet * dictionaryMask,
                           StreamSet * combinedBytes,
                           StreamSet * combinedMask,
                           unsigned strideBlocks = 8);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;
    unsigned mStrideBlocks;
};

class SymbolGroupDecompression final : public MultiBlockKernel {
public:
    SymbolGroupDecompression(BuilderRef b,
                             EncodingInfo encodingScheme,
                             unsigned numSym,
                             unsigned groupNo,
                             StreamSet * const codeWordMarks,
                             StreamSet * const hashMarks,
                             StreamSet * const byteData,
                             StreamSet * const result,
                             unsigned strideBlocks = 8);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;

    const EncodingInfo mEncodingScheme;
    const unsigned mGroupNo;
    const unsigned mNumSym;
};

}
#endif
