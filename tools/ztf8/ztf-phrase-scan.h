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
                           StreamSet * hashValues,
                           StreamSet * hashMarks,
                           unsigned strideBlocks = 8);
private:
    void generateMultiBlockLogic(BuilderRef iBuilder, llvm::Value * const numOfStrides) override;

    const EncodingInfo mEncodingScheme;
    const unsigned mGroupNo;
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
