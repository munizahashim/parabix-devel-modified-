/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */
#ifndef ZTF_PHRASELOGIC_H
#define ZTF_PHRASELOGIC_H

#include <pablo/pablo_kernel.h>
#include <kernel/core/kernel_builder.h>
#include "ztf-logic.h"

namespace kernel {

class UpdateNextHashMarks : public pablo::PabloKernel {
public:
    UpdateNextHashMarks(BuilderRef kb,
                StreamSet * extractionMask,
                StreamSet * hashMarksToUpdate,
                unsigned groupNo,
                StreamSet * hashMarksUpdated);
protected:
    void generatePabloMethod() override;
    unsigned mGroupNo;
};


class InverseStream : public pablo::PabloKernel {
public:
    InverseStream(BuilderRef kb,
                StreamSet * hashMarks,
                StreamSet * prevMarks,
                unsigned startLgIdx,
                unsigned groupNum,
                StreamSet * selected);
protected:
    void generatePabloMethod() override;
    unsigned mGroupNum;
    unsigned mStartIdx;
};

/*
Input : hashMarks, bixnum indicating length of k-symbol phrases
Output: Split the hashMarks across mEncodingScheme.minSymbolLength() and mEncodingScheme.maxSymbolLength()
Each stream in selectedHashMarksPos contains the hashMarks positions of phrases of same length.
*/
class LengthSelector final: public pablo::PabloKernel {
public:
    LengthSelector(BuilderRef b,
                 EncodingInfo & encodingScheme,
                 StreamSet * groupLenBixnum,
                 StreamSet * hashMarks,
                 StreamSet * selectedHashMarksPos);
protected:
    void generatePabloMethod() override;
    EncodingInfo & mEncodingScheme;
};

// 1. select non-overlapping phrases of same length
// 2. If any curLen phrase is overlapping prevSelected longer phrase, eliminate such phrase
class OverlappingLengthGroupMarker final: public pablo::PabloKernel {
public:
    OverlappingLengthGroupMarker(BuilderRef b,
                 unsigned groupNo,
                 StreamSet * lengthwiseHashMarks,
                 StreamSet * prevSelected,
                 StreamSet * selected);
protected:
    void generatePabloMethod() override;
    unsigned mGroupNo;
};

class OverlappingLookaheadMarker final: public pablo::PabloKernel {
public:
    OverlappingLookaheadMarker(BuilderRef b,
                 unsigned groupNo,
                 StreamSet * groupLenBixnum,
                 StreamSet * longerHashMarks,
                 StreamSet * selectedPart1,
                 StreamSet * selected);
protected:
    void generatePabloMethod() override;
    unsigned mGroupNo;
};

/*
hashMarksBixNum[i] contains the hashMarks for phrases in the length range(i+offset, 32)
*/
class BixnumHashMarks final: public pablo::PabloKernel {
public:
    BixnumHashMarks(BuilderRef b,
                 EncodingInfo & encodingScheme,
                 StreamSet * phraseLenBixnum,
                 StreamSet * hashMarks,
                 unsigned toUpdateHashMarks,
                 StreamSet * hashMarksBixNum);
protected:
    void generatePabloMethod() override;
    EncodingInfo & mEncodingScheme;
    unsigned mUpdateCount;
};

class ZTF_PhraseDecodeLengths : public pablo::PabloKernel {
public:
    ZTF_PhraseDecodeLengths(BuilderRef b,
                      EncodingInfo & encodingScheme,
                      StreamSet * basisBits,
                      StreamSet * hashtableSpan,
                      StreamSet * groupStreams,
                      StreamSet * hashtableStreams);
protected:
    void generatePabloMethod() override;
    EncodingInfo & mEncodingScheme;
    unsigned mDecodeStreamCount;
    unsigned mNumSym;
};

class ZTF_PhraseExpansionDecoder final: public pablo::PabloKernel {
public:
    ZTF_PhraseExpansionDecoder(BuilderRef b,
                         EncodingInfo & encodingScheme,
                         StreamSet * const basis,
                         StreamSet * insertBixNum,
                         StreamSet * hashtableSpan,
                         StreamSet * countStream);
protected:
    void generatePabloMethod() override;
    EncodingInfo & mEncodingScheme;
};

class codeword_index : public pablo::PabloKernel {
public:
    codeword_index(BuilderRef kb, StreamSet * Source, StreamSet * cwIndex);
protected:
    void generatePabloMethod() override;
};

}
#endif