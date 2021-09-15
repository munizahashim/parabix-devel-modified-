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

/*
Input: hashMarks, kSymBixnum, prevHashMask, phraseRuns
Output: hashMask
*/

class PhraseSelection : public pablo::PabloKernel {
public:
    PhraseSelection(BuilderRef kb,
                StreamSet * hashMarks,
                StreamSet * hashMarksBixNum,
                StreamSet * prevHashMarks,
                unsigned symNum,
                StreamSet * updatedHashMark);
protected:
    void generatePabloMethod() override;
    unsigned mSymNum;
};

class InverseStream : public pablo::PabloKernel {
public:
    InverseStream(BuilderRef kb,
                StreamSet * hashMarks,
                StreamSet * prevMarks,
                unsigned groupNum,
                StreamSet * selected);
protected:
    void generatePabloMethod() override;
    unsigned mGroupNum;
};

class LengthSelector final: public pablo::PabloKernel {
public:
    LengthSelector(BuilderRef b,
                 EncodingInfo & encodingScheme,
                 unsigned groupNo,
                 StreamSet * groupLenBixnum,
                 StreamSet * hashMarks,
                 StreamSet * selected);
protected:
    void generatePabloMethod() override;
    EncodingInfo & mEncodingScheme;
    unsigned mGroupNo;
};

// mark overlapping phrases within a length group
class OverlappingLengthGroupMarker final: public pablo::PabloKernel {
public:
    OverlappingLengthGroupMarker(BuilderRef b,
                 unsigned groupNo,
                 StreamSet * groupLenBixnum,
                 StreamSet * hashMarks,
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
                 StreamSet * prevSelected,
                 StreamSet * selected);
protected:
    void generatePabloMethod() override;
    unsigned mGroupNo;
};

/*
hashMarksBixNum[i] contains the hashMarks for phrases in the length range(i+4, 32)
*/
class BixnumHashMarks final: public pablo::PabloKernel {
public:
    BixnumHashMarks(BuilderRef b,
                 StreamSet * phraseLenBixnum,
                 StreamSet * hashMarks,
                 StreamSet * hashMarksBixNum);
protected:
    void generatePabloMethod() override;
};

}
#endif

