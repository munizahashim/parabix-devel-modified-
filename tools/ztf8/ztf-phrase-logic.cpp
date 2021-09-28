/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include "ztf-phrase-logic.h"
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pe_zeroes.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <pablo/pe_debugprint.h>
#include <re/ucd/ucd_compiler.hpp>
#include <re/unicode/resolve_properties.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>

using namespace pablo;
using namespace kernel;
using namespace llvm;

PhraseSelection::PhraseSelection(BuilderRef kb,
                StreamSet * hashMarks,
                StreamSet * hashMarksBixNum,
                StreamSet * prevHashMarks,
                unsigned symNum,
                StreamSet * updatedHashMark)
: PabloKernel(kb, "PhraseSelection",
            {Binding{"hashMarks", hashMarks, FixedRate(), LookAhead(32)},
             Binding{"hashMarksBixNum", hashMarksBixNum, FixedRate(), LookAhead(32)},
             Binding{"prevHashMarks", prevHashMarks}},
            {Binding{"updatedHashMark", updatedHashMark}}), mSymNum(symNum) { }

void PhraseSelection::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    PabloAST * toUpdateHashMarks = getInputStreamSet("prevHashMarks")[0];
    std::vector<PabloAST *> hashMarksBixNum = getInputStreamSet("hashMarksBixNum");
    // valid (k-1)-sym phrases after eliminating directly overlapping ones
    PabloAST * finalMask = hashMarks;
    for(unsigned pos = 1; pos < 32; pos++) {
        if (pos > 3) {
            PabloAST * lookaheadMarks = hashMarksBixNum[pos-4];
            finalMask = pb.createOr(finalMask, pb.createLookahead(lookaheadMarks, pos));
        }
        else {
            finalMask = pb.createOr(finalMask, pb.createLookahead(hashMarks, pos));
        }
    }
    PabloAST * result = pb.createXor(finalMask, toUpdateHashMarks);
    //pb.createDebugPrint(toUpdateHashMarks, "toUpdateHashMarks-before");
    toUpdateHashMarks = pb.createAnd(toUpdateHashMarks, pb.createXor(toUpdateHashMarks, hashMarks));
    //pb.createDebugPrint(finalMask, "finalMask");
    //pb.createDebugPrint(toUpdateHashMarks, "toUpdateHashMarks-after");
    //pb.createDebugPrint(result, "result");
    pb.createAssign(pb.createExtract(getOutputStreamVar("updatedHashMark"), pb.getInteger(0)), pb.createAnd(result, toUpdateHashMarks));
}

InverseStream::InverseStream(BuilderRef kb,
                StreamSet * hashMarks,
                StreamSet * prevMarks,
                unsigned groupNum,
                StreamSet * selected)
: PabloKernel(kb, "InverseStream" + std::to_string(groupNum),
            {Binding{"hashMarks", hashMarks},
             Binding{"prevMarks", prevMarks}},
            {Binding{"selected", selected}}), mGroupNum(groupNum) { }

void InverseStream::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    PabloAST * prevMarks = getInputStreamSet("prevMarks")[0];
    if (mGroupNum == 1) {
        prevMarks = pb.createNot(prevMarks);
    }
    PabloAST * result = pb.createNot(hashMarks);
    result = pb.createOr(result, prevMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("selected"), pb.getInteger(0)), result);

}

LengthSelector::LengthSelector(BuilderRef b,
                           EncodingInfo & encodingScheme,
                           unsigned groupNo,
                           StreamSet * groupLenBixnum,
                           StreamSet * hashMarks,
                           StreamSet * selectedHashMarksPos)
: PabloKernel(b, "LengthSelector" + std::to_string(groupNo),
              {Binding{"hashMarks", hashMarks, FixedRate(), LookAhead(1)},
               Binding{"groupLenBixnum", groupLenBixnum}},
              {Binding{"selectedHashMarksPos", selectedHashMarksPos}}), mEncodingScheme(encodingScheme), mGroupNo(groupNo) { }

void LengthSelector::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    Var * selectedHashMarksPosStreamVar = getOutputStreamVar("selectedHashMarksPos");
    LengthGroupInfo groupInfo = mEncodingScheme.byLength[mGroupNo];
    std::vector<PabloAST *> groupLenBixnum = getInputStreamSet("groupLenBixnum");
    std::vector<PabloAST *> selectedLengthMarks;
    unsigned offset = 2;
    unsigned lo = mEncodingScheme.minSymbolLength()+1;
    unsigned hi = mEncodingScheme.maxSymbolLength();
    unsigned groupSize = hi - lo + 1;
    std::string groupName = "lengthGroup" + std::to_string(lo) +  "_" + std::to_string(hi);
    for (unsigned i = lo; i <= hi; i++) {
        PabloAST * lenBixnum = bnc.EQ(groupLenBixnum, i - offset);
        selectedLengthMarks.push_back(lenBixnum);
        //pb.createDebugPrint(pb.createCount(pb.createInFile(lenBixnum)), "count"+std::to_string(i));
    }
    for (unsigned i = 0; i < groupSize; i++) {
        pb.createAssign(pb.createExtract(selectedHashMarksPosStreamVar, pb.getInteger(i)), pb.createAnd(hashMarks, selectedLengthMarks[i]));
    }
}

OverlappingLengthGroupMarker::OverlappingLengthGroupMarker(BuilderRef b,
                           unsigned groupNo,
                           StreamSet * lengthwiseHashMarks,
                           StreamSet * prevSelected,
                           StreamSet * selected)
: PabloKernel(b, "OverlappingLengthGroupMarker" + std::to_string(groupNo),
              {Binding{"lengthwiseHashMarks", lengthwiseHashMarks},
               Binding{"prevSelected", prevSelected}},
              {Binding{"selected", selected}}), mGroupNo(groupNo) { }

void OverlappingLengthGroupMarker::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    Var * selectedStreamVar = getOutputStreamVar("selected");
    std::vector<PabloAST *> lengthwiseHashMarks = getInputStreamSet("lengthwiseHashMarks");
    PabloAST * prevSelected = getInputStreamSet("prevSelected")[0];
    unsigned offset = 4;
    unsigned curPhraseLen = mGroupNo+offset;
    PabloAST * curLenPos = lengthwiseHashMarks[mGroupNo];
    PabloAST * selected = pb.createZeroes();
    PabloAST * toAdvance = curLenPos;
    for (unsigned loop = 0; loop < 2; loop++) { // loop depends on the max number of consecutive phrases of same length?
        PabloAST * notSelected = pb.createZeroes();
        for (unsigned i = 1; i < curPhraseLen; i++) {
            toAdvance = pb.createAdvance(toAdvance, 1);
            notSelected = pb.createOr(notSelected, pb.createAnd(toAdvance, curLenPos));
        }
        selected = pb.createOr(selected, pb.createXor(curLenPos, notSelected));
        toAdvance = selected;
    }

    PabloAST * toEliminate = pb.createZeroes();
    for (unsigned i = mGroupNo+1; i < lengthwiseHashMarks.size(); i++) {
        PabloAST * phrasePos = lengthwiseHashMarks[i];
        phrasePos = pb.createAnd(phrasePos, prevSelected); // update prevSelected correctly
        PabloAST * preceededByLongerPhrase = pb.createZeroes();
        for (unsigned j = 1; j < curPhraseLen; j++) {
           phrasePos = pb.createAdvance(phrasePos, 1);
           preceededByLongerPhrase = pb.createOr(preceededByLongerPhrase, pb.createAnd(phrasePos, selected));
        }
        toEliminate = pb.createOr(toEliminate, preceededByLongerPhrase);
    }
    pb.createAssign(pb.createExtract(selectedStreamVar, pb.getInteger(0)), pb.createXor(selected, toEliminate));
}


OverlappingLookaheadMarker::OverlappingLookaheadMarker(BuilderRef b,
                           unsigned groupNo,
                           StreamSet * groupLenBixnum,
                           StreamSet * longerHashMarks,
                           StreamSet * selectedPart1,
                           StreamSet * selected)
: PabloKernel(b, "OverlappingLookaheadMarker" + std::to_string(groupNo),
              {Binding{"groupLenBixnum", groupLenBixnum, FixedRate(), LookAhead(32)},
               Binding{"longerHashMarks", longerHashMarks},
               Binding{"selectedPart1", selectedPart1}},
              {Binding{"selected", selected}}), mGroupNo(groupNo) { }

void OverlappingLookaheadMarker::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * longerHashMarks = getInputStreamSet("longerHashMarks")[0];
    Var * selectedStreamVar = getOutputStreamVar("selected");
    std::vector<PabloAST *> groupLenBixnum = getInputStreamSet("groupLenBixnum");
    PabloAST * selectedPart1 = getInputStreamSet("selectedPart1")[0];
    PabloAST * selected = selectedPart1;
    unsigned offset = 4;
    if (mGroupNo < 28) {
        PabloAST * eliminateFinal = pb.createZeroes();
        for (unsigned i = mGroupNo+1; i < 29; i++) {
            //unsigned groupIdx = i - (mGroupNo + 1); ---> only when BixnumHashMarks kernel is used
            PabloAST * toEliminate = pb.createZeroes();
            PabloAST * lookaheadStream = groupLenBixnum[i];
            //pb.createDebugPrint(lookaheadStream, "lookaheadStream"+std::to_string(i));
            for (unsigned j = 1; j < i+offset; j++) {
                toEliminate = pb.createOr(toEliminate, pb.createAnd(selectedPart1, pb.createLookahead(lookaheadStream, j)));
            }
            eliminateFinal = pb.createOr(eliminateFinal, toEliminate);
        }
        //pb.createDebugPrint(selectedPart1, "selectedPart1");
        //pb.createDebugPrint(eliminateFinal, "eliminateFinal");
        selected = pb.createOr(longerHashMarks, pb.createXor(selectedPart1, eliminateFinal));
    }
    pb.createAssign(pb.createExtract(selectedStreamVar, pb.getInteger(0)), selected);
}

BixnumHashMarks::BixnumHashMarks(BuilderRef kb,
                EncodingInfo & encodingScheme,
                StreamSet * phraseLenBixnum,
                StreamSet * hashMarks,
                unsigned toUpdateHashMarks,
                StreamSet * accumHashMarks)
: PabloKernel(kb, "BixnumHashMarks"+std::to_string(toUpdateHashMarks),
            {Binding{"phraseLenBixnum", phraseLenBixnum},
             Binding{"hashMarks", hashMarks}},
            {Binding{"accumHashMarks", accumHashMarks}}), mUpdateCount(toUpdateHashMarks), mEncodingScheme(encodingScheme) { }

void BixnumHashMarks::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> phraseLenBixnum = getInputStreamSet("phraseLenBixnum");
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    std::vector<PabloAST *> hashMarksUpdated;
    unsigned maxLength = mEncodingScheme.maxSymbolLength();
    unsigned toUpdateHashMarks = maxLength - mUpdateCount + 1;
    unsigned offset = 4;
    for (unsigned i = mUpdateCount; i <= maxLength; i++) {
        PabloAST * curHashMarksBixnum = pb.createZeroes();
        curHashMarksBixnum = pb.createOr(curHashMarksBixnum, pb.createAnd(hashMarks, phraseLenBixnum[i-offset]));
        hashMarksUpdated.push_back(curHashMarksBixnum);
    }
    for (unsigned i = 0; i < toUpdateHashMarks; i++) {
        pb.createAssign(pb.createExtract(getOutputStreamVar("accumHashMarks"), pb.getInteger(i)), hashMarksUpdated[i]);
    }
}

HashGroupSelector::HashGroupSelector(BuilderRef b,
                           EncodingInfo & encodingScheme,
                           unsigned groupNo,
                           StreamSet * hashMarks,
                           StreamSet * const lengthBixNum,
                           StreamSet * overflow,
                           StreamSet * selected)
: PabloKernel(b, "HashGroupSelector" + groupNo,
              {Binding{"hashMarks", hashMarks, FixedRate(), LookAhead(1)},
                  Binding{"lengthBixNum", lengthBixNum},
                  Binding{"overflow", overflow}},
              {Binding{"selected", selected}}), mEncodingScheme(encodingScheme), mGroupNo(groupNo) { }

void HashGroupSelector::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    std::vector<PabloAST *> lengthBixNum = getInputStreamSet("lengthBixNum");
    PabloAST * overflow = getInputStreamSet("overflow")[0];
    //PabloAST* hashMarksFinal = pb.createAnd(hashMarks, pb.createNot(overflow));
    Var * groupStreamVar = getOutputStreamVar("selected");
    LengthGroupInfo groupInfo = mEncodingScheme.byLength[mGroupNo];
    // hashMarks index codes count from 0 on the 2nd byte of a symbol.
    // So the length is 2 more than the bixnum.
    unsigned offset = 2;
    unsigned lo = groupInfo.lo;
    unsigned hi = groupInfo.hi;
    std::string groupName = "lengthGroup" + std::to_string(lo) +  "_" + std::to_string(hi);
    PabloAST * groupStream = pb.createAnd3(bnc.UGE(lengthBixNum, lo - offset), bnc.ULE(lengthBixNum, hi - offset), hashMarks, groupName);
    pb.createAssign(pb.createExtract(groupStreamVar, pb.getInteger(0)), groupStream);
}

ZTF_PhraseExpansionDecoder::ZTF_PhraseExpansionDecoder(BuilderRef b,
                                           EncodingInfo & encodingScheme,
                                           StreamSet * const basis,
                                           StreamSet * insertBixNum,
                                           StreamSet * countStream)
: pablo::PabloKernel(b, "ZTF_PhraseExpansionDecoder" + encodingScheme.uniqueSuffix(),
                     {Binding{"basis", basis, FixedRate(), LookAhead(encodingScheme.maxEncodingBytes() - 1)}},
                     {Binding{"insertBixNum", insertBixNum}, Binding{"countStream", countStream}}),
    mEncodingScheme(encodingScheme)  {}

void ZTF_PhraseExpansionDecoder::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    std::vector<PabloAST *> count;
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::vector<PabloAST *> ASCII_lookaheads;
    PabloAST * ASCII_lookahead = pb.createNot(pb.createLookahead(basis[7], 1)); // for lg 0,1,2
    ASCII_lookaheads.push_back(ASCII_lookahead);
    //pb.createDebugPrint(ASCII_lookahead, "ASCII_lookahead");
    // for lg 3,4
    for (unsigned i = 2; i < mEncodingScheme.maxEncodingBytes(); i++) {
        PabloAST * ASCII_lookahead_multibyte = pb.createNot(pb.createLookahead(basis[7], pb.getInteger(i)));
        ASCII_lookaheads.push_back(ASCII_lookahead_multibyte);
        //pb.createDebugPrint(pb.createLookahead(basis[7], pb.getInteger(i)), "ASCII_lookahead_multibyte");
    }
    /*
        CODEWORDS                                                    | VALID UTF-8
        2-byte -> non-ASCII ASCII               > bit 7 -> 1 0       | 2-byte -> non-ASCII 80-BF               > bit 7 -> 1 1
        3-byte -> non-ASCII ASCII ASCII         > bit 7 -> 1 0 0     | 3-byte -> non-ASCII 80-BF 80-BF         > bit 7 -> 1 1 1
        3-byte -> non-ASCII ASCII ASCII ASCII   > bit 7 -> 1 0 0 0   | 3-byte -> non-ASCII 80-BF 80-BF 80-BF   > bit 7 -> 1 1 1 1
        10000000 - 10111111
    */

    /*
    3    |  0xC0-0xC7               (192-199) 0000 0001 0010 0011 0100 0101 0110 0111
    4    |  0xC8-0xCF               (200-208) 1000 1001 1010 1011 1100 1101 1110 1111
    5    |  0xD0, 0xD4, 0xD8, 0xDC  } - base = 0,4,8,12  0000 0100 1000 1100 // low 2 bits + (lo - encoding_bytes)
    6    |  0xD1, 0xD5, 0xD9, 0xDD  } - base = 1,5,9,13  0001 0101 1001 1101
    7    |  0xD2, 0xD6, 0xDA, 0xDE  } - base = 2,6,10,14 0010 0110 1010 1110
    8    |  0xD3, 0xD7, 0xDB, 0xDF  } - base = 3,7,11,15 0011 0111 1011 1111
    9-16 |  0xE0 - 0xEF (3-bytes)   } - lo - encoding_bytes = 9 - 3 = 6
                                        length = low 3 bits + (lo - encoding_bytes)
    17-32|  0xF0 - 0xFF (4-bytes)   } - lo - encoding_bytes = 17 - 4 = 13
                                        length = pfx-base + (lo - encoding_bytes)
    */

    BixNum insertLgth(5, pb.createZeroes());
    for (unsigned i = 0; i < mEncodingScheme.byLength.size(); i++) {
        BixNum relative(5, pb.createZeroes());
        BixNum toInsert(5, pb.createZeroes());
        LengthGroupInfo groupInfo = mEncodingScheme.byLength[i];
        unsigned lo = groupInfo.lo;
        unsigned hi = groupInfo.hi;
        unsigned base = groupInfo.prefix_base;
        unsigned next_base;
        PabloAST * inGroup = pb.createZeroes();
        if (i < 3) {
            if (i < 2) {
                next_base = base + 8;
                inGroup = pb.createOr(inGroup, pb.createAnd3(ASCII_lookaheads[0], bnc.UGE(basis, base), bnc.ULT(basis, next_base)));
                count.push_back(inGroup);
            }
            else {
                next_base = base + 16;
                #if 0
                for (unsigned c = base; c < base+4; c++) {
                    PabloAST * countStream = pb.createZeroes();
                    unsigned c1 = c;
                    for (unsigned idx = 0; idx < 4; idx++) {
                         countStream = pb.createOr(countStream, bnc.EQ(basis, c1));
                         c1 += 4;
                    }
                    count.push_back(countStream);
                }
                #endif
                inGroup = pb.createOr(inGroup, pb.createAnd3(ASCII_lookaheads[0], bnc.UGE(basis, base), bnc.ULT(basis, next_base)));
                BixNum diff = bnc.SubModular(basis, base); // SubModular range (0-7)
                for (unsigned extractIdx = 0; extractIdx < 2; extractIdx++) { // extract low 2 bits
                    relative[extractIdx] = pb.createOr(relative[extractIdx], diff[extractIdx]);
                }
            }
            //pb.createDebugPrint(inGroup, "inGroup["+std::to_string(i)+"]");
            toInsert = bnc.AddModular(relative, lo - groupInfo.encoding_bytes);
        }
        else {
            next_base = base + 16;
            inGroup = pb.createOr(inGroup, pb.createAnd3(ASCII_lookaheads[i-2], bnc.UGE(basis, base), bnc.ULT(basis, next_base)));
            //pb.createDebugPrint(inGroup, "inGroup["+std::to_string(i)+"]");
            if (i == 3) {
                #if 0
                for (unsigned c = base; c < base+8; c++) {
                    PabloAST * countStream = pb.createZeroes();
                    unsigned c1 = c;
                    for (unsigned idx = 0; idx < 2; idx++) {
                         countStream = pb.createOr(countStream, bnc.EQ(basis, c1));
                         c1 += 8;
                    }
                    count.push_back(countStream);
                }
                #endif

                BixNum diff = bnc.SubModular(basis, base); // 0,8; 1,9; 2,10; etc...
                for (unsigned extractIdx = 0; extractIdx < 3; extractIdx++) {
                    relative[extractIdx] = pb.createOr(relative[extractIdx], diff[extractIdx]);
                }
            }
            else {
                #if 0
                for (unsigned c = base; c < next_base; c++) {
                    PabloAST * countStream = bnc.EQ(basis, c);
                    count.push_back(countStream);
                }
                #endif
                BixNum diff = bnc.SubModular(basis, base); // SubModular range (0-7)
                for (unsigned extractIdx = 0; extractIdx < 4; extractIdx++) { // extract low 4 bits
                    relative[extractIdx] = pb.createOr(relative[extractIdx], diff[extractIdx]);
                }
            }
            toInsert = bnc.AddModular(relative, lo - groupInfo.encoding_bytes);
        }
        for (unsigned j = 0; j < 5; j++) {
            insertLgth[j] = pb.createSel(inGroup, toInsert[j], insertLgth[j], "insertLgth[" + std::to_string(j) + "]");
        }
    }
    Var * lengthVar = getOutputStreamVar("insertBixNum");
    for (unsigned i = 0; i < 5; i++) {
        //pb.createDebugPrint(insertLgth[i], "insertLgth["+std::to_string(i)+"]");
        pb.createAssign(pb.createExtract(lengthVar, pb.getInteger(i)), insertLgth[i]);
    }
    pb.createAssign(pb.createExtract(getOutputStreamVar("countStream"), pb.getInteger(0)), count[0]);
}
