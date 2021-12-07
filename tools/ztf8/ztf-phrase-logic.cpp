/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include "ztf-phrase-logic.h"
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pablo_intrinsic.h>
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

UpdateNextHashMarks::UpdateNextHashMarks(BuilderRef kb,
                    StreamSet * extractionMask,
                    StreamSet * hashMarksToUpdate,
                    unsigned groupNo,
                    StreamSet * hashMarksUpdated)
: PabloKernel(kb, "UpdateNextHashMarks"+std::to_string(groupNo),
            {Binding{"extractionMask", extractionMask},
             Binding{"hashMarksToUpdate", hashMarksToUpdate}},
            {Binding{"hashMarksUpdated", hashMarksUpdated}}), mGroupNo(groupNo) { }

void UpdateNextHashMarks::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * extractionMask = getInputStreamSet("extractionMask")[0]; // marks all the compressible byte positions with 0
    PabloAST * hashMarksToUpdate = getInputStreamSet("hashMarksToUpdate")[0];
    PabloAST * result = hashMarksToUpdate;

    PabloAST * compressedMarks = pb.createNot(extractionMask);
    // eliminate any (k-1)-sym phrases in the region of compressed bytes
    result = pb.createAnd(extractionMask, result);
    // eliminate any direct overlapping hashMarks between k-sym and (k-1)-sym phrases
    // every compressedMark in compressedMarks has 2/3/4 byte codewords written at the last 2/3/4 bytes of the phrase
    // Advance compressedMarks by 2,3,4 pos and check if any hashMarksToUpdate marked in the codeword position
    unsigned advPos = 0;
    if (mGroupNo < 3) {
        advPos = 2;
    }
    else if (mGroupNo == 3) {
        advPos = 3;
    }
    else { //mGroupNo = 4
        advPos = 4;
    }
    extractionMask = pb.createAdvance(extractionMask, advPos); // min codeword sequence length
    result = pb.createAnd(result, extractionMask, "result");
    //pb.createDebugPrint(result, "result");
    /*
        1111............11 => extractionMask
        111111............ => adv(extractionMask,2)
        ....111111111111.. => compressedMarks
        ...1.....1.......1 => hashMarksToUpdate
        ...1.............1 => result
        ...1.............. => codewordPositions
    */
    // figure out what hashmarks need to be eliminated and XOR with received hashmarks
    pb.createAssign(pb.createExtract(getOutputStreamVar("hashMarksUpdated"), pb.getInteger(0)), result);
}

InverseStream::InverseStream(BuilderRef kb,
                StreamSet * hashMarks,
                StreamSet * prevMarks,
                unsigned startLgIdx,
                unsigned groupNum,
                StreamSet * selected)
: PabloKernel(kb, "InverseStream_" + std::to_string(startLgIdx) + std::to_string(groupNum),
            {Binding{"hashMarks", hashMarks},
             Binding{"prevMarks", prevMarks}},
            {Binding{"selected", selected}}), mGroupNum(groupNum), mStartIdx(startLgIdx) { }

void InverseStream::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    PabloAST * prevMarks = getInputStreamSet("prevMarks")[0];
    if (mGroupNum == mStartIdx+1) {
        prevMarks = pb.createNot(prevMarks);
    }
    PabloAST * result = pb.createNot(hashMarks);
    result = pb.createOr(result, prevMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("selected"), pb.getInteger(0)), result);

}

LengthSelector::LengthSelector(BuilderRef b,
                           EncodingInfo & encodingScheme,
                           StreamSet * groupLenBixnum,
                           StreamSet * hashMarks,
                           StreamSet * selectedHashMarksPos)
: PabloKernel(b, "LengthSelector" + encodingScheme.uniqueSuffix(),
              {Binding{"hashMarks", hashMarks, FixedRate(), LookAhead(1)},
               Binding{"groupLenBixnum", groupLenBixnum}},
              {Binding{"selectedHashMarksPos", selectedHashMarksPos}}), mEncodingScheme(encodingScheme) { }

void LengthSelector::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    Var * selectedHashMarksPosStreamVar = getOutputStreamVar("selectedHashMarksPos");
    std::vector<PabloAST *> groupLenBixnum = getInputStreamSet("groupLenBixnum");
    unsigned offset = 2;
    unsigned lo = mEncodingScheme.minSymbolLength()+6; // min k-sym phrase length = 9 bytes
    unsigned hi = mEncodingScheme.maxSymbolLength();
    unsigned groupSize = hi - lo + 1;
    std::vector<PabloAST *> selectedLengthMarks(groupSize);
    for (unsigned i = lo; i <= hi; i++) {
        PabloAST * lenBixnum = bnc.EQ(groupLenBixnum, i - offset);
        pb.createAssign(pb.createExtract(selectedHashMarksPosStreamVar, pb.getInteger(i-lo)), pb.createAnd(hashMarks, lenBixnum));
    }
}


///TODO: Complexity of OverlappingLengthGroupMarker:
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
    unsigned offset = 9;
    unsigned curPhraseLen = mGroupNo;
    unsigned idx = mGroupNo-offset;
    /*
    Check for overlap with phrases of same length:
        Adv every lengthwiseHashMarks pos by curLen-1 positions (one bit at a time).
        If any of the phrases are being overlapped by preceeding phrase of same length, eliminate them.
        Do this atleast twice - if there are consecutive phrases overlapped by one another,
        select the max number of non-overlapping phrases.
    */
    /// TODO: add an assertion to check mGroupNo is valid array index
    PabloAST * curLenPos = pb.createAnd(lengthwiseHashMarks[idx], prevSelected);
    PabloAST * toAdvance = curLenPos;
    //for (unsigned loop = 0; loop < 2; loop++) { // loop depends on the max number of consecutive phrases of same length
        PabloAST * notSelected = pb.createZeroes();
        for (unsigned i = 1; i < curPhraseLen; i++) {
            toAdvance = pb.createAdvance(toAdvance, 1);
            notSelected = pb.createOr(notSelected, pb.createAnd(toAdvance, curLenPos));
        }
        PabloAST * selected = pb.createXor(curLenPos, notSelected);
        //toAdvance = selected;
    //}

    PabloAST * toEliminate = pb.createZeroes();
    for (unsigned i = idx+1; i < lengthwiseHashMarks.size(); i++) {
        PabloAST * phrasePos = pb.createAnd(lengthwiseHashMarks[i], prevSelected);
        // get all the phrases marked for compression of length i+3 Eg: 29+3 -> mGroupNo+3 = 32
        //phrasePos = pb.createAnd(phrasePos, prevSelected); // update prevSelected correctly
        PabloAST * preceededByLongerPhrase = pb.createZeroes();
        // check if any of the current selected phrases (of length Eg: 16) are preceeded by any longer length phrases (> 16)
        // already marked for compression; Eliminate such phrases from current selected group
        for (unsigned j = 1; j < curPhraseLen; j++) {
           phrasePos = pb.createAdvance(phrasePos, 1);
           preceededByLongerPhrase = pb.createOr(preceededByLongerPhrase, pb.createAnd(phrasePos, selected));
        }
        toEliminate = pb.createOr(toEliminate, preceededByLongerPhrase);
    }
    pb.createAssign(pb.createExtract(selectedStreamVar, pb.getInteger(0)), pb.createXor(selected, toEliminate));
    //pb.createAssign(pb.createExtract(getOutputStreamVar("countStream1"), pb.getInteger(0)), selected);
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
    /*
    1. eliminate any phrases of curLen in the OR(preceding region) of longer phrases.
        eliminateFinal = zeroes()
        for len in ((mGroupNo + 1), 32)
            lookaheadStream = hashMarks[len]
            toEliminate = zeroes()
            for lookaheadPos in (1, len)
                toEliminate = OR(toEliminate, AND( selected(curLen), lookahead(lookaheadStream, lookaheadPos)))
            eliminateFinal = OR(eliminateFinal, toEliminate)
        selected(curLen) = XOR(eliminateFinal, selected(curLen))
        selected(curLen) = OR(selected(curLen), selectedLongerPhrases)
    */
    unsigned offset = 9;
    unsigned phraseLenIdx = mGroupNo - offset;
    if (phraseLenIdx+1 < groupLenBixnum.size()) {
        PabloAST * eliminateFinal = pb.createZeroes();
        for (unsigned i = phraseLenIdx+1; i < groupLenBixnum.size(); i++) { // Use encoding scheme to determine the longest len
            /*
                mGroupNo = 5; max = 8
                => selectedPart1 = phrases of len = 5 where any longer length phrase does not occur in the preceeding overlapping
                region of len 5 phrases.
                => for l in range 6-8:
                    lookaheadStream = hashMarks of len l
                    lookahead l-1 positions to check if any len 5 compressible phrase preceeds phrase of len l
                    mark such len 5 phrases to not compress
                => OR with already selected hashMarks
            */
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
        //PabloAST * countStream = pb.createXor(selectedPart1, eliminateFinal);
    }
    pb.createAssign(pb.createExtract(selectedStreamVar, pb.getInteger(0)), selected);
    //pb.createAssign(pb.createExtract(getOutputStreamVar("countStream"), pb.getInteger(0)), countStream);
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

ZTF_PhraseDecodeLengths::ZTF_PhraseDecodeLengths(BuilderRef b,
                                                EncodingInfo & encodingScheme,
                                                unsigned numSym,
                                                StreamSet * basisBits,
                                                StreamSet * groupStreams,
                                                StreamSet * hashtableStreams)
: PabloKernel(b, "ZTF_PhraseDecodeLengths" + encodingScheme.uniqueSuffix(),
              {Binding{"basisBits", basisBits}},
              {Binding{"groupStreams", groupStreams},
               Binding{"hashtableStreams", hashtableStreams}}),
    mEncodingScheme(encodingScheme), mNumSym(numSym) { }

void ZTF_PhraseDecodeLengths::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    std::vector<PabloAST *> basis = getInputStreamSet("basisBits");
    std::vector<PabloAST *> groupStreams(mNumSym * mEncodingScheme.byLength.size());

    PabloAST * hashTableBoundary = pb.createAnd(pb.createAnd(basis[7], basis[6]), pb.createAnd(basis[5], basis[4]));
    hashTableBoundary = pb.createAnd(hashTableBoundary, pb.createAdvance(hashTableBoundary, 1));
    //pb.createDebugPrint(hashTableBoundary, "hashTableBoundary");
    PabloAST * fileStart = pb.createNot(pb.createAdvance(pb.createOnes(), 1));
    PabloAST * hashTableSpan = pb.createIntrinsicCall(pablo::Intrinsic::SpanUpTo, {fileStart, hashTableBoundary}); //includes first "ff" of the boundary
    PabloAST * includeBoundaryHTSpan = pb.createOr(hashTableSpan, hashTableBoundary);
    //pb.createDebugPrint(includeBoundaryHTSpan, "includeBoundaryHTSpan");
    //pb.createDebugPrint(hashTableSpan, "hashTableSpan");

    PabloAST * ASCII = bnc.ULT(basis, 0x80);
    PabloAST * suffix_80_BF = pb.createAnd(bnc.UGE(basis, 0x80), bnc.ULE(basis, 0xBF));
    Var * groupStreamVar = getOutputStreamVar("groupStreams");
    Var * hashTableStreamVar = getOutputStreamVar("hashtableStreams");
    for (unsigned i = 0; i < mEncodingScheme.byLength.size(); i++) {
        for (unsigned j = 0; j < mNumSym; j++) {
            unsigned idx = i + (j*mEncodingScheme.byLength.size());
            groupStreams[idx] = pb.createZeroes();
        }
        LengthGroupInfo groupInfo = mEncodingScheme.byLength[i];
        unsigned lo = groupInfo.lo;
        unsigned hi = groupInfo.hi;
        unsigned base = groupInfo.prefix_base;
        unsigned next_base = 0;
        if (i < 2) {
            next_base = base + 8;
        }
        else {
            next_base = base + 16;
        }
        PabloAST * inGroup = pb.createAnd(bnc.UGE(basis, base), bnc.ULT(basis, next_base));
        /* curGroupStream =>
        0 -> C0-C7 00-7F
        1 -> C8-CF 00-7F
        2 -> D0-DF 00-7F
        3 -> E0-EF 00-7F 00-7F / EO-EF 00-7F 80-BF
        4 -> F0-FF 00-7F 00-7F 00-7F / F0-FF 00-7F 00-7F 80-BF
        */
        PabloAST * curGroupStream = pb.createAnd(pb.createAdvance(inGroup, 1), ASCII); // PFX 00-7F
        groupStreams[i] = curGroupStream;
        for (unsigned j = 2; j < groupInfo.encoding_bytes; j++) {
            groupStreams[i] = pb.createAnd(pb.createAdvance(groupStreams[i], 1), ASCII);
            if (j+1 == groupInfo.encoding_bytes) {
                // PFX 00-7F{1,2} 80-BF
                curGroupStream = pb.createAnd(pb.createAdvance(curGroupStream, groupInfo.encoding_bytes-2), suffix_80_BF);
                groupStreams[i+mEncodingScheme.byLength.size()] = curGroupStream;
                //pb.createOr(groupStreams[i], curGroupStream);
            }
        }
    }
    for (unsigned i = 0; i < (mNumSym * mEncodingScheme.byLength.size()); i++) {
        if (i == 4) { // do not consider hash table boundary as codeword
            PabloAST * htSpan = pb.createOr(includeBoundaryHTSpan, pb.createAdvance(includeBoundaryHTSpan, 3));
            pb.createAssign(pb.createExtract(groupStreamVar, pb.getInteger(i)), pb.createAnd(groupStreams[i], pb.createNot(htSpan)));
            //pb.createDebugPrint(htSpan, "htSpan");
            //pb.createDebugPrint(pb.createAnd(groupStreams[i], pb.createNot(hashTableSpan)), "groupStreamVar["+std::to_string(i)+"]");
        }
        else {
            pb.createAssign(pb.createExtract(groupStreamVar, pb.getInteger(i)), pb.createAnd(groupStreams[i], pb.createNot(hashTableSpan)));

        }
        pb.createAssign(pb.createExtract(hashTableStreamVar, pb.getInteger(i)), pb.createAnd(groupStreams[i], hashTableSpan));
        //pb.createDebugPrint(pb.createAnd(groupStreams[i], hashTableSpan), "hashTableBoundary["+std::to_string(i)+"]");
    }
}


ZTF_PhraseExpansionDecoder::ZTF_PhraseExpansionDecoder(BuilderRef b,
                                           EncodingInfo & encodingScheme,
                                           StreamSet * const basis,
                                           StreamSet * insertBixNum,
                                           StreamSet * countStream)
: pablo::PabloKernel(b, "ZTF_PhraseExpansionDecoder" + encodingScheme.uniqueSuffix(),
                     {Binding{"basis", basis, FixedRate(), LookAhead(encodingScheme.maxEncodingBytes() - 1)}},
                     {Binding{"insertBixNum", insertBixNum},
                      Binding{"countStream", countStream}}),
    mEncodingScheme(encodingScheme)  {}

void ZTF_PhraseExpansionDecoder::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    //std::vector<PabloAST *> count;
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    std::vector<PabloAST *> ASCII_lookaheads;
    std::vector<PabloAST *> sfx_80_BF_lookaheads;
    PabloAST * ASCII_lookahead = pb.createNot(pb.createLookahead(basis[7], 1)); // for lg 0,1,2
    ASCII_lookaheads.push_back(ASCII_lookahead);
    //pb.createDebugPrint(ASCII_lookahead, "ASCII_lookahead");
    // for lg 3,4
    PabloAST * hashTableBoundary = pb.createAnd(pb.createAnd(basis[7], basis[6]), pb.createAnd(basis[5], basis[4]));
    //pb.createDebugPrint(hashTableBoundary, "hashTableBoundary");
    hashTableBoundary = pb.createAnd(hashTableBoundary, pb.createAdvance(hashTableBoundary, 1));
    //pb.createDebugPrint(hashTableBoundary, "hashTableBoundary");
    PabloAST * fileStart = pb.createNot(pb.createAdvance(pb.createOnes(), 1));
    PabloAST * hashTableRange = pb.createIntrinsicCall(pablo::Intrinsic::SpanUpTo, {fileStart, hashTableBoundary});
    hashTableRange = pb.createOr(hashTableRange, hashTableBoundary);

    for (unsigned i = 2; i < mEncodingScheme.maxEncodingBytes(); i++) {
        PabloAST * ASCII_lookahead_multibyte = pb.createAnd(ASCII_lookahead, pb.createNot(pb.createLookahead(basis[7], pb.getInteger(i))));
        ASCII_lookaheads.push_back(ASCII_lookahead_multibyte);
        //pb.createDebugPrint(pb.createLookahead(basis[7], pb.getInteger(i)), "ASCII_lookahead_multibyte");
    }
    for(unsigned i = 2; i < mEncodingScheme.maxEncodingBytes(); i++) {
        PabloAST * suffix_lookahead = pb.createAnd(pb.createLookahead(basis[7], pb.getInteger(i)), pb.createNot(pb.createLookahead(basis[6], pb.getInteger(i)))); // 80-BF
        suffix_lookahead = pb.createAnd(suffix_lookahead, ASCII_lookahead); // E0-EF 00-7F 80-BF; F0-FF 00-7F 00-7F 80-BF
        sfx_80_BF_lookaheads.push_back(suffix_lookahead);
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
        PabloAST * groupRange = pb.createZeroes();
        if (i < 3) {
            if (i < 2) {
                next_base = base + 8;
                groupRange = pb.createAnd(bnc.UGE(basis, base), bnc.ULT(basis, next_base));
                inGroup = pb.createOr(inGroup, pb.createAnd(ASCII_lookaheads[0], groupRange));
                //count.push_back(inGroup);
            }
            else {
                next_base = base + 16;
                groupRange = pb.createAnd(bnc.UGE(basis, base), bnc.ULT(basis, next_base));
                inGroup = pb.createOr(inGroup, pb.createAnd(ASCII_lookaheads[0], groupRange));
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
            groupRange = pb.createAnd(bnc.UGE(basis, base), bnc.ULT(basis, next_base));
            PabloAST * lookahead_accum = pb.createOr(ASCII_lookaheads[i-2], sfx_80_BF_lookaheads[i-3]);
            inGroup = pb.createOr(inGroup, pb.createAnd(lookahead_accum, groupRange));
            //pb.createDebugPrint(inGroup, "inGroup["+std::to_string(i)+"]");
            if (i == 3) {
                BixNum diff = bnc.SubModular(basis, base); // 0,8; 1,9; 2,10; etc...
                for (unsigned extractIdx = 0; extractIdx < 3; extractIdx++) {
                    relative[extractIdx] = pb.createOr(relative[extractIdx], diff[extractIdx]);
                }
            }
            else {
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
        //if(i == 0) {
        //    pb.createAssign(pb.createExtract(getOutputStreamVar("countStream"), pb.getInteger(0)), pb.createAnd(pb.createNot(hashTableRange), insertLgth[i]));
        //}
        pb.createAssign(pb.createExtract(lengthVar, pb.getInteger(i)), pb.createAnd(pb.createNot(hashTableRange), insertLgth[i]));
    }

    //pb.createDebugPrint(hashTableRange, "hashTableRange");
    pb.createAssign(pb.createExtract(getOutputStreamVar("countStream"), pb.getInteger(0)), hashTableBoundary);
}