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
                           StreamSet * bixnumLenMarks)
: PabloKernel(b, "LengthSelector" + std::to_string(groupNo),
              {Binding{"hashMarks", hashMarks, FixedRate(), LookAhead(1)},
               Binding{"groupLenBixnum", groupLenBixnum}},
              {Binding{"bixnumLenMarks", bixnumLenMarks}}), mEncodingScheme(encodingScheme), mGroupNo(groupNo) { }

void LengthSelector::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    Var * bixnumLenMarksStreamVar = getOutputStreamVar("bixnumLenMarks");
    LengthGroupInfo groupInfo = mEncodingScheme.byLength[mGroupNo];
    std::vector<PabloAST *> groupLenBixnum = getInputStreamSet("groupLenBixnum");
    std::vector<PabloAST *> selectedLengthMarks;
    unsigned offset = 2;
    unsigned lo = groupInfo.lo;
    unsigned hi = groupInfo.hi;
    unsigned groupSize = hi - lo + 1;
    std::string groupName = "lengthGroup" + std::to_string(lo) +  "_" + std::to_string(hi);
    hashMarks = pb.createNot(hashMarks);
    for (unsigned i = lo; i <= hi; i++) {
        PabloAST * lenBixnum = bnc.EQ(groupLenBixnum, i - offset);
        selectedLengthMarks.push_back(lenBixnum);
        //pb.createDebugPrint(pb.createCount(pb.createInFile(lenBixnum)), "count"+std::to_string(i));
    }
    for (unsigned i = 0; i < groupSize; i++) {
        pb.createAssign(pb.createExtract(bixnumLenMarksStreamVar, pb.getInteger(i)), pb.createAnd(hashMarks, selectedLengthMarks[i]));
    }
}

OverlappingLengthGroupMarker::OverlappingLengthGroupMarker(BuilderRef b,
                           unsigned groupNo,
                           StreamSet * groupLenBixnum,
                           StreamSet * hashMarks,
                           StreamSet * prevSelected,
                           StreamSet * selected)
: PabloKernel(b, "OverlappingLengthGroupMarker" + std::to_string(groupNo),
              {Binding{"groupLenBixnum", groupLenBixnum},
               Binding{"hashMarks", hashMarks, FixedRate(), LookAhead(1)},
               Binding{"prevSelected", prevSelected}},
              {Binding{"selected", selected}}), mGroupNo(groupNo) { }

void OverlappingLengthGroupMarker::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    Var * selectedStreamVar = getOutputStreamVar("selected");
    std::vector<PabloAST *> groupLenBixnum = getInputStreamSet("groupLenBixnum");
    PabloAST * prevSelected = getInputStreamSet("prevSelected")[0];
    /// TODO: add an assertion to check mGroupNo is valid array index
    PabloAST * curLenPos = pb.createAnd(groupLenBixnum[mGroupNo], hashMarks);
    PabloAST * selected = pb.createZeroes();
    PabloAST * toAdvance = curLenPos;
    for (unsigned loop = 0; loop < 2; loop++) { // depends on the number of consecutive phrases of same length?
        PabloAST * notSelected = pb.createZeroes();
        for (unsigned i = 1; i < mGroupNo+4; i++) {
            toAdvance = pb.createAdvance(toAdvance, 1);
            notSelected = pb.createOr(notSelected, pb.createAnd(toAdvance, curLenPos));
        }
        selected = pb.createOr(selected, pb.createXor(curLenPos, notSelected));
        toAdvance = selected;
    }
    // non-overlapping phrases of length mGroupNo
    selected = toAdvance;

    PabloAST * toEliminate = pb.createZeroes();
    for (unsigned i = mGroupNo+1; i <= 28; i++) {
        PabloAST * phrasePos = groupLenBixnum[i];
        phrasePos = pb.createAnd(phrasePos, prevSelected);
        PabloAST * preceededByLongerPhrase = pb.createZeroes();
        for (unsigned j = 1; j < mGroupNo+4; j++) {
           phrasePos = pb.createAdvance(phrasePos, 1);
           preceededByLongerPhrase = pb.createOr(preceededByLongerPhrase, pb.createAnd(phrasePos, selected));
        }
        toEliminate = pb.createOr(toEliminate, preceededByLongerPhrase);
    }
    selected = pb.createXor(selected, toEliminate);
    pb.createAssign(pb.createExtract(selectedStreamVar, pb.getInteger(0)), selected);
}


OverlappingLookaheadMarker::OverlappingLookaheadMarker(BuilderRef b,
                           unsigned groupNo,
                           StreamSet * groupLenBixnum,
                           StreamSet * longerHashMarks,
                           StreamSet * prevSelected,
                           StreamSet * selected)
: PabloKernel(b, "OverlappingLookaheadMarker" + std::to_string(groupNo),
              {Binding{"groupLenBixnum", groupLenBixnum, FixedRate(), LookAhead(32)},
               Binding{"longerHashMarks", longerHashMarks},
               Binding{"prevSelected", prevSelected}},
              {Binding{"selected", selected}}), mGroupNo(groupNo) { }

void OverlappingLookaheadMarker::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * longerHashMarks = getInputStreamSet("longerHashMarks")[0];
    Var * selectedStreamVar = getOutputStreamVar("selected");
    std::vector<PabloAST *> groupLenBixnum = getInputStreamSet("groupLenBixnum");
    PabloAST * prevSelected = getInputStreamSet("prevSelected")[0];
    PabloAST * selected = prevSelected;
    if (mGroupNo < 28) {
        PabloAST * eliminateFinal = pb.createZeroes();
        for (unsigned i = mGroupNo+1; i < 29; i++) {
            PabloAST * toEliminate = pb.createZeroes();
            PabloAST * lookaheadPos = groupLenBixnum[i];
            //pb.createDebugPrint(lookaheadPos, "lookaheadPos"+std::to_string(i));
            for (unsigned j = 1; j < i+4; j++) {
                toEliminate = pb.createOr(toEliminate, pb.createAnd(prevSelected, pb.createLookahead(lookaheadPos, j)));
            }
            eliminateFinal = pb.createOr(eliminateFinal, toEliminate);
        }
        //pb.createDebugPrint(prevSelected, "prevSelected");
        //pb.createDebugPrint(eliminateFinal, "eliminateFinal");
        selected = pb.createOr(longerHashMarks, pb.createXor(prevSelected, eliminateFinal));
    }
    pb.createAssign(pb.createExtract(selectedStreamVar, pb.getInteger(0)), selected);
}

BixnumHashMarks::BixnumHashMarks(BuilderRef kb,
                StreamSet * phraseLenBixnum,
                StreamSet * hashMarks,
                StreamSet * hashMarksBixNum)
: PabloKernel(kb, "BixnumHashMarks",
            {Binding{"phraseLenBixnum", phraseLenBixnum},
             Binding{"hashMarks", hashMarks}},
            {Binding{"hashMarksBixNum", hashMarksBixNum}}) { }

void BixnumHashMarks::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> phraseLenBixnum = getInputStreamSet("phraseLenBixnum");
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    std::vector<PabloAST *> hashMarksUpdated;
    for (unsigned i = 0; i < 29; i++) {
        PabloAST * curHashMarksBixnum = phraseLenBixnum[i];
        for (unsigned j = i; j < 29; j++) {
            curHashMarksBixnum = pb.createOr(curHashMarksBixnum, pb.createAnd(hashMarks, phraseLenBixnum[j]));
        }
        hashMarksUpdated.push_back(curHashMarksBixnum);
    }
    for (unsigned i = 0; i < 29; i++) {
        pb.createAssign(pb.createExtract(getOutputStreamVar("hashMarksBixNum"), pb.getInteger(i)), hashMarksUpdated[i]);
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