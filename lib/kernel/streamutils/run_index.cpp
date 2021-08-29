/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include <kernel/streamutils/run_index.h>
#include <kernel/core/kernel_builder.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>          // for Ones
#include <pablo/pe_var.h>           // for Var
#include <pablo/pe_infile.h>

#include <pablo/bixnum/bixnum.h>

using namespace pablo;

namespace kernel {

Bindings RunIndexOutputBindings(StreamSet * runIndex, StreamSet * overflow) {
    if (overflow == nullptr) return {Binding{"runIndex", runIndex}};
    return {Binding{"runIndex", runIndex}, Binding{"overflow", overflow}};
}
    
RunIndex::RunIndex(BuilderRef b,
                   StreamSet * const runMarks, StreamSet * runIndex, StreamSet * overflow, bool invert)
    : PabloKernel(b, "RunIndex-" + std::to_string(runIndex->getNumElements()) + (overflow == nullptr ? "" : "overflow") + (invert ? "" : "_invert"),
           // input
{Binding{"runMarks", runMarks}},
           // output
RunIndexOutputBindings(runIndex, overflow)),
mIndexCount(runIndex->getNumElements()),
mOverflow(overflow != nullptr),
mInvertMask(invert) {
    assert(mIndexCount > 0);
    assert(mIndexCount <= 5);
}

void RunIndex::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    Var * runMarksVar = pb.createExtract(getInputStreamVar("runMarks"), pb.getInteger(0));
    PabloAST * runMarks = mInvertMask ? pb.createInFile(pb.createNot(runMarksVar)) : runMarksVar;
    PabloAST * runStart = pb.createAnd(runMarks, pb.createNot(pb.createAdvance(runMarks, 1)), "runStart");
    PabloAST * selectZero = runMarks;
    PabloAST * outputEnable = runMarks;
    Var * runIndexVar = getOutputStreamVar("runIndex");
    std::vector<PabloAST *> runIndex(mIndexCount);
    PabloAST * even = nullptr;
    PabloAST * overflow = nullptr;
    if (mOverflow) {
        overflow = pb.createAnd(pb.createAdvance(runMarks, 1), runMarks);
    }
    for (unsigned i = 0; i < mIndexCount; i++) {
        switch (i) {
            case 0: even = pb.createRepeat(1, pb.getInteger(0x55, 8)); break;
            case 1: even = pb.createRepeat(1, pb.getInteger(0x33, 8)); break;
            case 2: even = pb.createRepeat(1, pb.getInteger(0x0F, 8)); break;
            case 3: even = pb.createRepeat(1, pb.getInteger(0x00FF, 16)); break;
            case 4: even = pb.createRepeat(1, pb.getInteger(0x0000FFFF, 32)); break;
            case 5: even = pb.createRepeat(1, pb.getInteger(0x00000000FFFFFFFF, 64)); break;
        }
        PabloAST * odd = pb.createNot(even);
        PabloAST * evenStart = pb.createAnd(even, runStart);
        PabloAST * oddStart = pb.createAnd(odd, runStart);
        PabloAST * idx = pb.createOr(pb.createAnd(pb.createMatchStar(evenStart, runMarks), odd),
                                     pb.createAnd(pb.createMatchStar(oddStart, runMarks), even));
        idx = pb.createAnd(idx, selectZero);
        for (unsigned j = 0; j < i; j++) {
            idx = pb.createOr(idx, pb.createAdvance(idx, 1<<j));
        }
        runIndex[i] = pb.createAnd(idx, outputEnable, "runidx[" + std::to_string(i) + "]");
        pb.createAssign(pb.createExtract(runIndexVar, pb.getInteger(i)), runIndex[i]);
        if (i < mIndexCount - 1) {
            selectZero = pb.createAnd(selectZero, pb.createNot(idx), "selectZero");
            outputEnable = pb.createAnd(outputEnable, pb.createAdvance(outputEnable, 1<<i), "outputEnable");
        }
        if (mOverflow) {
            overflow = pb.createAnd(overflow, pb.createAdvance(overflow, 1<<i), "overflow");
        }
    }
    if (mOverflow) {
        pb.createAssign(pb.createExtract(getOutputStreamVar("overflow"), pb.getInteger(0)), overflow);
    }
}

Bindings AccumRunIndexOutputBindings(StreamSet * accumRunIndex, StreamSet * accumOverflow) {
    if (accumOverflow == nullptr) return {Binding{"accumRunIndex", accumOverflow}};
    return {Binding{"accumRunIndex", accumRunIndex}, Binding{"accumOverflow", accumOverflow}};
}

AccumRunIndex::AccumRunIndex(BuilderRef b,
                   StreamSet * const runMarks, StreamSet * runIndex, StreamSet * overflow, StreamSet * accumRunIndex, StreamSet * accumOverflow)
    : PabloKernel(b, "AccumRunIndex-" + std::to_string(runIndex->getNumElements()) + (overflow == nullptr ? "" : "overflow"),
           // input
{Binding{"runMarks", runMarks}, Binding{"runIndex", runIndex}, Binding{"overflow", overflow}},
           // output
AccumRunIndexOutputBindings(accumRunIndex, accumOverflow)),
mIndexCount(accumRunIndex->getNumElements()),
mOverflow(accumOverflow != nullptr) {
    assert(mIndexCount > 0);
    assert(mIndexCount <= 5);
}

void AccumRunIndex::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * runMarks = getInputStreamSet("runMarks")[0];
    PabloAST * symEndPos = pb.createNot(runMarks); // further validation needed?
    std::vector<PabloAST *> runIndex = getInputStreamSet("runIndex");
    PabloAST * overflow = pb.createExtract(getInputStreamVar("overflow"), pb.getInteger(0));
    // output
    std::vector<PabloAST *> accumRunIndex(mIndexCount);
    Var * accumRunIndexVar = getOutputStreamVar("accumRunIndex");
    BixNumCompiler bnc(pb);
    BixNum curSymLen(mIndexCount);
    BixNum prevSymLen(mIndexCount);

    for (unsigned i = 0; i < mIndexCount; i++) {
        // check ri[i] against symEndPos; set 0 if symEndPos is 0 => all the RI bixnum prior to sym end set to 0
        runIndex[i] = pb.createAnd(runIndex[i], symEndPos);
    }
    //propagte last sym's length to next sym stopping right before next sym's start pos
    for (unsigned int i = 0; i < 32; i++) { // max sym byte_length restricted to 32 (allow initializng at compile time)
        for (unsigned ii = 0; ii < mIndexCount; ii++) {
            PabloAST * priorBits = pb.createAdvance(runIndex[ii], 1);
            priorBits = pb.createAnd(priorBits, runMarks);
            runIndex[ii] = pb.createOr(runIndex[ii], priorBits);
        }
    }
    // remove 1 byte symbols like LF
    PabloAST * byteLenSym = pb.createAnd(symEndPos, pb.createAdvance(symEndPos, 1));
    // IndexedAdvance by consecutive k-1 positions to mark invalid k-gram phrase positions
    byteLenSym = pb.createIndexedAdvance(byteLenSym, symEndPos, 1);
    symEndPos = pb.createAnd(symEndPos, pb.createXor(symEndPos, pb.createAdvance(symEndPos, 1)));
    for (unsigned i = 0; i < mIndexCount; i++) {
        curSymLen[i] = pb.createAnd(runIndex[i], symEndPos);
        prevSymLen[i] = pb.createAnd(pb.createAdvance(runIndex[i], 1), symEndPos);
    }
    // first symbol has no prev symbol
    // IndexedAdvance by consecutive k-1 positions to mark invalid k-gram phrase positions
    PabloAST * notFirstSymSum = pb.createIndexedAdvance(symEndPos, symEndPos, 1);
    PabloAST * inRangeFinal = pb.createAnd(pb.createNot(byteLenSym), notFirstSymSum);
    BixNum sum = bnc.AddFull(bnc.AddFull(prevSymLen, curSymLen), 2);

    for (unsigned i = 0; i < mIndexCount; i++) {
        // if either of the operands is 0, ignore the sum in final run -> avoids 1-byte symbols like LF
        pb.createAssign(pb.createExtract(accumRunIndexVar, pb.getInteger(i)), pb.createAnd3(sum[i], symEndPos, inRangeFinal));
    }

}


}

