#include "ztf-phrase-scan.h"
#include "common.h"
#include <llvm/IR/Function.h>                      // for Function, Function...
#include <llvm/IR/Module.h>                        // for Module
#include <llvm/Support/CommandLine.h>              // for ParseCommandLineOp...
#include <llvm/Support/Debug.h>                    // for dbgs
#include <kernel/core/kernel_builder.h>
#include <kernel/core/streamset.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <boost/intrusive/detail/math.hpp>
#include <sstream>

#if 0
#define DEBUG_PRINT(title,value) b->CallPrintInt(title, value)
#else
#define DEBUG_PRINT(title,value)
#endif

#if 0
#define CHECK_COMPRESSION_DECOMPRESSION_STORE
#endif

using namespace kernel;
using namespace llvm;

using BuilderRef = Kernel::BuilderRef;

MarkRepeatedHashvalue::MarkRepeatedHashvalue(BuilderRef b,
                                    EncodingInfo encodingScheme,
                                    unsigned groupNo,
                                    StreamSet * symbolMarks,
                                    StreamSet * hashValues,
                                    StreamSet * hashMarks,
                                    unsigned strideBlocks)
: MultiBlockKernel(b, "MarkRepeatedHashvalue" + std::to_string(groupNo) + lengthGroupSuffix(encodingScheme, groupNo),
                   {Binding{"symbolMarks", symbolMarks},
                    Binding{"hashValues", hashValues}},
                   {}, {}, {}, {InternalScalar{b->getBitBlockType(), "pendingMaskInverted"},
                       InternalScalar{ArrayType::get(b->getInt8Ty(), /*hashTableSize(encodingScheme.byLength[groupNo])*/333334), "hashTable"}}),
mEncodingScheme(encodingScheme), mGroupNo(groupNo) {
    if (DelayedAttributeIsSet()) {
        mOutputStreamSets.emplace_back("hashMarks", hashMarks, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
    } else {
        mOutputStreamSets.emplace_back("hashMarks", hashMarks, BoundedRate(0,1));
        addInternalScalar(ArrayType::get(b->getInt8Ty(), encodingScheme.byLength[groupNo].hi), "pendingOutput");
    }
    setStride(std::min(b->getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
}

void MarkRepeatedHashvalue::generateMultiBlockLogic(BuilderRef b, Value * const numOfStrides) {
    ScanWordParameters sw(b, mStride);
    LengthGroupParameters lg(b, mEncodingScheme, mGroupNo);
    Constant * sz_DELAYED = b->getSize(mEncodingScheme.maxSymbolLength());
    Constant * sz_STRIDE = b->getSize(mStride);
    Constant * sz_BLOCKS_PER_STRIDE = b->getSize(mStride/b->getBitBlockWidth());
    Constant * sz_ZERO = b->getSize(0);
    Constant * sz_ONE = b->getSize(1);
    Constant * sz_TWO = b->getSize(2);
    Constant * sz_BITS = b->getSize(SIZE_T_BITS);
    Constant * sz_BLOCKWIDTH = b->getSize(b->getBitBlockWidth());
    ConstantInt * const i1_FALSE = b->getFalse();
    ConstantInt * const i1_TRUE = b->getTrue();
    Constant * sz_TABLEMASK = b->getSize((1U << 15) -1);
    Constant * INT32_1 = b->getInt32(1);

    Type * sizeTy = b->getSizeTy();
    Type * bitBlockPtrTy = b->getBitBlockType()->getPointerTo();
    Type * const boolTy = b->getInt1Ty();
    Type * countTy = b->getIntNTy(8U * 4);
    Type * countPtrTy = countTy->getPointerTo();

    BasicBlock * const entryBlock = b->GetInsertBlock();
    BasicBlock * const stridePrologue = b->CreateBasicBlock("stridePrologue");
    BasicBlock * const strideMasksReady = b->CreateBasicBlock("strideMasksReady");
    BasicBlock * const keyHashProcessingLoop = b->CreateBasicBlock("keyHashProcessingLoop");
    BasicBlock * const markHashEntry = b->CreateBasicBlock("markHashEntry");
    BasicBlock * const storeHashFlag = b->CreateBasicBlock("storeHashFlag");
    BasicBlock * const markHashRepeat = b->CreateBasicBlock("markHashRepeat");
    BasicBlock * const nextHash = b->CreateBasicBlock("nextHash");
    BasicBlock * const keyHashesDone = b->CreateBasicBlock("keyHashesDone");
    BasicBlock * const stridesDone = b->CreateBasicBlock("stridesDone");
    BasicBlock * const updatePending = b->CreateBasicBlock("updatePending");
    BasicBlock * const hashMarksDone = b->CreateBasicBlock("hashMarksDone");

    //common to all the input streams
    Value * initialPos = b->getProcessedItemCount("symbolMarks");
    Value * avail = b->getAvailableItemCount("symbolMarks");
    Value * initialProduced = b->getProducedItemCount("hashMarks");
    Value * producedPtr = b->CreateBitCast(b->getRawOutputPointer("hashMarks", initialProduced), bitBlockPtrTy);
    Value * pendingMask = b->CreateNot(b->getScalarField("pendingMaskInverted"));
    b->CreateStore(pendingMask, producedPtr);
    Value * hashMarksPtr = b->CreateBitCast(b->getRawOutputPointer("hashMarks", initialPos), bitBlockPtrTy);
    Value * hashTableBasePtr = b->CreateBitCast(b->getScalarFieldPtr("hashTable"), b->getInt8PtrTy());

    b->CreateBr(stridePrologue);
    b->SetInsertPoint(stridePrologue);
    PHINode * const strideNo = b->CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);

    Value * stridePos = b->CreateAdd(initialPos, b->CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b->CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * nextStrideNo = b->CreateAdd(strideNo, sz_ONE);

    std::vector<Value *> hashMasks;
    hashMasks = initializeCompressionMasks(b, sw, sz_BLOCKS_PER_STRIDE, 1, strideBlockOffset, hashMarksPtr, strideMasksReady);
    Value * hashMask = hashMasks[0];

    b->SetInsertPoint(strideMasksReady);
    Value * keywordBasePtr = b->getInputStreamBlockPtr("symbolMarks", sz_ZERO, strideBlockOffset);
    keywordBasePtr = b->CreateBitCast(keywordBasePtr, sw.pointerTy);

    b->CreateUnlikelyCondBr(b->CreateICmpEQ(hashMask, sz_ZERO), keyHashesDone, keyHashProcessingLoop);
    b->SetInsertPoint(keyHashProcessingLoop);

    PHINode * hashMaskPhi = b->CreatePHI(sizeTy, 2);
    hashMaskPhi->addIncoming(hashMask, strideMasksReady);
    PHINode * hashWordPhi = b->CreatePHI(sizeTy, 2);
    hashWordPhi->addIncoming(sz_ZERO, strideMasksReady);

    Value * hashWordIdx = b->CreateCountForwardZeroes(hashMaskPhi, "hashWordIdx");
    Value * nextHashWord = b->CreateZExtOrTrunc(b->CreateLoad(b->CreateGEP(keywordBasePtr, hashWordIdx)), sizeTy);
    Value * theHashWord = b->CreateSelect(b->CreateICmpEQ(hashWordPhi, sz_ZERO), nextHashWord, hashWordPhi);
    Value * hashWordPos = b->CreateAdd(stridePos, b->CreateMul(hashWordIdx, sw.WIDTH));
    Value * hashMarkPosInWord = b->CreateCountForwardZeroes(theHashWord);
    Value * hashMarkPos = b->CreateAdd(hashWordPos, hashMarkPosInWord, "keyEndPos");

    // get the hashVal bytes corresponding to the length of the keyword/phrase
    Value * hashValue = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("hashValues", hashMarkPos)), sizeTy);
    // calculate keyLength from hashValue's bixnum part
    Value * keyLength = b->CreateAdd(b->CreateLShr(hashValue, lg.MAX_HASH_BITS), sz_TWO, "keyLength");
    // get start position of the keyword/phrase

    Value * keyHash = b->CreateAnd(hashValue, lg.HASH_MASK, "keyHash");
    //b->CallPrintInt("keyHash", keyHash);
    Value * tableIdxHash = b->CreateAnd(hashValue, sz_TABLEMASK, "tableIdx");
    //b->CallPrintInt("hashValue", hashValue);
    //b->CallPrintInt("tableIdxHash", tableIdxHash);

    Value * tableEntryPtr = b->CreateInBoundsGEP(hashTableBasePtr, tableIdxHash);
    Value * tblPtr1 = b->CreateBitCast(tableEntryPtr, b->getInt8PtrTy());
    Value * entry = b->CreateMonitoredScalarFieldLoad("hashTable", tblPtr1);

    Value * entryVal = b->CreateTrunc(entry, b->getInt8Ty());
    Value * entryExists = b->CreateICmpEQ(entryVal, b->getInt8(0x1));
    b->CreateCondBr(entryExists, markHashRepeat, markHashEntry);

    b->SetInsertPoint(markHashRepeat);

    Value * const hashMarkBasePtr = b->CreateBitCast(b->getRawOutputPointer("hashMarks", hashMarkPos), sizeTy->getPointerTo());
    Value * initialMark = b->CreateAlignedLoad(hashMarkBasePtr, 1);
    //b->CallPrintInt("initialMark", initialMark);
    Value * updated = b->CreateXor(initialMark, sz_ONE);
    //b->CallPrintInt("updated", updated);
    b->CreateAlignedStore(updated, hashMarkBasePtr, 1);
    b->CreateBr(nextHash);

    b->SetInsertPoint(markHashEntry);
    Value * isEmptyEntry = b->CreateICmpEQ(entryVal, Constant::getNullValue(b->getInt8Ty()));
    b->CreateCondBr(isEmptyEntry, storeHashFlag, nextHash);
    b->SetInsertPoint(storeHashFlag);
    b->CreateMonitoredScalarFieldStore("hashTable", b->getInt8(0x1), tblPtr1);
    b->CreateBr(nextHash);

    b->SetInsertPoint(nextHash);
    Value * dropHash = b->CreateResetLowestBit(theHashWord);
    Value * thisWordDone = b->CreateICmpEQ(dropHash, sz_ZERO);
    // There may be more hashes in the key mask.
    Value * nextHashMask = b->CreateSelect(thisWordDone, b->CreateResetLowestBit(hashMaskPhi), hashMaskPhi);
    BasicBlock * currentBB = b->GetInsertBlock();
    hashMaskPhi->addIncoming(nextHashMask, currentBB);
    hashWordPhi->addIncoming(dropHash, currentBB);

    b->CreateCondBr(b->CreateICmpNE(nextHashMask, sz_ZERO), keyHashProcessingLoop, keyHashesDone);

    b->SetInsertPoint(keyHashesDone);
    strideNo->addIncoming(nextStrideNo, keyHashesDone);
    b->CreateCondBr(b->CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);

    b->SetInsertPoint(stridesDone);
    // In the next segment, we may need to access hash codes in the last
    // length group's hi bytes of this segment.
    if (DeferredAttributeIsSet()) {
        Value * processed = b->CreateSub(avail, lg.HI);
        b->setProcessedItemCount("hashValues", processed);
    }
    // Although we have written the last block mask, we do not include it as
    // produced, because we may need to update it in the event that there is
    // a compressible symbol starting in this segment and finishing in the next.
    Value * produced = b->CreateSelect(b->isFinal(), avail, b->CreateSub(avail, sz_BLOCKWIDTH));
    b->setProducedItemCount("hashMarks", produced);
    b->CreateCondBr(b->isFinal(), hashMarksDone, updatePending);
    b->SetInsertPoint(updatePending);
    Value * pendingPtr = b->CreateBitCast(b->getRawOutputPointer("hashMarks", produced), bitBlockPtrTy);
    Value * lastMask = b->CreateBlockAlignedLoad(pendingPtr);
    b->setScalarField("pendingMaskInverted", b->CreateNot(lastMask));
    b->CreateBr(hashMarksDone);
    b->SetInsertPoint(hashMarksDone);
}
