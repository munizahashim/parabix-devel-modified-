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

    Value * markBase = b->CreateSub(hashMarkPos, b->CreateURem(hashMarkPos, sz_BITS));
    Value * markOffset = b->CreateSub(hashMarkPos, markBase);
    Value * const hashMarkBasePtr = b->CreateBitCast(b->getRawOutputPointer("hashMarks", markBase), sizeTy->getPointerTo());
    Value * initialMark = b->CreateAlignedLoad(hashMarkBasePtr, 1);
    //b->CallPrintInt("initialMark", initialMark);
    Value * updated = b->CreateXor(initialMark, b->CreateShl(sz_ONE, markOffset));
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

// use a mask stream to indicate the k-symbol sequences already compressed
// to eliminate compressing any overlapping l-symbol (l < k) sequences
SymbolGroupCompression::SymbolGroupCompression(BuilderRef b,
                                               EncodingInfo encodingScheme,
                                               unsigned numSym,
                                               StreamSet * symbolMarks,
                                               StreamSet * hashValues,
                                               StreamSet * const byteData,
                                               StreamSet * compressionMask,
                                               StreamSet * encodedBytes,
                                               unsigned strideBlocks)
: MultiBlockKernel(b, "SymbolGroupCompression" + std::to_string(numSym) /*lengthGroupSuffix(encodingScheme) + (PrefixCheckIsSet() ? "_prefix" : "")*/,
                   {Binding{"symbolMarks", symbolMarks},
                       Binding{"hashValues", hashValues},
                       Binding{"byteData", byteData, FixedRate(), LookBehind(32+1)}},
                   {}, {}, {},
                   {InternalScalar{b->getBitBlockType(), "pendingMaskInverted"}}),
mEncodingScheme(encodingScheme), mNumSym(numSym) {
    if (DelayedAttributeIsSet()) {
        mOutputStreamSets.emplace_back("compressionMask", compressionMask, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
        mOutputStreamSets.emplace_back("encodedBytes", encodedBytes, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
    } else {
        mOutputStreamSets.emplace_back("compressionMask", compressionMask, BoundedRate(0,1));
        mOutputStreamSets.emplace_back("encodedBytes", encodedBytes, BoundedRate(0,1));
        addInternalScalar(ArrayType::get(b->getInt8Ty(), 32), "pendingOutput");
    }
    Type * phraseType = ArrayType::get(b->getInt8Ty(), 32);
    mInternalScalars.emplace_back(ArrayType::get(phraseType, 333334), "hashTable");
    setStride(std::min(b->getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
}

void SymbolGroupCompression::generateMultiBlockLogic(BuilderRef b, Value * const numOfStrides) {
    ScanWordParameters sw(b, mStride);
    LengthGroupParameters lg(b, mEncodingScheme, 4); // use lg 4 for all any k-symbol sequence
    Constant * sz_DELAYED = b->getSize(mEncodingScheme.maxSymbolLength());
    Constant * sz_STRIDE = b->getSize(mStride);
    Constant * sz_BLOCKS_PER_STRIDE = b->getSize(mStride/b->getBitBlockWidth());
    Constant * sz_ZERO = b->getSize(0);
    Constant * sz_ONE = b->getSize(1);
    Constant * sz_TWO = b->getSize(2);
    Constant * sz_BITS = b->getSize(SIZE_T_BITS);
    Constant * sz_BLOCKWIDTH = b->getSize(b->getBitBlockWidth());
    Constant * sz_SYM = b->getSize(mNumSym);
    Constant * sz_TABLEMASK = b->getSize((1U << 15) -1);

    Type * sizeTy = b->getSizeTy();
    Type * bitBlockPtrTy = b->getBitBlockType()->getPointerTo();

    BasicBlock * const entryBlock = b->GetInsertBlock();
    BasicBlock * const stridePrologue = b->CreateBasicBlock("stridePrologue");
    BasicBlock * const strideMasksReady = b->CreateBasicBlock("strideMasksReady");
    BasicBlock * const keyProcessingLoop = b->CreateBasicBlock("keyProcessingLoop");
    BasicBlock * const tryStore = b->CreateBasicBlock("tryStore");
    BasicBlock * const storeKey = b->CreateBasicBlock("storeKey");
    BasicBlock * const markCompression = b->CreateBasicBlock("markCompression");
    BasicBlock * const nextKey = b->CreateBasicBlock("nextKey");
    BasicBlock * const keysDone = b->CreateBasicBlock("keysDone");
    BasicBlock * const stridesDone = b->CreateBasicBlock("stridesDone");
    BasicBlock * const updatePending = b->CreateBasicBlock("updatePending");
    BasicBlock * const compressionMaskDone = b->CreateBasicBlock("compressionMaskDone");

    Value * const initialPos = b->getProcessedItemCount("symbolMarks");
    Value * const avail = b->getAvailableItemCount("symbolMarks");
    Value * const initialProduced = b->getProducedItemCount("compressionMask");
    Value * pendingMask = b->CreateNot(b->getScalarField("pendingMaskInverted"));
    Value * producedPtr = b->CreateBitCast(b->getRawOutputPointer("compressionMask", initialProduced), bitBlockPtrTy);
    b->CreateStore(pendingMask, producedPtr);
    Value * compressMaskPtr = b->CreateBitCast(b->getRawOutputPointer("compressionMask", initialPos), bitBlockPtrTy);
    Type * phraseType = ArrayType::get(b->getInt8Ty(), 32);
    Value * hashTableBasePtr = b->CreateBitCast(b->getScalarFieldPtr("hashTable"), phraseType->getPointerTo());
    if (!DelayedAttributeIsSet()) {
        // Copy pending output data.
        Value * const initialProduced = b->getProducedItemCount("result");
        b->CreateMemCpy(b->getRawOutputPointer("encodedBytes", initialProduced), b->getScalarFieldPtr("pendingOutput"), sz_DELAYED, 1);
    }
    // Copy all new input to the output buffer; this will be then
    // overwritten when and as necessary for decompression of ZTF codes.
    Value * toCopy = b->CreateMul(numOfStrides, sz_STRIDE);
    b->CreateMemCpy(b->getRawOutputPointer("encodedBytes", initialPos), b->getRawInputPointer("byteData", initialPos), toCopy, 1);
    b->CreateBr(stridePrologue);

    b->SetInsertPoint(stridePrologue);
    // Set up the loop variables as PHI nodes at the beginning of each stride.
    PHINode * const strideNo = b->CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    Value * stridePos = b->CreateAdd(initialPos, b->CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b->CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * nextStrideNo = b->CreateAdd(strideNo, sz_ONE);
    std::vector<Value *> keyMasks = initializeCompressionMasks(b, sw, sz_BLOCKS_PER_STRIDE, 1, strideBlockOffset, compressMaskPtr, strideMasksReady);
    Value * keyMask = keyMasks[0];
    b->SetInsertPoint(strideMasksReady);
    // Iterate through key symbols and update the hash table as appropriate.
    // As symbols are encountered, the hash value is retrieved from the
    // hashValues stream.   There are then three possibilities:
    //   1.  The hashTable has no entry for this hash value.
    //       In this case, the current symbol is copied into the table.
    //   2.  The hashTable has an entry for this hash value, and
    //       that entry is equal to the current symbol.    Mark the
    //       symbol for compression.
    //   3.  The hashTable has an entry for this hash value, but
    //       that entry is not equal to the current symbol.    Skip the
    //       symbol.
    //
    Value * keyWordBasePtr = b->getInputStreamBlockPtr("symbolMarks", sz_ZERO, strideBlockOffset);
    keyWordBasePtr = b->CreateBitCast(keyWordBasePtr, sw.pointerTy);
    b->CreateUnlikelyCondBr(b->CreateICmpEQ(keyMask, sz_ZERO), keysDone, keyProcessingLoop);

    b->SetInsertPoint(keyProcessingLoop);
    PHINode * const keyMaskPhi = b->CreatePHI(sizeTy, 2);
    keyMaskPhi->addIncoming(keyMask, strideMasksReady);
    PHINode * const keyWordPhi = b->CreatePHI(sizeTy, 2);
    keyWordPhi->addIncoming(sz_ZERO, strideMasksReady);
    Value * keyWordIdx = b->CreateCountForwardZeroes(keyMaskPhi, "keyWordIdx");
    Value * nextKeyWord = b->CreateZExtOrTrunc(b->CreateLoad(b->CreateGEP(keyWordBasePtr, keyWordIdx)), sizeTy);
    Value * theKeyWord = b->CreateSelect(b->CreateICmpEQ(keyWordPhi, sz_ZERO), nextKeyWord, keyWordPhi);
    //b->CallPrintInt("keyWordIdx", keyWordIdx);
    Value * keyWordPos = b->CreateAdd(stridePos, b->CreateMul(keyWordIdx, sw.WIDTH));
    Value * keyMarkPosInWord = b->CreateCountForwardZeroes(theKeyWord);
    Value * keyMarkPos = b->CreateAdd(keyWordPos, keyMarkPosInWord, "keyEndPos");
    /* Determine the key length. */
    Value * hashValue = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("hashValues", keyMarkPos)), sizeTy);
    Value * keyLength = b->CreateAdd(b->CreateLShr(hashValue, lg.MAX_HASH_BITS), sz_TWO, "keyLength");
    Value * keyStartPos = b->CreateSub(keyMarkPos, b->CreateSub(keyLength, sz_ONE), "keyStartPos");
    // keyOffset for accessing the final half of an entry.
    Value * keyOffset = b->CreateSub(keyLength, lg.HALF_LENGTH);
    // Get the hash of this key.
    Value * keyHash = b->CreateAnd(hashValue, lg.HASH_MASK, "keyHash");

    Value * tableIdxHash = b->CreateAnd(hashValue, sz_TABLEMASK, "tableIdx");
    Value * tblEntryPtr = b->CreateGEP(hashTableBasePtr, b->CreateAdd(tableIdxHash, b->getSize(15)));
    //Value * tblEntryPtr = b->CreateGEP(hashTablePtr, b->CreateMul(keyHash, lg.HI));
    // Use two 8-byte loads to get hash and symbol values.
    //b->CallPrintInt("tblEntryPtr", tblEntryPtr);
    Value * tblPtr1 = b->CreateBitCast(tblEntryPtr, lg.halfSymPtrTy);
    Value * tblPtr2 = b->CreateBitCast(b->CreateGEP(tblEntryPtr, keyOffset), lg.halfSymPtrTy);
    Value * symPtr1 = b->CreateBitCast(b->getRawInputPointer("byteData", keyStartPos), lg.halfSymPtrTy);
    Value * symPtr2 = b->CreateBitCast(b->getRawInputPointer("byteData", b->CreateAdd(keyStartPos, keyOffset)), lg.halfSymPtrTy);
    // Check to see if the hash table entry is nonzero (already assigned).
    Value * sym1 = b->CreateAlignedLoad(symPtr1, 1);
    Value * sym2 = b->CreateAlignedLoad(symPtr2, 1);
    Value * entry1 = b->CreateMonitoredScalarFieldLoad("hashTable", tblPtr1);
    Value * entry2 = b->CreateMonitoredScalarFieldLoad("hashTable", tblPtr2);
    Value * symIsEqEntry = b->CreateAnd(b->CreateICmpEQ(entry1, sym1), b->CreateICmpEQ(entry2, sym2));
    Value * maskLength = nullptr;

    //b->CreateCondBr(symIsEqEntry, markCompression, tryStore);
    b->CreateBr(markCompression);

    b->SetInsertPoint(markCompression);
    maskLength = b->CreateZExt(b->CreateSub(keyLength, lg.ENC_BYTES, "maskLength"), sizeTy);

    // Compute a mask of bits, with zeroes marking positions to eliminate.
    // The entire symbols will be replaced, but we need to keep the required
    // number of positions for the encoded ZTF sequence.
    Value * mask = b->CreateSub(b->CreateShl(sz_ONE, maskLength), sz_ONE);
    // Determine a base position from which both the keyStart and the keyEnd
    // are accessible within SIZE_T_BITS - 8, and which will not overflow
    // the buffer.
    assert(SIZE_T_BITS - 8 > 2 * lg.groupHalfLength);
    Value * startBase = b->CreateSub(keyStartPos, b->CreateURem(keyStartPos, b->getSize(8)));
    Value * markBase = b->CreateSub(keyMarkPos, b->CreateURem(keyMarkPos, sz_BITS));
    Value * keyBase = b->CreateSelect(b->CreateICmpULT(startBase, markBase), startBase, markBase);
    Value * bitOffset = b->CreateSub(keyStartPos, keyBase);
    //b->CallPrintInt("bitOffset", bitOffset);
    mask = b->CreateShl(mask, bitOffset);
    //b->CallPrintInt("mask", mask);
    Value * const keyBasePtr = b->CreateBitCast(b->getRawOutputPointer("compressionMask", keyBase), sizeTy->getPointerTo());

    Value * initialMask = b->CreateAlignedLoad(keyBasePtr, 1);
    //b->CallPrintInt("initialMask", initialMask);
    Value * updated = b->CreateAnd(initialMask, b->CreateNot(mask));
    //b->CallPrintInt("updated", updated);
    b->CreateAlignedStore(b->CreateAnd(updated, b->CreateNot(mask)), keyBasePtr, 1);
    Value * curPos = keyMarkPos;
    Value * curHash = keyHash;  // Add hash extension bits later.
    // Write the suffixes.
    for (unsigned i = 0; i < mNumSym/*lg.groupInfo.encoding_bytes - 1*/; i++) {
        Value * ZTF_suffix = b->CreateTrunc(/*b->CreateAnd(curHash, lg.SUFFIX_MASK, "ZTF_suffix")*/sz_SYM, b->getInt8Ty());
        b->CreateStore(ZTF_suffix, b->getRawOutputPointer("encodedBytes", curPos));
        curPos = b->CreateSub(curPos, sz_ONE);
        curHash = b->CreateLShr(curHash, lg.SUFFIX_BITS);
    }
    // Now prepare the prefix - PREFIX_BASE + ... + remaining hash bits.
    //Value * lgthBase = b->CreateShl(b->CreateSub(keyLength, lg.LO), lg.PREFIX_LENGTH_OFFSET);
    //Value * ZTF_prefix = b->CreateAdd(b->CreateAdd(lg.PREFIX_BASE, lgthBase), curHash, "ZTF_prefix");
    //b->CreateStore(b->CreateTrunc(ZTF_prefix, b->getInt8Ty()), b->getRawOutputPointer("encodedBytes", curPos));

    //b->CreateBr(nextKey);
    b->CreateBr(tryStore);

    b->SetInsertPoint(tryStore);
    Value * isEmptyEntry = b->CreateICmpEQ(b->CreateOr(entry1, entry2), Constant::getNullValue(lg.halfLengthTy));
    b->CreateCondBr(isEmptyEntry, storeKey, nextKey);

    b->SetInsertPoint(storeKey);
#ifdef CHECK_COMPRESSION_DECOMPRESSION_STORE
    b->CallPrintInt("hashCode", keyHash);
    b->CallPrintInt("keyStartPos", keyStartPos);
    b->CallPrintInt("keyLength", keyLength);
#endif
    // We have a new symbol that allows future occurrences of the symbol to
    // be compressed using the hash code.
    b->CreateMonitoredScalarFieldStore("hashTable", sym1, tblPtr1);
    b->CreateMonitoredScalarFieldStore("hashTable", sym2, tblPtr2);
    b->CreateBr(nextKey);

    b->SetInsertPoint(nextKey);
    Value * dropKey = b->CreateResetLowestBit(theKeyWord);
    Value * thisWordDone = b->CreateICmpEQ(dropKey, sz_ZERO);
    // There may be more keys in the key mask.
    Value * nextKeyMask = b->CreateSelect(thisWordDone, b->CreateResetLowestBit(keyMaskPhi), keyMaskPhi);
    BasicBlock * currentBB = b->GetInsertBlock();
    keyMaskPhi->addIncoming(nextKeyMask, currentBB);
    keyWordPhi->addIncoming(dropKey, currentBB);
    b->CreateCondBr(b->CreateICmpNE(nextKeyMask, sz_ZERO), keyProcessingLoop, keysDone);

    b->SetInsertPoint(keysDone);
    strideNo->addIncoming(nextStrideNo, keysDone);
    b->CreateCondBr(b->CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);

    b->SetInsertPoint(stridesDone);
    // In the next segment, we may need to access byte data in the last
    // 16 bytes of this segment.
    if (DeferredAttributeIsSet()) {
        Value * processed = b->CreateSub(avail, lg.HI);
        b->setProcessedItemCount("byteData", processed);
    }
    // Although we have written the last block mask, we do not include it as
    // produced, because we may need to update it in the event that there is
    // a compressible symbol starting in this segment and finishing in the next.
    Value * produced = b->CreateSelect(b->isFinal(), avail, b->CreateSub(avail, sz_BLOCKWIDTH));
    b->setProducedItemCount("compressionMask", produced);
    b->CreateCondBr(b->isFinal(), compressionMaskDone, updatePending);
    b->SetInsertPoint(updatePending);
    Value * pendingPtr = b->CreateBitCast(b->getRawOutputPointer("compressionMask", produced), bitBlockPtrTy);
    //b->CallPrintInt("pendingPtr", pendingPtr);
    Value * lastMask = b->CreateBlockAlignedLoad(pendingPtr);
    b->setScalarField("pendingMaskInverted", b->CreateNot(lastMask));
    b->CreateBr(compressionMaskDone);
    b->SetInsertPoint(compressionMaskDone);
}
