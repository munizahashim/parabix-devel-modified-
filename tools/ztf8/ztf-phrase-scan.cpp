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
                                    unsigned numSyms,
                                    unsigned groupNo,
                                    StreamSet * symbolMarks,
                                    StreamSet * hashValues,
                                    StreamSet * hashMarks,
                                    unsigned strideBlocks)
: MultiBlockKernel(b, "MarkRepeatedHashvalue" + std::to_string(groupNo) + lengthGroupSuffix(encodingScheme, groupNo),
                   {Binding{"symbolMarks", symbolMarks},
                    Binding{"hashValues", hashValues}},
                   {}, {}, {}, {InternalScalar{b->getBitBlockType(), "pendingMaskInverted"},
                       InternalScalar{ArrayType::get(b->getInt8Ty(), 20000), "hashTable"}}),
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
    Constant * sz_STRIDE = b->getSize(mStride);
    Constant * sz_BLOCKS_PER_STRIDE = b->getSize(mStride/b->getBitBlockWidth());
    Constant * sz_ZERO = b->getSize(0);
    Constant * sz_ONE = b->getSize(1);
    Constant * sz_BITS = b->getSize(SIZE_T_BITS);
    Constant * sz_BLOCKWIDTH = b->getSize(b->getBitBlockWidth());
    Constant * sz_TABLEMASK = b->getSize((1U << 14) -1);

    Type * sizeTy = b->getSizeTy();
    Type * bitBlockPtrTy = b->getBitBlockType()->getPointerTo();

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
    // Value * keyLength = b->CreateAdd(b->CreateLShr(hashValue, lg.MAX_HASH_BITS), sz_TWO, "keyLength");
    // get start position of the keyword/phrase

    Value * keyHash = b->CreateAnd(hashValue, lg.HASH_MASK_NEW, "keyHash");
    //b->CallPrintInt("keyHash", keyHash);
    Value * tableIdxHash = b->CreateAnd(b->CreateLShr(keyHash, 8), sz_TABLEMASK, "tableIdx");
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

    // Mark the first symbol with any unique hashcode (used to make hash table entry) while marking repeated hashcodes
    // ==> avoids collision with another codeword
    // Even if the phrase is not repeated, it will be registered as the first phrase with codeword C
    Value * markBase1 = b->CreateSub(hashMarkPos, b->CreateURem(hashMarkPos, sz_BITS));
    Value * markOffset1 = b->CreateSub(hashMarkPos, markBase1);
    Value * const hashMarkBasePtr1 = b->CreateBitCast(b->getRawOutputPointer("hashMarks", markBase1), sizeTy->getPointerTo());
    Value * initialMark1 = b->CreateAlignedLoad(hashMarkBasePtr1, 1);
    Value * updated1 = b->CreateXor(initialMark1, b->CreateShl(sz_ONE, markOffset1));
    b->CreateAlignedStore(updated1, hashMarkBasePtr1, 1);

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

SymbolGroupCompression::SymbolGroupCompression(BuilderRef b,
                                               EncodingInfo encodingScheme,
                                               unsigned numSyms,
                                               unsigned groupNo,
                                               StreamSet * symbolMarks,
                                               StreamSet * hashValues,
                                               StreamSet * const byteData,
                                               StreamSet * compressionMask,
                                               StreamSet * encodedBytes,
                                               unsigned strideBlocks)
: MultiBlockKernel(b, "SymbolGroupCompression" + std::to_string(numSyms) + std::to_string(groupNo) + lengthGroupSuffix(encodingScheme, groupNo),
                   {Binding{"symbolMarks", symbolMarks},
                    Binding{"hashValues", hashValues},
                    Binding{"byteData", byteData, FixedRate(), LookBehind(32+1)}},
                   {}, {}, {},
                   {InternalScalar{b->getBitBlockType(), "pendingMaskInverted"},
                    InternalScalar{ArrayType::get(ArrayType::get(b->getInt8Ty(), 32), (20000 * (encodingScheme.byLength[groupNo].hi - encodingScheme.byLength[groupNo].lo + 1))), "hashTable"}}),
mEncodingScheme(encodingScheme), mGroupNo(groupNo), mNumSym(numSyms) {
    if (DelayedAttributeIsSet()) {
        mOutputStreamSets.emplace_back("compressionMask", compressionMask, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
        mOutputStreamSets.emplace_back("encodedBytes", encodedBytes, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
    } else {
        mOutputStreamSets.emplace_back("compressionMask", compressionMask, BoundedRate(0,1));
        mOutputStreamSets.emplace_back("encodedBytes", encodedBytes, BoundedRate(0,1));
        addInternalScalar(ArrayType::get(b->getInt8Ty(), 32), "pendingOutput");
    }
    Type * phraseType = ArrayType::get(b->getInt8Ty(), 32);
    //mInternalScalars.emplace_back(ArrayType::get(phraseType, (20000 * (encodingScheme.byLength[groupNo].hi - encodingScheme.byLength[groupNo].lo + 1))), "hashTable");
    setStride(std::min(b->getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
}

void SymbolGroupCompression::generateMultiBlockLogic(BuilderRef b, Value * const numOfStrides) {
    ScanWordParameters sw(b, mStride);
    LengthGroupParameters lg(b, mEncodingScheme, mGroupNo, mNumSym);
    Constant * sz_DELAYED = b->getSize(mEncodingScheme.maxSymbolLength());
    Constant * sz_STRIDE = b->getSize(mStride);
    Constant * sz_BLOCKS_PER_STRIDE = b->getSize(mStride/b->getBitBlockWidth());
    Constant * sz_ZERO = b->getSize(0);
    Constant * sz_ONE = b->getSize(1);
    Constant * sz_TWO = b->getSize(2);
    Constant * sz_BITS = b->getSize(SIZE_T_BITS);
    Constant * sz_BLOCKWIDTH = b->getSize(b->getBitBlockWidth());
    Constant * sz_TABLEMASK = b->getSize((1U << 14) -1);
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
    ///TODO: optimize if there are no hashmarks in the keyMasks stream
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
    Value * keyHash = b->CreateAnd(hashValue, lg.HASH_MASK_NEW, "keyHash");
    /*
    For 2-byte codeword, extract 32-bits of hashvalue
    hi 8 bits of both 16-bits are length part -> discarded
    Hence HASH_MASK_NEW -> mask of FFFFFFFF, FFFFFFFFFFFF, FFFFFFFFFFFFFFFF for LG {0,1,2}, 3, 4 respectively.
    */
    // Build up a single encoded value for table lookup from the hashcode sequence.
    Value * codewordVal = b->CreateAdd(lg.LAST_SUFFIX_BASE, b->CreateAnd(hashValue, lg.LAST_SUFFIX_MASK));
    Value * hashcodePos = keyMarkPos;
    for (unsigned j = 1; j < lg.groupInfo.encoding_bytes - 1/* + mNumSym*/; j++) { // same # encoding_bytes for k-sym phrases
        hashcodePos = b->CreateSub(hashcodePos, sz_ONE);
        Value * suffixByte = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("hashValues", hashcodePos)), sizeTy);
        codewordVal = b->CreateOr(b->CreateShl(codewordVal, lg.MAX_HASH_BITS), b->CreateAnd(suffixByte, lg.SUFFIX_MASK));
    }
    // add PREFIX_LENGTH_MASK bits for larger index space
    hashcodePos = b->CreateSub(hashcodePos, sz_ONE);
    Value * pfxByte = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("hashValues", hashcodePos)), sizeTy);

    Value * writtenVal = codewordVal;
    codewordVal = b->CreateOr(b->CreateShl(codewordVal, lg.MAX_HASH_BITS), b->CreateAnd(pfxByte, lg.PREFIX_LENGTH_MASK));

    pfxByte = b->CreateTrunc(b->CreateAnd(pfxByte, lg.PREFIX_LENGTH_MASK), b->getInt64Ty());

    Value * lgthBase1 = b->CreateSub(keyLength, lg.LO);
    Value * pfxOffset1 = b->CreateAdd(lg.PREFIX_BASE, lgthBase1);
    Value * multiplier1 = b->CreateMul(lg.RANGE, pfxByte);

    Value * ZTF_prefix1 = b->CreateAdd(pfxOffset1, multiplier1);

    Value * subTablePtr = b->CreateGEP(hashTableBasePtr, b->CreateMul(b->CreateSub(keyLength, lg.LO), b->getSize(20000)));
    /// TODO: experiment the number of bits in sz_TABLEMASK
    Value * tableIdxHash = b->CreateAnd(b->CreateLShr(codewordVal, 8), sz_TABLEMASK, "tableIdx");
    Value * tblEntryPtr = b->CreateInBoundsGEP(subTablePtr, tableIdxHash);

    //b->CallPrintInt("->codewordVal-1", codewordVal);
    //b->CallPrintInt("->codewordVal-2", b->CreateAnd(codewordVal, sz_TABLEMASK));

    // Use two 8-byte loads to get hash and symbol values.
    Value * tblPtr1 = b->CreateBitCast(tblEntryPtr, lg.halfSymPtrTy);
    Value * tblPtr2 = b->CreateBitCast(b->CreateGEP(tblEntryPtr, keyOffset), lg.halfSymPtrTy);
    Value * symPtr1 = b->CreateBitCast(b->getRawInputPointer("byteData", keyStartPos), lg.halfSymPtrTy);
    Value * symPtr2 = b->CreateBitCast(b->getRawInputPointer("byteData", b->CreateAdd(keyStartPos, keyOffset)), lg.halfSymPtrTy);
    // Check to see if the hash table entry is nonzero (already assigned).
    Value * sym1 = b->CreateAlignedLoad(symPtr1, 1);
    Value * sym2 = b->CreateAlignedLoad(symPtr2, 1);
    Value * entry1 = b->CreateMonitoredScalarFieldLoad("hashTable", tblPtr1);
    Value * entry2 = b->CreateMonitoredScalarFieldLoad("hashTable", tblPtr2);
/*
All the marked symMarks indicate hashMarks for only repeated phrases.
Among those marks,
1. If any symbol is being seen for the first time and has no hash table entry, store that hashcode in the hashtable
and mark its compression mask.
2. If the hashcode exists in the hashtable but the current phrase and hash table entry do not match, go to next symbol.
*/

    Value * symIsEqEntry = b->CreateAnd(b->CreateICmpEQ(entry1, sym1), b->CreateICmpEQ(entry2, sym2));
    b->CreateCondBr(symIsEqEntry, markCompression, tryStore);

    b->SetInsertPoint(tryStore);

    Value * isEmptyEntry = b->CreateICmpEQ(b->CreateOr(entry1, entry2), Constant::getNullValue(lg.halfLengthTy));
    b->CreateCondBr(isEmptyEntry, storeKey, nextKey);

    b->SetInsertPoint(storeKey);
//#if 0
    b->CreateWriteCall(b->getInt32(STDERR_FILENO), symPtr1, keyLength);

    // write writtenVal into a buffer and then write into STDOUT
    writtenVal = b->CreateOr(b->CreateShl(writtenVal, lg.MAX_HASH_BITS), ZTF_prefix1, "writtenVal");
    Value * const copyLen = b->CreateAdd(lg.ENC_BYTES, sz_ZERO);
    Value * outputCodeword = b->CreateAlloca(b->getInt8Ty(), copyLen);
    b->CreateAlignedStore(writtenVal, outputCodeword, 1);
    //b->CallPrintInt("outputCodeword", outputCodeword);
    b->CreateWriteCall(b->getInt32(STDERR_FILENO), outputCodeword, copyLen);
//#endif
    // We have a new symbol that allows future occurrences of the symbol to
    // be compressed using the hash code.
    b->CreateMonitoredScalarFieldStore("hashTable", sym1, tblPtr1);
    b->CreateMonitoredScalarFieldStore("hashTable", sym2, tblPtr2);

    // markCompression even for the first occurrence
    //b->CreateBr(nextKey);
    b->CreateBr(markCompression);
    b->SetInsertPoint(markCompression);

    Value * maskLength = b->CreateZExt(b->CreateSub(keyLength, lg.ENC_BYTES, "maskLength"), sizeTy);
    //b->CallPrintInt("maskLength", maskLength);
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
    Value * const keyBasePtr = b->CreateBitCast(b->getRawOutputPointer("compressionMask", keyBase), sizeTy->getPointerTo());
    Value * initialMask = b->CreateAlignedLoad(keyBasePtr, 1);
    Value * updated = b->CreateAnd(initialMask, b->CreateNot(mask));
    //b->CallPrintInt("updated", updated);
    b->CreateAlignedStore(updated, keyBasePtr, 1);
    Value * curPos = keyMarkPos;
    Value * curHash = keyHash;
    // Write the suffixes.

    ///TODO: codeword version 2.0 - write extra suffix byte
    Value * last_suffix = b->CreateTrunc(b->CreateAdd(lg.LAST_SUFFIX_BASE, b->CreateAnd(curHash, lg.LAST_SUFFIX_MASK, "ZTF_suffix_last")), b->getInt8Ty());
    b->CreateStore(last_suffix, b->getRawOutputPointer("encodedBytes", curPos));
    curPos = b->CreateSub(curPos, sz_ONE);
#if 0
    Value * writtenVal = b->CreateZExt(b->CreateAdd(lg.LAST_SUFFIX_BASE, b->CreateAnd(curHash, lg.LAST_SUFFIX_MASK, "ZTF_suffix_last")), sizeTy);
#endif

    for (unsigned i = 0; i < lg.groupInfo.encoding_bytes - 2; i++) {
        Value * ZTF_suffix = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("hashValues", curPos)), sizeTy);
#if 0
        writtenVal = b->CreateOr(b->CreateShl(writtenVal, lg.MAX_HASH_BITS), b->CreateAnd(ZTF_suffix, lg.SUFFIX_MASK));
#endif
        b->CreateStore(b->CreateTrunc(b->CreateAnd(ZTF_suffix, lg.SUFFIX_MASK), b->getInt8Ty()), b->getRawOutputPointer("encodedBytes", curPos));
        curPos = b->CreateSub(curPos, sz_ONE);
    }
    // Now prepare the prefix - PREFIX_BASE + ... + remaining hash bits.
    /*
            3    |  0xC0-0xC7
            4    |  0xC8-0xCF
            5    |  0xD0, 0xD4, 0xD8, 0xDC
            6    |  0xD1, 0xD5, 0xD9, 0xDD
            7    |  0xD2, 0xD6, 0xDA, 0xDE
            8    |  0xD3, 0xD7, 0xDB, 0xDF
            9-16 |  0xE0 - 0xEF (3-bytes)
            17-32|  0xF0 - 0xFF (4-bytes)

                (length - lo) = row of the prefix table
            LG    RANGE         xHashBits      hashMask     numRows
            0      0               3              111          8
            1      0               3              111          8
            2      0-3             2               11          4
            3      0-7             1                1          8
            4      0-15            0                0         16
            (PFX_BASE + RANGE) + (numRows * (keyHash AND hashMask))
    */
    Value * pfxLgthMask = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("hashValues", curPos)), sizeTy);
    pfxLgthMask = b->CreateTrunc(b->CreateAnd(pfxLgthMask, lg.PREFIX_LENGTH_MASK), b->getInt64Ty());

    Value * lgthBase = b->CreateSub(keyLength, lg.LO);
    Value * pfxOffset = b->CreateAdd(lg.PREFIX_BASE, lgthBase);
    Value * multiplier = b->CreateMul(lg.RANGE, pfxLgthMask);

    Value * ZTF_prefix = b->CreateAdd(pfxOffset, multiplier, "ZTF_prefix");
    //b->CallPrintInt("prefix", b->CreateTrunc(pfxOffset, b->getInt8Ty()));
    //b->CallPrintInt("ZTF_prefix", ZTF_prefix);

    b->CreateStore(b->CreateTrunc(ZTF_prefix, b->getInt8Ty()), b->getRawOutputPointer("encodedBytes", curPos));

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
    Value * lastMask = b->CreateBlockAlignedLoad(pendingPtr);
    b->setScalarField("pendingMaskInverted", b->CreateNot(lastMask));
    b->CreateBr(compressionMaskDone);
    b->SetInsertPoint(compressionMaskDone);
}


SymbolGroupDecompression::SymbolGroupDecompression(BuilderRef b,
                                                   EncodingInfo encodingScheme,
                                                   unsigned numSym,
                                                   unsigned groupNo,
                                                   StreamSet * const codeWordMarks,
                                                   StreamSet * const hashMarks, StreamSet * const byteData,
                                                   StreamSet * const result, unsigned strideBlocks)
: MultiBlockKernel(b, "SymbolGroupDecompression" + lengthGroupSuffix(encodingScheme, groupNo),
                   {Binding{"keyMarks0", codeWordMarks},
                       Binding{"hashMarks0", hashMarks},
                       Binding{"byteData", byteData, BoundedRate(0,1), LookBehind(32+1)}
                   },
                   {}, {}, {},
                   {// Hash table 8 length-based tables with 256 16-byte entries each.
                    InternalScalar{ArrayType::get(ArrayType::get(b->getInt8Ty(), 32), (20000 * (encodingScheme.byLength[groupNo].hi - encodingScheme.byLength[groupNo].lo + 1))), "hashTable"}}),
    mEncodingScheme(encodingScheme), mGroupNo(groupNo), mNumSym(numSym) {
    setStride(std::min(b->getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
    if (DelayedAttributeIsSet()) {
        mOutputStreamSets.emplace_back("result", result, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
    } else {
        mOutputStreamSets.emplace_back("result", result, BoundedRate(0,1));
    }
    //Type * phraseType = ArrayType::get(b->getInt8Ty(), 32);
   // addInternalScalar(ArrayType::get(phraseType, (20000 * (encodingScheme.byLength[groupNo].hi - encodingScheme.byLength[groupNo].lo + 1))), "hashTable");
}

void SymbolGroupDecompression::generateMultiBlockLogic(BuilderRef b, Value * const numOfStrides) {

    ScanWordParameters sw(b, mStride);
    LengthGroupParameters lg(b, mEncodingScheme, mGroupNo);
    Constant * sz_STRIDE = b->getSize(mStride);
    Constant * sz_BLOCKS_PER_STRIDE = b->getSize(mStride/b->getBitBlockWidth());
    Constant * sz_ZERO = b->getSize(0);
    Constant * sz_ONE = b->getSize(1);
    Constant * sz_TWO = b->getSize(2);
    Constant * sz_TABLEMASK = b->getSize((1U << 14) -1);
    Type * sizeTy = b->getSizeTy();

    BasicBlock * const entryBlock = b->GetInsertBlock();
    BasicBlock * const stridePrologue = b->CreateBasicBlock("stridePrologue");
    BasicBlock * const strideMasksReady = b->CreateBasicBlock("strideMasksReady");
    BasicBlock * const keyProcessingLoop = b->CreateBasicBlock("keyProcessingLoop");
    BasicBlock * const storeKey = b->CreateBasicBlock("storeKey");
    BasicBlock * const nextKey = b->CreateBasicBlock("nextKey");
    BasicBlock * const keysDone = b->CreateBasicBlock("keysDone");
    BasicBlock * const hashProcessingLoop = b->CreateBasicBlock("hashProcessingLoop");
    BasicBlock * const lookupSym = b->CreateBasicBlock("lookupSym");
    BasicBlock * const nextHash = b->CreateBasicBlock("nextHash");
    BasicBlock * const hashesDone = b->CreateBasicBlock("hashesDone");
    BasicBlock * const stridesDone = b->CreateBasicBlock("stridesDone");

    Value * const initialPos = b->getProcessedItemCount("keyMarks0");
    Value * const avail = b->getAvailableItemCount("keyMarks0");

    // Copy all new input to the output buffer; this will be then
    // overwritten when and as necessary for decompression of ZTF codes.
    ///TODO: only copy the decompressed data, not the hashtable from the compressed data
    Value * toCopy = b->CreateMul(numOfStrides, sz_STRIDE);
    b->CreateMemCpy(b->getRawOutputPointer("result", initialPos), b->getRawInputPointer("byteData", initialPos), toCopy, 1);

    Type * phraseType = ArrayType::get(b->getInt8Ty(), 32);
    Value * hashTableBasePtr = b->CreateBitCast(b->getScalarFieldPtr("hashTable"), phraseType->getPointerTo());
    b->CreateBr(stridePrologue);

    b->SetInsertPoint(stridePrologue);
    // Set up the loop variables as PHI nodes at the beginning of each stride.
    PHINode * const strideNo = b->CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    Value * stridePos = b->CreateAdd(initialPos, b->CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b->CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * nextStrideNo = b->CreateAdd(strideNo, sz_ONE);

    std::vector<Value *> keyMasks(1);
    std::vector<Value *> hashMasks(1);
    initializeDecompressionMasks(b, sw, sz_BLOCKS_PER_STRIDE, 1, strideBlockOffset, keyMasks, hashMasks, strideMasksReady);
    Value * keyMask = keyMasks[0];
    Value * hashMask = hashMasks[0];

    b->SetInsertPoint(strideMasksReady);

    Value * keyWordBasePtr = b->getInputStreamBlockPtr("keyMarks0", sz_ZERO, strideBlockOffset);
    keyWordBasePtr = b->CreateBitCast(keyWordBasePtr, sw.pointerTy);
    DEBUG_PRINT("keyMask", keyMask);
    b->CreateUnlikelyCondBr(b->CreateICmpEQ(keyMask, sz_ZERO), keysDone, keyProcessingLoop);

    b->SetInsertPoint(keyProcessingLoop);
    PHINode * const keyMaskPhi = b->CreatePHI(sizeTy, 2);
    keyMaskPhi->addIncoming(keyMask, strideMasksReady);
    PHINode * const keyWordPhi = b->CreatePHI(sizeTy, 2);
    keyWordPhi->addIncoming(sz_ZERO, strideMasksReady);
    Value * keyWordIdx = b->CreateCountForwardZeroes(keyMaskPhi, "keyWordIdx");
    Value * nextKeyWord = b->CreateZExtOrTrunc(b->CreateLoad(b->CreateGEP(keyWordBasePtr, keyWordIdx)), sizeTy);
    Value * theKeyWord = b->CreateSelect(b->CreateICmpEQ(keyWordPhi, sz_ZERO), nextKeyWord, keyWordPhi);
    Value * keyWordPos = b->CreateAdd(stridePos, b->CreateMul(keyWordIdx, sw.WIDTH));
    Value * keyMarkPosInWord = b->CreateCountForwardZeroes(theKeyWord);
    Value * keyMarkPos = b->CreateAdd(keyWordPos, keyMarkPosInWord, "keyEndPos");
    DEBUG_PRINT("keyMarkPos", keyMarkPos);
    /* Determine the key length. */
    // determine keyLength from the codeword prefix
    Value * pfxPos = b->CreateSub(keyMarkPos, lg.MAX_INDEX);
    Value * const thePfx = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("byteData", pfxPos)), sizeTy);
    Value * theGroupLen = b->CreateSub(thePfx, lg.PREFIX_BASE);
    Value * keyLength = b->CreateAdd(b->CreateAnd(theGroupLen, lg.PHRASE_EXTENSION_MASK), lg.LO, "keyLength");
    Value * keyStartPos = b->CreateSub(pfxPos, keyLength, "keyStartPos");
    DEBUG_PRINT("keyLength", keyLength);

    // fetch the phrase and corresponding codeword
    // calculate the hashtable index and store the phrase
    // step over to the next phrase of same length

    // keyOffset for accessing the final half of an entry.
    Value * keyOffset = b->CreateSub(keyLength, lg.HALF_LENGTH);
    DEBUG_PRINT("keyOffset", keyOffset);
    // Build up a single encoded value for table lookup from the hashcode sequence.
    Value * hashcodePos = keyMarkPos;
    Value * codewordVal = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("byteData", hashcodePos)), sizeTy);
    for (unsigned j = 1; j < lg.groupInfo.encoding_bytes - 1; j++) {
        hashcodePos = b->CreateSub(hashcodePos, sz_ONE);
        Value * sfxByte = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("byteData", hashcodePos)), sizeTy);
        codewordVal = b->CreateOr(b->CreateShl(codewordVal, lg.MAX_HASH_BITS), b->CreateAnd(sfxByte, lg.SUFFIX_MASK));
    }
    Value * keylen_range = b->CreateSub(keyLength, lg.LO);
    Value * thePfxOffset = b->CreateAdd(lg.PREFIX_BASE, keylen_range);
    Value * theMultiplier = b->CreateSub(thePfx, thePfxOffset);
    Value * thePfxHashBits = b->CreateUDiv(theMultiplier, lg.RANGE);
    /// CHECK: Assertion for CreateUDiv(multiplier, lg.RANGE)
    codewordVal = b->CreateOr(b->CreateShl(codewordVal, lg.MAX_HASH_BITS), b->CreateAnd(thePfxHashBits, lg.PREFIX_LENGTH_MASK));
#if 0
    b->CreateWriteCall(b->getInt32(STDERR_FILENO), b->getRawInputPointer("byteData", keyStartPos), keyLength);
    Value * hashLen = b->CreateAdd(lg.ENC_BYTES, sz_ZERO);
    Value * hashStart = b->CreateSub(keyMarkPos, hashLen);
    b->CreateWriteCall(b->getInt32(STDERR_FILENO), b->getRawInputPointer("byteData", hashStart), hashLen);
    Value * codewordVal_debug = codewordVal;
#endif

#if 0
    b->CallPrintInt("decmp-tableIdxHash", b->CreateAnd(codewordVal, sz_TABLEMASK));
#endif

    Value * subTablePtr = b->CreateGEP(hashTableBasePtr, b->CreateMul(b->CreateSub(keyLength, lg.LO), b->getSize(20000)));
    Value * tableIdxHash = b->CreateAnd(b->CreateLShr(codewordVal, 8), sz_TABLEMASK, "tableIdx");
    Value * tblEntryPtr = b->CreateInBoundsGEP(subTablePtr, tableIdxHash);

    // Use two halfSymLen loads to get hash and symbol values.
    Value * tblPtr1 = b->CreateBitCast(tblEntryPtr, lg.halfSymPtrTy);
    Value * tblPtr2 = b->CreateBitCast(b->CreateGEP(tblEntryPtr, keyOffset), lg.halfSymPtrTy);
    Value * symPtr1 = b->CreateBitCast(b->getRawInputPointer("byteData", keyStartPos), lg.halfSymPtrTy);
    Value * symPtr2 = b->CreateBitCast(b->getRawInputPointer("byteData", b->CreateAdd(keyStartPos, keyOffset)), lg.halfSymPtrTy);

    // Check to see if the hash table entry is nonzero (already assigned).
    Value * sym1 = b->CreateLoad(symPtr1);
    Value * sym2 = b->CreateLoad(symPtr2);
    Value * entry1 = b->CreateMonitoredScalarFieldLoad("hashTable", tblPtr1);
    Value * entry2 = b->CreateMonitoredScalarFieldLoad("hashTable", tblPtr2);
    // hash collisions may exists between k-symbol phrases, just replace the collisions with latest phrase
    // as we would have already replaced (k+1)-symbol phrase already mapped to the same hashtable index
    Value * isEmptyEntry = b->CreateIsNull(b->CreateOr(entry1, entry2));
    b->CreateCondBr(isEmptyEntry, storeKey, nextKey);
    b->SetInsertPoint(storeKey);
    // We have a new symbols that allows future occurrences of the symbol to
    // be compressed using the hash code.

#if 0
    Value * pfxLgthMask = pfxByteBits;
    pfxLgthMask = b->CreateTrunc(b->CreateAnd(pfxLgthMask, lg.PREFIX_LENGTH_MASK), b->getInt64Ty());
    Value * lgthBase = b->CreateSub(keyLength, lg.LO);
    Value * pfxOffset1 = b->CreateAdd(lg.PREFIX_BASE, lgthBase);
    Value * multiplier1 = b->CreateMul(lg.RANGE, pfxLgthMask);
    Value * ZTF_prefix = b->CreateAdd(pfxOffset1, multiplier1, "ZTF_prefix");
    //b->CreateWriteCall(b->getInt32(STDERR_FILENO), symPtr1, keyLength);
    //b->CallPrintInt("hashCode", b->CreateOr(b->CreateShl(codewordVal_debug, lg.MAX_HASH_BITS), ZTF_prefix));
#endif
#ifdef CHECK_COMPRESSION_DECOMPRESSION_STORE
    b->CallPrintInt("hashCode", keyHash);
    b->CallPrintInt("keyStartPos", keyStartPos);
    b->CallPrintInt("keyLength", keyLength);
#endif
    b->CreateMonitoredScalarFieldStore("hashTable", sym1, tblPtr1);
    b->CreateMonitoredScalarFieldStore("hashTable", sym2, tblPtr2);

    b->CreateBr(nextKey);

    b->SetInsertPoint(nextKey);
#if 0
    b->CreateWriteCall(b->getInt32(STDERR_FILENO), tblPtr1, b->CreateSub(keyLength, keyOffset));
    b->CreateWriteCall(b->getInt32(STDERR_FILENO), tblPtr2, keyOffset);
#endif
    Value * dropKey = b->CreateResetLowestBit(theKeyWord);
    Value * thisWordDone = b->CreateICmpEQ(dropKey, sz_ZERO);
    // There may be more keys in the key mask.
    Value * nextKeyMask = b->CreateSelect(thisWordDone, b->CreateResetLowestBit(keyMaskPhi), keyMaskPhi);
    BasicBlock * currentBB = b->GetInsertBlock();
    keyMaskPhi->addIncoming(nextKeyMask, currentBB);
    keyWordPhi->addIncoming(dropKey, currentBB);
    b->CreateCondBr(b->CreateICmpNE(nextKeyMask, sz_ZERO), keyProcessingLoop, keysDone);

    b->SetInsertPoint(keysDone);
    // replace codewords by decompressed phrases
    Value * hashWordBasePtr = b->getInputStreamBlockPtr("hashMarks0", sz_ZERO, strideBlockOffset);
    hashWordBasePtr = b->CreateBitCast(hashWordBasePtr, sw.pointerTy);
    b->CreateUnlikelyCondBr(b->CreateICmpEQ(hashMask, sz_ZERO), hashesDone, hashProcessingLoop);

    b->SetInsertPoint(hashProcessingLoop);
    PHINode * const hashMaskPhi = b->CreatePHI(sizeTy, 2);
    hashMaskPhi->addIncoming(hashMask, keysDone);
    PHINode * const hashWordPhi = b->CreatePHI(sizeTy, 2);
    hashWordPhi->addIncoming(sz_ZERO, keysDone);
    Value * hashWordIdx = b->CreateCountForwardZeroes(hashMaskPhi, "hashWordIdx");
    Value * nextHashWord = b->CreateZExtOrTrunc(b->CreateLoad(b->CreateGEP(hashWordBasePtr, hashWordIdx)), sizeTy);
    Value * theHashWord = b->CreateSelect(b->CreateICmpEQ(hashWordPhi, sz_ZERO), nextHashWord, hashWordPhi);
    Value * hashWordPos = b->CreateAdd(stridePos, b->CreateMul(hashWordIdx, sw.WIDTH));
    Value * hashPosInWord = b->CreateCountForwardZeroes(theHashWord);
    Value * hashMarkPos = b->CreateAdd(hashWordPos, hashPosInWord, "hashMarkPos");
    DEBUG_PRINT("hashMarkPos", hashMarkPos);
    Value * hashPfxPos = b->CreateSub(hashMarkPos, lg.MAX_INDEX);
    Value * const hashPfx = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("byteData", hashPfxPos)), sizeTy);
    DEBUG_PRINT("hashPfx", hashPfx);
    // Build up a single encoded value from the ZTF code sequence.
    Value * pfxGroupLen = b->CreateSub(hashPfx, lg.PREFIX_BASE, "encodedVal");
    /*
    pfxGroupLen                = 0-7, 0-7, 0-FF, 0-FF, 0-FF
    bits to calculate len      = 0,   0,   2,    3,    4
    */
    Value * curPos = hashMarkPos;
    Value * encodedVal = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("byteData", curPos)), sizeTy);
#if 0
    Value * encodedVal_debug = encodedVal;
#endif
    //b->CallPrintInt("lastSuffixByte", encodedVal);
    //encodedVal = b->CreateSub(encodedVal, lg.LAST_SUFFIX_BASE); //-> subtracting leads to incorrect tableIdx?
    for (unsigned i = 1; i < lg.groupInfo.encoding_bytes-1; i++) {
        curPos = b->CreateSub(curPos, sz_ONE);
        Value * suffixByte = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("byteData", curPos)), sizeTy);
        //b->CallPrintInt("suffixByte"+std::to_string(i), suffixByte);
        encodedVal = b->CreateOr(b->CreateShl(encodedVal, lg.MAX_HASH_BITS), suffixByte, "encodedVal");
#if 0
        encodedVal_debug = b->CreateOr(b->CreateShl(encodedVal_debug, lg.MAX_HASH_BITS), suffixByte);
#endif
    }
    /// WIP: add the logic to extract LAST_SUFFIX_MASK bits for k-symbol phrases
    Value * symLength = b->CreateAdd(b->CreateAnd(pfxGroupLen, lg.PHRASE_EXTENSION_MASK), lg.LO, "symLength");
    /*
    extract PREFIX_LENGTH_MASK bits from prefix -> required for dict index lookup
    * get the length_range from key length
    * key_len - lg.lo = length_range
    * PREFIX_BASE + length_range = pfxOffset
    * PFX - pfxOffset = multiplier
    * multiplier/lg.RANGE = PREFIX_LENGTH_MASK bits
    */
    Value * len_range = b->CreateSub(symLength, lg.LO);
    Value * pfxOffset = b->CreateAdd(lg.PREFIX_BASE, len_range);
    Value * multiplier = b->CreateSub(hashPfx, pfxOffset);
    Value * pfxHashBits = b->CreateUDiv(multiplier, lg.RANGE);
    /// CHECK: Assertion for CreateUDiv(multiplier, lg.RANGE)
    encodedVal = b->CreateOr(b->CreateShl(encodedVal, lg.MAX_HASH_BITS), b->CreateAnd(pfxHashBits, lg.PREFIX_LENGTH_MASK));
#if 0
    encodedVal_debug = b->CreateOr(b->CreateShl(encodedVal_debug, lg.MAX_HASH_BITS), hashPfx);
#endif
    Value * validLength = b->CreateAnd(b->CreateICmpUGE(symLength, lg.LO), b->CreateICmpULE(symLength, lg.HI));
    DEBUG_PRINT("symLength", symLength);
    b->CreateCondBr(validLength, lookupSym, nextHash);
    b->SetInsertPoint(lookupSym);
#if 0
    b->CallPrintInt("DhashVal-lookup", b->CreateAnd(encodedVal, sz_TABLEMASK));
#endif
    Value * symStartPos = b->CreateSub(hashMarkPos, b->CreateSub(symLength, sz_ONE), "symStartPos");
    Value * symOffset = b->CreateSub(symLength, lg.HALF_LENGTH);

    subTablePtr = b->CreateGEP(hashTableBasePtr, b->CreateMul(b->CreateSub(symLength, lg.LO), b->getSize(20000)/*lg.SUBTABLE_SIZE)*/));
    tableIdxHash = b->CreateAnd(b->CreateLShr(encodedVal, 8), sz_TABLEMASK);
    tblEntryPtr = b->CreateGEP(subTablePtr, tableIdxHash);
    // Use two halfSymLen loads to get hash and symbol values.
    tblPtr1 = b->CreateBitCast(tblEntryPtr, lg.halfSymPtrTy);
    tblPtr2 = b->CreateBitCast(b->CreateGEP(tblEntryPtr, symOffset), lg.halfSymPtrTy);
    entry1 = b->CreateAlignedLoad(tblPtr1, 1);
    entry2 = b->CreateAlignedLoad(tblPtr2, 1);
#if 0
    b->CreateWriteCall(b->getInt32(STDERR_FILENO), tblPtr1, b->CreateSub(symLength, symOffset));
    b->CallPrintInt("codewordRead", encodedVal_debug);
#endif
    symPtr1 = b->CreateBitCast(b->getRawOutputPointer("result", symStartPos), lg.halfSymPtrTy);
    symPtr2 = b->CreateBitCast(b->getRawOutputPointer("result", b->CreateAdd(symStartPos, symOffset)), lg.halfSymPtrTy);
    b->CreateAlignedStore(entry1, symPtr1, 1);
    b->CreateAlignedStore(entry2, symPtr2, 1);
    b->CreateBr(nextHash);
    b->SetInsertPoint(nextHash);
    Value * dropHash = b->CreateResetLowestBit(theHashWord);
    Value * hashMaskDone = b->CreateICmpEQ(dropHash, sz_ZERO);
    // There may be more hashes in the hash mask.
    Value * nextHashMask = b->CreateSelect(hashMaskDone, b->CreateResetLowestBit(hashMaskPhi), hashMaskPhi);
    BasicBlock * hashBB = b->GetInsertBlock();
    hashMaskPhi->addIncoming(nextHashMask, hashBB);
    hashWordPhi->addIncoming(dropHash, hashBB);
    b->CreateCondBr(b->CreateICmpNE(nextHashMask, sz_ZERO), hashProcessingLoop, hashesDone);

    b->SetInsertPoint(hashesDone);
    strideNo->addIncoming(nextStrideNo, hashesDone);
    b->CreateCondBr(b->CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);

    b->SetInsertPoint(stridesDone);
    // If the segment ends in the middle of a 2-byte codeword, we need to
    // make sure that we still have access to the codeword in the next block.
    if (DeferredAttributeIsSet()) {
        Value * processed = b->CreateSub(avail, lg.HI);
        b->setProcessedItemCount("byteData", processed);
    }
    //CHECK: Although we have written the full input stream to output, there may
    // be an incomplete symbol at the end of this block.   Store the
    // data that may be overwritten as pending and set the produced item
    // count to that which is guaranteed to be correct.
}
