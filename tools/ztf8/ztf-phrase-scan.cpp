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
                       InternalScalar{ArrayType::get(b->getInt8Ty(), phraseHashTableSize(encodingScheme.byLength[groupNo])), "hashTable"}}),
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
    Constant * sz_TWO = b->getSize(2);
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
                                               StreamSet * codewordMask,
                                               StreamSet * dictBoundaryMask,
                                               unsigned strideBlocks)
: MultiBlockKernel(b, "SymbolGroupCompression" + std::to_string(numSyms) + std::to_string(groupNo) + lengthGroupSuffix(encodingScheme, groupNo),
                   {Binding{"symbolMarks", symbolMarks},
                    Binding{"hashValues", hashValues},
                    Binding{"byteData", byteData, FixedRate(), LookBehind(encodingScheme.byLength[groupNo].hi+1)}},
                   {}, {}, {},
                   {InternalScalar{b->getBitBlockType(), "pendingMaskInverted"},
                    InternalScalar{b->getBitBlockType(), "pendingPhraseMask"},
                    InternalScalar{ArrayType::get(ArrayType::get(b->getInt8Ty(), encodingScheme.byLength[groupNo].hi), phraseHashTableSize(encodingScheme.byLength[groupNo])), "hashTable"}}),
mEncodingScheme(encodingScheme), mGroupNo(groupNo), mNumSym(numSyms) {
    if (DelayedAttributeIsSet()) {
        mOutputStreamSets.emplace_back("compressionMask", compressionMask, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
        mOutputStreamSets.emplace_back("encodedBytes", encodedBytes, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
        mOutputStreamSets.emplace_back("codewordMask", codewordMask, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
        mOutputStreamSets.emplace_back("dictBoundaryMask", dictBoundaryMask, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
        addInternalScalar(b->getInt64Ty(), "segmentSize");
    } else {
        mOutputStreamSets.emplace_back("compressionMask", compressionMask, BoundedRate(0,1));
        mOutputStreamSets.emplace_back("encodedBytes", encodedBytes, BoundedRate(0,1));
        addInternalScalar(ArrayType::get(b->getInt8Ty(), encodingScheme.byLength[groupNo].hi), "pendingOutput");
    }
    Type * phraseType = ArrayType::get(b->getInt8Ty(), encodingScheme.byLength[groupNo].hi);
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
    Constant * sz_HASH_TABLE_START = b->getInt8(254);
    Constant * sz_HASH_TABLE_END = b->getInt8(255);
    Constant * sz_BITS = b->getSize(SIZE_T_BITS);
    Constant * sz_BLOCKWIDTH = b->getSize(b->getBitBlockWidth());
    Constant * sz_TABLEMASK = b->getSize((1U << 14) -1);
    Type * sizeTy = b->getSizeTy();
    Type * bitBlockPtrTy = b->getBitBlockType()->getPointerTo();

    BasicBlock * const entryBlock = b->GetInsertBlock();
    BasicBlock * const stridePrologue = b->CreateBasicBlock("stridePrologue");
    BasicBlock * const writeHTStart = b->CreateBasicBlock("writeHTStart");
    BasicBlock * const nextBlock = b->CreateBasicBlock("nextBlock");
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
    BasicBlock * const writeHTEnd = b->CreateBasicBlock("writeHTEnd");
    BasicBlock * const proceed = b->CreateBasicBlock("proceed");

    Value * const initialPos = b->getProcessedItemCount("symbolMarks");
    Value * const avail = b->getAvailableItemCount("symbolMarks");
    Value * const initialProduced = b->getProducedItemCount("compressionMask");
    Value * const phrasesProduced = b->getProducedItemCount("codewordMask");

    Value * pendingPhraseMask = b->getScalarField("pendingPhraseMask");
    Value * phrasesProducedPtr = b->CreateBitCast(b->getRawOutputPointer("codewordMask", phrasesProduced), bitBlockPtrTy);
    b->CreateStore(pendingPhraseMask, phrasesProducedPtr);
    Value * phraseMaskPtr = b->CreateBitCast(b->getRawOutputPointer("codewordMask", initialPos), bitBlockPtrTy);
    Value * dictMaskPtr = b->CreateBitCast(b->getRawOutputPointer("dictBoundaryMask", phrasesProduced), bitBlockPtrTy);

    Value * pendingMask = b->CreateNot(b->getScalarField("pendingMaskInverted"));
    Value * producedPtr = b->CreateBitCast(b->getRawOutputPointer("compressionMask", initialProduced), bitBlockPtrTy);
    b->CreateStore(pendingMask, producedPtr);
    Value * compressMaskPtr = b->CreateBitCast(b->getRawOutputPointer("compressionMask", initialPos), bitBlockPtrTy);
    Type * phraseType = ArrayType::get(b->getInt8Ty(), mEncodingScheme.byLength[mGroupNo].hi);
    Value * hashTableBasePtr = b->CreateBitCast(b->getScalarFieldPtr("hashTable"), phraseType->getPointerTo());
    if (!DelayedAttributeIsSet()) {
        // Copy pending output data.
        Value * const initialProduced1 = b->getProducedItemCount("encodedBytes");
        b->CreateMemCpy(b->getRawOutputPointer("encodedBytes", initialProduced1), b->getScalarFieldPtr("pendingOutput"), sz_DELAYED, 1);
    }
    // Copy all new input to the output buffer; this will be then
    // overwritten when and as necessary for decompression of ZTF codes.
    Value * toCopy = b->CreateMul(numOfStrides, sz_STRIDE);
    b->CreateMemCpy(b->getRawOutputPointer("encodedBytes", initialProduced), b->getRawInputPointer("byteData", initialProduced), toCopy, 1);
    b->CreateBr(stridePrologue);

    b->SetInsertPoint(stridePrologue);
    // Set up the loop variables as PHI nodes at the beginning of each stride.
    PHINode * const strideNo = b->CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    Value * stridePos = b->CreateAdd(initialPos, b->CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b->CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * nextStrideNo = b->CreateAdd(strideNo, sz_ONE);

    Value * checkStartBoundary = b->CreateAnd(b->CreateICmpEQ(b->getSize(mGroupNo), b->getSize(4)), b->CreateICmpEQ(b->getSize(mNumSym), sz_ONE));
    checkStartBoundary = b->CreateAnd(checkStartBoundary, b->CreateICmpEQ(initialProduced, sz_ZERO));
    b->CreateCondBr(checkStartBoundary, writeHTStart, nextBlock);
    b->SetInsertPoint(writeHTStart);
    b->setScalarField("segmentSize", b->getSize(0xF4240));

    b->CreateBr(nextBlock);
    b->SetInsertPoint(nextBlock);


    ///TODO: optimize if there are no hashmarks in the keyMasks stream
    std::vector<Value *> keyMasks = initializeCompressionMasks(b, sw, sz_BLOCKS_PER_STRIDE, 1, strideBlockOffset, compressMaskPtr, phraseMaskPtr, dictMaskPtr, strideMasksReady);
    Value * keyMask = keyMasks[0];

    b->SetInsertPoint(strideMasksReady);

    // b->CreateBr(strideMasksReady);
    // b->SetInsertPoint(strideMasksReady);
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
    Value * keyWordPos = b->CreateAdd(stridePos, b->CreateMul(keyWordIdx, sw.WIDTH));
    Value * keyMarkPosInWord = b->CreateCountForwardZeroes(theKeyWord);
    Value * keyMarkPos = b->CreateAdd(keyWordPos, keyMarkPosInWord, "keyEndPos");
    /* Determine the key length. */
    Value * hashValue = b->CreateZExt(b->CreateLoad(b->getRawInputPointer("hashValues", keyMarkPos)), sizeTy);

    Value * keyLength = b->CreateAdd(b->CreateLShr(hashValue, lg.MAX_HASH_BITS), sz_ONE, "keyLength");
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

    //Value * writtenVal = codewordVal;
    codewordVal = b->CreateOr(b->CreateShl(codewordVal, lg.MAX_HASH_BITS), b->CreateAnd(pfxByte, lg.PREFIX_LENGTH_MASK));

    pfxByte = b->CreateTrunc(b->CreateAnd(pfxByte, lg.PREFIX_LENGTH_MASK), b->getInt64Ty());

    Value * lgthBase1 = b->CreateSub(keyLength, lg.LO);
    Value * pfxOffset1 = b->CreateAdd(lg.PREFIX_BASE, lgthBase1);
    Value * multiplier1 = b->CreateMul(lg.RANGE, pfxByte);

    Value * ZTF_prefix1 = b->CreateAdd(pfxOffset1, multiplier1);

    Value * subTablePtr = b->CreateGEP(hashTableBasePtr, b->CreateMul(b->CreateSub(keyLength, lg.LO), lg.PHRASE_SUBTABLE_SIZE));
    /// TODO: experiment the number of bits in sz_TABLEMASK
    Value * tableIdxHash = b->CreateAnd(b->CreateLShr(codewordVal, 8), lg.TABLE_MASK, "tableIdx");
    Value * tblEntryPtr = b->CreateInBoundsGEP(subTablePtr, tableIdxHash);

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
    //b->CreateWriteCall(b->getInt32(STDERR_FILENO), symPtr1, keyLength);
    // writtenVal = b->CreateOr(b->CreateShl(writtenVal, lg.MAX_HASH_BITS), ZTF_prefix1, "writtenVal");
    // Value * const copyLen = b->CreateAdd(lg.ENC_BYTES, sz_ZERO);
    // Value * outputCodeword = b->CreateAlloca(b->getInt64Ty(), copyLen);
    // b->CreateAlignedStore(writtenVal, outputCodeword, 1);
    //b->CallPrintInt("outputCodeword", outputCodeword);
    // b->CreateWriteCall(b->getInt32(STDOUT_FILENO), outputCodeword, copyLen);

    // Mark the last byte of phrase
    Value * phraseMarkBase = b->CreateSub(keyMarkPos, b->CreateURem(keyMarkPos, sz_BITS));
    Value * markOffset = b->CreateSub(keyMarkPos, phraseMarkBase);
    Value * const codewordMaskBasePtr = b->CreateBitCast(b->getRawOutputPointer("codewordMask", phraseMarkBase), sizeTy->getPointerTo());
    Value * initialMark = b->CreateAlignedLoad(codewordMaskBasePtr, 1);
    Value * phraseEndPos = b->CreateSelect(b->CreateICmpEQ(b->getSize(mNumSym), sz_ONE), sz_ZERO, sz_ONE);
    Value * updatedMask = b->CreateOr(initialMark, b->CreateShl(sz_ONE, b->CreateSub(markOffset, phraseEndPos)));
    b->CreateAlignedStore(updatedMask, codewordMaskBasePtr, 1);

    // We have a new symbol that allows future occurrences of the symbol to
    // be compressed using the hash code.
    b->CreateMonitoredScalarFieldStore("hashTable", sym1, tblPtr1);
    b->CreateMonitoredScalarFieldStore("hashTable", sym2, tblPtr2);

    // markCompression even for the first occurrence
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

    mask = b->CreateShl(mask, bitOffset);
    Value * const keyBasePtr = b->CreateBitCast(b->getRawOutputPointer("compressionMask", keyBase), sizeTy->getPointerTo());
    Value * initialMask = b->CreateAlignedLoad(keyBasePtr, 1);
    Value * updated = b->CreateAnd(initialMask, b->CreateNot(mask));
    b->CreateAlignedStore(updated, keyBasePtr, 1);
    Value * curPos = keyMarkPos;
    Value * curHash = keyHash;
    // Write the suffixes.
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
    // 32 bytes of this segment.
    if (DeferredAttributeIsSet()) {
        Value * processed = b->CreateSub(avail, lg.HI);
        b->setProcessedItemCount("byteData", processed);
    }

    // Although we have written the last block mask, we do not include it as
    // produced, because we may need to update it in the event that there is
    // a compressible symbol starting in this segment and finishing in the next.
    Value * produced = b->CreateSelect(b->isFinal(), avail, b->CreateSub(avail, sz_BLOCKWIDTH));
    b->setProducedItemCount("compressionMask", produced);
    b->setProducedItemCount("codewordMask", produced);
    b->setProducedItemCount("dictBoundaryMask", produced);

    Value * checkBoundary = b->CreateAnd(b->CreateICmpEQ(b->getSize(mGroupNo), b->getSize(4)), b->CreateICmpEQ(b->getSize(mNumSym), sz_ONE));
    checkBoundary = b->CreateAnd(checkBoundary, b->CreateICmpUGE(produced, b->getScalarField("segmentSize")));
    b->CreateCondBr(checkBoundary, writeHTEnd, proceed);
    b->SetInsertPoint(writeHTEnd);
    b->setScalarField("segmentSize", b->CreateAdd(b->getScalarField("segmentSize"), b->getSize(0xF4240)));
    Value * const dictPtr = b->CreateBitCast(b->getRawOutputPointer("dictBoundaryMask", produced), sizeTy->getPointerTo() );
    Value * initMask = b->CreateAlignedLoad(dictPtr, 1);
    Value * update = b->CreateOr(initMask, sz_ONE);
    b->CreateAlignedStore(update, dictPtr, 1);
    //b->CallPrintInt("update", update);

    b->CreateBr(proceed);
    b->SetInsertPoint(proceed);

    b->CreateCondBr(b->isFinal(), compressionMaskDone, updatePending);
    b->SetInsertPoint(updatePending);
    Value * pendingPtr = b->CreateBitCast(b->getRawOutputPointer("compressionMask", produced), bitBlockPtrTy);
    Value * lastMask = b->CreateBlockAlignedLoad(pendingPtr);
    b->setScalarField("pendingMaskInverted", b->CreateNot(lastMask));

    Value * pendingCWmaskPtr = b->CreateBitCast(b->getRawOutputPointer("codewordMask", produced), bitBlockPtrTy);
    Value * lastCWMask = b->CreateBlockAlignedLoad(pendingCWmaskPtr);
    b->setScalarField("pendingPhraseMask", lastCWMask);
    b->CreateBr(compressionMaskDone);
    b->SetInsertPoint(compressionMaskDone);
}

// Assumes phrases with frequency >= 2 are compressed
WriteDictionary::WriteDictionary(BuilderRef b,
                                EncodingInfo encodingScheme,
                                unsigned numSyms,
                                StreamSet * byteData,
                                StreamSet * codedBytes,
                                StreamSet * extractionMask,
                                StreamSet * phraseMask,
                                StreamSet * allLenHashValues,
                                StreamSet * dictionaryBytes,
                                StreamSet * dictionaryMask,
                                unsigned strideBlocks)
: MultiBlockKernel(b, "WriteDictionary",
                   {Binding{"phraseMask", phraseMask},
                    Binding{"codewordMask", extractionMask},
                    Binding{"byteData", byteData, FixedRate(), LookBehind(33)},
                    Binding{"codedBytes", codedBytes, FixedRate(), LookBehind(33)},
                    Binding{"lengthData", allLenHashValues, FixedRate(), LookBehind(33)}},
                   {}, {}, {},
                   {InternalScalar{b->getBitBlockType(), "pendingMaskInverted"},
                    InternalScalar{b->getBitBlockType(), "pendingPhraseMask"}}),
mNumSym(numSyms) {
    if (DelayedAttributeIsSet()) {
        mOutputStreamSets.emplace_back("dictionaryMask", dictionaryMask, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
        mOutputStreamSets.emplace_back("dictionaryBytes", dictionaryBytes, FixedRate(), Delayed(encodingScheme.maxSymbolLength()) );
    } else {
        mOutputStreamSets.emplace_back("dictionaryMask", dictionaryMask, BoundedRate(0,1));
        mOutputStreamSets.emplace_back("dictionaryBytes", dictionaryBytes, BoundedRate(0,1));
        addInternalScalar(ArrayType::get(b->getInt8Ty(), encodingScheme.maxSymbolLength()), "pendingOutput");
    }
    //setStride(std::min(b->getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
    setStride(102400);
}

void WriteDictionary::generateMultiBlockLogic(BuilderRef b, Value * const numOfStrides) {
    ScanWordParameters sw(b, mStride);
    Constant * sz_STRIDE = b->getSize(mStride);
    Constant * sz_BLOCKS_PER_STRIDE = b->getSize(mStride/b->getBitBlockWidth());
    Constant * sz_ZERO = b->getSize(0);
    Constant * sz_ONE = b->getSize(1);
    Constant * sz_TWO = b->getSize(2);
    Constant * sz_FOUR = b->getSize(4);
    Constant * sz_EIGHT = b->getSize(8);
    Constant * sz_SYM_MASK = b->getSize(0x8F);
    Constant * sz_HASH_TABLE_START = b->getSize(65278);
    Constant * sz_HASH_TABLE_END = b->getSize(65535);

    Type * sizeTy = b->getSizeTy();
    Type * halfLengthTy = b->getIntNTy(8U * 8);
    Type * halfSymPtrTy = halfLengthTy->getPointerTo();
    Type * bitBlockPtrTy = b->getBitBlockType()->getPointerTo();

    BasicBlock * const entryBlock = b->GetInsertBlock();
    BasicBlock * const stridePrologue = b->CreateBasicBlock("stridePrologue");
    BasicBlock * const writeHTStart = b->CreateBasicBlock("writeHTStart");
    BasicBlock * const writeFEFE = b->CreateBasicBlock("writeFEFE");
    BasicBlock * const FEFEDone = b->CreateBasicBlock("FEFEDone");
    BasicBlock * const firstPhrase = b->CreateBasicBlock("firstPhrase");
    BasicBlock * const firstPhraseDone = b->CreateBasicBlock("firstPhraseDone");
    BasicBlock * const firstCodeword = b->CreateBasicBlock("firstCodeword");
    BasicBlock * const firstCodewordDone = b->CreateBasicBlock("firstCodewordDone");
    BasicBlock * const tryWriteMask = b->CreateBasicBlock("tryWriteMask");
    BasicBlock * const writeMask = b->CreateBasicBlock("writeMask");
    BasicBlock * const strideMasksReady = b->CreateBasicBlock("strideMasksReady");
    BasicBlock * const dictProcessingLoop = b->CreateBasicBlock("dictProcessingLoop");
    BasicBlock * const writePhrase = b->CreateBasicBlock("writePhrase");
    BasicBlock * const writeSegPhrase = b->CreateBasicBlock("writeSegPhrase");
    BasicBlock * const phraseWritten = b->CreateBasicBlock("phraseWritten");
    BasicBlock * const writeCodeword = b->CreateBasicBlock("writeCodeword");
    BasicBlock * const codewordWritten = b->CreateBasicBlock("codewordWritten");
    BasicBlock * const tryUpdateMask = b->CreateBasicBlock("tryUpdateMask");
    BasicBlock * const updateMask = b->CreateBasicBlock("updateMask");
    BasicBlock * const nextPhrase = b->CreateBasicBlock("nextPhrase");
    BasicBlock * const writeHTEnd = b->CreateBasicBlock("writeHTEnd");
    BasicBlock * const checkLoopCond = b->CreateBasicBlock("checkLoopCond");
    BasicBlock * const phrasesDone = b->CreateBasicBlock("phrasesDone");
    BasicBlock * const stridesDone = b->CreateBasicBlock("stridesDone");
    // BasicBlock * const updatePending = b->CreateBasicBlock("updatePending");
    // BasicBlock * const compressionMaskDone = b->CreateBasicBlock("compressionMaskDone");

    Value * const initialPos = b->getProcessedItemCount("phraseMask");
    Value * const avail = b->getAvailableItemCount("phraseMask");
    Value * const initialProduced = b->getProducedItemCount("dictionaryMask");

    // Value * pendingMask = b->CreateNot(b->getScalarField("pendingMaskInverted"));
    Value * producedPtr = b->CreateBitCast(b->getRawOutputPointer("dictionaryMask", initialProduced), bitBlockPtrTy);
    Value * toCopy = b->CreateMul(numOfStrides, sz_STRIDE);
    //b->CreateMemCpy(b->getRawOutputPointer("dictionaryBytes", initialProduced), b->getRawInputPointer("byteData", initialProduced), toCopy, 1);
    b->CreateBr(stridePrologue);

    b->SetInsertPoint(stridePrologue);
    // Set up the loop variables as PHI nodes at the beginning of each stride.
    PHINode * const strideNo = b->CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    //b->CallPrintInt("strideNo", strideNo);
    Value * stridePos = b->CreateAdd(initialPos, b->CreateMul(strideNo, sz_STRIDE));
    Value * strideBlockOffset = b->CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);
    Value * nextStrideNo = b->CreateAdd(strideNo, sz_ONE);

    std::vector<Value *> phraseMasks = initializeCompressionMasks1(b, sw, sz_BLOCKS_PER_STRIDE, 1, strideBlockOffset, producedPtr, strideMasksReady);
    Value * phraseMask = phraseMasks[0];

    b->SetInsertPoint(strideMasksReady);
    Value * phraseWordBasePtr = b->getInputStreamBlockPtr("phraseMask", sz_ZERO, strideBlockOffset);
    phraseWordBasePtr = b->CreateBitCast(phraseWordBasePtr, sw.pointerTy);
    b->CreateUnlikelyCondBr(b->CreateICmpEQ(phraseMask, sz_ZERO), phrasesDone, dictProcessingLoop);

    b->SetInsertPoint(dictProcessingLoop);
    PHINode * const phraseMaskPhi = b->CreatePHI(sizeTy, 2);
    phraseMaskPhi->addIncoming(phraseMask, strideMasksReady);
    PHINode * const phraseWordPhi = b->CreatePHI(sizeTy, 2);
    phraseWordPhi->addIncoming(sz_ZERO, strideMasksReady);
    PHINode * writePosPhi = b->CreatePHI(sizeTy, 2);
    writePosPhi->addIncoming(sz_ZERO, strideMasksReady);

    Value * phraseWordIdx = b->CreateCountForwardZeroes(phraseMaskPhi, "phraseWordIdx");
    Value * nextphraseWord = b->CreateZExtOrTrunc(b->CreateLoad(b->CreateGEP(phraseWordBasePtr, phraseWordIdx)), sizeTy);
    Value * thephraseWord = b->CreateSelect(b->CreateICmpEQ(phraseWordPhi, sz_ZERO), nextphraseWord, phraseWordPhi);
    Value * phraseWordPos = b->CreateAdd(stridePos, b->CreateMul(phraseWordIdx, sw.WIDTH));
    Value * phraseMarkPosInWord = b->CreateCountForwardZeroes(thephraseWord);
    Value * phraseMarkPos = b->CreateAdd(phraseWordPos, phraseMarkPosInWord, "phraseEndPos");
    /* Determine the phrase length. */
    Value * phraseLength = b->CreateZExtOrTrunc(b->CreateLoad(b->getRawInputPointer("lengthData", phraseMarkPos)), sizeTy);
    Value * numSym = phraseLength;
    numSym = b->CreateAnd(b->CreateLShr(numSym, b->getSize(5)), b->getSize(7));
    //b->CallPrintInt("numSym", numSym);

    //b->CallPrintInt("phraseLength-read", phraseLength);
    phraseLength = b->CreateAnd(phraseLength, sz_SYM_MASK);
    Value * phraseEndPos = b->CreateSelect(b->CreateICmpULT(numSym, b->getSize(mNumSym)), sz_ONE, sz_ZERO);
    //b->CallPrintInt("phraseEndPos", phraseEndPos);
    phraseLength = b->CreateAdd(phraseLength, phraseEndPos);
    //b->CallPrintInt("phraseLength-final", phraseLength);

    Value * phraseMarkPosFinal = b->CreateAdd(phraseMarkPos, b->CreateSub(b->getSize(mNumSym), numSym));
    Value * phraseStartPos = b->CreateSub(phraseMarkPosFinal, b->CreateSub(phraseLength, sz_ONE), "phraseStartPos");

    Value * codeWordLen = b->getSize(2);
    codeWordLen = b->CreateSelect(b->CreateICmpUGT(phraseLength, b->getSize(8)), b->CreateAdd(codeWordLen, sz_ONE), codeWordLen);
    codeWordLen = b->CreateSelect(b->CreateICmpUGT(phraseLength, b->getSize(16)), b->CreateAdd(codeWordLen, sz_ONE), codeWordLen);
    // Write phrase followed by codeword
    Value * codeWordStartPos =  b->CreateSub(phraseMarkPosFinal, b->CreateSub(codeWordLen, sz_ONE));
    Value * checkStartBoundary = b->CreateICmpEQ(writePosPhi, sz_ZERO);
    // Write initial hashtable boundary "fefe" and update dictionaryMask
    b->CreateBr(writeHTStart);
    //b->CreateCondBr(checkStartBoundary, writeHTStart, writeSegPhrase);
    b->SetInsertPoint(writeHTStart);
    PHINode * curWritePos = b->CreatePHI(sizeTy, 2);
    PHINode * loopIdx = b->CreatePHI(sizeTy, 2);
    curWritePos->addIncoming(writePosPhi, dictProcessingLoop);
    loopIdx->addIncoming(sz_ZERO, dictProcessingLoop);
    //b->CallPrintInt("loopIdx", loopIdx);
    Value * writeLen = b->CreateSelect(b->CreateICmpEQ(loopIdx, sz_ZERO), sz_TWO, phraseLength);
    writeLen = b->CreateSelect(b->CreateICmpEQ(loopIdx, sz_ONE), writeLen, codeWordLen);
    writeLen = b->CreateSelect(checkStartBoundary, writeLen, sz_ZERO);
    Value * nextLoopIdx = b->CreateAdd(loopIdx, sz_ONE);
    Value * updateWritePos = b->CreateAdd(curWritePos, writeLen);
    Value * maxLoopIdx = b->getSize(3);

    b->CreateCondBr(b->CreateAnd(checkStartBoundary, b->CreateICmpEQ(loopIdx, sz_ZERO)), writeFEFE, FEFEDone);
    b->SetInsertPoint(writeFEFE);
    Value * const startBoundary = sz_TWO;
    Value * sBoundaryCodeword = b->CreateAlloca(b->getInt64Ty(), startBoundary);
    b->CreateAlignedStore(sz_HASH_TABLE_START, sBoundaryCodeword, 1);
    b->CreateMemCpy(b->getRawOutputPointer("dictionaryBytes", curWritePos), sBoundaryCodeword, startBoundary, 1);
    b->CreateBr(FEFEDone);
    // Write start boundary
    b->SetInsertPoint(FEFEDone);
    b->CreateCondBr(b->CreateAnd(checkStartBoundary, b->CreateICmpEQ(loopIdx, sz_ONE)), firstPhrase, firstPhraseDone);
    b->SetInsertPoint(firstPhrase);
    // Value * symPtr1 = b->CreateBitCast(b->getRawInputPointer("byteData", phraseStartPos), halfSymPtrTy);
    // b->CreateWriteCall(b->getInt32(STDERR_FILENO), symPtr1, phraseLength);
    b->CreateMemCpy(b->getRawOutputPointer("dictionaryBytes", curWritePos), b->getRawInputPointer("byteData", phraseStartPos), phraseLength, 1);
    b->CreateBr(firstPhraseDone);
    // Write phrase
    b->SetInsertPoint(firstPhraseDone);
    b->CreateCondBr(b->CreateAnd(checkStartBoundary, b->CreateICmpEQ(loopIdx, sz_TWO)), firstCodeword, firstCodewordDone);
    b->SetInsertPoint(firstCodeword);
    // Value * symPtr2 = b->CreateBitCast(b->getRawInputPointer("codedBytes", codeWordStartPos), halfSymPtrTy);
    // b->CreateWriteCall(b->getInt32(STDERR_FILENO), symPtr2, codeWordLen);
    b->CreateMemCpy(b->getRawOutputPointer("dictionaryBytes", curWritePos), b->getRawInputPointer("codedBytes", codeWordStartPos), codeWordLen, 1);
    b->CreateBr(firstCodewordDone);
    // Write codeword
    b->SetInsertPoint(firstCodewordDone);
    BasicBlock * thisBB = b->GetInsertBlock();
    loopIdx->addIncoming(nextLoopIdx, thisBB);
    curWritePos->addIncoming(updateWritePos, thisBB);
    //b->CallPrintInt("updateWritePos", updateWritePos);
    b->CreateCondBr(b->CreateAnd(checkStartBoundary, b->CreateICmpNE(nextLoopIdx, maxLoopIdx)), writeHTStart, tryWriteMask);

    b->SetInsertPoint(tryWriteMask);
    // Update dictionaryMask
    b->CreateCondBr(checkStartBoundary, writeMask, writeSegPhrase);
    b->SetInsertPoint(writeMask);
    Value * const maskLength = b->CreateZExt(b->CreateAdd(sz_TWO, b->CreateAdd(phraseLength, codeWordLen)), sizeTy);
    Value * mask = b->CreateSub(b->CreateShl(sz_ONE, maskLength), sz_ONE);
    Value * const maskBasePtr = b->CreateBitCast(b->getRawOutputPointer("dictionaryMask", writePosPhi), sizeTy->getPointerTo());
    Value * initialBoundaryMask = b->CreateAlignedLoad(maskBasePtr, 1);
    Value * updatedBoundaryMask = b->CreateOr(initialBoundaryMask, mask);
    //b->CallPrintInt("updatedBoundaryMask", updatedBoundaryMask);
    b->CreateAlignedStore(updatedBoundaryMask, maskBasePtr, 1);
    b->CreateBr(writeSegPhrase);

    b->SetInsertPoint(writeSegPhrase);
    // If not first phrase of the segment
    // Write phrase followed by codeword
    PHINode * segWritePos = b->CreatePHI(sizeTy, 3);
    PHINode * segLoopIdx = b->CreatePHI(sizeTy, 3);
    segWritePos->addIncoming(writePosPhi, writeMask);
    segLoopIdx->addIncoming(sz_ZERO, writeMask);
    segWritePos->addIncoming(writePosPhi, tryWriteMask);
    segLoopIdx->addIncoming(sz_ZERO, tryWriteMask);
    // b->CallPrintInt("segWritePos", segWritePos);
    // b->CallPrintInt("segLoopIdx", segLoopIdx);
    Value * segWriteLen = b->CreateSelect(b->CreateICmpEQ(segLoopIdx, sz_ZERO), phraseLength, codeWordLen);
    segWriteLen = b->CreateSelect(b->CreateNot(checkStartBoundary), segWriteLen, sz_ZERO);
    Value * nextSegLoopIdx = b->CreateAdd(segLoopIdx, sz_ONE);
    Value * updateSegWritePos = b->CreateAdd(segWritePos, segWriteLen);

    b->CreateCondBr(b->CreateAnd(b->CreateNot(checkStartBoundary), b->CreateICmpEQ(segLoopIdx, sz_ZERO)), writePhrase, phraseWritten);
    // Write phrase
    b->SetInsertPoint(writePhrase);
    // Value * symPtr3 = b->CreateBitCast(b->getRawInputPointer("byteData", phraseStartPos), halfSymPtrTy);
    // b->CreateWriteCall(b->getInt32(STDERR_FILENO), symPtr3, phraseLength);
    b->CreateMemCpy(b->getRawOutputPointer("dictionaryBytes", segWritePos), b->getRawInputPointer("byteData", phraseStartPos), phraseLength, 1);
    b->CreateBr(phraseWritten);

    b->SetInsertPoint(phraseWritten);
    b->CreateCondBr(b->CreateAnd(b->CreateNot(checkStartBoundary), b->CreateICmpEQ(segLoopIdx, sz_ONE)), writeCodeword, codewordWritten);
    // Write codeword
    b->SetInsertPoint(writeCodeword);
    // b->CallPrintInt("codeWordLen", codeWordLen);
    // b->CallPrintInt("codeWordStartPos", codeWordStartPos);
    // Value * symPtr4 = b->CreateBitCast(b->getRawInputPointer("codedBytes", codeWordStartPos), halfSymPtrTy);
    // b->CreateWriteCall(b->getInt32(STDERR_FILENO), symPtr4, codeWordLen);
    b->CreateMemCpy(b->getRawOutputPointer("dictionaryBytes", segWritePos), b->getRawInputPointer("codedBytes", codeWordStartPos), codeWordLen, 1);
    b->CreateBr(codewordWritten);

    b->SetInsertPoint(codewordWritten);
    BasicBlock * thisSegBB = b->GetInsertBlock();
    segLoopIdx->addIncoming(nextSegLoopIdx, thisSegBB);
    segWritePos->addIncoming(updateSegWritePos, thisSegBB);
    //b->CallPrintInt("updateSegWritePos", updateSegWritePos);
    b->CreateCondBr(b->CreateICmpNE(nextSegLoopIdx, b->getSize(2)), writeSegPhrase, tryUpdateMask);

    b->SetInsertPoint(tryUpdateMask);
    b->CreateCondBr(checkStartBoundary, nextPhrase, updateMask);
    b->SetInsertPoint(updateMask);
    Value * phraseMaskLength = b->CreateZExt(b->CreateAdd(phraseLength, codeWordLen), sizeTy);
    Value * lastMask = b->CreateSub(b->CreateShl(sz_ONE, phraseMaskLength), sz_ONE);
    Value * dictBase = b->CreateSub(writePosPhi, b->CreateURem(writePosPhi, sz_EIGHT));
    Value * bitOffset1 = b->CreateSub(writePosPhi, dictBase);
    lastMask = b->CreateShl(lastMask, bitOffset1);
    Value * const dictPhraseBasePtr = b->CreateBitCast(b->getRawOutputPointer("dictionaryMask", dictBase), sizeTy->getPointerTo());
    Value * initialdictPhraseMask = b->CreateAlignedLoad(dictPhraseBasePtr, 1);
    Value * updatedDictPhraseMask = b->CreateOr(initialdictPhraseMask, lastMask);
    //b->CallPrintInt("updatedDictPhraseMask", updatedDictPhraseMask);
    b->CreateAlignedStore(updatedDictPhraseMask, dictPhraseBasePtr, 1);
    b->CreateBr(nextPhrase);

    b->SetInsertPoint(nextPhrase);
    Value * dropPhrase = b->CreateResetLowestBit(thephraseWord);
    Value * thisWordDone = b->CreateICmpEQ(dropPhrase, sz_ZERO);
    // There may be more phrases in the phrase mask.
    Value * nextphraseMask = b->CreateSelect(thisWordDone, b->CreateResetLowestBit(phraseMaskPhi), phraseMaskPhi);
    Value * nextWritePos = b->CreateAdd(writePosPhi, b->CreateAdd(codeWordLen, phraseLength));
    nextWritePos = b->CreateSelect(checkStartBoundary, b->CreateAdd(nextWritePos, sz_TWO), nextWritePos);
    b->CreateCondBr(b->CreateAnd(b->CreateICmpEQ(nextStrideNo, numOfStrides), b->CreateICmpEQ(nextphraseMask, sz_ZERO)), writeHTEnd, checkLoopCond);
    b->SetInsertPoint(writeHTEnd);
    // Write hashtable end boundary FFFF
    Value * const copyLen = sz_TWO;
    Value * boundaryCodeword = b->CreateAlloca(b->getInt64Ty(), copyLen);
    b->CreateAlignedStore(sz_HASH_TABLE_END, boundaryCodeword, 1);
    b->CreateMemCpy(b->getRawOutputPointer("dictionaryBytes", nextWritePos), boundaryCodeword, copyLen, 1);
    Value * lastBoundaryBase = b->CreateSub(nextWritePos, b->CreateURem(nextWritePos, sz_EIGHT));
    Value * lastBoundaryBitOffset1 = b->CreateSub(nextWritePos, lastBoundaryBase);
    Value * boundaryBits = b->getSize(0x3);
    boundaryBits = b->CreateShl(boundaryBits, lastBoundaryBitOffset1);
    Value * const dictPtr = b->CreateBitCast(b->getRawOutputPointer("dictionaryMask", lastBoundaryBase), sizeTy->getPointerTo());
    Value * initMask = b->CreateAlignedLoad(dictPtr, 1);
    Value * update = b->CreateOr(initMask, boundaryBits);
    b->CreateAlignedStore(update, dictPtr, 1);
    b->CreateBr(checkLoopCond);

    b->SetInsertPoint(checkLoopCond);
    BasicBlock * currentBB = b->GetInsertBlock();
    phraseMaskPhi->addIncoming(nextphraseMask, currentBB);
    phraseWordPhi->addIncoming(dropPhrase, currentBB);
    writePosPhi->addIncoming(nextWritePos, currentBB);
    b->CreateCondBr(b->CreateICmpEQ(nextphraseMask, sz_ZERO), phrasesDone, dictProcessingLoop);
    b->SetInsertPoint(phrasesDone);
    strideNo->addIncoming(nextStrideNo, phrasesDone);
    b->CreateCondBr(b->CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, stridesDone);
    b->SetInsertPoint(stridesDone);

    // b->CreateCondBr(b->isFinal(), compressionMaskDone, updatePending);
    // b->SetInsertPoint(updatePending);
    // // Value * pendingPtr = b->CreateBitCast(b->getRawOutputPointer("dictionaryMask", produced), bitBlockPtrTy);
    // // Value * lastMask = b->CreateBlockAlignedLoad(pendingPtr);
    // // b->setScalarField("pendingMaskInverted", b->CreateNot(lastMask));

    // b->CreateBr(compressionMaskDone);
    // b->SetInsertPoint(compressionMaskDone);
}


InterleaveCompressionSegment::InterleaveCompressionSegment(BuilderRef b,
                                    StreamSet * dictData,
                                    StreamSet * codedBytes,
                                    StreamSet * extractionMask,
                                    StreamSet * dictionaryMask,
                                    StreamSet * combinedBytes,
                                    StreamSet * combinedMask,
                                    unsigned strideBlocks)
: MultiBlockKernel(b, "InterleaveCompressionSegment",
                   {Binding{"dictData", dictData, FixedRate(1), LookBehind(32)},
                    Binding{"codedBytes", codedBytes, FixedRate(1), LookBehind(32)},
                    Binding{"compressionMask", extractionMask},
                    Binding{"dictionaryMask", dictionaryMask}},
                   {}, {}, {}, {InternalScalar{b->getBitBlockType(), "pendingMaskInverted"}}),
mStrideBlocks(strideBlocks) {
    if (DelayedAttributeIsSet()) {
        mOutputStreamSets.emplace_back("combinedBytes", combinedBytes, BoundedRate(1, 2), Delayed(32) );
        mOutputStreamSets.emplace_back("combinedMask", combinedMask, BoundedRate(1, 2), Delayed(32) );
    } else {
        mOutputStreamSets.emplace_back("combinedBytes", combinedBytes, FixedRate(2), Delayed(32) );
        mOutputStreamSets.emplace_back("combinedMask", combinedMask, FixedRate(2), Delayed(32) );
    }
    setStride(102400);
}

void InterleaveCompressionSegment::generateMultiBlockLogic(BuilderRef b, Value * const numOfStrides) {
    Constant * sz_ZERO = b->getSize(0);
    Constant * sz_STRIDE = b->getSize(mStride);
    Constant * sz_BLOCKS_PER_STRIDE = b->getSize(mStride/b->getBitBlockWidth());
    Constant * sz_ONE = b->getSize(1);
    Type * sizeTy = b->getSizeTy();
    BasicBlock * const entryBlock = b->GetInsertBlock();
    BasicBlock * const stridePrologue = b->CreateBasicBlock("stridePrologue");
    BasicBlock * const stridePrecomputation = b->CreateBasicBlock("stridePrecomputation");
    BasicBlock * const strideMasksReady = b->CreateBasicBlock("strideMasksReady");
    BasicBlock * const writeBlock = b->CreateBasicBlock("writeBlock");
    BasicBlock * const writeDoneBlock = b->CreateBasicBlock("writeDoneBlock");
    Value * const initialProduced = b->getProducedItemCount("combinedBytes");
    Value * const dictAvail = b->getAccessibleItemCount("dictData");
    Value * const dictMaskAvail = b->getAccessibleItemCount("dictionaryMask");
    Value * const cmpAvail = b->getAccessibleItemCount("codedBytes");
    Value * const dictProcessed = b->getProcessedItemCount("dictData");
    Value * const cmpProcessed = b->getProcessedItemCount("codedBytes");
    // b->CallPrintInt("initialProduced", initialProduced);
    // b->CallPrintInt("dictAvail", dictAvail);
    // b->CallPrintInt("dictMaskAvail", dictMaskAvail);
    // b->CallPrintInt("cmpAvail", cmpAvail);
    // b->CallPrintInt("dictProcessed", dictProcessed);
    // b->CallPrintInt("cmpProcessed", cmpProcessed);
    b->CreateBr(stridePrologue);
    b->SetInsertPoint(stridePrologue);

    PHINode * const strideNo = b->CreatePHI(b->getSizeTy(), 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    PHINode* const curDictAvailable = b->CreatePHI(b->getSizeTy(), 2);
    curDictAvailable->addIncoming(dictAvail, entryBlock);
    PHINode* const curCmpAvailable = b->CreatePHI(b->getSizeTy(), 2);
    curCmpAvailable->addIncoming(cmpAvail, entryBlock);
    Value * strideBlockOffset = b->CreateMul(strideNo, sz_BLOCKS_PER_STRIDE);

    b->CreateBr(stridePrecomputation);
    // Precompute partial sum popcount of the dictionary mask to be copied.
    b->SetInsertPoint(stridePrecomputation);
    PHINode * const dictMaskAccum = b->CreatePHI(sizeTy, 2);
    dictMaskAccum->addIncoming(sz_ZERO, stridePrologue);
    PHINode * const blockNo = b->CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, stridePrologue);

    Value * strideBlockIndex = b->CreateAdd(strideBlockOffset, blockNo);
    Value * dictionaryBlock = b->loadInputStreamBlock("dictionaryMask", sz_ZERO, strideBlockIndex);
    Value * const anyDictEntry = b->CreateAdd(dictMaskAccum, b->bitblock_popcount(dictionaryBlock));
    Value * const nextBlockNo = b->CreateAdd(blockNo, sz_ONE);
    dictMaskAccum->addIncoming(anyDictEntry, stridePrecomputation);
    blockNo->addIncoming(nextBlockNo, stridePrecomputation);
    b->CreateCondBr(b->CreateICmpNE(nextBlockNo, sz_BLOCKS_PER_STRIDE), stridePrecomputation, strideMasksReady);

    b->SetInsertPoint(strideMasksReady);

    Value * writeCond = b->CreateAnd(b->CreateICmpEQ(dictAvail, sz_ZERO), b->CreateICmpEQ(cmpAvail, sz_ZERO));
    b->CreateCondBr(writeCond, writeDoneBlock, writeBlock);
    b->SetInsertPoint(writeBlock);

    // b->CallPrintInt("curDictAvailable", curDictAvailable);
    // b->CallPrintInt("curCmpAvailable", curCmpAvailable);
    // b->CallPrintInt("strideNo", strideNo);
    Value * toCopyDict = b->CreateSelect(b->CreateICmpUGT(curDictAvailable, sz_STRIDE), sz_STRIDE, curDictAvailable);
    Value * toCopyCmp = b->CreateSelect(b->CreateICmpUGT(curCmpAvailable, sz_STRIDE), sz_STRIDE, curCmpAvailable);
    Value * const toCopyFinalDict = anyDictEntry;
    // b->CallPrintInt("toCopyDict", toCopyDict);
    // b->CallPrintInt("toCopyCmp", toCopyCmp);
    // b->CallPrintInt("anyDictEntry", anyDictEntry);

    Value * const bytesCopyOffset = b->CreateAdd(b->getProducedItemCount("combinedBytes"), toCopyFinalDict);
    Value * const maskCopyOffset = b->CreateAdd(b->getProducedItemCount("combinedMask"), toCopyFinalDict);
    // b->CallPrintInt("bytesCopyOffset", bytesCopyOffset);
    // b->CallPrintInt("maskCopyOffset", maskCopyOffset);
    Value * maskBase = b->CreateSub(maskCopyOffset, b->CreateURem(maskCopyOffset, b->getSize(8)));
    Value * maskBitOffset = b->CreateSub(maskCopyOffset, maskBase);
    // b->CallPrintInt("maskBitOffset", maskBitOffset);
    Value * byteOffset = b->CreateSub(b->getSize(8), maskBitOffset);
    // b->CallPrintInt("byteOffset", byteOffset);
    // Interleave dictionary bytes followed by compressed bytes
    b->CreateMemCpy(b->getRawOutputPointer("combinedBytes", b->getProducedItemCount("combinedBytes")),
                    b->getRawInputPointer("dictData", b->getProcessedItemCount("dictData")),
                    toCopyFinalDict, 1);
    b->CreateMemCpy(b->getRawOutputPointer("combinedBytes", b->CreateAdd(byteOffset, bytesCopyOffset)),
                    b->getRawInputPointer("codedBytes", b->getProcessedItemCount("codedBytes")),
                    toCopyCmp, 1);
    // Interleave dictionary mask followed by compression mask
    b->CreateMemCpy(b->getRawOutputPointer("combinedMask", b->getProducedItemCount("combinedMask")),
                    b->getRawInputPointer("dictionaryMask", b->getProcessedItemCount("dictionaryMask")),
                    toCopyFinalDict, 1);
    b->CreateMemCpy(b->getRawOutputPointer("combinedMask", b->CreateAdd(maskBitOffset, maskCopyOffset)),
                    b->getRawInputPointer("compressionMask", b->getProcessedItemCount("compressionMask")),
                    toCopyCmp, 1);

    b->setProcessedItemCount("dictData", b->CreateAdd(b->getProcessedItemCount("dictData"), toCopyDict));
    b->setProcessedItemCount("dictionaryMask", b->CreateAdd(b->getProcessedItemCount("dictionaryMask"), toCopyDict));
    b->setProcessedItemCount("codedBytes", b->CreateAdd(b->getProcessedItemCount("codedBytes"), toCopyCmp));
    b->setProcessedItemCount("compressionMask", b->CreateAdd(b->getProcessedItemCount("compressionMask"), toCopyCmp));
    Value * producedBytesThisStride = b->CreateAdd(b->getProducedItemCount("combinedBytes"), b->CreateAdd(b->CreateAdd(byteOffset, anyDictEntry), toCopyCmp));
    b->setProducedItemCount("combinedBytes", producedBytesThisStride);
    Value * producedMaskThisStride = b->CreateAdd(b->getProducedItemCount("combinedMask"), b->CreateAdd(b->CreateAdd(maskBitOffset, anyDictEntry), toCopyCmp));
    b->setProducedItemCount("combinedMask", producedMaskThisStride);
    // b->CallPrintInt("combinedBytes", b->getProducedItemCount("combinedBytes"));
    // b->CallPrintInt("dictData", b->getProcessedItemCount("dictData"));
    // b->CallPrintInt("codedBytes", b->getProcessedItemCount("codedBytes"));
    // b->CallPrintInt("combinedMask-procudedFin", b->getProducedItemCount("combinedMask"));
    Value * const nextStrideNo = b->CreateAdd(strideNo, b->getSize(1));
    strideNo->addIncoming(nextStrideNo, writeBlock);
    Value * const updateDictAvail = b->CreateSub(dictAvail, toCopyDict);
    curDictAvailable->addIncoming(updateDictAvail, writeBlock);
    Value * const updateCmpAvail = b->CreateSub(cmpAvail, toCopyDict);
    curCmpAvailable->addIncoming(updateCmpAvail, writeBlock);
    b->CreateCondBr(b->CreateICmpNE(nextStrideNo, numOfStrides), stridePrologue, writeDoneBlock);

    b->SetInsertPoint(writeDoneBlock);
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
                       Binding{"byteData", byteData, BoundedRate(0,1)} //, {LookBehind(32+1)} }//, Deferred()} //FixedRate(), LookBehind(32+1)}//, Deferred()}
                   },
                   {}, {}, {},
                   {InternalScalar{ArrayType::get(b->getInt8Ty(), encodingScheme.byLength[groupNo].hi), "pendingOutput"},
                    // Hash table 8 length-based tables with 256 16-byte entries each.
                    InternalScalar{ArrayType::get(ArrayType::get(b->getInt8Ty(), encodingScheme.byLength[groupNo].hi), phraseHashTableSize(encodingScheme.byLength[groupNo])), "hashTable"}}),
    mEncodingScheme(encodingScheme), mGroupNo(groupNo), mNumSym(numSym) {
    setStride(std::min(b->getBitBlockWidth() * strideBlocks, SIZE_T_BITS * SIZE_T_BITS));
    if (DelayedAttributeIsSet()) {
        mOutputStreamSets.emplace_back("result", result, BoundedRate(0,1), Delayed(encodingScheme.maxSymbolLength())); //FixedRate(), Delayed(encodingScheme.maxSymbolLength()));
    } else {
        mOutputStreamSets.emplace_back("result", result, BoundedRate(0,1));
    }
    //Type * phraseType = ArrayType::get(b->getInt8Ty(), 32);
   // addInternalScalar(ArrayType::get(phraseType, phraseHashTableSize(encodingScheme.byLength[groupNo])), "hashTable");
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

    Value * const initialProduced = b->getProducedItemCount("result");
    b->CreateMemCpy(b->getRawOutputPointer("result", initialProduced), b->getScalarFieldPtr("pendingOutput"), lg.HI, 1);
    //b->CreateWriteCall(b->getInt32(STDERR_FILENO), b->getScalarFieldPtr("pendingOutput"), b->CreateAdd(lg.HI, sz_ZERO));

    // Copy all new input to the output buffer; this will be then
    // overwritten when and as necessary for decompression of ZTF codes.
    ///TODO: only copy the decompressed data, not the hashtable from the compressed data
    Value * toCopy = b->CreateMul(numOfStrides, sz_STRIDE);
    b->CreateMemCpy(b->getRawOutputPointer("result", initialPos), b->getRawInputPointer("byteData", initialPos), toCopy, 1);

    Type * phraseType = ArrayType::get(b->getInt8Ty(), mEncodingScheme.byLength[mGroupNo].hi);
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
    b->CallPrintInt("decmp-tableIdxHash", b->CreateAnd(codewordVal, lg.TABLE_MASK));
#endif

    Value * subTablePtr = b->CreateGEP(hashTableBasePtr, b->CreateMul(b->CreateSub(keyLength, lg.LO), lg.PHRASE_SUBTABLE_SIZE));
    Value * tableIdxHash = b->CreateAnd(b->CreateLShr(codewordVal, 8), lg.TABLE_MASK, "tableIdx");
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
    b->CallPrintInt("DhashVal-lookup", b->CreateAnd(encodedVal, lg.TABLE_MASK));
#endif
    Value * symStartPos = b->CreateSub(hashMarkPos, b->CreateSub(symLength, sz_ONE), "symStartPos");
    Value * symOffset = b->CreateSub(symLength, lg.HALF_LENGTH);

    subTablePtr = b->CreateGEP(hashTableBasePtr, b->CreateMul(b->CreateSub(symLength, lg.LO), lg.PHRASE_SUBTABLE_SIZE));
    tableIdxHash = b->CreateAnd(b->CreateLShr(encodedVal, 8), lg.TABLE_MASK);
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
    Value * processed = b->CreateSub(avail, lg.HI);
    b->setProcessedItemCount("byteData", processed);

    Value * guaranteedProduced = b->CreateSub(avail, lg.HI);
    b->CreateMemCpy(b->getScalarFieldPtr("pendingOutput"), b->getRawOutputPointer("result", guaranteedProduced), lg.HI, 1);
    b->setProducedItemCount("result", b->CreateSelect(b->isFinal(), avail, guaranteedProduced));
    //b->CreateWriteCall(b->getInt32(STDERR_FILENO), b->getRawOutputPointer("result", b->CreateSub(guaranteedProduced, lg.HI)), b->CreateAdd(lg.HI, sz_ZERO));
    //b->CallPrintInt("processed", processed);
    //CHECK: Although we have written the full input stream to output, there may
    // be an incomplete symbol at the end of this block.   Store the
    // data that may be overwritten as pending and set the produced item
    // count to that which is guaranteed to be correct.
}
