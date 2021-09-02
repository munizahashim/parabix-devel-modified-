
#include "common.h"
#include <boost/intrusive/detail/math.hpp>

using namespace kernel;
using namespace llvm;

static cl::opt<bool> DeferredAttribute("deferred", cl::desc("Use Deferred attribute instead of Lookbehind for source data"), cl::init(false));
static cl::opt<bool> DelayedAttribute("delayed", cl::desc("Use Delayed Attribute instead of BoundedRate for output"), cl::init(true));
static cl::opt<bool> PrefixCheck("prefix-check-mode", cl::desc("Use experimental prefix check mode"), cl::init(false));

bool LLVM_READONLY DeferredAttributeIsSet() {
    return DeferredAttribute;
}

bool LLVM_READONLY DelayedAttributeIsSet() {
    return DelayedAttribute;
}

bool LLVM_READONLY PrefixCheckIsSet() {
    return PrefixCheck;
}

using BuilderRef = Kernel::BuilderRef;

ScanWordParameters::ScanWordParameters(BuilderRef b, unsigned stride) :
#ifdef PREFER_NARROW_SCANWIDTH
    width(std::max(BITS_PER_BYTE, stride/SIZE_T_BITS)),
#else
    width(std::min(SIZE_T_BITS, stride/BITS_PER_BYTE)),
#endif
    indexWidth(stride/width),
    Ty(b->getIntNTy(width)),
    pointerTy(Ty->getPointerTo()),
    WIDTH(b->getSize(width)),
    ix_MAXBIT(b->getSize(indexWidth - 1)),
    WORDS_PER_BLOCK(b->getSize(b->getBitBlockWidth()/width)),
    WORDS_PER_STRIDE(b->getSize(indexWidth))
    {   //  The stride must be a power of 2 and a multiple of the BitBlock width.
        assert((((stride & (stride - 1)) == 0) && (stride >= b->getBitBlockWidth()) && (stride <= SIZE_T_BITS * SIZE_T_BITS)));
    }

LengthGroupParameters::LengthGroupParameters(BuilderRef b, EncodingInfo encodingScheme, unsigned groupNo) :
    groupInfo(encodingScheme.byLength[groupNo]),
    MAX_HASH_BITS(b->getSize(encodingScheme.MAX_HASH_BITS)),
    SUFFIX_BITS(b->getSize(7)),
    SUFFIX_MASK(b->getSize(0x7F)),
    groupHalfLength(1UL << boost::intrusive::detail::floor_log2(groupInfo.lo)),
    halfLengthTy(b->getIntNTy(8U * groupHalfLength)),
    halfSymPtrTy(halfLengthTy->getPointerTo()),
    HALF_LENGTH(b->getSize(groupHalfLength)),
    LO(b->getSize(groupInfo.lo)),
    HI(b->getSize(groupInfo.hi)),
    // All subtables are sized the same.
    SUBTABLE_SIZE(b->getSize((1UL << groupInfo.hash_bits) * groupInfo.hi)),
    HASH_BITS(b->getSize(groupInfo.hash_bits)),
    EXTENDED_BITS(b->getSize(std::max((groupInfo.hash_bits + groupInfo.length_extension_bits), ((groupInfo.encoding_bytes - 1U) * 7U)))),
    HASH_MASK(b->getSize((1UL << ((groupInfo.hash_bits >> 1UL) * groupInfo.encoding_bytes)) - 1UL)),
    ENC_BYTES(b->getSize(groupInfo.encoding_bytes)),
    MAX_INDEX(b->getSize(groupInfo.encoding_bytes - 1UL)),
    PREFIX_BASE(b->getSize(groupInfo.prefix_base)),
    PREFIX_LENGTH_OFFSET(b->getSize(encodingScheme.prefixLengthOffset(groupInfo.lo))),
    LENGTH_MASK(b->getSize(2UL * groupHalfLength - 1UL)),
    EXTENSION_MASK(b->getSize((1UL << groupInfo.length_extension_bits) - 1UL)) {
        assert(groupInfo.hi <= (1UL << (boost::intrusive::detail::floor_log2(groupInfo.lo) + 1UL)));
    }

unsigned hashTableSize(LengthGroupInfo g) {
    unsigned numSubTables = (g.hi - g.lo + 1);
    return numSubTables * g.hi * (1<<g.hash_bits);
}

unsigned phraseHashTableSize(LengthGroupInfo g, unsigned sz_stride) {
    unsigned segmentSizeInBytes = 8 * sz_stride; // numStrides * sz_stride
    unsigned maxSyms = segmentSizeInBytes / g.lo;
    return 32768; // temp

}

std::string lengthRangeSuffix(EncodingInfo encodingScheme, unsigned lo, unsigned hi) {
    std::stringstream suffix;
    suffix << encodingScheme.uniqueSuffix() << "_" << lo << "_" << hi;
    if (DeferredAttributeIsSet()) suffix << "deferred";
    if (DelayedAttributeIsSet()) suffix << "_delayed";
    return suffix.str();
}

std::string lengthGroupSuffix(EncodingInfo encodingScheme, unsigned groupNo) {
    LengthGroupInfo g = encodingScheme.byLength[groupNo];
    return lengthRangeSuffix(encodingScheme, g.lo, g.hi);
}

// indicate which block contain a symbol to be considered for compression
// skip the block that does not contain any symbol
std::vector<Value *> initializeCompressionMasks(BuilderRef b,
                                                ScanWordParameters & sw,
                                                Constant * sz_BLOCKS_PER_STRIDE,
                                                unsigned maskCount,
                                                Value * strideBlockOffset,
                                                Value * compressMaskPtr,
                                                BasicBlock * strideMasksReady) {
    Constant * sz_ZERO = b->getSize(0);
    Constant * sz_ONE = b->getSize(1);
    Type * sizeTy = b->getSizeTy();
    std::vector<Value *> keyMasks(maskCount);
    BasicBlock * const entryBlock = b->GetInsertBlock();
    BasicBlock * const maskInitialization = b->CreateBasicBlock("maskInitialization");
    b->CreateBr(maskInitialization);
    b->SetInsertPoint(maskInitialization);
    std::vector<PHINode *> keyMaskAccum(maskCount);
    for (unsigned i = 0; i < maskCount; i++) {
        keyMaskAccum[i] = b->CreatePHI(sizeTy, 2);
        keyMaskAccum[i]->addIncoming(sz_ZERO, entryBlock);
    }
    PHINode * const blockNo = b->CreatePHI(sizeTy, 2);
    blockNo->addIncoming(sz_ZERO, entryBlock);
    Value * strideBlockIndex = b->CreateAdd(strideBlockOffset, blockNo);
    for (unsigned i = 0; i < maskCount; i++) {
        Value * keyBitBlock = b->loadInputStreamBlock("symbolMarks" + (i > 0 ? std::to_string(i) : ""), sz_ZERO, strideBlockIndex);
        Value * const anyKey = b->simd_any(sw.width, keyBitBlock);
        Value * keyWordMask = b->CreateZExtOrTrunc(b->hsimd_signmask(sw.width, anyKey), sizeTy);
        //b->CallPrintRegister("keyBitBlock", keyBitBlock);
        //b->CallPrintRegister("anyKey", anyKey);
        //b->CallPrintInt("keyWordMask", keyWordMask);
        // number of symbols in a block at 64 bit boundaries
        keyMasks[i] = b->CreateOr(keyMaskAccum[i], b->CreateShl(keyWordMask, b->CreateMul(blockNo, sw.WORDS_PER_BLOCK)));
        //b->CallPrintInt("keyMasks"+std::to_string(i), keyMasks[i]);
        keyMaskAccum[i]->addIncoming(keyMasks[i], maskInitialization);
    }
    // Initialize the compression mask.
    // Default initial compression mask is all ones (no zeroes => no compression).
    b->CreateBlockAlignedStore(b->allOnes(), b->CreateGEP(compressMaskPtr, strideBlockIndex));
    Value * const nextBlockNo = b->CreateAdd(blockNo, sz_ONE);
    blockNo->addIncoming(nextBlockNo, maskInitialization);
    // Default initial compression mask is all ones (no zeroes => no compression).
    b->CreateCondBr(b->CreateICmpNE(nextBlockNo, sz_BLOCKS_PER_STRIDE), maskInitialization, strideMasksReady);
    return keyMasks;
}

