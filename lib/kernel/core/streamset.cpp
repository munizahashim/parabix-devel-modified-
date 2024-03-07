/*
 *  Copyright (c) 2016 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include <kernel/core/streamset.h>

#include <kernel/core/kernel.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>
#include <kernel/core/kernel_builder.h>
#include <toolchain/toolchain.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Format.h>
#include <llvm/ADT/Twine.h>
#include <boost/intrusive/detail/math.hpp>
#include <llvm/Analysis/ConstantFolding.h>

#include <array>

#define _GNU_SOURCE
#include <sys/mman.h>

#define TEST_MMAPPED_CIRCULAR_BUFFERS

#ifdef TEST_MMAPPED_CIRCULAR_BUFFERS

#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>

//#include <sys/memfd.h>

size_t get_lcm(const size_t A, const size_t B) {
    return boost::lcm<size_t>(A, B);
}


inline static int create_memfd() {
#if defined(__FreeBSD__)
return shm_open(SHM_ANON, O_RDWR, 0);
#elif defined(__NR_memfd_create)
return syscall(__NR_memfd_create, "shm_anon", (unsigned int)(MFD_CLOEXEC));
#else
char nm[] = "/tmp/streamset-XXXXXX";
#if defined(__OpenBSD__)
const int fd = shm_mkstemp(nm);
if (shm_unlink(nm) != 0) {
    close(fd);
    return -1;
}
#else
const int fd = mkstemp(nm);
if (unlink(nm) != 0) {
    close(fd);
    return -1;
}
#endif
return fd;
#endif
}

uint8_t * make_circular_buffer(const size_t underflow, const size_t size, const size_t overflow) {

    //if (underflow || overflow) {

        assert ((size % getpagesize()) == 0);

        const auto memfd = create_memfd();
        if (memfd == -1) {
            llvm::report_fatal_error("failed to create memfd");
        }

        if (ftruncate(memfd, size) == -1) {
            llvm::report_fatal_error(llvm::Twine{"failed to truncate to ", std::to_string(size)});
        }

        assert (underflow <= size);
        assert (overflow <= size);
        size_t n = underflow ? 1 : 0;
        // size_t m = 1 + n + (overflow ? 1 : 0);
        size_t m = 2 + n;

        uint8_t * base = (uint8_t*)mmap(NULL, m * size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (base == MAP_FAILED) {
            llvm::report_fatal_error("failed to allocate virtual address space");
        }

//        llvm::errs() << "allocating ";
//        llvm::errs().write_hex((uintptr_t)base) << " (" << size << "x" << m << ")\n";

        for (size_t i = 0; i < m; i++) {
            void* p = mmap(base + (i * size), size, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_SHARED, memfd, 0);
            if (p == MAP_FAILED) {
                llvm::report_fatal_error("failed to map virtual address space");
            }
        }
        close(memfd);
        return base + (n * size);
//    } else {

//        const auto ret = aligned_alloc(256 / 8, size);
//        llvm::errs().write_hex((uintptr_t)ret + size) << "\n";
//        assert (ret);
//        return (uint8_t *)ret;
//    }
}

void destroy_circular_buffer(uint8_t * base, const size_t underflow, const size_t size, const size_t overflow) {
    if (LLVM_UNLIKELY(base == nullptr)) {
        return;
    }
//    if (underflow || overflow) {
        size_t n = underflow ? 1 : 0;
        // size_t m = 1 + n + (overflow ? 1 : 0);
        size_t m = n + 2;
        const auto ptr = base - (n * size);
//        llvm::errs() << "releasing ";
//        llvm::errs().write_hex((uintptr_t)base) << " (" << size << "x" << m << ")\n";
        munmap(ptr, m * size);
//    } else {
//        free(base);
//    }
}

#endif


namespace llvm { class Constant; }
namespace llvm { class Function; }

using namespace llvm;
using namespace IDISA;
using IDISA::IDISA_Builder;

using boost::intrusive::detail::is_pow2;
using boost::intrusive::detail::floor_log2;

#define ANON_MMAP_SIZE (2ULL * 1048576ULL)

inline bool isConstantOne(const Value * const index) {
    return isa<ConstantInt>(index) ? cast<ConstantInt>(index)->isOne() : false;
}

inline bool isCapacityGuaranteed(const Value * const index, const size_t capacity) {
    return isa<ConstantInt>(index) ? cast<ConstantInt>(index)->getLimitedValue() < capacity : false;
}

namespace kernel {

using Rational = KernelBuilder::Rational;

using BuilderPtr = StreamSetBuffer::BuilderPtr;

[[noreturn]] void unsupported(const char * const function, const char * const bufferType) {
    report_fatal_error(StringRef{function} + " is not supported by " + bufferType + "Buffers");
}

LLVM_READNONE inline unsigned getItemWidth(const Type * ty ) {
    if (LLVM_LIKELY(isa<ArrayType>(ty))) {
        ty = ty->getArrayElementType();
    }
    ty = cast<FixedVectorType>(ty)->getElementType();
    return cast<IntegerType>(ty)->getBitWidth();
}

LLVM_READNONE inline size_t getArraySize(const Type * ty) {
    if (LLVM_LIKELY(isa<ArrayType>(ty))) {
        return ty->getArrayNumElements();
    } else {
        return 1;
    }
}

LLVM_READNONE inline Value * addUnderflow(BuilderPtr & b, Value * ptr, const unsigned underflow) {
    if (LLVM_LIKELY(underflow == 0)) {
        return ptr;
    } else {
        assert ("unspecified module" && b.get() && b->getModule());
        DataLayout DL(b->getModule());
        IntegerType * const intPtrTy = DL.getIntPtrType(b->getContext());
        Constant * offset = ConstantInt::get(intPtrTy, underflow);
        return b->CreateInBoundsGEP(ptr->getType(), ptr, offset);
    }
}

LLVM_READNONE inline Value * subtractUnderflow(BuilderPtr & b, Value * ptr, const unsigned underflow) {
    if (LLVM_LIKELY(underflow == 0)) {
        return ptr;
    } else {
        DataLayout DL(b->getModule());
        IntegerType * const intPtrTy = DL.getIntPtrType(b->getContext());
        Constant * offset = ConstantExpr::getNeg(ConstantInt::get(intPtrTy, underflow));
        return b->CreateInBoundsGEP(ptr->getType(), ptr, offset);
    }
}

void StreamSetBuffer::assertValidStreamIndex(BuilderPtr b, Value * streamIndex) const {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value * const count = getStreamSetCount(b);
        Value * const index = b->CreateZExtOrTrunc(streamIndex, count->getType());
        Value * const withinSet = b->CreateICmpULT(index, count);
        b->CreateAssert(withinSet, "out-of-bounds stream access: %i of %i", index, count);
    }
}

Value * StreamSetBuffer::getStreamBlockPtr(BuilderPtr b, Value * const baseAddress, Value * const streamIndex, Value * const blockIndex) const {
   // assertValidStreamIndex(b, streamIndex);
    return b->CreateInBoundsGEP(mType, baseAddress, {blockIndex, streamIndex});
}

Value * StreamSetBuffer::getStreamPackPtr(BuilderPtr b, Value * const baseAddress, Value * const streamIndex, Value * blockIndex, Value * const packIndex) const {
   // assertValidStreamIndex(b, streamIndex);
    return b->CreateInBoundsGEP(mType, baseAddress, {blockIndex, streamIndex, packIndex});
}

llvm::Value * StreamSetBuffer::loadStreamBlock(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex, const bool unaligned) const {
    Value * addr = getStreamBlockPtr(b, baseAddress, streamIndex, blockIndex);
    return b->CreateAlignedLoad(b->getBitBlockType(), addr, unaligned ? 1U : (b->getBitBlockWidth() / 8));
}

llvm::Value * StreamSetBuffer::loadStreamPack(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex, llvm::Value * packIndex, const bool unaligned) const {
    Value * addr = getStreamPackPtr(b, baseAddress, streamIndex, blockIndex, packIndex);
    return b->CreateAlignedLoad(b->getBitBlockType(), addr, unaligned ? 1U : (b->getBitBlockWidth() / 8));
}

Value * StreamSetBuffer::getStreamSetCount(BuilderPtr b) const {
    size_t count = 1;
    if (isa<ArrayType>(getBaseType())) {
        count = getBaseType()->getArrayNumElements();
    }
    return b->getSize(count);
}

size_t StreamSetBuffer::getUnderflowCapacity(BuilderPtr b) const {
    return mUnderflow * b->getBitBlockWidth();
}

size_t StreamSetBuffer::getOverflowCapacity(BuilderPtr b) const {
    return mOverflow * b->getBitBlockWidth();
}

bool StreamSetBuffer::isEmptySet() const {
    return getArraySize(mBaseType) == 0;
}

unsigned StreamSetBuffer::getFieldWidth() const {
    return getItemWidth(mBaseType);
}

/**
 * @brief getRawItemPointer
 *
 * get a raw pointer the iN field at position absoluteItemPosition of the stream number streamIndex of the stream set.
 * In the case of a stream whose fields are less than one byte (8 bits) in size, the pointer is to the containing byte.
 * The type of the pointer is i8* for fields of 8 bits or less, otherwise iN* for N-bit fields.
 */
Value * StreamSetBuffer::getRawItemPointer(BuilderPtr b, Value * streamIndex, Value * absolutePosition) const {
    Type * const elemTy = cast<ArrayType>(mBaseType)->getElementType();
    Type * itemTy = cast<VectorType>(elemTy)->getElementType();
    #if LLVM_VERSION_CODE < LLVM_VERSION_CODE(12, 0, 0)
    const unsigned itemWidth = itemTy->getPrimitiveSizeInBits();
    #else
    const unsigned itemWidth = itemTy->getPrimitiveSizeInBits().getFixedSize();
    #endif
    IntegerType * const sizeTy = b->getSizeTy();
    absolutePosition = b->CreateZExt(absolutePosition, sizeTy);
    streamIndex = b->CreateZExt(streamIndex, sizeTy);

    Value * pos = nullptr;
    Value * addr = nullptr;
    Value * const streamCount = getStreamSetCount(b);
    if (LLVM_LIKELY(isConstantOne(streamCount))) {
        addr = getBaseAddress(b);
        pos = absolutePosition;
        if (!isLinear()) {
            pos = b->CreateURem(pos, getCapacity(b));
        }
    } else {
        Constant * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
        Value * blockIndex = b->CreateUDiv(absolutePosition, BLOCK_WIDTH);
        addr = getStreamBlockPtr(b, getBaseAddress(b), streamIndex, blockIndex);
        pos = b->CreateURem(absolutePosition, BLOCK_WIDTH);
    }
    if (LLVM_UNLIKELY(itemWidth < 8)) {
        const Rational itemsPerByte{8, itemWidth};
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            b->CreateAssertZero(b->CreateURemRational(absolutePosition, itemsPerByte),
                                "absolutePosition (%" PRIu64 " * %" PRIu64 "x%" PRIu64 ") must be byte aligned",
                                absolutePosition, streamCount, b->getSize(itemWidth));
        }
        pos = b->CreateUDivRational(pos, itemsPerByte);
        itemTy = b->getInt8Ty();
    }
    addr = b->CreatePointerCast(addr, itemTy->getPointerTo(mAddressSpace));
    return b->CreateInBoundsGEP(itemTy, addr, pos);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addOverflow
 ** ------------------------------------------------------------------------------------------------------------- */
Value * StreamSetBuffer::addOverflow(BuilderPtr b, Value * const bufferCapacity, Value * const overflowItems, Value * const consumedOffset) const {
    if (overflowItems) {
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            Value * const overflowCapacity = b->getSize(getOverflowCapacity(b));
            Value * const valid = b->CreateICmpULE(overflowItems, overflowCapacity);
            b->CreateAssert(valid, "overflow items exceeds overflow capacity");
        }
        if (consumedOffset) {
            // limit the overflow so that we do not overwrite our unconsumed data during a copyback
            Value * const effectiveOverflow = b->CreateUMin(consumedOffset, overflowItems);
            return b->CreateAdd(bufferCapacity, effectiveOverflow);
        } else {
            return b->CreateAdd(bufferCapacity, overflowItems);
        }
    } else { // no overflow
        return bufferCapacity;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief resolveType
 ** ------------------------------------------------------------------------------------------------------------- */
Type * StreamSetBuffer::resolveType(BuilderPtr b, Type * const streamSetType) {
    unsigned numElements = 1;
    Type * type = streamSetType;
    if (LLVM_LIKELY(type->isArrayTy())) {
        numElements = type->getArrayNumElements();
        type = type->getArrayElementType();
    }
    if (LLVM_LIKELY(type->isVectorTy() && cast<FixedVectorType>(type)->getNumElements() == 0)) {
        type = cast<FixedVectorType>(type)->getElementType();
        if (LLVM_LIKELY(type->isIntegerTy())) {
            const auto fieldWidth = cast<IntegerType>(type)->getBitWidth();
            type = b->getBitBlockType();
            if (fieldWidth != 1) {
                type = ArrayType::get(type, fieldWidth);
            }
            return ArrayType::get(type, numElements);
        }
    }
    std::string tmp;
    raw_string_ostream out(tmp);
    streamSetType->print(out);
    out << " is an unvalid stream set buffer type.";
    report_fatal_error(Twine(out.str()));
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkFunctions
 ** ------------------------------------------------------------------------------------------------------------- */
#ifdef TEST_MMAPPED_CIRCULAR_BUFFERS
constexpr char __MAKE_CIRCULAR_BUFFER[] = "__make_circular_buffer";
constexpr char __DESTROY_CIRCULAR_BUFFER[] = "__destroy_circular_buffer";
constexpr char __GET_LCM[] = "__get_lcm";
#endif

void StreamSetBuffer::linkFunctions(BuilderPtr b) {
#ifdef TEST_MMAPPED_CIRCULAR_BUFFERS
b->LinkFunction(__MAKE_CIRCULAR_BUFFER, make_circular_buffer);
b->LinkFunction(__DESTROY_CIRCULAR_BUFFER, destroy_circular_buffer);
b->LinkFunction(__GET_LCM, get_lcm);
#endif
}

// External Buffer

StructType * ExternalBuffer::getHandleType(BuilderPtr b) const {
    if (mHandleType == nullptr) {
        FixedArray<Type *, 2> fields;
        fields[0] = getPointerType();
        fields[1] = b->getSizeTy();
        mHandleType = StructType::get(b->getContext(), fields);
    }
    return mHandleType;
}

void ExternalBuffer::allocateBuffer(BuilderPtr /* b */, Value * const /* capacityMultiplier */) {
    unsupported("allocateBuffer", "External");
}

void ExternalBuffer::releaseBuffer(BuilderPtr /* b */) const {
    // this buffer is not responsible for free-ing th data associated with it
}

void ExternalBuffer::destroyBuffer(BuilderPtr b, llvm::Value * baseAddress, llvm::Value *capacity) const {

}

void ExternalBuffer::setBaseAddress(BuilderPtr b, Value * const addr) const {
    assert (mHandle && "has not been set prior to calling setBaseAddress");
    Value * const p = b->CreateInBoundsGEP(mHandleType, mHandle, {b->getInt32(0), b->getInt32(BaseAddress)});
    b->CreateStore(b->CreatePointerBitCastOrAddrSpaceCast(addr, getPointerType()), p);
}

Value * ExternalBuffer::getBaseAddress(BuilderPtr b) const {
    assert (mHandle && "has not been set prior to calling getBaseAddress");
    Value * const p = b->CreateInBoundsGEP(mHandleType, mHandle, {b->getInt32(0), b->getInt32(BaseAddress)});
    return b->CreateLoad(getPointerType(), p);
}

Value * ExternalBuffer::getOverflowAddress(BuilderPtr b) const {
    assert (mHandle && "has not been set prior to calling getBaseAddress");
    Value * const p = b->CreateInBoundsGEP(mHandleType, mHandle, {b->getInt32(0), b->getInt32(EffectiveCapacity)});
    return b->CreateLoad(b->getSizeTy(), p);
}

void ExternalBuffer::setCapacity(BuilderPtr b, Value * const capacity) const {
    assert (mHandle && "has not been set prior to calling setCapacity");
    Value *  const p = b->CreateInBoundsGEP(mHandleType, mHandle, {b->getInt32(0), b->getInt32(EffectiveCapacity)});
    b->CreateStore(b->CreateZExt(capacity, b->getSizeTy()), p);
}

Value * ExternalBuffer::getCapacity(BuilderPtr b) const {
    assert (mHandle && "has not been set prior to calling getCapacity");
    Value * const p = b->CreateInBoundsGEP(mHandleType, mHandle, {b->getInt32(0), b->getInt32(EffectiveCapacity)});
    return b->CreateLoad(b->getSizeTy(), p);
}

Value * ExternalBuffer::getInternalCapacity(BuilderPtr b) const {
    return getCapacity(b);
}

Value * ExternalBuffer::modByCapacity(BuilderPtr /* b */, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    return offset;
}

Value * ExternalBuffer::getLinearlyAccessibleItems(BuilderPtr b, Value * const fromPosition, Value * const totalItems, Value * /* overflowItems */) const {
    assert (totalItems);
    assert (fromPosition);
    return b->CreateSub(totalItems, fromPosition);
}

Value * ExternalBuffer::getLinearlyWritableItems(BuilderPtr b, Value * const fromPosition, Value * const /* consumed */, Value * /* overflowItems */) const {
    assert (fromPosition);
    Value * const capacity = getCapacity(b);
    assert (fromPosition->getType() == capacity->getType());
    return b->CreateSub(capacity, fromPosition);
}

Value * ExternalBuffer::getVirtualBasePtr(BuilderPtr b, Value * baseAddress, Value * const /* transferredItems */) const {
    Constant * const sz_ZERO = b->getSize(0);
    Value * const addr = StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, sz_ZERO);
    return b->CreatePointerCast(addr, getPointerType());
}

inline void ExternalBuffer::assertValidBlockIndex(BuilderPtr b, Value * blockIndex) const {
    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value * const blockCount = b->CreateCeilUDiv(getCapacity(b), b->getSize(b->getBitBlockWidth()));
        blockIndex = b->CreateZExtOrTrunc(blockIndex, blockCount->getType());
        Value * const withinCapacity = b->CreateICmpULT(blockIndex, blockCount);
        b->CreateAssert(withinCapacity, "blockIndex exceeds buffer capacity");
    }
}

Value * ExternalBuffer::requiresExpansion(BuilderPtr b, Value * produced, Value * consumed, Value * required) const {
    unsupported("requiresExpansion", "External");
}

void ExternalBuffer::linearCopyBack(BuilderPtr b, Value * produced, Value * consumed, Value * required) const {
    unsupported("linearCopyBack", "External");
}

Value * ExternalBuffer::expandBuffer(BuilderPtr /* b */, Value * /* produced */, Value * /* consumed */, Value * const /* required */) const  {
    unsupported("expandBuffer", "External");
}

Value * ExternalBuffer::getMallocAddress(BuilderPtr /* b */) const {
    unsupported("getMallocAddress", "External");
}

// Internal Buffer

Value * InternalBuffer::getStreamBlockPtr(BuilderPtr b, Value * const baseAddress, Value * const streamIndex, Value * const blockIndex) const {
    Value * offset = nullptr;
    if (mLinear) {
        offset = blockIndex;
    } else {
        offset = modByCapacity(b, blockIndex);
    }
    return StreamSetBuffer::getStreamBlockPtr(b, baseAddress, streamIndex, offset);
}

Value * InternalBuffer::getStreamPackPtr(BuilderPtr b, Value * const baseAddress, Value * const streamIndex, Value * const blockIndex, Value * const packIndex) const {
    Value * offset = nullptr;
    if (mLinear) {
        offset = blockIndex;
    } else {
        offset = modByCapacity(b, blockIndex);
    }
    return StreamSetBuffer::getStreamPackPtr(b, baseAddress, streamIndex, offset, packIndex);
}

Value * InternalBuffer::getVirtualBasePtr(BuilderPtr b, Value * const baseAddress, Value * const transferredItems) const {
    Constant * const sz_ZERO = b->getSize(0);
    Value * baseBlockIndex = nullptr;
    if (mLinear) {
        // NOTE: the base address of a linear buffer is always the virtual base ptr; just return it.
        baseBlockIndex = sz_ZERO;
    } else {
        Constant * const LOG_2_BLOCK_WIDTH = b->getSize(floor_log2(b->getBitBlockWidth()));
        Value * const blockIndex = b->CreateLShr(transferredItems, LOG_2_BLOCK_WIDTH);
        baseBlockIndex = b->CreateSub(modByCapacity(b, blockIndex), blockIndex);
    }
    Value * addr = StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, baseBlockIndex);
    return b->CreatePointerCast(addr, getPointerType());
}

Value * InternalBuffer::getLinearlyAccessibleItems(BuilderPtr b, Value * const processedItems, Value * const totalItems, Value * const overflowItems) const {
    if (mLinear) {
        return b->CreateSub(totalItems, processedItems);
    } else {
        Value * const capacity = getCapacity(b);
        Value * const fromOffset = b->CreateURem(processedItems, capacity);
        Value * const capacityWithOverflow = addOverflow(b, capacity, overflowItems, nullptr);
        Value * const linearSpace = b->CreateSub(capacityWithOverflow, fromOffset);
        Value * const availableItems = b->CreateSub(totalItems, processedItems);
        return b->CreateUMin(availableItems, linearSpace);
    }
}

Value * InternalBuffer::getLinearlyWritableItems(BuilderPtr b, Value * const producedItems, Value * const consumedItems, Value * const overflowItems) const {
    Value * const capacity = getCapacity(b);
    ConstantInt * const ZERO = b->getSize(0);
    if (mLinear) {
        Value * const capacityWithOverflow = addOverflow(b, capacity, overflowItems, nullptr);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            Value * const valid = b->CreateICmpULE(producedItems, capacityWithOverflow);
            b->CreateAssert(valid, "produced item count (%" PRIu64 ") exceeds capacity (%" PRIu64 ").",
                            producedItems, capacityWithOverflow);
        }
        return b->CreateSub(capacityWithOverflow, producedItems);
     } else {
        Value * const unconsumedItems = b->CreateSub(producedItems, consumedItems);
        Value * const full = b->CreateICmpUGE(unconsumedItems, capacity);
        Value * const fromOffset = b->CreateURem(producedItems, capacity);
        Value * const consumedOffset = b->CreateURem(consumedItems, capacity);
        Value * const toEnd = b->CreateICmpULE(consumedOffset, fromOffset);
        Value * const capacityWithOverflow = addOverflow(b, capacity, overflowItems, consumedOffset);
        Value * const limit = b->CreateSelect(toEnd, capacityWithOverflow, consumedOffset);
        Value * const remaining = b->CreateSub(limit, fromOffset);
        return b->CreateSelect(full, ZERO, remaining);
    }
}


// Static Buffer

StructType * StaticBuffer::getHandleType(BuilderPtr b) const {
    if (mHandleType == nullptr) {
        auto & C = b->getContext();
        PointerType * const typePtr = getPointerType();
        FixedArray<Type *, 4> types;
        types[BaseAddress] = typePtr;
        IntegerType * const sizeTy = b->getSizeTy();
        if (mLinear) {
            types[EffectiveCapacity] = sizeTy;
            types[MallocedAddress] = typePtr;
        } else {
            Type * const emptyTy = StructType::get(C);
            types[EffectiveCapacity] = emptyTy;
            types[MallocedAddress] = emptyTy;
        }
        types[InternalCapacity] = sizeTy;
        mHandleType = StructType::get(C, types);
    }
    return mHandleType;
}

void StaticBuffer::allocateBuffer(BuilderPtr b, Value * const capacityMultiplier) {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    Value * const handle = getHandle();
    assert (handle && "has not been set prior to calling allocateBuffer");
    Value * const capacity = b->CreateMul(capacityMultiplier, b->getSize(mCapacity));

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b->CreateAssert(capacity, "Static buffer capacity cannot be 0.");
    }

    indices[1] = b->getInt32(InternalCapacity);
    StructType * const handleTy = mHandleType;
    Value * const intCapacityField = b->CreateInBoundsGEP(handleTy, handle, indices);
    b->CreateStore(capacity, intCapacityField);

    indices[1] = b->getInt32(BaseAddress);
    Value * const size = b->CreateAdd(capacity, b->getSize(mUnderflow + mOverflow));
    Value * const mallocAddr = b->CreatePageAlignedMalloc(mType, size, mAddressSpace);
    Value * const buffer = addUnderflow(b, mallocAddr, mUnderflow);
    Value * const baseAddressField = b->CreateInBoundsGEP(handleTy, handle, indices);
    b->CreateStore(buffer, baseAddressField);

    if (mLinear) {
        indices[1] = b->getInt32(EffectiveCapacity);
        Value * const capacityField = b->CreateInBoundsGEP(handleTy, handle, indices);
        b->CreateStore(capacity, capacityField);

        indices[1] = b->getInt32(MallocedAddress);
        Value * const concreteAddrField = b->CreateInBoundsGEP(handleTy, handle, indices);
        b->CreateStore(buffer, concreteAddrField);
    }
}

void StaticBuffer::releaseBuffer(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(mLinear ? MallocedAddress : BaseAddress);
    Value * const addressField = b->CreateInBoundsGEP(mHandleType, getHandle(), indices);
    Value * buffer = b->CreateLoad(mType, addressField);
    destroyBuffer(b, buffer, nullptr);
    b->CreateStore(ConstantPointerNull::get(cast<PointerType>(mType)), addressField);
}

void StaticBuffer::destroyBuffer(BuilderPtr b, Value * baseAddress, llvm::Value *capacity) const {
    b->CreateFree(subtractUnderflow(b, baseAddress, mUnderflow));
}

Value * StaticBuffer::modByCapacity(BuilderPtr b, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    if (LLVM_UNLIKELY(mLinear || isCapacityGuaranteed(offset, mCapacity))) {
        return offset;
    } else {
        FixedArray<Value *, 2> indices;
        indices[0] = b->getInt32(0);
        indices[1] = b->getInt32(InternalCapacity);
        Value * ptr = b->CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const capacity = b->CreateLoad(b->getSizeTy(), ptr);
        return b->CreateURem(offset, capacity);
    }
}

Value * StaticBuffer::getCapacity(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(mLinear ? EffectiveCapacity : InternalCapacity);
    Value * ptr = b->CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
    Value * const capacity = b->CreateLoad(b->getSizeTy(), ptr);
    assert (capacity->getType()->isIntegerTy());
    return b->CreateMul(capacity, BLOCK_WIDTH, "capacity");
}

Value * StaticBuffer::getInternalCapacity(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(InternalCapacity);
    Value * const intCapacityField = b->CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
    Value * const capacity = b->CreateLoad(b->getSizeTy(), intCapacityField);
    assert (capacity->getType()->isIntegerTy());
    return b->CreateMul(capacity, BLOCK_WIDTH, "internalCapacity");
}

void StaticBuffer::setCapacity(BuilderPtr b, Value * capacity) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(InternalCapacity);
    Value * const handle = getHandle(); assert (handle);
    Value * capacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    ConstantInt * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
    assert (capacity->getType()->isIntegerTy());
    Value * const cap = b->CreateExactUDiv(capacity, BLOCK_WIDTH);
    b->CreateStore(cap, capacityField);
    if (mLinear) {
        indices[1] = b->getInt32(EffectiveCapacity);
        Value * const effCapacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
        b->CreateStore(cap, effCapacityField);
    }
}

Value * StaticBuffer::getBaseAddress(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    Value * const base = b->CreateInBoundsGEP(mHandleType, handle, indices);
    return b->CreateLoad(getPointerType(), base, "baseAddress");
}

void StaticBuffer::setBaseAddress(BuilderPtr b, Value * addr) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    b->CreateStore(addr, b->CreateInBoundsGEP(mHandleType, handle, indices));
    if (mLinear) {
         indices[1] = b->getInt32(MallocedAddress);
         b->CreateStore(addr, b->CreateInBoundsGEP(mHandleType, handle, indices));
    }
}

Value * StaticBuffer::getMallocAddress(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(mLinear ? MallocedAddress : BaseAddress);
    return b->CreateLoad(getPointerType(), b->CreateInBoundsGEP(mHandleType, getHandle(), indices));
}

Value * StaticBuffer::getOverflowAddress(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(mLinear ? MallocedAddress : BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    Value * const base = b->CreateLoad(getPointerType(), b->CreateInBoundsGEP(mHandleType, handle, indices));
    indices[1] = b->getInt32(InternalCapacity);
    Value * const capacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const capacity = b->CreateLoad(b->getSizeTy(), capacityField);
    return b->CreateInBoundsGEP(mType, base, capacity, "overflow");
}

Value * StaticBuffer::requiresExpansion(BuilderPtr b, Value * produced, Value * consumed, Value * required) const {
    return b->getFalse();
}

void StaticBuffer::linearCopyBack(BuilderPtr b, Value * produced, Value * consumed, Value * required) const {
    if (mLinear) {
        const auto blockWidth = b->getBitBlockWidth();
        assert (is_pow2(blockWidth));

        ConstantInt * const BLOCK_WIDTH = b->getSize(blockWidth);

        FixedArray<Value *, 2> indices;
        indices[0] = b->getInt32(0);
        indices[1] = b->getInt32(EffectiveCapacity);
        Value * const capacityField = b->CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * consumedChunks = b->CreateUDiv(consumed, BLOCK_WIDTH);

        indices[1] = b->getInt32(MallocedAddress);
        Value * const mallocedAddrField = b->CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * const bufferStart = b->CreateLoad(getPointerType(), mallocedAddrField);
        assert (bufferStart->getType()->isPointerTy());
        Value * const newBaseAddress = b->CreateGEP(getPointerType(), bufferStart, b->CreateNeg(consumedChunks));
        Value * const effectiveCapacity = b->CreateAdd(consumedChunks, b->getSize(mCapacity));

        indices[1] = b->getInt32(BaseAddress);
        Value * const baseAddrField = b->CreateInBoundsGEP(mHandleType, mHandle, indices);

        b->CreateStore(newBaseAddress, baseAddrField);
        b->CreateStore(effectiveCapacity, capacityField);
    }
}

Value * StaticBuffer::expandBuffer(BuilderPtr b, Value * produced, Value * consumed, Value * const required) const  {
    if (mLinear) {

        SmallVector<char, 200> buf;
        raw_svector_ostream name(buf);

        assert ("unspecified module" && b.get() && b->getModule());

        name << "__StaticLinearBuffer_linearCopyBack_";

        Type * ty = getBaseType();
        const auto streamCount = ty->getArrayNumElements();
        name << streamCount << 'x';
        ty = ty->getArrayElementType();
        ty = cast<FixedVectorType>(ty)->getElementType();;
        const auto itemWidth = ty->getIntegerBitWidth();
        name << itemWidth << '_' << mAddressSpace;

        Value * const myHandle = getHandle();


        Module * const m = b->getModule();
        IntegerType * const sizeTy = b->getSizeTy();
        FunctionType * funcTy = FunctionType::get(b->getVoidTy(), {myHandle->getType(), sizeTy, sizeTy, sizeTy, sizeTy}, false);
        Function * func = m->getFunction(name.str());
        if (func == nullptr) {

            const auto ip = b->saveIP();

            LLVMContext & C = m->getContext();
            func = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
            if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
                #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
                func->setHasUWTable();
                #else
                func->setUWTableKind(UWTableKind::Default);
                #endif
            }
            b->SetInsertPoint(BasicBlock::Create(C, "entry", func));

            auto arg = func->arg_begin();
            auto nextArg = [&]() {
                assert (arg != func->arg_end());
                Value * const v = &*arg;
                std::advance(arg, 1);
                return v;
            };

            Value * const handle = nextArg();
            handle->setName("handle");
            Value * const produced = nextArg();
            produced->setName("produced");
            Value * const consumed = nextArg();
            consumed->setName("consumed");
            Value * const underflow = nextArg();
            underflow->setName("underflow");
            Value * const overflow = nextArg();
            overflow->setName("overflow");
            assert (arg == func->arg_end());

            setHandle(handle);

            const auto blockWidth = b->getBitBlockWidth();
            assert (is_pow2(blockWidth));
            const auto blockSize = blockWidth / 8;

            ConstantInt * const BLOCK_WIDTH = b->getSize(blockWidth);
            Constant * const CHUNK_SIZE = b->getTypeSize(mType);

            FixedArray<Value *, 2> indices;
            indices[0] = b->getInt32(0);

            Value * const consumedChunks = b->CreateUDiv(consumed, BLOCK_WIDTH);
            Value * const producedChunks = b->CreateCeilUDiv(produced, BLOCK_WIDTH);
            Value * const unconsumedChunks = b->CreateSub(producedChunks, consumedChunks);

            StructType * const handleTy = getHandleType(b);
            IntegerType * const sizeTy = b->getSizeTy();

            indices[1] = b->getInt32(BaseAddress);
            Value * const virtualBaseField = b->CreateInBoundsGEP(handleTy, handle, indices);
            Value * const virtualBase = b->CreateLoad(getPointerType(), virtualBaseField);

            indices[1] = b->getInt32(MallocedAddress);
            Value * const mallocAddrField = b->CreateInBoundsGEP(handleTy, handle, indices);
            Value * const mallocAddress = b->CreateLoad(getPointerType(), mallocAddrField);
            Value * const bytesToCopy = b->CreateMul(unconsumedChunks, CHUNK_SIZE);
            Value * const unreadDataPtr = b->CreateInBoundsGEP(mType, virtualBase, consumedChunks);

            indices[1] = b->getInt32(InternalCapacity);
            Value * const intCapacityField = b->CreateInBoundsGEP(handleTy, handle, indices);
            Value * const bufferCapacity = b->CreateLoad(sizeTy, intCapacityField);

            b->CreateMemCpy(mallocAddress, unreadDataPtr, bytesToCopy, blockSize);

            Value * const newBaseAddress = b->CreateGEP(mType, mallocAddress, b->CreateNeg(consumedChunks));
            b->CreateStore(newBaseAddress, virtualBaseField);

            indices[1] = b->getInt32(EffectiveCapacity);

            Value * const capacityField = b->CreateInBoundsGEP(handleTy, handle, indices);
            Value * const effectiveCapacity = b->CreateAdd(consumedChunks, bufferCapacity);
            b->CreateStore(effectiveCapacity, capacityField);
            b->CreateRetVoid();

            b->restoreIP(ip);
            setHandle(myHandle);
        }

        b->CreateCall(funcTy, func, { myHandle, produced, consumed, b->getSize(mUnderflow), b->getSize(mOverflow) });
    }

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        Value * const writable = getLinearlyWritableItems(b, produced, consumed, b->getSize(mOverflow * b->getBitBlockWidth()));
        b->CreateAssert(b->CreateICmpULE(required, writable),
                        "Static buffer does not have sufficient capacity "
                        "(%" PRId64 ") for required items (%" PRId64 ")",
                        writable, required);
    }

    return nullptr;
}

// Dynamic Buffer

StructType * DynamicBuffer::getHandleType(BuilderPtr b) const {
    if (mHandleType == nullptr) {
        auto & C = b->getContext();
        PointerType * const typePtr = getPointerType();
        IntegerType * const sizeTy = b->getSizeTy();
        FixedArray<Type *, 5> types;

        types[BaseAddress] = typePtr;
        types[InternalCapacity] = sizeTy;

        if (mLinear) {
            types[MallocedAddress] = typePtr;
            types[EffectiveCapacity] = sizeTy;
            types[InitialConsumedCount] = sizeTy;
        } else {
            Type * const emptyTy = StructType::get(C);
            types[MallocedAddress] = emptyTy;
            types[EffectiveCapacity] = emptyTy;
            types[InitialConsumedCount] = emptyTy;
        }

        mHandleType = StructType::get(C, types);
    }
    return mHandleType;
}

void DynamicBuffer::allocateBuffer(BuilderPtr b, Value * const capacityMultiplier) {
    assert (mHandle && "has not been set prior to calling allocateBuffer");
    // note: when adding extensible stream sets, make sure to set the initial count here.
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);

    Value * const handle = getHandle();
    Value * capacity = b->CreateMul(capacityMultiplier, b->getSize(mInitialCapacity));

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b->CreateAssert(capacity, "Dynamic buffer capacity cannot be 0.");
    }

    indices[1] = b->getInt32(BaseAddress);
    assert (mHandleType->getElementType(BaseAddress)->isPointerTy());
    Value * const baseAddressField = b->CreateInBoundsGEP(mHandleType, handle, indices);

    indices[1] = b->getInt32(InternalCapacity);
    assert (mHandleType->getElementType(InternalCapacity)->isIntegerTy());
    Value * const capacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);

    #ifdef TEST_MMAPPED_CIRCULAR_BUFFERS
    if (mLinear) {

        Value * size = b->CreateAdd(capacity, b->getSize(mUnderflow + mOverflow));
        Value * baseAddress = b->CreatePageAlignedMalloc(mType, size, mAddressSpace);
        Value * const adjBaseAddress = addUnderflow(b, baseAddress, mUnderflow);
        b->CreateStore(adjBaseAddress, baseAddressField);
        b->CreateStore(capacity, capacityField);

        indices[1] = b->getInt32(MallocedAddress);
        Value * const initialField = b->CreateInBoundsGEP(mHandleType, handle, indices);
        b->CreateStore(adjBaseAddress, initialField);

        indices[1] = b->getInt32(EffectiveCapacity);
        Value * const effCapacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
        b->CreateStore(capacity, effCapacityField);

        indices[1] = b->getInt32(InitialConsumedCount);
        Value * const reqSegNoField = b->CreateInBoundsGEP(mHandleType, handle, indices);
        b->CreateStore(b->getSize(0), reqSegNoField);

    } else {

        ConstantInt * typeSize = b->getTypeSize(mType);

        Module * m = b->getModule();
        Rational stepSize{getpagesize(), typeSize->getLimitedValue()};
        capacity = b->CreateRoundUp(capacity, b->getSize(stepSize.numerator()));
        Value * capacityBytes = b->CreateMul(typeSize, capacity);

        FixedArray<Value *, 3> makeArgs;
        makeArgs[0] = b->getSize(mUnderflow);
        makeArgs[1] = capacityBytes;
        makeArgs[2] = b->getSize(mUnderflow);
        Function * makeBuffer = m->getFunction(__MAKE_CIRCULAR_BUFFER); assert (makeBuffer);
        Value * base = b->CreateCall(makeBuffer, makeArgs);
        b->CreateStore(base, baseAddressField);
        b->CreateStore(capacity, capacityField);

    }

    #else

    Value * size = b->CreateAdd(capacity, b->getSize(mUnderflow + mOverflow));
    Value * baseAddress = b->CreatePageAlignedMalloc(mType, size, mAddressSpace);
    Value * const adjBaseAddress = addUnderflow(b, baseAddress, mUnderflow);
    b->CreateStore(adjBaseAddress, baseAddressField);

    indices[1] = b->getInt32(InternalCapacity);
    Value * const capacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    b->CreateStore(capacity, capacityField);

    if (mLinear) {
        indices[1] = b->getInt32(MallocedAddress);
        Value * const initialField = b->CreateInBoundsGEP(mHandleType, handle, indices);
        b->CreateStore(adjBaseAddress, initialField);

        indices[1] = b->getInt32(EffectiveCapacity);
        Value * const effCapacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
        b->CreateStore(capacity, effCapacityField);

        indices[1] = b->getInt32(InitialConsumedCount);
        Value * const reqSegNoField = b->CreateInBoundsGEP(mHandleType, handle, indices);
        b->CreateStore(b->getSize(0), reqSegNoField);
    }

    #endif

}

void DynamicBuffer::releaseBuffer(BuilderPtr b) const {
    /* Free the dynamically allocated buffer(s). */
    Value * const handle = getHandle();
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(mLinear ? MallocedAddress : BaseAddress);
    Value * const baseAddressField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const baseAddress = subtractUnderflow(b, b->CreateLoad(getPointerType(), baseAddressField), mUnderflow);

    #ifdef TEST_MMAPPED_CIRCULAR_BUFFERS
    if (mLinear) {
        b->CreateFree(baseAddress);
    } else {
        indices[1] = b->getInt32(InternalCapacity);
        Value * const capacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
        Value * const capacity = b->CreateLoad(b->getSizeTy(), capacityField);
        destroyBuffer(b, baseAddress, capacity);
    }
    #else
    b->CreateFree(baseAddress);
    #endif
    b->CreateStore(ConstantPointerNull::get(getPointerType()), baseAddressField);
}

void DynamicBuffer::destroyBuffer(BuilderPtr b, Value * baseAddress, Value * capacity) const {
    if (mLinear) {
        b->CreateFree(baseAddress);
    } else {
        FixedArray<Value *, 4> destroyArgs;
        destroyArgs[0] = baseAddress;
        destroyArgs[1] = b->getSize(mUnderflow);
        destroyArgs[2] = b->CreateMul(b->getTypeSize(mType), capacity);
        destroyArgs[3] = b->getSize(mUnderflow);
        b->CreateCall(b->getModule()->getFunction(__DESTROY_CIRCULAR_BUFFER), destroyArgs);
    }
}

void DynamicBuffer::setBaseAddress(BuilderPtr /* b */, Value * /* addr */) const {
    unsupported("setBaseAddress", "Dynamic");
}

Value * DynamicBuffer::getBaseAddress(BuilderPtr b) const {
    assert (getHandle());
    Value * const ptr = b->CreateInBoundsGEP(mHandleType, getHandle(), {b->getInt32(0), b->getInt32(BaseAddress)});
    return b->CreateLoad(getPointerType(), ptr);
}

Value * DynamicBuffer::getMallocAddress(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(mLinear ? MallocedAddress : BaseAddress);
    return b->CreateLoad(getPointerType(), b->CreateInBoundsGEP(mHandleType, getHandle(), indices));
}

Value * DynamicBuffer::getOverflowAddress(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(mLinear ? MallocedAddress : BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    Value * const base = b->CreateLoad(getPointerType(), b->CreateInBoundsGEP(mHandleType, handle, indices));
    indices[1] = b->getInt32(mLinear ? EffectiveCapacity : InternalCapacity);
    Value * const capacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const capacity = b->CreateLoad(b->getSizeTy(), capacityField);
    return b->CreateInBoundsGEP(mType, base, capacity);
}

Value * DynamicBuffer::modByCapacity(BuilderPtr b, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    if (mLinear || isCapacityGuaranteed(offset, mInitialCapacity)) {
        return offset;
    } else {
        assert (getHandle());
        FixedArray<Value *, 2> indices;
        indices[0] = b->getInt32(0);
        indices[1] = b->getInt32(InternalCapacity);
        Value * const capacityPtr = b->CreateInBoundsGEP(mHandleType, getHandle(), indices);
        Value * const capacity = b->CreateLoad(b->getSizeTy(), capacityPtr);
        assert (capacity->getType()->isIntegerTy());
        return b->CreateURem(offset, capacity);
    }
}

Value * DynamicBuffer::getCapacity(BuilderPtr b) const {
    assert (getHandle());
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(mLinear ? EffectiveCapacity : InternalCapacity);
    Value * ptr = b->CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
    Value * const capacity = b->CreateLoad(b->getSizeTy(), ptr);
    assert (capacity->getType()->isIntegerTy());
    return b->CreateMul(capacity, BLOCK_WIDTH, "capacity");
}

Value * DynamicBuffer::getInternalCapacity(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(InternalCapacity);
    Value * const intCapacityField = b->CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
    Value * const capacity = b->CreateLoad(b->getSizeTy(), intCapacityField);
    assert (capacity->getType()->isIntegerTy());
    return b->CreateMul(capacity, BLOCK_WIDTH);
}

void DynamicBuffer::setCapacity(BuilderPtr /* b */, Value * /* capacity */) const {
    unsupported("setCapacity", "Dynamic");
}

Value * DynamicBuffer::getLinearlyAccessibleItems(BuilderPtr b, Value * const processedItems, Value * const totalItems, Value * const overflowItems) const {
    #ifdef TEST_MMAPPED_CIRCULAR_BUFFERS
    return b->CreateSub(totalItems, processedItems);
    #else
    return InternalBuffer::getLinearlyAccessibleItems(b, processedItems, totalItems, overflowItems);
    #endif
}

Value * DynamicBuffer::getLinearlyWritableItems(BuilderPtr b, Value * const producedItems, Value * const consumedItems, Value * const overflowItems) const {
    if (mLinear) {

        FixedArray<Value *, 2> indices;
        indices[0] = b->getInt32(0);
        indices[1] = b->getInt32(InternalCapacity);
        IntegerType * const sizeTy = b->getSizeTy();
        Value * const intCapacityField = b->CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * const internalCapacity = b->CreateLoad(sizeTy, intCapacityField);
        indices[1] = b->getInt32(EffectiveCapacity);
        Value * const effCapacityField = b->CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * const effCapacity = b->CreateLoad(sizeTy, effCapacityField);
        indices[1] = b->getInt32(InitialConsumedCount);
        Value * const initConsumedField = b->CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * const initConsumed = b->CreateLoad(sizeTy, initConsumedField);
        ConstantInt * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
        Value * const consumedChunks = b->CreateUDiv(consumedItems, BLOCK_WIDTH);
        Value * const reclaimedSinceCopyBack = b->CreateSub(consumedChunks, initConsumed);
        Value * const reclaimedChunks = b->CreateAdd(effCapacity, reclaimedSinceCopyBack);
        Value * const reclaimed = b->CreateMul(reclaimedChunks, BLOCK_WIDTH);
        Value * const maxChunks = b->CreateAdd(initConsumed, internalCapacity);
        Value * maxCapacity = b->CreateMul(maxChunks, BLOCK_WIDTH);
        if (overflowItems) {
            maxCapacity = b->CreateAdd(maxCapacity, overflowItems);
        }
        Value * const actualCapacity = b->CreateUMin(reclaimed, maxCapacity);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            Value * const valid = b->CreateICmpULE(producedItems, actualCapacity);
            b->CreateAssert(valid, "DynamicBuffer produced item count (%" PRIu64 ") exceeds capacity (%" PRIu64 ").",
                            producedItems, actualCapacity);
        }
        return b->CreateSub(actualCapacity, producedItems);
     } else {
        Value * const capacity = getCapacity(b);
        Value * const unconsumedItems = b->CreateSub(producedItems, consumedItems);
        #ifdef TEST_MMAPPED_CIRCULAR_BUFFERS
        Value * const remaining = b->CreateSub(capacity, unconsumedItems);
        Value * const retVal = b->CreateSaturatingSub(capacity, unconsumedItems);

        return retVal;
        #else
        Value * const full = b->CreateICmpULT(capacity, unconsumedItems);
        ConstantInt * const ZERO = b->getSize(0);
        Value * const fromOffset = b->CreateURem(producedItems, capacity);
        Value * const consumedOffset = b->CreateURem(consumedItems, capacity);
        Value * const toEnd = b->CreateICmpULE(consumedOffset, fromOffset);
        Value * const capacityWithOverflow = addOverflow(b, capacity, overflowItems, consumedOffset);
        Value * const limit = b->CreateSelect(toEnd, capacityWithOverflow, consumedOffset);
        Value * const remaining = b->CreateSub(limit, fromOffset);
        return b->CreateSelect(full, ZERO, remaining);
        #endif

    }
}

Value * DynamicBuffer::requiresExpansion(BuilderPtr b, Value * produced, Value * consumed, Value * required) const {

    if (mLinear) {

        const auto blockWidth = b->getBitBlockWidth();
        assert (is_pow2(blockWidth));

        FixedArray<Value *, 2> indices;
        indices[0] = b->getInt32(0);

        ConstantInt * const BLOCK_WIDTH = b->getSize(blockWidth);

        Value * const consumedChunks = b->CreateUDiv(consumed, BLOCK_WIDTH);

        indices[1] = b->getInt32(BaseAddress);
        Value * const virtualBaseField = b->CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * const virtualBase = b->CreateLoad(getPointerType(), virtualBaseField);
        Value * startOfUsedBuffer = b->CreateInBoundsGEP(mType, virtualBase, consumedChunks);
        DataLayout DL(b->getModule());
        Type * const intPtrTy = DL.getIntPtrType(virtualBase->getType());
        startOfUsedBuffer = b->CreatePtrToInt(startOfUsedBuffer, intPtrTy);

        indices[1] = b->getInt32(MallocedAddress);
        Value * const mallocedAddressField = b->CreateInBoundsGEP(mHandleType, mHandle, indices);
        Value * const mallocedAddress = b->CreateLoad(getPointerType(), mallocedAddressField);
        Value * const newPos = b->CreateAdd(produced, required);
        Value * const newChunks = b->CreateCeilUDiv(newPos, BLOCK_WIDTH);
        Value * const requiredChunks = b->CreateSub(newChunks, consumedChunks);
        Value * requiresUpToPosition = b->CreateInBoundsGEP(mType, mallocedAddress, requiredChunks);
        requiresUpToPosition = b->CreatePtrToInt(requiresUpToPosition, intPtrTy);

        return b->CreateICmpUGT(requiresUpToPosition, startOfUsedBuffer);

    } else { // Circular
        return b->getTrue();
    }

}

void DynamicBuffer::linearCopyBack(BuilderPtr b, Value * produced, Value * consumed, Value * required) const {

    if (mLinear) {

        SmallVector<char, 200> buf;
        raw_svector_ostream name(buf);

        assert ("unspecified module" && b.get() && b->getModule());

        name << "__DynamicBuffer_linearCopyBack";

        Type * ty = getBaseType();
        const auto streamCount = ty->getArrayNumElements();
        name << streamCount << 'x';
        ty = ty->getArrayElementType();
        ty = cast<FixedVectorType>(ty)->getElementType();
        const auto itemWidth = ty->getIntegerBitWidth();
        name << itemWidth << '@' << mAddressSpace;


        Value * const myHandle = getHandle();

        Module * const m = b->getModule();

        Function * func = m->getFunction(name.str());
        if (func == nullptr) {

            IntegerType * const sizeTy = b->getSizeTy();
            FixedArray<Type *, 3> params;
            params[0] = myHandle->getType();
            params[1] = sizeTy;
            params[2] = sizeTy;

            FunctionType * funcTy = FunctionType::get(b->getVoidTy(), params, false);

            const auto ip = b->saveIP();

            LLVMContext & C = m->getContext();
            func = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
            if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
                #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
                func->setHasUWTable();
                #else
                func->setUWTableKind(UWTableKind::Default);
                #endif
            }
            b->SetInsertPoint(BasicBlock::Create(C, "entry", func));

            auto arg = func->arg_begin();
            auto nextArg = [&]() {
                assert (arg != func->arg_end());
                Value * const v = &*arg;
                std::advance(arg, 1);
                return v;
            };

            Value * const handle = nextArg();
            handle->setName("handle");
            Value * const produced = nextArg();
            produced->setName("produced");
            Value * const consumed = nextArg();
            consumed->setName("consumed");
//            Value * const requiredSegNoBeforeAnotherCopyBack = nextArg();
//            requiredSegNoBeforeAnotherCopyBack->setName("requiredSegNoBeforeAnotherCopyBack");
            assert (arg == func->arg_end());

            setHandle(handle);

            const auto blockWidth = b->getBitBlockWidth();
            assert (is_pow2(blockWidth));
            const auto blockSize = blockWidth / 8;
            const auto sizeTyWidth = sizeTy->getBitWidth() / 8;

            ConstantInt * const BLOCK_WIDTH = b->getSize(blockWidth);
            Constant * const CHUNK_SIZE = b->getTypeSize(mType);

            FixedArray<Value *, 2> indices;
            indices[0] = b->getInt32(0);

            Value * const consumedChunks = b->CreateUDiv(consumed, BLOCK_WIDTH);
            Value * const producedChunks = b->CreateCeilUDiv(produced, BLOCK_WIDTH);

            Value * const unconsumedChunks = b->CreateSub(producedChunks, consumedChunks);
            Value * const bytesToCopy = b->CreateMul(unconsumedChunks, CHUNK_SIZE);

            indices[1] = b->getInt32(BaseAddress);
            Value * const virtualBaseField = b->CreateInBoundsGEP(mHandleType, handle, indices);
            Value * const virtualBase = b->CreateLoad(getPointerType(), virtualBaseField);

            indices[1] = b->getInt32(MallocedAddress);
            Value * const mallocedAddressField = b->CreateInBoundsGEP(mHandleType, handle, indices);
            Value * const mallocedAddress = b->CreateAlignedLoad(getPointerType(), mallocedAddressField, sizeTyWidth);

            Value * const unreadDataPtr = b->CreateInBoundsGEP(mType, virtualBase, consumedChunks);
            b->CreateMemCpy(mallocedAddress, unreadDataPtr, bytesToCopy, blockSize);
            Value * const newVirtualAddress = b->CreateGEP(mType, mallocedAddress, b->CreateNeg(consumedChunks));
            b->CreateAlignedStore(newVirtualAddress, virtualBaseField, sizeTyWidth);
            indices[1] = b->getInt32(InternalCapacity);
            Value * const intCapacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
            Value * const internalCapacity = b->CreateAlignedLoad(b->getSizeTy(), intCapacityField, sizeTyWidth);
            Value * const effectiveCapacity = b->CreateSub(b->CreateAdd(consumedChunks, internalCapacity), unconsumedChunks);
            indices[1] = b->getInt32(EffectiveCapacity);
            Value * const effCapacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
            b->CreateAlignedStore(effectiveCapacity, effCapacityField, sizeTyWidth);
            indices[1] = b->getInt32(InitialConsumedCount);
            Value * const initialConsumedField = b->CreateInBoundsGEP(mHandleType, handle, indices);
            b->CreateAlignedStore(consumedChunks, initialConsumedField, sizeTyWidth);
            b->CreateRetVoid();

            b->restoreIP(ip);
            setHandle(myHandle);
        }

        FixedArray<Value *, 3> args;
        args[0] = myHandle;
        args[1] = produced;
        args[2] = consumed;
        b->CreateCall(func->getFunctionType(), func, args);

    }
}


Value * DynamicBuffer::expandBuffer(BuilderPtr b, Value * const produced, Value * const consumed, Value * const required) const {

    SmallVector<char, 200> buf;
    raw_svector_ostream name(buf);

    assert ("unspecified module" && b.get() && b->getModule());

    name << "__DynamicBuffer_";
    if (mLinear) {
        name << "linear";
    } else {
        name << "circular";
    }
    name << "Expand_";

    Type * ty = getBaseType();
    const auto streamCount = ty->getArrayNumElements();
    name << streamCount << 'x';
    ty = ty->getArrayElementType();
    ty = cast<FixedVectorType>(ty)->getElementType();
    const auto itemWidth = ty->getIntegerBitWidth();
    name << itemWidth << '@' << mAddressSpace;

    Value * const myHandle = getHandle();

    Module * const m = b->getModule();

    Function * func = m->getFunction(name.str());
    if (func == nullptr) {

        IntegerType * const sizeTy = b->getSizeTy();
        PointerType * const voidPtrTy = b->getVoidPtrTy();

        FixedArray<Type *, 2> retTypes;
        retTypes[0] = voidPtrTy;
        retTypes[1] = sizeTy;
        StructType * retValTy = StructType::create(b->getContext(), retTypes);

        FunctionType * funcTy = FunctionType::get(retValTy, {myHandle->getType(), sizeTy, sizeTy, sizeTy, sizeTy, sizeTy}, false);

        const auto ip = b->saveIP();

        LLVMContext & C = m->getContext();
        func = Function::Create(funcTy, Function::InternalLinkage, name.str(), m);
        if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
            #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
            func->setHasUWTable();
            #else
            func->setUWTableKind(UWTableKind::Default);
            #endif
        }
        b->SetInsertPoint(BasicBlock::Create(C, "entry", func));

        auto arg = func->arg_begin();
        auto nextArg = [&]() {
            assert (arg != func->arg_end());
            Value * const v = &*arg;
            std::advance(arg, 1);
            return v;
        };

        Value * const handle = nextArg();
        handle->setName("handle");
        Value * const produced = nextArg();
        produced->setName("produced");
        Value * const consumed = nextArg();
        consumed->setName("consumed");
        Value * const required = nextArg();
        required->setName("required");
        Value * const underflow = nextArg();
        underflow->setName("underflow");
        Value * const overflow = nextArg();
        overflow->setName("overflow");
        assert (arg == func->arg_end());

        setHandle(handle);


        const auto blockWidth = b->getBitBlockWidth();
        assert (is_pow2(blockWidth));
        const auto blockSize = blockWidth / 8;

        ConstantInt * const BLOCK_WIDTH = b->getSize(blockWidth);
        Constant * const typeSize = b->getTypeSize(mType);

        FixedArray<Value *, 2> indices;
        indices[0] = b->getInt32(0);


        Value * const consumedChunks = b->CreateUDiv(consumed, BLOCK_WIDTH);
        Value * const producedChunks = b->CreateCeilUDiv(produced, BLOCK_WIDTH);
        Value * const requiredCapacity = b->CreateAdd(produced, required);
        Value * const requiredChunks = b->CreateCeilUDiv(requiredCapacity, BLOCK_WIDTH);
\
        Value * const unconsumedChunks = b->CreateSub(producedChunks, consumedChunks);
        Value * const bytesToCopy = b->CreateMul(unconsumedChunks, typeSize);

        indices[1] = b->getInt32(BaseAddress);

        StructType * handleTy = getHandleType(b);

        Value * const virtualBaseField = b->CreateInBoundsGEP(handleTy, handle, indices);
        Value * const virtualBase = b->CreateLoad(getPointerType(), virtualBaseField);

        DataLayout DL(b->getModule());
        Type * const intPtrTy = DL.getIntPtrType(virtualBase->getType());

        const auto sizeTyWidth = sizeTy->getBitWidth() / 8;

        indices[1] = b->getInt32(InternalCapacity);
        Value * const intCapacityField = b->CreateInBoundsGEP(handleTy, handle, indices);
        Value * const internalCapacity = b->CreateAlignedLoad(sizeTy, intCapacityField, sizeTyWidth);

        Value * const chunksToReserve = b->CreateSub(requiredChunks, consumedChunks);
        // newInternalCapacity tends to be 2x internalCapacity
        Value * const reserveCapacity = b->CreateAdd(chunksToReserve, internalCapacity);
        Value * const newInternalCapacity = b->CreateRoundUp(reserveCapacity, internalCapacity);

        Value * retVal = nullptr;

        if (mLinear) {


            Value * const additionalCapacity = b->CreateAdd(underflow, overflow);
            Value * const mallocCapacity = b->CreateAdd(newInternalCapacity, additionalCapacity);
            Value * const mallocSize = b->CreateMul(mallocCapacity, typeSize);
            Value * expandedBuffer = b->CreatePointerCast(b->CreatePageAlignedMalloc(mallocSize), getPointerType());
            expandedBuffer = b->CreateInBoundsGEP(mType, expandedBuffer, underflow);

            Value * const unreadDataPtr = b->CreateInBoundsGEP(mType, virtualBase, consumedChunks);
            b->CreateMemCpy(expandedBuffer, unreadDataPtr, bytesToCopy, blockSize);

            b->CreateAlignedStore(newInternalCapacity, intCapacityField, sizeTyWidth);

            indices[1] = b->getInt32(MallocedAddress);
            Value * const mallocedAddressField = b->CreateInBoundsGEP(handleTy, handle, indices);
            Value * const mallocedAddress = b->CreateAlignedLoad(getPointerType(), mallocedAddressField, sizeTyWidth);

            b->CreateAlignedStore(expandedBuffer, mallocedAddressField, sizeTyWidth);

            Value * const effectiveCapacity = b->CreateAdd(consumedChunks, newInternalCapacity);
            indices[1] = b->getInt32(EffectiveCapacity);
            Value * const effCapacityField = b->CreateInBoundsGEP(handleTy, handle, indices);
            b->CreateAlignedStore(effectiveCapacity, effCapacityField, sizeTyWidth);
            Value * const newVirtualAddress = b->CreateGEP(mType, expandedBuffer, b->CreateNeg(consumedChunks));
            b->CreateAlignedStore(newVirtualAddress, virtualBaseField, sizeTyWidth);

            indices[1] = b->getInt32(InitialConsumedCount);
            Value * const initConsumedField = b->CreateInBoundsGEP(handleTy, handle, indices);
            b->CreateAlignedStore(consumedChunks, initConsumedField, sizeTyWidth);

            retVal = mallocedAddress;


            if (underflow) {
                retVal = b->CreateInBoundsGEP(mType, retVal, b->CreateNeg(underflow));
            }

        } else { // Circular

            indices[1] = b->getInt32(InternalCapacity);

            b->CreateAlignedStore(newInternalCapacity, intCapacityField, sizeTyWidth);

            #ifdef TEST_MMAPPED_CIRCULAR_BUFFERS

            Module * const m = b->getModule();

            FixedArray<Value *, 3> makeArgs;
            makeArgs[0] = b->getSize(mUnderflow);
            makeArgs[1] = b->CreateMul(newInternalCapacity, typeSize);
            makeArgs[2] = b->getSize(mUnderflow);
            Value * const newBuffer = b->CreateCall(m->getFunction(__MAKE_CIRCULAR_BUFFER), makeArgs);

            Value * newBufferOffset = b->CreateURem(consumedChunks, newInternalCapacity);
            Value * const toWriteDataPtr = b->CreateInBoundsGEP(mType, newBuffer, newBufferOffset);
            b->CreateAlignedStore(newBuffer, virtualBaseField, sizeTyWidth);

            Value * const oldBufferOffset = b->CreateURem(consumedChunks, internalCapacity);
            Value * const unreadDataPtr = b->CreateInBoundsGEP(mType, virtualBase, oldBufferOffset);

            b->CreateMemCpy(toWriteDataPtr, unreadDataPtr, bytesToCopy, blockSize);

            retVal = virtualBase;

            #else

            b->CreateAlignedStore(newCapacity, intCapacityField, sizeTyWidth);

            Value * const additionalCapacity = b->CreateAdd(underflow, overflow);
            Value * const requiredCapacity = b->CreateAdd(newCapacity, additionalCapacity);

            Value * const mallocSize = b->CreateMul(requiredCapacity, CHUNK_SIZE);
            Value * newBuffer = b->CreatePointerCast(b->CreatePageAlignedMalloc(mallocSize), getPointerType());
            newBuffer = b->CreateInBoundsGEP(mType, newBuffer, underflow);

            Value * const consumedOffset = b->CreateURem(consumedChunks, internalCapacity);
            Value * const producedOffset = b->CreateURem(producedChunks, internalCapacity);
            Value * const newConsumedOffset = b->CreateURem(consumedChunks, newCapacity);
            Value * const newProducedOffset = b->CreateURem(producedChunks, newCapacity);
            Value * const consumedOffsetEnd = b->CreateAdd(consumedOffset, unconsumedChunks);
            Value * const sourceLinear = b->CreateICmpULE(consumedOffsetEnd, producedOffset);
            Value * const newConsumedOffsetEnd = b->CreateAdd(newConsumedOffset, unconsumedChunks);
            Value * const targetLinear = b->CreateICmpULE(newConsumedOffsetEnd, newProducedOffset);
            Value * const linearCopy = b->CreateAnd(sourceLinear, targetLinear);

            Value * const consumedOffsetPtr = b->CreateInBoundsGEP(mType, virtualBase, consumedOffset);
            Value * const newConsumedOffsetPtr = b->CreateInBoundsGEP(mType, newBuffer, newConsumedOffset);

            BasicBlock * const copyLinear = BasicBlock::Create(C, "copyLinear", func);
            BasicBlock * const copyNonLinear = BasicBlock::Create(C, "copyNonLinear", func);
            BasicBlock * const storeNewBuffer = BasicBlock::Create(C, "storeNewBuffer", func);
            b->CreateCondBr(linearCopy, copyLinear, copyNonLinear);

            b->SetInsertPoint(copyLinear);
            b->CreateMemCpy(newConsumedOffsetPtr, consumedOffsetPtr, bytesToCopy, blockSize);
            b->CreateBr(storeNewBuffer);

            b->SetInsertPoint(copyNonLinear);
            Value * const bufferLength1 = b->CreateSub(internalCapacity, consumedOffset);
            Value * const newBufferLength1 = b->CreateSub(newCapacity, newConsumedOffset);
            Value * const partialLength1 = b->CreateUMin(bufferLength1, newBufferLength1);
            Value * const copyEndPtr = b->CreateInBoundsGEP(mType, virtualBase, b->CreateAdd(consumedOffset, partialLength1));
            Value * const copyEndPtrInt = b->CreatePtrToInt(copyEndPtr, intPtrTy);
            Value * const consumedOffsetPtrInt = b->CreatePtrToInt(consumedOffsetPtr, intPtrTy);
            Value * const bytesToCopy1 = b->CreateSub(copyEndPtrInt, consumedOffsetPtrInt);
            b->CreateMemCpy(newConsumedOffsetPtr, consumedOffsetPtr, bytesToCopy1, blockSize);
            Value * const sourceOffset = b->CreateURem(b->CreateAdd(consumedOffset, partialLength1), internalCapacity);
            Value * const sourcePtr = b->CreateInBoundsGEP(mType, virtualBase, sourceOffset);
            Value * const targetOffset = b->CreateURem(b->CreateAdd(newConsumedOffset, partialLength1), newCapacity);
            Value * const targetPtr = b->CreateInBoundsGEP(mType, newBuffer, targetOffset);
            Value * const bytesToCopy2 = b->CreateSub(bytesToCopy, bytesToCopy1);
            b->CreateMemCpy(targetPtr, sourcePtr, bytesToCopy2, blockSize);
            b->CreateBr(storeNewBuffer);

            b->SetInsertPoint(storeNewBuffer);
            b->CreateStore(newBuffer, virtualBaseField);
            b->CreateAlignedStore(newCapacity, intCapacityField, sizeTyWidth);

            retVal = virtualBase;


            if (underflow) {
                retVal = b->CreateInBoundsGEP(mType, retVal, b->CreateNeg(underflow));
            }

            #endif


        }

        FixedArray<Value *, 2> retVals;
        retVals[0] = b->CreatePointerCast(retVal, voidPtrTy);
        retVals[1] = internalCapacity;

        b->CreateAggregateRet(retVals.data(), 2);

        b->restoreIP(ip);
        setHandle(myHandle);
    }

    FixedArray<Value *, 6> args;
    args[0] = myHandle;
    args[1] = produced;
    args[2] = consumed;
    args[3] = required;
    args[4] = b->getSize(mUnderflow);
    args[5] = b->getSize(mOverflow);
    return b->CreateCall(func->getFunctionType(), func, args);
}

// MMapped Buffer

StructType * MMapedBuffer::getHandleType(BuilderPtr b) const {
    if (mHandleType == nullptr) {
        auto & C = b->getContext();
        PointerType * const typePtr = getPointerType();
        IntegerType * const sizeTy = b->getSizeTy();
        FixedArray<Type *, 4> types;
        types[BaseAddress] = typePtr;
        types[Capacity] = sizeTy;
        types[Released] = sizeTy;
        types[Fd] = b->getInt32Ty();
        mHandleType = StructType::get(C, types);
    }
    return mHandleType;
}

void MMapedBuffer::allocateBuffer(BuilderPtr b, Value * const capacityMultiplier) {
    assert (mHandle && "has not been set prior to calling allocateBuffer");
    // note: when adding extensible stream sets, make sure to set the initial count here.
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);

    Value * const handle = getHandle();
    Value * capacity = b->CreateMul(capacityMultiplier, b->getSize(mInitialCapacity));

    if (LLVM_UNLIKELY(codegen::DebugOptionIsSet(codegen::EnableAsserts))) {
        b->CreateAssert(capacity, "Dynamic buffer capacity cannot be 0.");
    }

    indices[1] = b->getInt32(BaseAddress);
    Value * const baseAddressField = b->CreateInBoundsGEP(mHandleType, handle, indices);

    Constant * const typeSize = b->getTypeSize(mType);
    Value * const minCapacity = b->CreateCeilUDiv(b->getSize(ANON_MMAP_SIZE), typeSize);

    capacity = b->CreateUMax(capacity, minCapacity);

    Value * size = b->CreateAdd(capacity, b->getSize(mUnderflow + mOverflow));

    Value * const fileSize = b->CreateMul(typeSize, size);

    Value * const fd = b->CreateMemFdCreate(b->GetString("streamset"), b->getInt32(0));

    b->CreateFTruncate(fd, fileSize);

    PointerType * const voidPtrTy = b->getVoidPtrTy();
    ConstantInt * const prot =  b->getInt32(PROT_READ | PROT_WRITE);
    ConstantInt * const intflags =  b->getInt32(MAP_PRIVATE | MAP_NORESERVE);
    Constant * const sz_ZERO = b->getSize(0);
    Value * baseAddress = b->CreateMMap(ConstantPointerNull::getNullValue(voidPtrTy), fileSize, prot, intflags, fd, sz_ZERO);

    baseAddress = b->CreatePointerCast(baseAddress, mType->getPointerTo(mAddressSpace));

    Value * const adjBaseAddress = addUnderflow(b, baseAddress, mUnderflow);
    b->CreateStore(adjBaseAddress, baseAddressField);

    indices[1] = b->getInt32(Capacity);
    Value * const capacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    b->CreateStore(capacity, capacityField);

    indices[1] = b->getInt32(Fd);
    Value * const fdField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    b->CreateStore(fd, fdField);

}

void MMapedBuffer::releaseBuffer(BuilderPtr b) const {
    /* Free the dynamically allocated buffer(s). */
    Value * const handle = getHandle();
    Constant * const typeSize = b->getTypeSize(mType);
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(BaseAddress);
    Value * const baseAddressField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const baseAddress = subtractUnderflow(b, b->CreateLoad(getPointerType(), baseAddressField), mUnderflow);
    indices[1] = b->getInt32(Capacity);
    Value * const capacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const capacity = b->CreateLoad(b->getSizeTy(), capacityField);
    Value * const size = b->CreateAdd(capacity, b->getSize(mUnderflow + mOverflow));
    Value * const fileSize = b->CreateMul(typeSize, size);
    b->CreateMUnmap(baseAddress, fileSize);
    b->CreateStore(ConstantPointerNull::get(getPointerType()), baseAddressField);
    indices[1] = b->getInt32(Fd);
    Value * const fdField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    b->CreateCloseCall(b->CreateLoad(b->getInt32Ty(), fdField));
}

void MMapedBuffer::destroyBuffer(BuilderPtr b, Value * baseAddress, Value *capacity) const {

}

void MMapedBuffer::setBaseAddress(BuilderPtr /* b */, Value * /* addr */) const {
    unsupported("setBaseAddress", "MMaped");
}

Value * MMapedBuffer::getBaseAddress(BuilderPtr b) const {
    assert (getHandle());
    Value * const ptr = b->CreateInBoundsGEP(mHandleType, getHandle(), {b->getInt32(0), b->getInt32(BaseAddress)});
    return b->CreateLoad(getPointerType(), ptr);
}

Value * MMapedBuffer::getMallocAddress(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(BaseAddress);
    return b->CreateLoad(getPointerType(), b->CreateInBoundsGEP(mHandleType, getHandle(), indices));
}

Value * MMapedBuffer::getOverflowAddress(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    Value * const base = b->CreateLoad(getPointerType(), b->CreateInBoundsGEP(mHandleType, handle, indices));
    indices[1] = b->getInt32(Capacity);
    Value * const capacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const capacity = b->CreateLoad(b->getSizeTy(), capacityField);
    assert (capacity->getType() == b->getSizeTy());
    return b->CreateInBoundsGEP(getPointerType(), base, capacity);
}

Value * MMapedBuffer::modByCapacity(BuilderPtr b, Value * const offset) const {
    assert (offset->getType()->isIntegerTy());
    return offset;
}

Value * MMapedBuffer::getCapacity(BuilderPtr b) const {
    assert (getHandle());
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(Capacity);
    Value * ptr = b->CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
    Value * const capacity = b->CreateLoad(b->getSizeTy(), ptr);
    assert (capacity->getType()->isIntegerTy());
    return b->CreateMul(capacity, BLOCK_WIDTH, "capacity");
}

Value * MMapedBuffer::getInternalCapacity(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(Capacity);
    Value * const intCapacityField = b->CreateInBoundsGEP(mHandleType, getHandle(), indices);
    ConstantInt * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
    Value * const capacity = b->CreateLoad(b->getSizeTy(), intCapacityField);
    assert (capacity->getType()->isIntegerTy());
    return b->CreateMul(capacity, BLOCK_WIDTH);
}

void MMapedBuffer::setCapacity(BuilderPtr /* b */, Value * /* capacity */) const {
    unsupported("setCapacity", "MMaped");
}

Value * MMapedBuffer::requiresExpansion(BuilderPtr b, Value * produced, Value * consumed, Value * required) const {

    assert (mLinear);

    const auto blockWidth = b->getBitBlockWidth();
    assert (is_pow2(blockWidth));
    ConstantInt * const BLOCK_WIDTH = b->getSize(blockWidth);

    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(BaseAddress);
    Value * const virtualBaseField = b->CreateInBoundsGEP(mHandleType, mHandle, indices);
    Value * const virtualBase = b->CreateLoad(getPointerType(), virtualBaseField);
    Value * const consumedChunks = b->CreateUDiv(consumed, BLOCK_WIDTH);

    DataLayout DL(b->getModule());
    Type * const intPtrTy = DL.getIntPtrType(virtualBase->getType());
    Value * const virtualBaseInt = b->CreatePtrToInt(virtualBase, intPtrTy);
    Value * startOfUsedBuffer = b->CreatePtrToInt(b->CreateInBoundsGEP(mType, virtualBase, consumedChunks), intPtrTy);
    Value * unnecessaryBytes = b->CreateSub(startOfUsedBuffer, virtualBaseInt);
    unnecessaryBytes = b->CreateRoundDown(unnecessaryBytes, b->getSize(b->getPageSize()));
    // assume that we can always discard memory
    b->CreateMAdvise(virtualBase, unnecessaryBytes, MADV_DONTNEED);

    indices[1] = b->getInt32(Capacity);
    Value * const capacityField = b->CreateInBoundsGEP(mHandleType, mHandle, indices);
    Value * const capacity = b->CreateLoad(b->getSizeTy(), capacityField);

    return b->CreateICmpUGE(b->CreateAdd(produced, required), b->CreateMul(capacity, BLOCK_WIDTH));

}

void MMapedBuffer::linearCopyBack(BuilderPtr b, Value * produced, Value * consumed, Value * required) const {
    /* do nothing */
}


Value * MMapedBuffer::expandBuffer(BuilderPtr b, Value * const produced, Value * const consumed, Value * const required) const {

    Value * const handle = getHandle();

    Constant * const typeSize = b->getTypeSize(mType);
    Value * const expandStepSize = b->CreateCeilUDiv(b->getSize(ANON_MMAP_SIZE), typeSize);

    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(Capacity);
    Value * const capacityField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    const auto blockWidth = b->getBitBlockWidth();
    assert (is_pow2(blockWidth));
    ConstantInt * const BLOCK_WIDTH = b->getSize(blockWidth);

    Value * newCapacity = b->CreateCeilUDiv(b->CreateAdd(produced, required), BLOCK_WIDTH);
    if (mOverflow) {
        newCapacity = b->CreateAdd(newCapacity, b->getSize(mOverflow));
    }
    newCapacity = b->CreateRoundUp(newCapacity, expandStepSize);
    b->CreateStore(newCapacity, capacityField);

    indices[1] = b->getInt32(Fd);
    Value * const fdField = b->CreateInBoundsGEP(mHandleType, handle, indices);
    Value * const fd = b->CreateLoad(b->getInt32Ty(), fdField);
    b->CreateFTruncate(fd, b->CreateMul(newCapacity, typeSize));

    return nullptr;
}

// Repeating Buffer

StructType * RepeatingBuffer::getHandleType(BuilderPtr b) const {
    if (mHandleType == nullptr) {
        auto & C = b->getContext();
        FixedArray<Type *, 1> types;
        types[BaseAddress] = getPointerType();
        mHandleType = StructType::get(C, types);
    }
    return mHandleType;
}

void RepeatingBuffer::allocateBuffer(BuilderPtr b, Value * const capacityMultiplier) {
    unsupported("allocateBuffer", "Repeating");
}

void RepeatingBuffer::releaseBuffer(BuilderPtr b) const {
    unsupported("releaseBuffer", "Repeating");
}

void RepeatingBuffer::destroyBuffer(BuilderPtr b, Value * baseAddress, llvm::Value *capacity) const {

}

Value * RepeatingBuffer::modByCapacity(BuilderPtr b, Value * const offset) const {
    Value * const capacity = b->CreateExactUDiv(mModulus, b->getSize(b->getBitBlockWidth()));
    return b->CreateURem(offset, capacity);
}

Value * RepeatingBuffer::getCapacity(BuilderPtr b) const {
    return mModulus;
}

Value * RepeatingBuffer::getInternalCapacity(BuilderPtr b) const {
    return mModulus;
}

void RepeatingBuffer::setCapacity(BuilderPtr b, Value * capacity) const {
    unsupported("setCapacity", "Repeating");
}


Value * RepeatingBuffer::getVirtualBasePtr(BuilderPtr b, Value * const baseAddress, Value * const transferredItems) const {
    Value * addr = nullptr;
    Constant * const LOG_2_BLOCK_WIDTH = b->getSize(floor_log2(b->getBitBlockWidth()));
    if (mUnaligned) {
        assert (isConstantOne(getStreamSetCount(b)));
        Value * offset = b->CreateSub(transferredItems, b->CreateURem(transferredItems, mModulus));
        Type * const elemTy = cast<ArrayType>(mBaseType)->getElementType();
        Type * itemTy = cast<VectorType>(elemTy)->getElementType();
        #if LLVM_VERSION_CODE < LLVM_VERSION_CODE(12, 0, 0)
        const unsigned itemWidth = itemTy->getPrimitiveSizeInBits();
        #else
        const unsigned itemWidth = itemTy->getPrimitiveSizeInBits().getFixedSize();
        #endif
        PointerType * itemPtrTy = nullptr;
        if (LLVM_UNLIKELY(itemWidth < 8)) {
            const Rational itemsPerByte{8, itemWidth};
            offset = b->CreateUDivRational(offset, itemsPerByte);
            itemTy = b->getInt8Ty();
        }
        itemPtrTy = itemTy->getPointerTo(mAddressSpace);
        addr = b->CreatePointerCast(baseAddress, itemPtrTy);
        addr = b->CreateInBoundsGEP(itemTy, addr, b->CreateNeg(offset));
    } else {
        Value * const transferredBlocks = b->CreateLShr(transferredItems, LOG_2_BLOCK_WIDTH);
        Constant * const BLOCK_WIDTH = b->getSize(b->getBitBlockWidth());
        Value * const capacity = b->CreateExactUDiv(mModulus, BLOCK_WIDTH);
        Value * offset = b->CreateURem(transferredBlocks, capacity);
        offset = b->CreateSub(offset, transferredBlocks);
        Constant * const sz_ZERO = b->getSize(0);
        addr = StreamSetBuffer::getStreamBlockPtr(b, baseAddress, sz_ZERO, offset);
    }
    return b->CreatePointerCast(addr, getPointerType());
}

Value * RepeatingBuffer::getBaseAddress(BuilderPtr b) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    Value * const base = b->CreateInBoundsGEP(mHandleType, handle, indices);
    return b->CreateLoad(getPointerType(), base, "baseAddress");
}

void RepeatingBuffer::setBaseAddress(BuilderPtr b, Value * addr) const {
    FixedArray<Value *, 2> indices;
    indices[0] = b->getInt32(0);
    indices[1] = b->getInt32(BaseAddress);
    Value * const handle = getHandle(); assert (handle);
    b->CreateStore(addr, b->CreateInBoundsGEP(mHandleType, handle, indices));
}

Value * RepeatingBuffer::getMallocAddress(BuilderPtr b) const {
    return getBaseAddress(b);
}

Value * RepeatingBuffer::getOverflowAddress(BuilderPtr b) const {
    Value * const capacity = b->CreateExactUDiv(mModulus, b->getSize(b->getBitBlockWidth()));
    return b->CreateGEP(getPointerType(), getBaseAddress(b), capacity);
}

Value * RepeatingBuffer::requiresExpansion(BuilderPtr b, Value * produced, Value * consumed, Value * required) const {
    return b->getFalse();
}

void RepeatingBuffer::linearCopyBack(BuilderPtr b, Value * produced, Value * consumed, Value * required) const {
    unsupported("linearCopyBack", "Repeating");
}

Value * RepeatingBuffer::expandBuffer(BuilderPtr b, Value * produced, Value * consumed, Value * const required) const  {
    unsupported("linearCopyBack", "Repeating");
}

// Constructors

ExternalBuffer::ExternalBuffer(const unsigned id, BuilderPtr b, Type * const type,
                               const bool linear,
                               const unsigned AddressSpace)
: StreamSetBuffer(id, BufferKind::ExternalBuffer, b, type, 0, 0, linear, AddressSpace) {

}

StaticBuffer::StaticBuffer(const unsigned id, BuilderPtr b, Type * const type,
                           const size_t capacity, const size_t overflowSize, const size_t underflowSize,
                           const bool linear, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::StaticBuffer, b, type, overflowSize, underflowSize, linear, AddressSpace)
, mCapacity(capacity) {
}

DynamicBuffer::DynamicBuffer(const unsigned id, BuilderPtr b, Type * const type,
                             const size_t initialCapacity, const size_t overflowSize, const size_t underflowSize,
                             const bool linear, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::DynamicBuffer, b, type, overflowSize, underflowSize, linear, AddressSpace)
, mInitialCapacity(initialCapacity) {
}

MMapedBuffer::MMapedBuffer(const unsigned id, BuilderPtr b, Type * const type,
                             const size_t initialCapacity, const size_t overflowSize, const size_t underflowSize,
                             const bool linear, const unsigned AddressSpace)
: InternalBuffer(id, BufferKind::MMapedBuffer, b, type, overflowSize, underflowSize, linear, AddressSpace)
, mInitialCapacity(initialCapacity) {

}

RepeatingBuffer::RepeatingBuffer(const unsigned id, BuilderPtr b, Type * const type, const bool unaligned)
: InternalBuffer(id, BufferKind::RepeatingBuffer, b, type, 0, 0, false, 0)
, mUnaligned(unaligned) {

}


inline InternalBuffer::InternalBuffer(const unsigned id, const BufferKind k, BuilderPtr b, Type * const baseType,
                                      const size_t overflowSize, const size_t underflowSize,
                                      const bool linear, const unsigned AddressSpace)
: StreamSetBuffer(id, k, b, baseType, overflowSize, underflowSize, linear, AddressSpace) {

}

inline StreamSetBuffer::StreamSetBuffer(const unsigned id, const BufferKind k, BuilderPtr b, Type * const baseType,
                                        const size_t overflowSize, const size_t underflowSize,
                                        const bool linear, const unsigned AddressSpace)
: mId(id)
, mBufferKind(k)
, mHandle(nullptr)
, mType(resolveType(b, baseType))
, mBaseType(baseType)
, mHandleType(nullptr)
, mOverflow(overflowSize)
, mUnderflow(underflowSize)
, mAddressSpace(AddressSpace)
, mLinear(linear || isEmptySet()) {

}

StreamSetBuffer::~StreamSetBuffer() { }

}
