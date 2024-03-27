/*
 *  Copyright (c) 2016 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#ifndef STREAMSET_H
#define STREAMSET_H

#include <llvm/IR/Type.h>  // for Type
#include <llvm/IR/DerivedTypes.h>  // for Type
#include <kernel/core/ptrwrapper.hpp>

namespace IDISA { class IDISA_Builder; }
namespace llvm { class Value; }
namespace llvm { class Constant; }

namespace kernel {

class Kernel;
class PipelineKernel;
class KernelBuilder;

class StreamSetBuffer {
public:

    enum class BufferKind : unsigned {
        ExternalBuffer
        , RepeatingBuffer
        , StaticBuffer
        , DynamicBuffer
        , MMapedBuffer
    };

    using BuilderPtr = PtrWrapper<kernel::KernelBuilder>;

    using ScalarRef = std::pair<llvm::Value *, llvm::Type *>;

    BufferKind getBufferKind() const {
        return mBufferKind;
    }

    llvm::Type * getType() const {
        return mType;
    }

    llvm::Type * getBaseType() const {
        return mBaseType;
    }

    unsigned getAddressSpace() const {
        return mAddressSpace;
    }

    __attribute__((const)) llvm::PointerType * getPointerType()  const {
        return getType()->getPointerTo(getAddressSpace());
    }

    bool isLinear() const {
        return mLinear;
    }

    unsigned getId() const {
        return mId;
    }

    unsigned getFieldWidth() const;

    bool isEmptySet() const;

    bool isDynamic() const {
        return (mBufferKind == BufferKind::DynamicBuffer) || (mBufferKind == BufferKind::MMapedBuffer);
    }

    virtual ~StreamSetBuffer() = 0;

    llvm::Value * getHandle() const {
        return mHandle;
    }

    void setHandle(llvm::Value * const handle) const {
        mHandle = handle;
    }

    void setHandle(ScalarRef handle) const {
        mHandle = handle.first;
        assert (handle.second == mHandleType);
    }

    virtual void allocateBuffer(BuilderPtr b, llvm::Value * const capacityMultiplier) = 0;

    virtual void releaseBuffer(BuilderPtr b) const = 0;

    virtual void destroyBuffer(BuilderPtr b, llvm::Value * baseAddress, llvm::Value *capacity) const = 0;

    // The number of items that cam be linearly accessed from a given logical stream position.
    virtual llvm::Value * getLinearlyAccessibleItems(BuilderPtr b, llvm::Value * fromPosition, llvm::Value * totalItems) const = 0;

    virtual llvm::Value * getLinearlyWritableItems(BuilderPtr b, llvm::Value * fromPosition, llvm::Value * consumedItems) const = 0;

    virtual llvm::StructType * getHandleType(BuilderPtr b) const = 0;

    llvm::PointerType * getHandlePointerType(BuilderPtr b) const {
        return getHandleType(b)->getPointerTo(getAddressSpace());
    }

    virtual llvm::Value * getStreamBlockPtr(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex) const;

    virtual llvm::Value * getStreamPackPtr(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex, llvm::Value * packIndex) const;

    virtual llvm::Value * loadStreamBlock(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex, const bool unaligned) const;

    virtual llvm::Value * loadStreamPack(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex, llvm::Value * packIndex, const bool unaligned) const;

    virtual llvm::Value * getStreamSetCount(BuilderPtr b) const;

    virtual llvm::Value * getBaseAddress(BuilderPtr b) const = 0;

    virtual llvm::Value * getMallocAddress(BuilderPtr b) const = 0;

    virtual void setBaseAddress(BuilderPtr b, llvm::Value * addr) const = 0;

    virtual void setCapacity(BuilderPtr b, llvm::Value * size) const = 0;

    virtual llvm::Value * getCapacity(BuilderPtr b) const = 0;

    virtual llvm::Value * getInternalCapacity(BuilderPtr b) const = 0;

    virtual llvm::Value * modByCapacity(BuilderPtr b, llvm::Value * const offset) const = 0;

    virtual llvm::Value * getRawItemPointer(BuilderPtr b, llvm::Value * streamIndex, llvm::Value * absolutePosition) const;

    virtual llvm::Value * getVirtualBasePtr(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * const transferredItems) const = 0;

    virtual llvm::Value * requiresExpansion(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const = 0;

    virtual void linearCopyBack(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const = 0;

    virtual llvm::Value * expandBuffer(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const = 0;

    static llvm::Type * resolveType(BuilderPtr b, llvm::Type * const streamSetType);

    static void linkFunctions(BuilderPtr b); // temporary function

protected:

    StreamSetBuffer(const unsigned id, const BufferKind k, BuilderPtr b, llvm::Type * baseType, const bool linear, const unsigned AddressSpace);

private:

    void assertValidStreamIndex(BuilderPtr b, llvm::Value * streamIndex) const;

protected:

    const unsigned                  mId;
    const BufferKind                mBufferKind;
    // Each StreamSetBuffer object is local to the Kernel (or pipeline) object at (pre-JIT) "compile time" but
    // by sharing the same handle will refer to the same stream set at (post-JIT) run time.
    mutable llvm::Value *           mHandle;
    llvm::Type * const              mType;
    llvm::Type * const              mBaseType;
    mutable llvm::StructType *      mHandleType;
    const unsigned                  mAddressSpace;
    const bool                      mLinear;
};

class ExternalBuffer final : public StreamSetBuffer {
public:
    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() == BufferKind::ExternalBuffer;
    }

    enum Field { BaseAddress, EffectiveCapacity };

    ExternalBuffer(const unsigned id, BuilderPtr b, llvm::Type * const type, const bool linear, const unsigned AddressSpace);

    void allocateBuffer(BuilderPtr b, llvm::Value * const capacityMultiplier) override;

    void releaseBuffer(BuilderPtr b) const override;

    void destroyBuffer(BuilderPtr b, llvm::Value * baseAddress, llvm::Value *capacity) const override;

    llvm::Value * getVirtualBasePtr(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * const transferredItems) const override;

    llvm::Value * getLinearlyAccessibleItems(BuilderPtr b, llvm::Value * fromPosition, llvm::Value * totalItems) const override;

    llvm::Value * getLinearlyWritableItems(BuilderPtr b, llvm::Value * fromPosition, llvm::Value * consumedItems) const override;

    llvm::StructType * getHandleType(BuilderPtr b) const override;

    llvm::Value * getBaseAddress(BuilderPtr b) const override;

    llvm::Value * getMallocAddress(BuilderPtr b) const override;

    void setCapacity(BuilderPtr b, llvm::Value * capacity) const override;

    llvm::Value * getCapacity(BuilderPtr b) const override;

    llvm::Value * getInternalCapacity(BuilderPtr b) const override;

    llvm::Value * modByCapacity(BuilderPtr b, llvm::Value * const offset) const override;

    llvm::Value * requiresExpansion(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void linearCopyBack(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    llvm::Value * expandBuffer(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void setBaseAddress(BuilderPtr b, llvm::Value * addr) const override;

private:

    void assertValidBlockIndex(BuilderPtr b, llvm::Value * blockIndex) const;

};

class InternalBuffer : public StreamSetBuffer {
public:

    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() != BufferKind::ExternalBuffer;
    }

    llvm::Value * getStreamBlockPtr(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex) const final;

    llvm::Value * getStreamPackPtr(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * streamIndex, llvm::Value * blockIndex, llvm::Value * packIndex) const final;

    llvm::Value * getVirtualBasePtr(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * const transferredItems) const override;

    llvm::Value * getLinearlyAccessibleItems(BuilderPtr b, llvm::Value * fromPosition, llvm::Value * const totalItems) const override;

    llvm::Value * getLinearlyWritableItems(BuilderPtr b, llvm::Value * fromPosition, llvm::Value * consumedItems) const override;

protected:

    InternalBuffer(const unsigned id, const BufferKind k, BuilderPtr b, llvm::Type * baseType,
                   const bool linear, const unsigned AddressSpace);


};

class StaticBuffer final : public InternalBuffer {
public:
    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() == BufferKind::StaticBuffer;
    }

    StaticBuffer(const unsigned id, BuilderPtr b, llvm::Type * const type,
                 const size_t capacity,
                 const bool linear, const unsigned AddressSpace);

    enum Field { BaseAddress, EffectiveCapacity, MallocedAddress, InternalCapacity, PriorAddress };

    void allocateBuffer(BuilderPtr b, llvm::Value * const capacityMultiplier) override;

    void releaseBuffer(BuilderPtr b) const override;

    void destroyBuffer(BuilderPtr b, llvm::Value * baseAddress, llvm::Value *capacity) const override;

    llvm::StructType * getHandleType(BuilderPtr b) const override;

    llvm::Value * getBaseAddress(BuilderPtr b) const override;

    llvm::Value * getMallocAddress(BuilderPtr b) const override;

    void setBaseAddress(BuilderPtr b, llvm::Value * addr) const override;

    void setCapacity(BuilderPtr b, llvm::Value * capacity) const override;

    llvm::Value * getCapacity(BuilderPtr b) const override;

    llvm::Value * getInternalCapacity(BuilderPtr b) const override;

    llvm::Value * modByCapacity(BuilderPtr b, llvm::Value * const offset) const override;

    llvm::Value * requiresExpansion(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void linearCopyBack(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    llvm::Value * expandBuffer(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    size_t getCapacity() const {
        return mCapacity;
    }

private:

    const size_t    mCapacity;

};

class DynamicBuffer final : public InternalBuffer {

    enum { LinearMallocedAddress = 0,
           LinearInternalCapacity = 1,
           LinearBaseAddress = 2,
           LinearEffectiveCapacity = 3,
           LinearFields = 4,
           // -------------------------------
           CircularAddressSelector = 0,
           CircularBaseAddress = 1,
           CircularSecondaryBaseAddress = 2,
           CircularInternalCapacity = 3,
           CircularSecondaryInternalCapacity = 4,
           CircularFields = 5 };

public:

    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() == BufferKind::DynamicBuffer;
    }

    DynamicBuffer(const unsigned id, BuilderPtr b, llvm::Type * type, const size_t initialCapacity,
                  const bool hasUnderflow,
                  const bool linear, const unsigned AddressSpace);

    void allocateBuffer(BuilderPtr b, llvm::Value * const capacityMultiplier) override;

    void releaseBuffer(BuilderPtr b) const override;

    void destroyBuffer(BuilderPtr b, llvm::Value * baseAddress, llvm::Value *capacity) const override;

    llvm::Value * getMallocAddress(BuilderPtr b) const override;

    llvm::Value * getCapacity(BuilderPtr b) const override;

    llvm::Value * getInternalCapacity(BuilderPtr b) const override;

    void setCapacity(BuilderPtr b, llvm::Value * capacity) const override;

    llvm::Value * modByCapacity(BuilderPtr b, llvm::Value * const offset) const final;

    llvm::Value * requiresExpansion(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void linearCopyBack(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    llvm::Value * expandBuffer(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    size_t getInitialCapacity() const {
        return mInitialCapacity;
    }

    llvm::StructType * getHandleType(BuilderPtr b) const override;

    llvm::Value * getBaseAddress(BuilderPtr b) const override;

    void setBaseAddress(BuilderPtr b, llvm::Value * addr) const override;

private:

    const size_t    mInitialCapacity;
    const bool      mHasUnderflow;
};

class MMapedBuffer final : public InternalBuffer {

    enum Field { BaseAddress, Capacity, Released, Fd };

public:

    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() == BufferKind::MMapedBuffer;
    }

    MMapedBuffer(const unsigned id, BuilderPtr b, llvm::Type * type, const size_t initialCapacity,
                  const size_t overflowSize, const size_t underflowSize,
                  const bool linear, const unsigned AddressSpace);

    void allocateBuffer(BuilderPtr b, llvm::Value * const capacityMultiplier) override;

    void releaseBuffer(BuilderPtr b) const override;

    void destroyBuffer(BuilderPtr b, llvm::Value * baseAddress, llvm::Value *capacity) const override;

    llvm::Value * getMallocAddress(BuilderPtr b) const override;

    llvm::Value * getCapacity(BuilderPtr b) const override;

    llvm::Value * getInternalCapacity(BuilderPtr b) const override;

    void setCapacity(BuilderPtr b, llvm::Value * capacity) const override;

    llvm::Value * modByCapacity(BuilderPtr b, llvm::Value * const offset) const final;

    llvm::Value * requiresExpansion(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void linearCopyBack(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    llvm::Value * expandBuffer(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    size_t getInitialCapacity() const {
        return mInitialCapacity;
    }

    llvm::StructType * getHandleType(BuilderPtr b) const override;

    llvm::Value * getBaseAddress(BuilderPtr b) const override;

    void setBaseAddress(BuilderPtr b, llvm::Value * addr) const override;

private:

    const size_t    mInitialCapacity;

};

class RepeatingBuffer final : public InternalBuffer {
public:
    static inline bool classof(const StreamSetBuffer * b) {
        return b->getBufferKind() == BufferKind::RepeatingBuffer;
    }

    enum Field { BaseAddress };

    RepeatingBuffer(const unsigned id, BuilderPtr b, llvm::Type * const type, const bool unaligned);

    llvm::Value * modByCapacity(BuilderPtr b, llvm::Value * const offset) const override;

    llvm::Value * getVirtualBasePtr(BuilderPtr b, llvm::Value * baseAddress, llvm::Value * const transferredItems) const override;

    void allocateBuffer(BuilderPtr b, llvm::Value * const capacityMultiplier) override;

    void releaseBuffer(BuilderPtr b) const override;

    void destroyBuffer(BuilderPtr b, llvm::Value * baseAddress, llvm::Value *capacity) const override;

    llvm::StructType * getHandleType(BuilderPtr b) const override;

    llvm::Value * getBaseAddress(BuilderPtr b) const override;

    llvm::Value * getMallocAddress(BuilderPtr b) const override;

    void setBaseAddress(BuilderPtr b, llvm::Value * addr) const override;

    void setCapacity(BuilderPtr b, llvm::Value * capacity) const override;

    llvm::Value * getCapacity(BuilderPtr b) const override;

    llvm::Value * getInternalCapacity(BuilderPtr b) const override;

    llvm::Value * requiresExpansion(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void linearCopyBack(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    llvm::Value * expandBuffer(BuilderPtr b, llvm::Value * produced, llvm::Value * consumed, llvm::Value * required) const override;

    void setModulus(llvm::Value * const modulus) {
        mModulus = modulus;
    }

    llvm::Value * getModulus() const {
        return mModulus;
    }

private:

    llvm::Value * mModulus;
    const bool mUnaligned;

};

}
#endif // STREAMSET_H
