/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <toolchain/toolchain.h>
#include <codegen/FunctionTypeBuilder.h>
#include <codegen/LLVMTypeSystemInterface.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Constants.h>
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(16, 0, 0)
#include <llvm/TargetParser/Triple.h>
#else
#include <llvm/ADT/Triple.h>
#endif
#ifndef NDEBUG
#include <llvm/IR/Function.h>
#endif
#include <unistd.h>
#include <util/not_null.h>

namespace kernel { class KernelBuilder; }
namespace kernel { class PipelineKernel; }
namespace llvm { class Function; }
namespace llvm { class IntegerType; }
namespace llvm { class Module; }
namespace llvm { class ModulePass; }
namespace llvm { class PointerType; }
namespace llvm { class Type; }
namespace llvm { class Value; }

class BaseDriver;

inline bool is_power_2(const uint64_t n) {
    return ((n & (n - 1)) == 0) && n;
}

extern "C" void free_debug_wrapper(void * ptr);



class CBuilder : public llvm::IRBuilder<> {

    friend class kernel::PipelineKernel;

public:

    CBuilder(llvm::LLVMContext & C);

    virtual ~CBuilder() {}

    llvm::Module * getModule() const {
        #ifndef NDEBUG
        llvm::BasicBlock * const bb = GetInsertBlock();
        if (bb) {
            llvm::Function * const f = bb->getParent();
            assert ("CBuilder has an insert point that is not contained within a Function" && f);
            assert ("CBuilder module differs from insertion point module" && (mModule == f->getParent()));
        }
        #endif
        return mModule;
    }

    void setModule(not_null<llvm::Module *> module) {
        mModule = module;
        ClearInsertionPoint();
    }

    // UDiv and URem with optimization for division by power-of-2 constants
    llvm::Value * CreateUDiv(llvm::Value * number, llvm::Value * divisor, const llvm::Twine Name = "");
    llvm::Value * CreateURem(llvm::Value * number, llvm::Value * divisor, const llvm::Twine Name = "");

    // Division with rounding up to the ceiling
    // Equivalent to CreateUDiv(CreateAdd(number, CreateSub(divisor, ConstantInt::get(divisor->getType(), 1))), divisor)
    llvm::Value * CreateCeilUDiv(llvm::Value * number, llvm::Value * divisor, const llvm::Twine Name = "");

    // Round down to a multiple of divisor.
    llvm::Value * CreateRoundDown(llvm::Value * number, llvm::Value * divisor, const llvm::Twine Name = "");

    // Round up to a multiple of divisor.
    llvm::Value * CreateRoundUp(llvm::Value * number, llvm::Value * divisor, const llvm::Twine Name = "");

    // Get minimum of two unsigned numbers
    llvm::Value * CreateUMin(llvm::Value * const a, llvm::Value * const b, const llvm::Twine Name = "") {
        if (LLVM_UNLIKELY(a == nullptr || a == b)) return b;
        if (LLVM_UNLIKELY(b == nullptr)) return a;
        assert (a->getType() == b->getType());
        return CreateSelect(CreateICmpULT(a, b), a, b, Name);
    }

    // Get minimum of two signed numbers
    llvm::Value * CreateSMin(llvm::Value * const a, llvm::Value * const b, const llvm::Twine Name = "") {
        if (LLVM_UNLIKELY(a == nullptr || a == b)) return b;
        if (LLVM_UNLIKELY(b == nullptr)) return a;
        assert (a->getType() == b->getType());
        return CreateSelect(CreateICmpSLT(a, b), a, b, Name);
    }

    // Get maximum of two unsigned numbers
    llvm::Value * CreateUMax(llvm::Value * const a, llvm::Value * const b, const llvm::Twine Name = "") {
        if (LLVM_UNLIKELY(a == nullptr || a == b)) return b;
        if (LLVM_UNLIKELY(b == nullptr)) return a;
        assert (a->getType() == b->getType());
        return CreateSelect(CreateICmpUGT(a, b), a, b, Name);
    }

    // Get maximum of two signed numbers
    llvm::Value * CreateSMax(llvm::Value * const a, llvm::Value * const b, const llvm::Twine Name = "") {
        if (LLVM_UNLIKELY(a == nullptr || a == b)) return b;
        if (LLVM_UNLIKELY(b == nullptr)) return a;
        assert (a->getType() == b->getType());
        return CreateSelect(CreateICmpSGT(a, b), a, b, Name);
    }

    llvm::Value * CreateSaturatingAdd(llvm::Value * const a, llvm::Value * const b, const llvm::Twine Name = "");

    llvm::Value * CreateSaturatingSub(llvm::Value * const a, llvm::Value * const b, const llvm::Twine Name = "");

    llvm::Value * CreateMalloc(llvm::Value * const size);

    llvm::Value * CreateAlignedMalloc(llvm::Value * const size, const unsigned alignment);

    llvm::Value * CreateAlignedMalloc(llvm::Type * const type, llvm::Value * const ArraySize, const unsigned addressSpace, const unsigned alignment);

    llvm::Value * CreatePageAlignedMalloc(llvm::Type * const type, llvm::Value * const ArraySize = nullptr, const unsigned addressSpace = 0);

    llvm::Value * CreatePageAlignedMalloc(llvm::Value * const size);

    llvm::Value * CreateCacheAlignedMalloc(llvm::Type * const type, llvm::Value * const ArraySize = nullptr, const unsigned addressSpace = 0);

    llvm::Value * CreateCacheAlignedMalloc(llvm::Value * const size);

    void CreateFree(llvm::Value * const ptr);

    llvm::Value * CreateRealloc(llvm::Type * const type, llvm::Value * const base, llvm::Value * const ArraySize);

    llvm::Value * CreateRealloc(llvm::Value * const base, llvm::Value * const size);

    llvm::CallInst * CreateMemZero(llvm::Value * const ptr, llvm::Value * const size, const unsigned alignment = unsigned{1}) {
        return CreateMemSet(ptr, getInt8(0), size, alignment);
    }

    llvm::Value * CreateMemChr(llvm::Value * ptr, llvm::Value * byteVal, llvm::Value * num);

    llvm::CallInst * CreateMemCmp(llvm::Value * ptr1, llvm::Value * ptr2, llvm::Value * num);

   llvm::AllocaInst * CreateAllocaAtEntryPoint(llvm::Type * Ty, llvm::Value * ArraySize = nullptr, const llvm::Twine Name = "");

    llvm::AllocaInst * CreateAlignedAlloca(llvm::Type * const Ty, const unsigned alignment, llvm::Value * const ArraySize = nullptr);

    llvm::AllocaInst * CreateCacheAlignedAlloca(llvm::Type * const Ty, llvm::Value * const ArraySize = nullptr) {
        return CreateAlignedAlloca(Ty, getCacheAlignment(), ArraySize);
    }

    llvm::AllocaInst * CreateAlignedAllocaAtEntryPoint(llvm::Type * const Ty, const unsigned alignment, llvm::Value * const ArraySize = nullptr);

    // stdio.h functions
    //
    //  Create a call to:  FILE * fopen(const char *filename, const char *mode);
    llvm::Value * CreateFOpenCall(llvm::Value * filename, llvm::Value * mode);
    //  Create a call to:  size_t fread(void *ptr, size_t size, size_t nitems, FILE *stream);
    llvm::Value * CreateFReadCall(llvm::Value * ptr, llvm::Value * size, llvm::Value * nitems, llvm::Value * stream);
    //  Create a call to:  size_t fwrite(const void *ptr, size_t size, size_t nitems, FILE *stream));
    llvm::Value * CreateFWriteCall(llvm::Value * ptr, llvm::Value * size, llvm::Value * nitems, llvm::Value * stream);
    //  Create a call to:  int fclose ( FILE * stream );
    llvm::Value * CreateFCloseCall(llvm::Value * stream);
    //  Create a call to:  int remove(const char *path);
    llvm::Value * CreateRemoveCall(llvm::Value * path);

    //  Create a call to:  int rename(const char *old, const char *new);
    llvm::Value * CreateRenameCall(llvm::Value * oldName, llvm::Value * newName);

    llvm::Function * GetPrintf();

    // Create call to int printf(const char *format, ...);
    template <typename ... Args>
    llvm::CallInst * CreatePrintfCall(const llvm::StringRef format, Args ... args) {
        std::initializer_list<llvm::Value *> a{std::forward<Args>(args)...};
        return __CreatePrintfCall(GetString(format), a);
    }

    llvm::Function * GetDprintf();

    // Create call to int dprintf(int fd, const char *format, ...);
    template <typename ... Args>
    llvm::CallInst * CreateDprintfCall(llvm::Value * const fd, const llvm::StringRef format, Args ... args) {
        std::initializer_list<llvm::Value *> a{std::forward<Args>(args)...};
        return __CreateDprintfCall(fd, GetString(format), a);
    }

    // Create call to int sprintf(char * str, const char *format, ...);
    template <typename ... Args>
    llvm::CallInst * CreateSprintfCall(llvm::Value * const str, const llvm::StringRef format, Args ... args) {
        std::initializer_list<llvm::Value *> a{std::forward<Args>(args)...};
        return __CreateSprintfCall(str, GetString(format), a);
    }

    //  Create calls to unistd.h functions.
    //
    //  Create a call to:  int open(const char *filename, int oflag, ...);
    llvm::Value * CreateOpenCall(llvm::Value * filename, llvm::Value * oflag, llvm::Value * mode);
    //  Create a call to:  ssize_t write(int fildes, const void *buf, size_t nbyte);
    llvm::Value * CreateWriteCall(llvm::Value * fileDescriptor, llvm::Value * buf, llvm::Value * nbyte);
    //  Create a call to:  ssize_t read(int fildes, void *buf, size_t nbyte);
    llvm::Value * CreateReadCall(llvm::Value * fileDescriptor, llvm::Value * buf, llvm::Value * nbyte);
    //  Create a call to:  int close(int filedes);
    llvm::Value * CreateCloseCall(llvm::Value * fileDescriptor);
    //  Create a call to:  int unlink(const char *path);
    llvm::Value * CreateUnlinkCall(llvm::Value * path);

    llvm::Value * CreatePosixFAllocate(llvm::Value * fileDescriptor, llvm::Value * offset, llvm::Value * len);

    // llvm::Value * CreateFileSize(llvm::Value * fileDescriptor);

    llvm::Value * CreateFSync(llvm::Value * fileDescriptor);

    //  Create calls to stdlib.h functions.
    //
    //  Create a call to:  int mkstemp (char *template);
    llvm::Value * CreateMkstempCall(llvm::Value * ftemplate);

    //  Create a call to:  size_t strlen(const char *str);
    llvm::Value * CreateStrlenCall(llvm::Value * str);

    llvm::Value * CreateAnonymousMMap(llvm::Value * size, const unsigned flags = 0);

    llvm::Value * CreateFileSourceMMap(llvm::Value * fd, llvm::Value * size);

    llvm::Value * CreateMAdvise(llvm::Value * addr, llvm::Value * length, const int advice);

    llvm::Value * CreateMMap(llvm::Value * const addr, llvm::Value * size, llvm::Value * const prot, llvm::Value * const flags, llvm::Value * const fd, llvm::Value * const offset);

    llvm::Value * CreateMemFdCreate(llvm::Value * const name, llvm::Value * const flags);

    llvm::Value * CreateFTruncate(llvm::Value * const fd, llvm::Value * size);

    llvm::Value * CreateMRemap(llvm::Value * addr, llvm::Value * oldSize, llvm::Value * newSize);

    llvm::Value * CreateMUnmap(llvm::Value * addr, llvm::Value * size);

    enum Protect {
        NONE = 0
        , READ = 1
        , WRITE = 2
        , EXEC = 4
    };

    llvm::Value * CreateMProtect(llvm::Type * type, llvm::Value * addr, const Protect protect) {
        return CreateMProtect(addr, getTypeSize(type), protect);
    }

    llvm::Value * CreateMProtect(llvm::Value * addr, llvm::Value * size, const Protect protect);

    enum class STD_FD {
        STD_IN = STDIN_FILENO
        , STD_OUT = STDOUT_FILENO
        , STD_ERR = STDERR_FILENO
    };

    void CallPrintIntCond(llvm::StringRef name, llvm::Value * const value, llvm::Value * const cond, const STD_FD fd = STD_FD::STD_ERR);

    llvm::CallInst * CallPrintInt(llvm::StringRef name, llvm::Value * const value, const STD_FD fd = STD_FD::STD_ERR);

    llvm::Constant * GetString(llvm::StringRef Str);

    inline llvm::IntegerType * getSizeTy() const {
        assert (mSizeType);
        return mSizeType;
    }

    inline llvm::ConstantInt * LLVM_READNONE getSize(const size_t value) {
        return llvm::ConstantInt::get(getSizeTy(), value);
    }

    llvm::IntegerType * LLVM_READNONE getIntAddrTy() const;

    llvm::PointerType * LLVM_READNONE getVoidPtrTy(const unsigned AddressSpace = 0) const;

    llvm::PointerType * LLVM_READNONE getFILEptrTy();

    llvm::ConstantInt * LLVM_READNONE getTypeSize(llvm::Type * type, llvm::IntegerType * valType = nullptr) const;

    static uintptr_t LLVM_READNONE getTypeSize(const llvm::DataLayout & DL, llvm::Type * type);

    inline unsigned getCacheAlignment() const {
        return mCacheLineAlignment;
    }

    static LLVM_READNONE unsigned getPageSize();

    virtual llvm::LoadInst* CreateAtomicLoadAcquire(llvm::Type * type, llvm::Value * ptr);

    virtual llvm::StoreInst *  CreateAtomicStoreRelease(llvm::Value * val, llvm::Value * ptr);

    llvm::Value * CreateAtomicFetchAndAdd(llvm::Value * val, llvm::Value * ptr, llvm::MaybeAlign align = llvm::MaybeAlign{});

    llvm::Value * CreateAtomicFetchAndSub(llvm::Value * val, llvm::Value * ptr, llvm::MaybeAlign align = llvm::MaybeAlign{});

    void CreateAssert(llvm::Value * assertion, const llvm::Twine failureMessage) {
        return __CreateAssert(CreateIsNotNull(assertion), failureMessage, {});
    }

    template <typename ... Args>
    void CreateAssert(llvm::Value * assertion, const llvm::Twine failureMessage, Args ... args) {
        std::initializer_list<llvm::Value *> a{std::forward<Args>(args)...};
        assert (a.size() > 0);
        return __CreateAssert(CreateIsNotNull(assertion), failureMessage, a);
    }

    void CreateAssertZero(llvm::Value * assertion, const llvm::Twine failureMessage) {
        return __CreateAssert(CreateIsNull(assertion), failureMessage, {});
    }

    template <typename ... Args>
    void CreateAssertZero(llvm::Value * assertion, const llvm::Twine failureMessage, Args ... args) {
        std::initializer_list<llvm::Value *> a{std::forward<Args>(args)...};
        assert (a.size() > 0);
        return __CreateAssert(CreateIsNull(assertion), failureMessage, a);
    }

    void CreateExit(const int exitCode);

    llvm::BranchInst * CreateLikelyCondBr(llvm::Value * Cond, llvm::BasicBlock * True, llvm::BasicBlock * False, const int probability = 90);

    llvm::BranchInst * CreateUnlikelyCondBr(llvm::Value * Cond, llvm::BasicBlock * True, llvm::BasicBlock * False, const int probability = 90) {
        return CreateLikelyCondBr(Cond, True, False, 100 - probability);
    }

    llvm::BasicBlock * CreateBasicBlock(const llvm::StringRef name = "", llvm::BasicBlock * insertBefore = nullptr);

    virtual bool supportsIndirectBr() const;

    llvm::Value * CreatePopcount(llvm::Value * bits);

    // TODO: AVX512 offers these as vector instructions
    llvm::Value * CreateCountForwardZeroes(llvm::Value * value, const llvm::Twine Name = "", const no_conversion<bool> guaranteedNonZero = false);
    llvm::Value * CreateCountReverseZeroes(llvm::Value * value, const llvm::Twine Name = "", const no_conversion<bool> guaranteedNonZero = false);

    // Useful bit manipulation operations
    llvm::Value * CreateResetLowestBit(llvm::Value * bits, const llvm::Twine Name = "");

    llvm::Value * CreateIsolateLowestBit(llvm::Value * bits, const llvm::Twine Name = "");

    llvm::Value * CreateMaskToLowestBitInclusive(llvm::Value * bits, const llvm::Twine Name = "");

    llvm::Value * CreateMaskToLowestBitExclusive(llvm::Value * bits, const llvm::Twine Name = "");

    virtual llvm::Value * CreateZeroHiBitsFrom(llvm::Value * bits, llvm::Value * pos, const llvm::Twine Name = "");

    virtual llvm::Value * CreateExtractBitField(llvm::Value * bits, llvm::Value * start, llvm::Value * length, const llvm::Twine Name = "");

    llvm::Value * CreateCeilLog2(llvm::Value * value, const llvm::Twine Name = "");

    llvm::Value * CreateLog2(llvm::Value * value, const llvm::Twine Name = "");

    llvm::Value * CreateReadCycleCounter();

    template <typename ExternalFunctionType>
    llvm::Function * LinkFunction(llvm::StringRef name, ExternalFunctionType & functionPtr) const;

    llvm::Function * LinkFunction(llvm::StringRef name, llvm::FunctionType * type, void * functionPtr) const;

    // Set the nontemporal metadata attribute for a store instruction.
    void setNontemporal(llvm::StoreInst * s);

    enum class PrefetchRW {Read, Write};
    enum class CacheType {Instruction, Data};
    llvm::Value * CreatePrefetch(llvm::Value * ptr, PrefetchRW mode, unsigned locality, CacheType c);

    virtual llvm::LoadInst * CreateLoad(llvm::Type * type, llvm::Value * Ptr, const char * Name);

    virtual llvm::LoadInst * CreateLoad(llvm::Type * type, llvm::Value * Ptr, const llvm::Twine Name = "");

    virtual llvm::LoadInst * CreateLoad(llvm::Type * type, llvm::Value * Ptr, bool isVolatile, const llvm::Twine Name = "");

    virtual llvm::StoreInst * CreateStore(llvm::Value * Val, llvm::Value * Ptr, bool isVolatile = false);

    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(15, 0, 0)
    llvm::Value * CreateGEP(llvm::Type * Ty, llvm::Value * Ptr, llvm::ArrayRef<llvm::Value *> IdxList, const llvm::Twine & Name = "", bool IsInBounds = false);

    llvm::Value * CreateInBoundsGEP(llvm::Type *Ty, llvm::Value *Ptr, llvm::ArrayRef<llvm::Value *> IdxList, const llvm::Twine &Name = "") {
        return CreateGEP(Ty, Ptr, IdxList, Name, /* IsInBounds */ true);
    }
    #endif

    llvm::LoadInst * CreateAlignedLoad(llvm::Type * type, llvm::Value * Ptr, const unsigned Align, const char * Name);

    llvm::LoadInst * CreateAlignedLoad(llvm::Type * type, llvm::Value * Ptr, const unsigned Align, const llvm::Twine Name = "");

    llvm::LoadInst * CreateAlignedLoad(llvm::Type * type, llvm::Value * Ptr, const unsigned Align, bool isVolatile, const llvm::Twine Name = "");

    llvm::StoreInst * CreateAlignedStore(llvm::Value * Val, llvm::Value * Ptr, const unsigned Align, bool isVolatile = false);

    llvm::CallInst * CreateMemMove(llvm::Value *Dst, llvm::Value *Src, uint64_t Size, const unsigned Align,
                            bool isVolatile = false, llvm::MDNode *TBAATag = nullptr,
                            llvm::MDNode *ScopeTag = nullptr,
                            llvm::MDNode *NoAliasTag = nullptr) {
        return CreateMemMove(Dst, Src, getInt64(Size), Align, isVolatile, TBAATag, ScopeTag, NoAliasTag);
    }

    llvm::CallInst * CreateMemMove(llvm::Value *Dst, llvm::Value *Src, llvm::Value *Size, const unsigned Align,
                            bool isVolatile = false, llvm::MDNode *TBAATag = nullptr,
                            llvm::MDNode *ScopeTag = nullptr,
                            llvm::MDNode *NoAliasTag = nullptr);

    llvm::CallInst * CreateMemCpy(llvm::Value *Dst, llvm::Value *Src, uint64_t Size, const unsigned Align,
                           bool isVolatile = false, llvm::MDNode *TBAATag = nullptr,
                           llvm::MDNode *TBAAStructTag = nullptr,
                           llvm::MDNode *ScopeTag = nullptr,
                           llvm::MDNode *NoAliasTag = nullptr) {
        return CreateMemCpy(Dst, Src, getInt64(Size), Align, isVolatile, TBAATag, TBAAStructTag, ScopeTag, NoAliasTag);
    }

    llvm::CallInst * CreateMemCpy(llvm::Value *Dst, llvm::Value *Src, llvm::Value *Size, const unsigned Align,
                           bool isVolatile = false, llvm::MDNode *TBAATag = nullptr,
                           llvm::MDNode *TBAAStructTag = nullptr,
                           llvm::MDNode *ScopeTag = nullptr,
                           llvm::MDNode *NoAliasTag = nullptr);

    llvm::CallInst * CreateMemSet(llvm::Value *Ptr, llvm::Value *Val, uint64_t Size, const unsigned Align,
                           bool isVolatile = false, llvm::MDNode *TBAATag = nullptr,
                           llvm::MDNode *ScopeTag = nullptr,
                           llvm::MDNode *NoAliasTag = nullptr) {
        return CreateMemSet(Ptr, Val, getInt64(Size), Align, isVolatile, TBAATag, ScopeTag, NoAliasTag);
    }

    llvm::CallInst * CreateMemSet(llvm::Value *Ptr, llvm::Value *Val, llvm::Value *Size, const unsigned Align,
                           bool isVolatile = false, llvm::MDNode *TBAATag = nullptr,
                           llvm::MDNode *ScopeTag = nullptr,
                           llvm::MDNode *NoAliasTag = nullptr);

    llvm::Value * CreateExtractElement(llvm::Value *Vec, llvm::Value *Idx, const llvm::Twine Name = "");

    llvm::Value * CreateExtractElement(llvm::Value *Vec, uint64_t Idx, const llvm::Twine Name = "") {
        return CreateExtractElement(Vec, getInt64(Idx), Name);
    }

    llvm::Value * CreateInsertElement(llvm::Value *Vec, llvm::Value *NewElt, llvm::Value *Idx, const llvm::Twine Name = "");

    llvm::Value * CreateInsertElement(llvm::Value *Vec, llvm::Value *NewElt, uint64_t Idx, const llvm::Twine Name = "") {
        return CreateInsertElement(Vec, NewElt, getInt64(Idx), Name);
    }

    llvm::CallInst * CreateSRandCall(llvm::Value * randomSeed);
    llvm::CallInst * CreateRandCall();

    void setDriver(BaseDriver & ts) {
        mDriver = &ts;
    }

    BaseDriver & getDriver() const {
        return *mDriver;
    }

    llvm::BasicBlock * WriteDefaultRethrowBlock();

    void CheckAddress(llvm::Value * const Ptr, llvm::Value * const Size, llvm::StringRef Name) {
        CheckAddress(Ptr, Size, GetString(Name));
    }

    void CheckAddress(llvm::Value * const Ptr, llvm::Value * const Size, llvm::Constant * const Name);

    // LLVM 18 removed all qualified pointer types but if we want to support earlier LLVM versions, we must still still allow kernels to
    // construct them.
    #if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(18, 0, 0)
    #define ADD_POINTER_TYPE_ALIAS(Name) llvm::PointerType * get##Name##PtrTy(unsigned = 0) { return llvm::PointerType::getUnqual(getContext()); }

    ADD_POINTER_TYPE_ALIAS(Int8)
    ADD_POINTER_TYPE_ALIAS(Int16)
    ADD_POINTER_TYPE_ALIAS(Int32)
    ADD_POINTER_TYPE_ALIAS(Int64)
    ADD_POINTER_TYPE_ALIAS(Void)

    #undef ADD_POINTER_TYPE
    #endif

protected:

    llvm::CallInst * __CreatePrintfCall(llvm::Value * const format, std::initializer_list<llvm::Value *> args);

    llvm::CallInst * __CreateDprintfCall(llvm::Value * const fd, llvm::Value * const format, std::initializer_list<llvm::Value *> args);

    llvm::CallInst * __CreateSprintfCall(llvm::Value * const str, llvm::Value * const format, std::initializer_list<llvm::Value *> args);

    llvm::Function * getDefaultPersonalityFunction();

    llvm::Function * getThrow();

    llvm::Function * getAllocateException();

    llvm::Function * getBeginCatch();

    llvm::Function * getEndCatch();

    llvm::Function * getRethrow();

    static llvm::AllocaInst * resolveStackAddress(llvm::Value * Ptr);

    bool hasAddressSanitizer() const;

    virtual std::string getKernelName() const;

    void __CreateAssert(llvm::Value * assertion, const llvm::Twine failureMessage, std::initializer_list<llvm::Value *> args);

    virtual void linkAllNecessaryExternalFunctions() const;

protected:

    llvm::Module *                  mModule;
    unsigned                        mCacheLineAlignment;
    llvm::IntegerType * const       mSizeType;
    llvm::StructType *              mFILEtype;
    BaseDriver *                    mDriver;
    llvm::LLVMContext               mContext;
    const std::string               mTriple;
    #ifdef ENABLE_LIBBACKTRACE
    void *                          mBacktraceState = nullptr;
    #endif
};

template <typename ExternalFunctionType>
llvm::Function * CBuilder::LinkFunction(llvm::StringRef name, ExternalFunctionType & functionPtr) const {
    llvm::FunctionType * const type = FunctionTypeBuilder<ExternalFunctionType>::get(getContext());
    assert ("FunctionTypeBuilder did not resolve a function type." && type);
    return LinkFunction(name, type, reinterpret_cast<void *>(&functionPtr));
}

llvm::ModulePass * createRemoveRedundantAssertionsPass();

template <typename T, unsigned N>
using FixedArray = std::array<T, N>;


