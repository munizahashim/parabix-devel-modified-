/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <pablo/compressed_carry_manager.h>

#include <pablo/carry_data.h>
#include <pablo/codegenstate.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/Transforms/Utils/Local.h>
#include <pablo/branch.h>
#include <pablo/pablo_intrinsic.h>
#include <pablo/pe_advance.h>
#include <pablo/pe_scanthru.h>
#include <pablo/pe_matchstar.h>
#include <pablo/pe_var.h>
#include <kernel/core/kernel_builder.h>
#include <toolchain/toolchain.h>
#include <array>

#define LONG_ADVANCE_BREAKPOINT 64

using namespace llvm;

using BuilderRef = pablo::CompressedCarryManager::BuilderRef;

namespace pablo {

/* Local Helper Functions */

inline static bool isNonAdvanceCarryGeneratingStatement(const Statement * const stmt) {
    if (IntrinsicCall const * call = dyn_cast<IntrinsicCall>(stmt)) {
        return call->isCarryProducing() && !call->isAdvanceType();
    } else {
        return isa<CarryProducingStatement>(stmt) && !isa<Advance>(stmt) && !isa<IndexedAdvance>(stmt);
    }
}


static inline bool isNonRegularLanguage(const PabloBlock * const scope) {
    if (const Branch * br = scope->getBranch()) {
        return !br->isRegular();
    }
    return false;
}


static inline unsigned ceil_log2(const unsigned v) {
    assert ("log2(0) is undefined!" && v != 0);
    return (sizeof(unsigned) * CHAR_BIT) - __builtin_clz(v - 1U);
}


static inline unsigned nearest_pow2(const unsigned v) {
    assert (v > 0 && v < (UINT32_MAX / 2));
    return (v < 2) ? 1 : (1U << ceil_log2(v));
}


static inline unsigned ceil_udiv(const unsigned x, const unsigned y) {
    return (((x - 1) | (y - 1)) + 1) / y;
}

// Recursively determines the minimum summary size needed, in bits, for a given pablo block.
static int32_t analyseSummarySize(int32_t blockWidth, const PabloBlock * const scope) {
    int32_t carrySize = 0;
    for (const Statement * stmt : *scope) {
        if (LLVM_UNLIKELY(isa<Advance>(stmt) || isa<IndexedAdvance>(stmt))) {
            int64_t amount = isa<Advance>(stmt)
                ? cast<Advance>(stmt)->getAmount()
                : cast<IndexedAdvance>(stmt)->getAmount();
            if (LLVM_LIKELY(amount < 8)) {
                carrySize = std::max(8, carrySize);
            } else if (amount >= 8 && amount < LONG_ADVANCE_BREAKPOINT) {
                carrySize = std::max(64, carrySize);
            } else {
                carrySize = blockWidth;
            }
        } else if (LLVM_UNLIKELY(isNonAdvanceCarryGeneratingStatement(stmt))) {
            carrySize = std::max(8, carrySize);
        } else if (LLVM_UNLIKELY(isa<If>(stmt))) {
            carrySize = std::max(analyseSummarySize(blockWidth, cast<If>(stmt)->getBody()), carrySize);
        } else if (LLVM_UNLIKELY(isa<While>(stmt))) {
            carrySize = std::max(analyseSummarySize(blockWidth, cast<While>(stmt)->getBody()), carrySize);
        }
    }
    assert (carrySize >= 0 && carrySize <= blockWidth);
    return carrySize;
}


static Type * toSummaryType(BuilderRef b, int32_t summarySize) {
    switch (summarySize) {
    case 8:
        return b->getInt8Ty();
    case 64:
        return b->getInt64Ty();
    default:
        assert ("unexpected summary type" && (uint32_t) summarySize == b->getBitBlockWidth());
        return b->getBitBlockType();
    }
}


static Value * compressImplicitSummary(BuilderRef b, Value * summary) {
    if (summary->getType() == b->getBitBlockType()) {
        summary = b->CreateBitCast(summary, b->getIntNTy(b->getBitBlockWidth()));
    }
    return b->CreateICmpNE(summary, Constant::getNullValue(summary->getType()));
}


/* ===== Initialization ===== */


void CompressedCarryManager::initializeCodeGen(BuilderRef b) {

    assert(!mCarryMetadata.empty());
    mCarryInfo = &mCarryMetadata[0];
    assert (!mCarryInfo->hasSummary());
    mCurrentFrame = b->getScalarFieldPtr("carries");
    mCurrentFrameIndex = 0;
    mCarryScopes = 0;
    mCarryScopeIndex.push_back(0);
    assert (mCarryFrameStack.empty());
    assert (mCarrySummaryStack.empty());

    const auto baseSummarySize = analyseSummarySize(b->getBitBlockWidth(), mKernel->getEntryScope());
    mBaseSummaryType = baseSummarySize > 0 ? toSummaryType(b, baseSummarySize) : b->getInt8Ty();
    mCarrySummaryStack.push_back(Constant::getNullValue(mBaseSummaryType));

    if (mHasLoop) {
        mLoopSelector = b->getScalarField("selector");
        mNextLoopSelector = b->CreateXor(mLoopSelector, ConstantInt::get(mLoopSelector->getType(), 1));
    }
}

void CompressedCarryManager::enterIfScope(BuilderRef b, const PabloBlock * const /*scope*/) {
    ++mIfDepth;
    enterScope(b);
    // We zero-initialized the nested summary value and later OR in the current summary into the escaping summary
    // so that upon processing the subsequent block iteration, we branch into this If scope iff a carry out was
    // generated by a statement within this If scope and not by a dominating statement in the outer scope.
    if (mCarryInfo->hasExplicitSummary()) {
        Type * const summaryTy = getSummaryTypeFromCurrentFrame(b);
        mCarrySummaryStack.push_back(Constant::getNullValue(summaryTy)); // new carry out summary accumulator
    } else if (mCarryInfo->hasImplicitSummary()) {
        mCarrySummaryStack.push_back(convertFrameToImplicitSummaryPtr(b));
    }
}

void CompressedCarryManager::combineCarryOutSummary(BuilderRef b, const unsigned offset) {
    if (LLVM_LIKELY(mCarryInfo->hasSummary())) {
        const auto n = mCarrySummaryStack.size(); assert (n > 0);
        // combine the outer summary with the nested summary so that when
        // we leave the scope, we'll properly phi out the value of the new
        // outer summary
        if (n > 2) {
            Value * nested = mCarrySummaryStack[n - 1];
            if (mCarryInfo->hasImplicitSummary()) {
                assert (nested->getType()->isPointerTy());
                nested = b->CreateLoad(nested);
            }
            Value * const outer = mCarrySummaryStack[n - 2];
            if (nested->getType() != outer->getType()) {
                nested = compressImplicitSummary(b, nested);
                if (outer->getType() == b->getBitBlockType()) {
                    nested = b->bitCast(b->CreateZExt(nested, b->getIntNTy(b->getBitBlockWidth())));
                } else {
                    nested = b->CreateZExt(nested, outer->getType());
                }
            }
            mCarrySummaryStack[n - offset] = b->CreateOr(outer, nested);
        }
    }
}

/* ===== Operations ===== */

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief shortIndexedAdvanceCarryInCarryOut
 ** ------------------------------------------------------------------------------------------------------------- */
Value * CompressedCarryManager::shortIndexedAdvanceCarryInCarryOut(BuilderRef b, const unsigned shiftAmount, Value * const strm, Value * const index_strm) {
    Value * carryIn = getNextCarryIn(b);
    carryIn = b->CreateBitCast(b->CreateZExt(carryIn, b->getIntNTy(b->getBitBlockWidth())), b->getBitBlockType());
    Value * carryOut, * result;
    std::tie(carryOut, result) = b->bitblock_indexed_advance(strm, index_strm, carryIn, shiftAmount);
    const uint32_t fw = shiftAmount < 8 ? 8 : 64;
    carryOut = b->mvmd_extract(fw, carryOut, 0);
    setNextCarryOut(b, carryOut);
    return result;
}

/* ===== Summary Operations ===== */

Value * CompressedCarryManager::readCarryInSummary(BuilderRef b) const {
    assert (mCarryInfo->hasSummary());

    if (LLVM_LIKELY(mCarryInfo->hasImplicitSummary())) {
        return convertFrameToImplicitSummary(b);
    } else {
        assert (mCarryInfo->hasExplicitSummary());
        Constant * const ZERO = b->getInt32(0);
        FixedArray<Value *, 3> indices;
        indices[0] = ZERO;
        indices[1] = ZERO;
        indices[2] = mLoopDepth == 0 ? ZERO : mLoopSelector;

        Value * const ptr = b->CreateGEP(mCurrentFrame, indices);
        Value * summary = b->CreateLoad(ptr);
        if (mNestedLoopCarryInMaskPhi) {
            summary = b->CreateAnd(summary, mNestedLoopCarryInMaskPhi);
        }
        return summary;
    }
}

Value * CompressedCarryManager::convertFrameToImplicitSummary(BuilderRef b) const {
    Value * const ptr = convertFrameToImplicitSummaryPtr(b);
    Value * const summary = b->CreateLoad(ptr);
    return summary;
}

Value * CompressedCarryManager::convertFrameToImplicitSummaryPtr(BuilderRef b) const {
    assert (mCarryInfo->hasSummary() && mCarryInfo->hasImplicitSummary());

    Type * const frameTy = mCurrentFrame->getType()->getPointerElementType();
    assert (frameTy->isStructTy());
    auto DL = b->getModule()->getDataLayout();
    auto frameSize = DL.getStructLayout(cast<StructType>(frameTy))->getSizeInBits();
    assert (frameSize == 64 || frameSize == b->getBitBlockWidth());

    Type * const summaryTy = frameSize == 64 ? (Type *) b->getInt64Ty() : b->getBitBlockType();
    return b->CreatePointerCast(mCurrentFrame, summaryTy->getPointerTo());
}

Type * CompressedCarryManager::getSummaryTypeFromCurrentFrame(BuilderRef b) const {
    assert (mCurrentFrame->getType()->isPointerTy());
    assert (mCurrentFrame->getType()->getPointerElementType()->isStructTy());
    return mCurrentFrame->getType()->getPointerElementType()->getStructElementType(0)->getArrayElementType();
}


/* ==== Scope Analyse ===== */
StructType * CompressedCarryManager::analyse(BuilderRef b,
                                             const PabloBlock * const scope,
                                             const unsigned ifDepth,
                                             const unsigned whileDepth,
                                             const bool isNestedWithinNonCarryCollapsingLoop)
{
    assert ("scope cannot be null!" && scope);
    assert ("entry scope (and only the entry scope) must be in scope 0"
            && (mCarryScopes == 0 ? (scope == mKernel->getEntryScope()) : (scope != mKernel->getEntryScope())));
    assert (mCarryScopes < mCarryMetadata.size());

    Type * const i8Ty = b->getInt8Ty();
    Type * const i64Ty = b->getInt64Ty();
    Type * const blockTy = b->getBitBlockType();
    const uint32_t blockWidth = b->getBitBlockWidth();

    const uint32_t carryScopeIndex = mCarryScopes++;
    const bool nonCarryCollapsingMode = isNestedWithinNonCarryCollapsingLoop || isNonRegularLanguage(scope);

    const uint64_t packSize = whileDepth == 0 ? 1 : 2;
    Type * const i8PackTy = ArrayType::get(i8Ty, packSize);
    Type * const i64PackTy = ArrayType::get(i64Ty, packSize);

    bool canUseImplicitSummary = packSize == 1 && !nonCarryCollapsingMode;
    size_t carryProducingStatementCount = 0;
    const size_t maxNumSmallCarriesForImplicitSummary = blockWidth / 8;

    /* Get Carry Types */

    std::vector<Type *> state;
    for (const Statement * stmt : *scope) {
        if (LLVM_UNLIKELY(isa<Advance>(stmt) || isa<IndexedAdvance>(stmt))) {
            carryProducingStatementCount++;
            const auto amount = isa<Advance>(stmt)
                ? cast<Advance>(stmt)->getAmount()
                : cast<IndexedAdvance>(stmt)->getAmount();
            Type * type = i8PackTy;
            if (LLVM_UNLIKELY(amount >= 8 && amount < LONG_ADVANCE_BREAKPOINT)) {
                canUseImplicitSummary = false;
                type = i64PackTy;
            } else if (LLVM_UNLIKELY(amount >= LONG_ADVANCE_BREAKPOINT)) {
                canUseImplicitSummary = false;
                const auto blockWidth = b->getBitBlockWidth();
                const auto blocks = ceil_udiv(amount, blockWidth);
                type = ArrayType::get(blockTy, nearest_pow2(blocks + (isa<IndexedAdvance>(stmt) ? 1:0) + ((whileDepth != 0) ? 1 : 0)));
                if (LLVM_UNLIKELY(ifDepth > 0 && blocks != 1)) {
                    const auto summarySize = ceil_udiv(blocks, blockWidth);
                    // 1 bit will mark the presense of any bit in each block.
                    state.push_back(ArrayType::get(blockTy, summarySize));
                }
                mHasLongAdvance = true;
                if (isa<IndexedAdvance>(stmt)) {
                    mIndexedLongAdvanceTotal++;
                }
            }
            state.push_back(type);
        } else if (LLVM_UNLIKELY(isNonAdvanceCarryGeneratingStatement(stmt))) {
            carryProducingStatementCount++;
            state.push_back(i8PackTy);
        } else if (LLVM_UNLIKELY(isa<If>(stmt))) {
            canUseImplicitSummary = false;
            state.push_back(analyse(b, cast<If>(stmt)->getBody(), ifDepth+1, whileDepth, nonCarryCollapsingMode));
        } else if (LLVM_UNLIKELY(isa<While>(stmt))) {
            canUseImplicitSummary = false;
            mHasLoop = true;
            state.push_back(analyse(b, cast<While>(stmt)->getBody(), ifDepth, whileDepth+1, nonCarryCollapsingMode));
        }

        if (carryProducingStatementCount >= maxNumSmallCarriesForImplicitSummary)
            canUseImplicitSummary = false;
    }

    /* Construct Carry State Struct */
    CarryData & cd = mCarryMetadata[carryScopeIndex];
    StructType * carryStruct = nullptr;
    CarryData::SummaryType summaryType = CarryData::NoSummary;

    // if we have at least one non-empty carry state, check if we need a summary
    if (LLVM_UNLIKELY(isEmptyCarryStruct(state))) {
        carryStruct = StructType::get(b->getContext(), state);
    } else {

        Type * summaryTy = nullptr;

        if (LLVM_LIKELY(ifDepth > 0 || whileDepth > 0)) {
            if (LLVM_LIKELY(canUseImplicitSummary)) {
                summaryType = CarryData::ImplicitSummary;
                // If needed, pad the structure to 64 bits or the bitblock width. This allows us to bitcast the structure to an i64 or
                // bitblock to get the summary.
                carryStruct = StructType::get(b->getContext(), state);
                const DataLayout & DL = b->getModule()->getDataLayout();
                const auto structBitWidth = DL.getTypeAllocSize(carryStruct) * 8;
                const auto targetWidth = structBitWidth <= 64 ? 64 : blockWidth;
                if (structBitWidth < targetWidth) {
                    const auto padding = ceil_udiv(targetWidth - structBitWidth, 8);
                    ArrayType * const paddingTy = ArrayType::get(i8Ty, padding);
                    state.push_back(paddingTy);
                }
            } else {
                summaryType = CarryData::ExplicitSummary;

                // Insert the smallest possible summary for this scope.
                const auto summarySize = analyseSummarySize((int32_t) blockWidth, scope);
                summaryTy = ArrayType::get(toSummaryType(b, summarySize), packSize);
                state.insert(state.begin(), summaryTy);
            }
        }

        carryStruct = StructType::get(b->getContext(), state);

        // If we're in a loop and cannot use collapsing carry mode, convert the carry state struct into a capacity,
        // carry state pointer, and summary pointer struct.
        if (LLVM_UNLIKELY(nonCarryCollapsingMode)) {
            mHasNonCarryCollapsingLoops = true;
            carryStruct = StructType::get(b->getContext(), {b->getSizeTy(), carryStruct->getPointerTo(), summaryTy->getPointerTo()});
        }
        cd.setNonCollapsingCarryMode(nonCarryCollapsingMode);
    }

    cd.setSummaryType(summaryType);
    return carryStruct;
}

CompressedCarryManager::CompressedCarryManager() noexcept
: CarryManager()
{}

} // namespace pablo
