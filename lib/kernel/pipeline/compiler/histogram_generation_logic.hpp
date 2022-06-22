#ifndef HISTOGRAM_GENERATION_LOGIC_HPP
#define HISTOGRAM_GENERATION_LOGIC_HPP

#include "pipeline_compiler.hpp"

namespace kernel {

namespace {

bool inline __trackPort(const BufferPort & br) {
    const Binding & bd = br.Binding;
    const ProcessingRate & pr = bd.getRate();
    switch (pr.getKind()) {
        case RateId::Fixed:
            // fixed rate doesn't need to be tracked as the only one that wouldn't be the exact rate would be
            // the final partial one but that isn't a very interesting value to model.
        case RateId::Greedy:
        case RateId::Unknown:
            // TODO: to support these, we'd need to use a non-static length histogram array but ideally we'd
            // want a sparse one.
            return false;
        default:
            return true;
    }
}


}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordsAnyHistogramData
 ** ------------------------------------------------------------------------------------------------------------- */
inline bool PipelineCompiler::recordsAnyHistogramData() const {
    if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram)) {
        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            if (__trackPort(br)) {
                return true;
            }
        }
        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            if (__trackPort(br)) {
                return true;
            }
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addHistogramProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addHistogramProperties(BuilderRef b, const size_t kernelId, const size_t groupId) {

    assert (mGenerateTransferredItemCountHistogram);

    IntegerType * const sizeTy = b->getSizeTy();

    const auto makeThreadLocalProps = mIsStatelessKernel.test(kernelId) || mIsInternallySynchronized.test(kernelId);

    for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        if (__trackPort(br)) {
            const auto prefix = makeBufferName(kernelId, br.Port);
            Type * const histTy = ArrayType::get(sizeTy, ceiling(br.Maximum) + 1);
            if (LLVM_UNLIKELY(makeThreadLocalProps)) {
                mTarget->addThreadLocalScalar(histTy, prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX, groupId,
                                              ThreadLocalScalarAccumulationRule::Sum);
            } else {
                mTarget->addInternalScalar(histTy, prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX, groupId);
            }
        }
    }

    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        if (__trackPort(br)) {
            const auto prefix = makeBufferName(kernelId, br.Port);
            Type * const histTy = ArrayType::get(sizeTy, ceiling(br.Maximum) + 1);
            if (LLVM_UNLIKELY(makeThreadLocalProps)) {
                mTarget->addThreadLocalScalar(histTy, prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX, groupId,
                                              ThreadLocalScalarAccumulationRule::Sum);
            } else {
                mTarget->addInternalScalar(histTy, prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX, groupId);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateTransferredItemsForHistogramData
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateTransferredItemsForHistogramData(BuilderRef b) {

    assert (mGenerateTransferredItemCountHistogram);

    FixedArray<Value *, 2> args;
    args[0] = b->getSize(0);

    ConstantInt * const sz_ONE = b->getSize(1);

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        if (__trackPort(br)) {
            const auto inputPort = br.Port;
            const auto prefix = makeBufferName(mKernelId, inputPort);
            Value * const histBaseAddr = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
            Value * const diff = b->CreateSub(mProcessedItemCount[inputPort], mCurrentProcessedItemCountPhi[inputPort]);
            if (LLVM_UNLIKELY(CheckAssertions)) {
                ArrayType * ty = cast<ArrayType>(histBaseAddr->getType()->getPointerElementType());
                Value * const maxSize = b->getSize(ty->getArrayNumElements() - 1);
                Value * const valid = b->CreateICmpULE(diff, maxSize);
                Constant * const bindingName = b->GetString(br.Binding.get().getName());
                b->CreateAssert(valid, "%s.%s: attempting to update %" PRIu64 "-th value of histogram data "
                                       "but internal array only has %" PRIx64 " elements",
                                        mCurrentKernelName, bindingName, diff, maxSize);
            }
            args[1] = diff;
            Value * const toInc = b->CreateGEP(histBaseAddr, args);
            b->CreateStore(b->CreateAdd(b->CreateLoad(toInc), sz_ONE), toInc);
        }
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & br = mBufferGraph[e];
        if (__trackPort(br)) {
            const auto outputPort = br.Port;
            const auto prefix = makeBufferName(mKernelId, outputPort);
            Value * histBaseAddr = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
            Value * const diff = b->CreateSub(mProducedItemCount[outputPort], mCurrentProducedItemCountPhi[outputPort]);
            if (LLVM_UNLIKELY(CheckAssertions)) {
                ArrayType * ty = cast<ArrayType>(histBaseAddr->getType()->getPointerElementType());
                Value * const maxSize = b->getSize(ty->getArrayNumElements() - 1);
                Value * const valid = b->CreateICmpULE(diff, maxSize);
                Constant * const bindingName = b->GetString(br.Binding.get().getName());
                b->CreateAssert(valid, "%s.%s: attempting to update %" PRIu64 "-th value of histogram data "
                                       "but internal array only has %" PRIu64 " elements",
                                        mCurrentKernelName, bindingName, diff, maxSize);
            }
            args[1] = diff;
            Value * const toInc = b->CreateGEP(histBaseAddr, args);
            b->CreateStore(b->CreateAdd(b->CreateLoad(toInc), sz_ONE), toInc);
        }
    }

}


}

#endif // HISTOGRAM_GENERATION_LOGIC_HPP
