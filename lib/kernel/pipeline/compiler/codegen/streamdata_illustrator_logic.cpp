#include "../pipeline_compiler.hpp"

namespace kernel {


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief registerStreamSetIllustrator
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::registerStreamSetIllustrator(BuilderRef b, const size_t streamSet) const {

    const auto & illustratorBindings = cast<PipelineKernel>(mTarget)->getIllustratorBindings();
    const StreamSet * const ss = cast<StreamSet>(mStreamGraph[streamSet].Relationship);
    for (const auto & bind : illustratorBindings) {
        if (bind.StreamSet == ss) {

            const auto & bn = mBufferGraph[streamSet];

            assert (bn.IsLinear);

            StreamSetBuffer * const buffer = bn.Buffer; assert (buffer);

            assert (mKernelSharedHandle);

            assert (mKernel);

            // TODO: should buffers have row major streamsets?
            registerIllustrator(b,
                                b->getScalarField(KERNEL_ILLUSTRATOR_CALLBACK_OBJECT),
                                b->GetString(mKernel->getName()),
                                b->GetString(bind.Name),
                                mKernelSharedHandle,
                                buffer->getType(), MemoryOrdering::ColumnMajor,
                                bind.IllustratorType, bind.ReplacementCharacter[0], bind.ReplacementCharacter[1]);
            return;
        }
    }
    llvm_unreachable("could not find illustrated streamset in binding list?");
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief illustrateStreamSet
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::illustrateStreamSet(BuilderRef b, const size_t streamSet, Value * const initial, Value * const current) const {

    assert (mInternallySynchronizedSubsegmentNumber);
    const auto & illustratorBindings = cast<PipelineKernel>(mTarget)->getIllustratorBindings();
    const StreamSet * const ss = cast<StreamSet>(mStreamGraph[streamSet].Relationship);
    for (const auto & bind : illustratorBindings) {
        if (bind.StreamSet == ss) {

            const auto & bn = mBufferGraph[streamSet];

            assert (bn.IsLinear);

            StreamSetBuffer * const buffer = bn.Buffer;

            const auto & rt = mBufferGraph[in_edge(streamSet, mBufferGraph)];

            Value * baseProduced = nullptr;
            if (rt.isDeferred()) {
                baseProduced = mAlreadyProducedDeferredPhi[rt.Port];
            } else {
                baseProduced = mAlreadyProducedPhi[rt.Port];
            }

            Value * const vba = getVirtualBaseAddress(b, rt, bn, baseProduced, bn.isNonThreadLocal(), true);

            ExternalBuffer tmp(0, b, buffer->getBaseType(), true, buffer->getAddressSpace());
            Constant * const LOG_2_BLOCK_WIDTH = b->getSize(floor_log2(b->getBitBlockWidth()));
            Value * const blockIndex = b->CreateLShr(initial, LOG_2_BLOCK_WIDTH);
            Value * const addr = tmp.getStreamBlockPtr(b, vba, b->getSize(0), blockIndex);

            // TODO: if this kernel is state-free, we need to pass in some other value for the handle.
            // We can easily use kernel # for it here but what if we capture a value in the kernel itself?
            assert (mKernelSharedHandle);

            // TODO: should we pass the values of the min repetition vector to better group the output?

            // TODO: should buffers have row major streamsets?
            captureStreamData(b, mCurrentKernelName, b->GetString(bind.Name), mKernelSharedHandle,
                              mInternallySynchronizedSubsegmentNumber,
                              buffer->getType(), MemoryOrdering::ColumnMajor,
                              addr, initial, current);
            return;
        }
    }
    llvm_unreachable("could not find illustrated streamset in binding list?");
}


}
