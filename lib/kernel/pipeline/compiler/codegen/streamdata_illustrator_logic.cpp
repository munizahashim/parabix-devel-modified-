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
                                ss->getNumElements(), 1, ss->getFieldWidth(), MemoryOrdering::RowMajor,
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

            Value * produced = mCurrentProducedItemCountPhi[rt.Port];

            Value * const vba = getVirtualBaseAddress(b, rt, bn, produced, bn.isNonThreadLocal(), true);

            // TODO: if this kernel is state-free, we need to pass in some other value for the handle.
            // We can easily use kernel # for it here but what if we capture a value in the kernel itself?
            assert (mKernelSharedHandle);

            // TODO: should we pass the values of the min repetition vector to better group the output?

            // TODO: should buffers have row major streamsets?
            captureStreamData(b, b->GetString(mKernel->getName()), b->GetString(bind.Name), mKernelSharedHandle,
                              mInternallySynchronizedSubsegmentNumber,
                              buffer->getType(), MemoryOrdering::RowMajor,
                              vba, initial, current);
            return;
        }
    }
    llvm_unreachable("could not find illustrated streamset in binding list?");
}


}
