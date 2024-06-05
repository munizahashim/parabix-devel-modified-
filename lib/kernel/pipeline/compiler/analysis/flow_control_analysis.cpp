#include "pipeline_analysis.hpp"

#include <queue>

namespace kernel {

void PipelineAnalysis::addFlowControlAnnotations() {

    size_t firstPartitionId = KernelPartitionId[FirstKernel];
    size_t lastPartitionId = KernelPartitionId[LastKernel];

    AllowIOProcessThread = false;

    if (codegen::UseProcessThreadForIO && !IsNestedPipeline) {

        for (; firstPartitionId < lastPartitionId; ++firstPartitionId) {
            const auto kernelId = FirstKernelInPartition[firstPartitionId];
            if (in_degree(kernelId, mBufferGraph) > 0) {
                break;
            }
        }
        assert (firstPartitionId > 0);
        for (; lastPartitionId >= firstPartitionId; --lastPartitionId) {
            const auto kernelId = FirstKernelInPartition[lastPartitionId];
            if (out_degree(kernelId, mBufferGraph) != 0) {
                break;
            }
        }

        if (LLVM_UNLIKELY(KernelPartitionId[FirstKernel] == firstPartitionId && lastPartitionId == KernelPartitionId[LastKernel])) {
            // No kernels that can be isolated? outside of a nested pipeline, the only
            // way for this to occur is if all input and output are transferred through
            // pipeline I/O streamsets.
        } else {
            #ifndef NDEBUG
            for (auto kernel = FirstKernelInPartition[firstPartitionId]; kernel < FirstKernelInPartition[lastPartitionId + 1]; ++kernel) {
                assert (in_degree(kernel, mBufferGraph) > 0);
                assert (out_degree(kernel, mBufferGraph) > 0);
            }
            #endif
            for (auto kernel = PipelineInput; kernel < FirstKernelInPartition[firstPartitionId]; ++kernel) {
                assert (in_degree(kernel, mBufferGraph) == 0);
                for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(e, mBufferGraph);
                    auto & bn = mBufferGraph[streamSet];
                    bn.Type |= BufferType::CrossThreaded;
                    if (LLVM_UNLIKELY(bn.isThreadLocal())) {
                        bn.Locality = BufferLocality::GloballyShared;
                    }
                }
            }

            for (auto kernel = FirstKernelInPartition[lastPartitionId + 1]; kernel <= PipelineOutput; ++kernel) {
                assert (out_degree(kernel, mBufferGraph) == 0);
                for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    const auto streamSet = source(e, mBufferGraph);
                    auto & bn = mBufferGraph[streamSet];
                    bn.Type |= BufferType::CrossThreaded;
                    if (LLVM_UNLIKELY(bn.isThreadLocal())) {
                        bn.Locality = BufferLocality::GloballyShared;
                    }
                }
            }
            AllowIOProcessThread = true;
        }

    }

    assert (firstPartitionId <= lastPartitionId);
    assert (lastPartitionId < PartitionCount);

    FirstComputePartitionId = firstPartitionId;
    LastComputePartitionId = lastPartitionId;

    for (size_t partitionId = firstPartitionId; partitionId <= lastPartitionId; ++partitionId) {
        const auto kernelId = FirstKernelInPartition[partitionId];
        if (MinimumNumOfStrides[kernelId] != MaximumNumOfStrides[kernelId] || IsNestedPipeline) {
            mBufferGraph[kernelId].Type |= PermitSegmentSizeSlidingWindowing;
        }
    }


}

}
