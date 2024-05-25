#include "pipeline_analysis.hpp"

#include <queue>

namespace kernel {

void PipelineAnalysis::addFlowControlAnnotations() {

    flat_set<size_t> fixedSegmentLength;

    size_t firstPartitionId = KernelPartitionId[FirstKernel];
    size_t lastPartitionId = KernelPartitionId[LastKernel];

    AllowIOProcessThread = false;

    if (codegen::UseProcessThreadForIO && !IsNestedPipeline) {
        std::queue<size_t> Q;
        flat_set<size_t> consumers;
        flat_set<size_t> found;
        BitVector V(num_vertices(mBufferGraph));

        for (auto kernel = PipelineInput; kernel <= LastKernel; ++kernel) {
            if (in_degree(kernel, mBufferGraph) == 0) {
                assert (consumers.empty());
                assert (V.none());
                for (auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(e, mBufferGraph);
                    for (auto f : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                        const auto consumer = target(f, mBufferGraph);
                        assert (KernelPartitionId[kernel] != KernelPartitionId[consumer]);
                        consumers.insert(consumer);
                    }
                }
                for (auto s : consumers) {
                    if (V.test(s) == 0) {
                        // determine the first port to read from this kernel

                        for (auto e : make_iterator_range(in_edges(s, mBufferGraph))) {
                            const auto streamSet = source(e, mBufferGraph);
                            for (auto f : make_iterator_range(in_edges(streamSet, mBufferGraph))) {
                                const auto producer = source(f, mBufferGraph);
                                if (producer == kernel) {
                                    mBufferGraph[e].Flags |= BufferPortType::InitialIOThreadRead;
                                    goto found_match;
                                }
                            }
                        }

found_match:

                        const auto partitionId = KernelPartitionId[s];
                        fixedSegmentLength.insert(partitionId);
                        assert (FirstKernelInPartition[partitionId] <= s);
                        assert (s < FirstKernelInPartition[partitionId + 1]);
                        assert (Q.empty());
                        Q.push(s);
                        for (;;) {
                            const auto u = Q.front();
                            Q.pop();
                            for (auto e : make_iterator_range(out_edges(u, mBufferGraph))) {
                                const auto v = target(e, mBufferGraph);
                                if (!V.test(v)) {
                                    V.set(v);
                                    Q.push(s);
                                }
                            }
                            if (Q.empty()) {
                                break;
                            }
                        }
                    }
                }
                consumers.clear();
                V.reset();
            }
        }

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
                    mBufferGraph[streamSet].Type |= BufferType::CrossThreaded;
                }
            }

            for (auto kernel = FirstKernelInPartition[lastPartitionId + 1]; kernel <= PipelineOutput; ++kernel) {
                assert (out_degree(kernel, mBufferGraph) == 0);
                for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    const auto streamSet = source(e, mBufferGraph);
                    mBufferGraph[streamSet].Type |= BufferType::CrossThreaded;
                }
            }
            AllowIOProcessThread = true;
        }

    }

    assert (firstPartitionId <= lastPartitionId);
    assert (lastPartitionId < PartitionCount);

    FirstComputePartitionId = firstPartitionId;
    LastComputePartitionId = lastPartitionId;


    for (size_t partitionId = 0; partitionId < PartitionCount; ++partitionId) {
        const auto kernelId = FirstKernelInPartition[partitionId];
        if (fixedSegmentLength.count(partitionId)) {
            assert (!IsNestedPipeline);
            mBufferGraph[kernelId].Type |= InitialSourceConsumer;
        } else if (partitionId <= LastComputePartitionId) {
            if (MinimumNumOfStrides[kernelId] != MaximumNumOfStrides[kernelId] || IsNestedPipeline) {
                mBufferGraph[kernelId].Type |= PermitSegmentSizeSlidingWindowing;
            }
        }
    }

}

}
