#include "pipeline_analysis.hpp"

#include <queue>

namespace kernel {

void PipelineAnalysis::addFlowControlAnnotations() {

    flat_set<size_t> fixedSegmentLength;

    if (codegen::UseProcessThreadForIO && !IsNestedPipeline) {
        std::queue<size_t> Q;
        flat_set<size_t> S;
        BitVector V(LastKernel + 1);

        for (auto kernel = PipelineInput; kernel <= LastKernel; ++kernel) {
            if (in_degree(kernel, mBufferGraph) == 0) {
                assert (S.empty());
                assert (V.none());
                for (auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(e, mBufferGraph);
                    for (auto f : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                        const auto consumer = target(f, mBufferGraph);
                        S.insert(consumer);
                    }
                }
                for (auto s : S) {
                    if (V.test(s) == 0) {
                        const auto f = PartitionIds.find(s);
                        assert (f != PartitionIds.end());
                        const auto partitionId = f->second;
                        fixedSegmentLength.insert(partitionId);
                        assert (FirstKernelInPartition[partitionId] <= s);
                        assert (Q.empty());
                        Q.push(s);
                        for (;;) {
                            const auto u = Q.front();
                            for (auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
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
                S.clear();
                V.reset();
            }
        }
    }

    for (size_t partitionId = 0; partitionId < PartitionCount; ++partitionId) {
        const auto kernelId = FirstKernelInPartition[partitionId];
        if (fixedSegmentLength.count(partitionId)) {
            assert (!IsNestedPipeline);
            mBufferGraph[kernelId].Type |= InitialSourceConsumer;
        } else {
            if (MinimumNumOfStrides[kernelId] != MaximumNumOfStrides[kernelId] || IsNestedPipeline) {
                mBufferGraph[kernelId].Type |= PermitSegmentSizeSlidingWindowing;
            }
        }
    }

}

}
