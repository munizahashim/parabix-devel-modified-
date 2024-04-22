#include "pipeline_analysis.hpp"
#include <queue>

namespace kernel {

void PipelineAnalysis::analyzePrincipalRateIO() {

//    using Graph = adjacency_list<vecS, vecS, bidirectionalS, Rational, Rational>;

    for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {

        BufferGraph::in_edge_iterator ei_begin, ei_end;
        std::tie(ei_begin, ei_end) = in_edges(kernel, mBufferGraph);

        for (auto ei = ei_begin; ei != ei_end; ++ei) {
            BufferPort & bp = mBufferGraph[*ei];
            if (bp.isPrincipal()) {
                assert (bp.isFixed());
                const auto streamSet = source(*ei, mBufferGraph);
                const auto output = in_edge(streamSet, mBufferGraph);
                const auto producer = source(output, mBufferGraph);
                // if all of the fixed inputs are from the same partition as the principal input,
                // we can ignore it.
                const auto partId = KernelPartitionId[producer];
                bool fromSamePartition = true;
                for (auto ej = ei; ++ej != ei_end; ) {
                    const BufferPort & bp = mBufferGraph[*ej];
                    if (bp.isFixed()) {
                        const auto streamSet = source(*ej, mBufferGraph);
                        const auto producer = parent(streamSet, mBufferGraph);
                        if (KernelPartitionId[producer] != partId) {
                            fromSamePartition = false;
                            break;
                        }
                    }
                }

                if (fromSamePartition) {
                    bp.Flags &= ~BufferPortType::IsPrincipal;
                } else {





//                    if (LLVM_LIKELY(mBufferGraph[output].isFixed())) {

//                        Graph G(LastStreamSet + 1U);
//                        std::queue<BufferGraph::vertex_descriptor> Q;

//                        Q.push(streamSet);
//                        for (;;) {







//                        }


//                    }



                }
                break;
            }
            if (bp.isFixed()) {
                break;
            }
        }
    }

}


}
