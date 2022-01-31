#ifndef CONSUMER_ANALYSIS_HPP
#define CONSUMER_ANALYSIS_HPP

#include "pipeline_analysis.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyCrossHybridThreadStreamSets
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyCrossHybridThreadStreamSets() {
    flat_set<unsigned> crossThreadStreamSets;
    for (unsigned kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        if (LLVM_UNLIKELY(KernelOnHybridThread.test(kernel))) {
            for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                BufferNode & streamSet = mBufferGraph[source(e, mBufferGraph)];
                streamSet.CrossesHybridThreadBarrier = true;
                assert (streamSet.Locality == BufferLocality::GloballyShared);
            }
            for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto buffer = target(e, mBufferGraph);
                BufferNode & streamSet = mBufferGraph[target(e, mBufferGraph)];
                if (LLVM_LIKELY(out_degree(buffer, mBufferGraph) > 0)) {
                    streamSet.CrossesHybridThreadBarrier = true;
                    assert (streamSet.Locality == BufferLocality::GloballyShared);
                }
            }

        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeConsumerGraph
 *
 * Copy the buffer graph but amalgamate any multi-edges into a single one
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::makeConsumerGraph() {

    mConsumerGraph = ConsumerGraph(LastStreamSet + 1);

    flat_set<unsigned> observedGlobalPortIds;

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        // copy the producing edge
        const auto pe = in_edge(streamSet, mBufferGraph);
        const BufferPort & output = mBufferGraph[pe];
        const auto producer = source(pe, mBufferGraph);
        add_edge(producer, streamSet, ConsumerEdge{output.Port, 0, ConsumerEdge::None}, mConsumerGraph);

        // If we have no consumers, we do not want to update the consumer count on exit
        // as we would then have to retain a scalar for it.

        const BufferNode & streamSetNode = mBufferGraph[streamSet];
        if (streamSetNode.Locality != BufferLocality::GloballyShared) {
            assert (!streamSetNode.CrossesHybridThreadBarrier);
            continue;
        }

        if (LLVM_UNLIKELY(out_degree(streamSet, mBufferGraph) == 0)) {
            assert (!streamSetNode.CrossesHybridThreadBarrier);
            continue;
        }

        const auto partitionId = KernelPartitionId[producer];

        // TODO: check gb18030. we can reduce the number of tests by knowing that kernel processes
        // the same amount of data so we only need to update this value after invoking the last one.

        std::array<unsigned, 2> lastThreadConsumer = { PipelineInput, PipelineInput };

        unsigned index = 0;

        for (const auto ce : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const auto consumer = target(ce, mBufferGraph);

            const auto onHybrid = KernelOnHybridThread.test(consumer);
            const auto type = onHybrid ? 1 : 0;
            auto & lastConsumer = lastThreadConsumer[type];
            lastConsumer = std::max<unsigned>(lastConsumer, consumer);

            if (KernelPartitionId[consumer] != partitionId) {
                const auto onHybrid = KernelOnHybridThread.test(consumer);
                const auto type = onHybrid ? 1 : 0;
                auto & lastConsumer = lastThreadConsumer[type];
                lastConsumer = std::max<unsigned>(lastConsumer, consumer);
                const BufferPort & input = mBufferGraph[ce];
                add_edge(streamSet, consumer, ConsumerEdge{input.Port, ++index, ConsumerEdge::UpdatePhi}, mConsumerGraph);
            }
        }

        assert (lastThreadConsumer[0] != 0 || lastThreadConsumer[1] != 0);

        // Although we may already know the final consumed item count prior
        // to executing the last consumer, we need to defer writing the final
        // consumed item count until the very last consumer reads the data.

        for (unsigned type = 0; type < 2; ++type) {
            // const auto lastConsumer = LastKernel;
            const auto lastConsumer = lastThreadConsumer[type];
            if (lastConsumer) {
                ConsumerGraph::edge_descriptor e;
                bool exists;
                std::tie(e, exists) = edge(streamSet, lastConsumer, mConsumerGraph);
                const auto flags = ConsumerEdge::WriteConsumedCount;
                if (exists) {
                    ConsumerEdge & cn = mConsumerGraph[e];
                    cn.Flags |= flags;
                } else {
                    add_edge(streamSet, lastConsumer, ConsumerEdge{output.Port, 0, flags}, mConsumerGraph);
                }
            }
        }


    }

    // If this is a pipeline input, we want to update the count at the end of the loop.
    for (const auto e : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        ConsumerGraph::edge_descriptor f;
        bool exists;
        std::tie(f, exists) = edge(streamSet, PipelineOutput, mConsumerGraph);
        const auto flags = ConsumerEdge::UpdateExternalCount;
        if (exists) {
            ConsumerEdge & cn = mConsumerGraph[f];
            cn.Flags |= flags;
        } else {
            const BufferPort & br = mBufferGraph[e];
            add_edge(streamSet, PipelineOutput, ConsumerEdge{br.Port, 0, flags}, mConsumerGraph);
        }
    }

#if 0

    auto & out = errs();

    out << "digraph \"ConsumerGraph\" {\n";
    for (auto v : make_iterator_range(vertices(mConsumerGraph))) {
        out << "v" << v << " [label=\"" << v << "\"];\n";
    }
    for (auto e : make_iterator_range(edges(mConsumerGraph))) {
        const auto s = source(e, mConsumerGraph);
        const auto t = target(e, mConsumerGraph);
        out << "v" << s << " -> v" << t <<
               " [label=\"";
        const ConsumerEdge & c = mConsumerGraph[e];
        if (c.Flags & ConsumerEdge::UpdatePhi) {
            out << 'U';
        }
        if (c.Flags & ConsumerEdge::WriteConsumedCount) {
            out << 'W';
        }
        if (c.Flags & ConsumerEdge::UpdateExternalCount) {
            out << 'E';
        }
        out << "\"];\n";
    }

    out << "}\n\n";
    out.flush();

#endif

}

}

#endif // CONSUMER_ANALYSIS_HPP
