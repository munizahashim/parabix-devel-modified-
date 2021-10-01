#ifndef CONSUMER_ANALYSIS_HPP
#define CONSUMER_ANALYSIS_HPP

#include "pipeline_analysis.hpp"
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>

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

#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeConsumerGraph
 *
 * Copy the buffer graph but amalgamate any multi-edges into a single one
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::makeConsumerGraph() {

    std::vector<unsigned> level(PartitionCount);

    assert (PartitionCount > 1);

    level[PartitionCount - 1] = 0;
    for (unsigned i = PartitionCount - 1; i-- != 0; ) {
        const auto j = mPartitionJumpIndex[i];
        assert (j > i);
        level[i] = level[j] + 1;
    }

    std::vector<unsigned> firstKernelInPartition(PartitionCount);
    auto currentPartitionId = KernelPartitionId[PipelineInput];
    firstKernelInPartition[currentPartitionId] = PipelineInput;

    for (auto kernel = PipelineInput; kernel <= LastKernel; ++kernel) {
        const auto partId = KernelPartitionId[kernel];
        if (currentPartitionId != partId) {
            firstKernelInPartition[partId] = kernel;
            currentPartitionId = partId;
        }
    }
    firstKernelInPartition[PartitionCount - 1] = LastKernel;

    mConsumerGraph = ConsumerGraph(LastStreamSet + 1);

    BitVector forcedWritesToLast(LastStreamSet - FirstStreamSet + 1);

    if (LLVM_UNLIKELY(codegen::ForceStreamSetConsumerWriteToLastKernel != codegen::OmittedOption)) {

        tokenizer<escaped_list_separator<char>> ids(codegen::ForceStreamSetConsumerWriteToLastKernel);
        for (const auto & id : ids) {
            const auto num = std::stoi(id);
            if (num >= FirstStreamSet && num <= LastStreamSet) {
                forcedWritesToLast.set(num - FirstStreamSet);
            }
        }

    }

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        // copy the producing edge
        const auto pe = in_edge(streamSet, mBufferGraph);
        const BufferPort & output = mBufferGraph[pe];
        const unsigned producer = source(pe, mBufferGraph);
        add_edge(producer, streamSet, ConsumerEdge{output.Port, 0, ConsumerEdge::None}, mConsumerGraph);

        // If we have no consumers, we do not want to update the consumer count on exit
        // as we would then have to retain a scalar for it.

        const BufferNode & streamSetNode = mBufferGraph[streamSet];
        if (streamSetNode.Locality == BufferLocality::ThreadLocal) {
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

        std::array<unsigned, 2> consumerLCA = { producer, producer };



        unsigned index = 0;




        for (const auto ce : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
            const auto consumer = target(ce, mBufferGraph);
            const auto consPartId = KernelPartitionId[consumer];

            const auto onHybrid = KernelOnHybridThread.test(consumer);
            const auto type = onHybrid ? 1 : 0;

            auto & lastConsumer = consumerLCA[type];
            // find the LCA of the last known consumer and this one
            auto a = KernelPartitionId[lastConsumer];
            auto b = KernelPartitionId[consumer];
            auto t = std::max<unsigned>(lastConsumer, consumer);
            for (;;) {
                const auto la = level[a];
                const auto lb = level[b];
                if (la < lb) {
                    b = mPartitionJumpIndex[b]; // parent
                } else if (la > lb) {
                    a = mPartitionJumpIndex[a]; // parent
                } else {
                    t = std::max(t, firstKernelInPartition[a]);
                    break;
                }
            }
            lastConsumer = t;

            if (KernelPartitionId[consumer] != partitionId) {
                const BufferPort & input = mBufferGraph[ce];
                add_edge(streamSet, consumer, ConsumerEdge{input.Port, ++index, ConsumerEdge::UpdatePhi}, mConsumerGraph);
            }
        }

        // If this is a pipeline input, we always update the count at the end of the loop.
        if (LLVM_UNLIKELY(producer == PipelineOutput)) {
            consumerLCA[0] = PipelineOutput;
            consumerLCA[1] = PipelineOutput;
        }



        // Although we may already know the final consumed item count prior
        // to executing the last consumer, we need to defer writing the final
        // consumed item count until the very last consumer reads the data.

        for (unsigned type = 0; type < 2; ++type) {
            const auto ftl = forcedWritesToLast.test(streamSet - FirstStreamSet);
            auto lastConsumer = ftl ? 6U : consumerLCA[type];


            #ifdef FORCE_LAST_KERNEL_CONSUMER_WRITE_FOR_STREAMSETS
            lastConsumer = LastKernel;
            #endif

            if (lastConsumer != producer) {
                ConsumerGraph::edge_descriptor e;
                bool exists;
                std::tie(e, exists) = edge(streamSet, lastConsumer, mConsumerGraph);
                const auto flags = ConsumerEdge::WriteConsumedCount;
                if (exists) {
                    ConsumerEdge & cn = mConsumerGraph[e];
                    cn.Flags |= flags;
                } else {
                    add_edge(streamSet, lastConsumer, ConsumerEdge{output.Port, ++index, flags}, mConsumerGraph);
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

#else

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeConsumerGraph
 *
 * Copy the buffer graph but amalgamate any multi-edges into a single one
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::makeConsumerGraph() {

    using BitSet = dynamic_bitset<>;

    using PartitionJumpTree = adjacency_list<vecS, vecS, bidirectionalS>;


    std::vector<unsigned> level(PartitionCount);

    assert (PartitionCount > 1);

    level[PartitionCount - 1] = 0;
    for (unsigned i = PartitionCount - 1; i-- != 0; ) {
        const auto j = mPartitionJumpIndex[i];
        assert (j > i);
        level[i] = level[j] + 1;
    }

    std::vector<unsigned> firstKernelInPartition(PartitionCount);
    auto currentPartitionId = KernelPartitionId[PipelineInput];
    firstKernelInPartition[currentPartitionId] = PipelineInput;

    for (auto kernel = PipelineInput; kernel <= LastKernel; ++kernel) {
        const auto partId = KernelPartitionId[kernel];
        if (currentPartitionId != partId) {
            firstKernelInPartition[partId] = kernel;
            currentPartitionId = partId;
        }
    }
    firstKernelInPartition[PartitionCount - 1] = LastKernel;

    mConsumerGraph = ConsumerGraph(LastStreamSet + 1);

//    BitVector forcedWritesToLast(LastStreamSet - FirstStreamSet + 1);

//    if (LLVM_UNLIKELY(codegen::ForceStreamSetConsumerWriteToLastKernel != codegen::OmittedOption)) {

//        tokenizer<escaped_list_separator<char>> ids(codegen::ForceStreamSetConsumerWriteToLastKernel);
//        for (const auto & id : ids) {
//            const auto num = std::stoi(id);
//            if (num >= FirstStreamSet && num <= LastStreamSet) {
//                forcedWritesToLast.set(num - FirstStreamSet);
//            }
//        }

//    }

    PartitionJumpTree J(PartitionCount);

    for (auto i = 0U; i < (PartitionCount - 1); ++i) {
        const auto j = mPartitionJumpIndex[i];
        assert (j > i && j < PartitionCount);
        assert (PartitionOnHybridThread.test(i) == PartitionOnHybridThread.test(j) || j == KernelPartitionId[PipelineOutput]);
        add_edge(i, j, J);

    }

    std::vector<BitSet> ancestors(PartitionCount);

    for (auto i = 0U; i < PartitionCount; ++i) {
        ancestors[i].resize(PartitionCount);
    }

    std::vector<std::bitset<2>> onPath(PartitionCount);

    std::queue<unsigned> Q;

    std::map<BitSet, unsigned> M;

    flat_set<unsigned> I;

    std::vector<unsigned> lastSlottedConsumer(PartitionCount);

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        // copy the producing edge
        const auto pe = in_edge(streamSet, mBufferGraph);
        const BufferPort & output = mBufferGraph[pe];
        const unsigned producer = source(pe, mBufferGraph);
        add_edge(producer, streamSet, ConsumerEdge{output.Port, 0, ConsumerEdge::None}, mConsumerGraph);

        // If we have no consumers, we do not want to update the consumer count on exit
        // as we would then have to retain a scalar for it.

        const BufferNode & streamSetNode = mBufferGraph[streamSet];
        if (streamSetNode.Locality == BufferLocality::ThreadLocal) {
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

        if (LLVM_UNLIKELY(mTraceIndividualConsumedItemCounts)) {

            unsigned index = 0;

            for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const auto consumer = target(e, mBufferGraph);
                const auto consPartId = KernelPartitionId[consumer];

                if (consPartId != partitionId) {
                    const BufferPort & input = mBufferGraph[e];
                    constexpr auto flags = ConsumerEdge::UpdatePhi | ConsumerEdge::WriteConsumedCount;
                    add_edge(streamSet, consumer, ConsumerEdge{input.Port, ++index, flags}, mConsumerGraph);
                }
            }

        } else {

            for (auto i = 0U; i < PartitionCount; ++i) {
                ancestors[i].reset();
                onPath[i].reset();
            }

            assert (Q.empty());

            Q.push(partitionId);
            for (;;) {
                const auto r = Q.front();
                Q.pop();
                auto & A = onPath[r];
                A.set(1);
                assert (out_degree(r, J) < 2);
                if (out_degree(r, J) != 0) {
                    const auto t = child(r, J);
                    Q.push(t);
                } else if (Q.empty()) {
                    break;
                }
            }

            assert (I.empty());

            for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const auto consumer = target(e, mBufferGraph);
                I.insert(KernelPartitionId[consumer]);
            }

            for (auto r : I) {
                assert (Q.empty());
                for (;;) {
                    auto & A = onPath[r];
                    if (!A.test(0)) {
                        A.set(0);
                        for (auto e : make_iterator_range(in_edges(r, J))) {
                            Q.push(source(e, J));
                        }
                    }
                    if (Q.empty()) {
                        break;
                    }
                    r = Q.front();
                    Q.pop();
                }
            }

            assert (Q.empty());

            assert (KernelPartitionId[PipelineOutput] == PartitionCount - 1);

            Q.push(PartitionCount - 1);

            unsigned pathCount = 0;

            for (;;) {

                const auto r = Q.front();
                Q.pop();

                auto & A = ancestors[r];
                if (onPath[r].all()) {
                    if (A.none()) {
                        A.set(pathCount++);
                    }
                } else {
                    A.reset();
                }

                unsigned numAncestors = 0;
                for (auto e : make_iterator_range(in_edges(r, J))) {
                    const auto s = source(e, J);
                    if (onPath[s].all()) {
                        ++numAncestors;
                    }
                }

                for (auto e : make_iterator_range(in_edges(r, J))) {
                    const auto s = source(e, J);
                    assert (pathCount < PartitionCount);
                    auto & B = ancestors[s];
                    B |= A;
                    if (numAncestors > 1) {
                        B.set(pathCount++);
                    }
                    Q.push(s);
                }

                if (Q.empty()) {
                    break;
                }
            }

            assert (pathCount > 0);

            assert (M.empty());
            // 'I' acts to ensure a topological partition order
            for (const auto partId : I) {
                const auto & A = ancestors[partId];
                assert (A.any());
                auto f = M.find(A);
                if (f == M.end()) {
                    M.insert(std::make_pair(A, M.size()));
                }
            }

            I.clear();

            const auto m = M.size();

            assert (m < PartitionCount);

            std::fill_n(lastSlottedConsumer.begin(), m, PipelineInput);

            for (const auto e : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const auto consumer = target(e, mBufferGraph);
                assert (consumer != PipelineOutput);
                const auto consPartId = KernelPartitionId[consumer];
                const auto & A = ancestors[consPartId];
                assert (A.any());
                const auto f = M.find(A);
                assert (f != M.end());
                const auto index = f->second;
                auto & lastSlotted = lastSlottedConsumer[index];
                lastSlotted = std::max<unsigned>(lastSlotted, consumer);
                const BufferPort & input = mBufferGraph[e];
                add_edge(streamSet, consumer, ConsumerEdge{input.Port, index + 1U, ConsumerEdge::UpdatePhi}, mConsumerGraph);
            }

            M.clear();

            for (unsigned i = 0; i != m; ++i) {

                const auto lastConsumer = lastSlottedConsumer[i];
                assert (lastConsumer > producer);
//                if (forcedWritesToLast.test(streamSet - FirstStreamSet)) {
//                    lastConsumer = lastConsumer + 1;
//                }

                ConsumerGraph::edge_descriptor f;
                bool exists;
                std::tie(f, exists) = edge(streamSet, lastConsumer, mConsumerGraph);
                assert (exists);
                ConsumerEdge & cn = mConsumerGraph[f];
                cn.Flags |= ConsumerEdge::WriteConsumedCount;

            }

        }
    }

    // If this is a pipeline input, we want to update the count at the end of the loop.
    for (const auto e : make_iterator_range(out_edges(PipelineInput, mBufferGraph))) {
        const auto streamSet = target(e, mBufferGraph);
        ConsumerGraph::edge_descriptor f;
        bool exists;
        std::tie(f, exists) = edge(streamSet, PipelineOutput, mConsumerGraph);
        constexpr auto flags = ConsumerEdge::UpdateExternalCount;
        if (LLVM_UNLIKELY(exists)) {
            ConsumerEdge & cn = mConsumerGraph[f];
            cn.Flags |= flags;
        } else {
            const BufferPort & br = mBufferGraph[e];
            add_edge(streamSet, PipelineOutput, ConsumerEdge{br.Port, 0, flags}, mConsumerGraph);
        }
    }

#if 0

    BEGIN_SCOPED_REGION
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
    END_SCOPED_REGION

#endif



}

#endif

}

#endif // CONSUMER_ANALYSIS_HPP
