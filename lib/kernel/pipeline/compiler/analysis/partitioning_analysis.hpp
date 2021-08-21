#ifndef PARTITIONING_ANALYSIS_HPP
#define PARTITIONING_ANALYSIS_HPP

#include "pipeline_analysis.hpp"
#include <toolchain/toolchain.h>
#include <util/slab_allocator.h>

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyKernelPartitions
 ** ------------------------------------------------------------------------------------------------------------- */
PartitionGraph PipelineAnalysis::identifyKernelPartitions() {

    using BitSet = dynamic_bitset<>;

    using BindingVertex = RelationshipGraph::vertex_descriptor;

    using Graph = adjacency_list<vecS, vecS, bidirectionalS, BitSet, BindingVertex>;

    using PartitionMap = std::map<BitSet, unsigned>;

    const unsigned n = num_vertices(Relationships);

    std::vector<unsigned> sequence;
    sequence.reserve(n);

    std::vector<unsigned> mapping(n, -1U);

    unsigned numOfKernels = 2;

    BEGIN_SCOPED_REGION

    std::vector<unsigned> ordering;
    ordering.reserve(n);
    if (LLVM_UNLIKELY(!lexical_ordering(Relationships, ordering))) {
        report_fatal_error("Failed to generate acyclic partition graph from kernel ordering");
    }

    // Convert the relationship graph into a simpler graph G that we can annotate.
    // For simplicity, force the pipeline input to be the first and the pipeline output
    // to be the last one.

    // For some reason, the Mac C++ compiler cannot link the constexpr PipelineInput value?
    // Hardcoding 0 here as a temporary workaround.
    mapping[0] = 0;
    sequence.push_back(0);

    for (unsigned u : ordering) {
        const RelationshipNode & node = Relationships[u];
        switch (node.Type) {
            case RelationshipNode::IsKernel:
                BEGIN_SCOPED_REGION
                #ifndef NDEBUG
                const auto & R = Relationships[u];
                #endif
                if (u == PipelineInput || u == PipelineOutput) {
                    assert (R.Kernel == mPipelineKernel);
                } else {
                    assert (R.Kernel != mPipelineKernel);
                    mapping[u] = sequence.size();
                    sequence.push_back(u);
                    #ifndef NDEBUG
                    ++numOfKernels;
                    #endif
                }
                END_SCOPED_REGION
                break;
            case RelationshipNode::IsRelationship:
                BEGIN_SCOPED_REGION
                const Relationship * const ss = Relationships[u].Relationship;
                if (LLVM_LIKELY(isa<StreamSet>(ss))) {
                    mapping[u] = sequence.size();
                    sequence.push_back(u);
                }
                END_SCOPED_REGION
                break;
            default: break;
        }
    }

    mapping[PipelineOutput] = sequence.size();
    sequence.push_back(PipelineOutput);

    END_SCOPED_REGION
    const auto m = sequence.size();

    Graph G(m);

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {
            addKernelRelationshipsInReferenceOrdering(u, Relationships,
                [&](const PortType type, const unsigned binding, const unsigned streamSet) {
                    const auto j = mapping[streamSet];
                    assert (j < m);
                    assert (sequence[j] == streamSet);
                    auto a = i, b = j;
                    if (type == PortType::Input) {
                        a = j; b = i;
                    }
                    assert (a < b);
                    assert (Relationships[binding].Type == RelationshipNode::IsBinding);
                    add_edge(a, b, binding, G);
                }
            );
        }
    }

    // Stage 1: identify synchronous components

    // wcan through the graph and determine where every non-Fixed relationship exists
    // so that we can construct our initial set of partitions. The goal here is to act
    // as a naive first pass to simplify the problem before using Z3.

    // NOTE: any decisions made during this pass *must* be provably correct for any
    // situation because the choices will *not* be verified.

    for (unsigned i = 0; i < m; ++i) {
        BitSet & V = G[i];
        V.resize(n);
    }

    struct AttributeClassifier {

        dynamic_bitset<> BitSet;
        AttrId Type;
        unsigned Amount;

        AttributeClassifier(AttributeClassifier && a)
        : BitSet(std::move(a.BitSet)), Type(a.Type), Amount(a.Amount) { }

        AttributeClassifier(const dynamic_bitset<> & bitSet, AttrId type, const unsigned amount)
        : BitSet(bitSet), Type(type), Amount(amount) {

        }
    };

    struct AttributeClassifierComp {
        bool operator()(const AttributeClassifier & lhs, const AttributeClassifier & rhs) const {
            if (lhs.Type != rhs.Type) {
                return (unsigned)lhs.Type < (unsigned)rhs.Type;
            }
            if (lhs.Amount != rhs.Amount) {
                return lhs.Amount < rhs.Amount;
            }
            return lhs.BitSet < rhs.BitSet;
        }
    };

    std::map<AttributeClassifier, unsigned, AttributeClassifierComp> attrBitIds;

    unsigned nextRateId = 0;

    for (unsigned i = 0; i < m; ++i) {

        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];

        if (node.Type == RelationshipNode::IsKernel) {

            BitSet & V = G[i];

            if (in_degree(i, G) == 0) {
                if (out_degree(i, G) > 0) {
                    V.set(nextRateId++);
                } else {
                    assert (node.Kernel == mPipelineKernel);
                }
            } else {
                for (const auto e : make_iterator_range(in_edges(i, G))) {

                    const auto bindingId = G[e];

                    const RelationshipNode & rn = Relationships[bindingId];
                    assert (rn.Type == RelationshipNode::IsBinding);
                    const Binding & b = rn.Binding;
                    const ProcessingRate & rate = b.getRate();

                    // BitSet & I = G[source(e, G)];


                    for (const auto e : make_iterator_range(in_edges(i, G))) {
                        const BitSet & I = G[source(e, G)];
                        V |= I;
                    }

                    if (rate.getKind() == RateId::Fixed) {

                        // Check the attributes to see whether any impose a partition change
                        for (const Attribute & attr : b.getAttributes()) {
                            switch (attr.getKind()) {
                                case AttrId::Delayed:
                                case AttrId::BlockSize:
                                case AttrId::LookAhead:
                                    BEGIN_SCOPED_REGION
                                    AttributeClassifier key(V, attr.getKind(), attr.amount());
                                    const auto f = attrBitIds.find(key);
                                    if (f != attrBitIds.end()) {
                                        const auto k = f->second;
                                        assert (k < nextRateId);
                                        V.set(k);
                                    } else {
                                        attrBitIds.emplace(std::move(key), nextRateId);
                                        assert (nextRateId < V.capacity());
                                        V.set(nextRateId++);
                                    }
                                    END_SCOPED_REGION
                                default: break;
                            }
                        }

                    } else {
                        V.set(nextRateId++);
                    }

                }
            }


            const Kernel * const kernelObj = node.Kernel;

            assert (V.any() || kernelObj == mPipelineKernel);

            // Check whether this (internal) kernel could terminate early
            bool demarcateOutputs = (kernelObj == mPipelineKernel);
            if (kernelObj != mPipelineKernel) {
                // TODO: an internally synchronzied kernel with fixed rate I/O can be contained within a partition
                // but cannot be the root of a non-isolated partition. To permit them to be roots, they'd need
                // some way of informing the pipeline as to how many strides they executed or the pipeline
                // would need to know to calculate it from its outputs. Rather than handling this complication,
                // for now we simply prevent this case.
                for (const Attribute & attr : kernelObj->getAttributes()) {
                    switch (attr.getKind()) {
                        case AttrId::InternallySynchronized:
                            V.set(nextRateId++);
                        case AttrId::CanTerminateEarly:
                        case AttrId::MayFatallyTerminate:
                        case AttrId::MustExplicitlyTerminate:
                            demarcateOutputs = true;
                        default: break;
                    }
                }
            }

            unsigned demarcationId = 0;
            if (LLVM_UNLIKELY(demarcateOutputs)) {
                demarcationId = nextRateId++;
            }

            // Now iterate through the outputs
            for (const auto e : make_iterator_range(out_edges(i, G))) {

                const auto bindingId = G[e];

                const RelationshipNode & rn = Relationships[bindingId];
                assert (rn.Type == RelationshipNode::IsBinding);
                const Binding & b = rn.Binding;
                const ProcessingRate & rate = b.getRate();

                BitSet & O = G[target(e, G)];

                O |= V;

                if (LLVM_UNLIKELY(demarcateOutputs)) {
                    O.set(demarcationId);
                }

                if (rate.getKind() == RateId::Fixed) {

                    // Check the attributes to see whether any impose a partition change
                    for (const Attribute & attr : b.getAttributes()) {
                        switch (attr.getKind()) {
                            case AttrId::Delayed:
                            case AttrId::Deferred:
                            // A deferred output rate is closer to an bounded rate than a
                            // countable rate but a deferred input rate simply means the
                            // buffer must be dynamic.
                            case AttrId::BlockSize:
                                BEGIN_SCOPED_REGION
                                AttributeClassifier key(O, attr.getKind(), attr.amount());
                                const auto f = attrBitIds.find(key);
                                if (f != attrBitIds.end()) {
                                    O.set(f->second);
                                } else {
                                    attrBitIds.emplace(std::move(key), nextRateId);
                                    O.set(nextRateId++);
                                }
                                END_SCOPED_REGION
                            default: break;
                        }
                    }

                } else {
                    O.set(nextRateId++);
                }

            }
        } else { // just propagate the bitsets

            BitSet & V = G[i];

            for (const auto e : make_iterator_range(in_edges(i, G))) {
                const BitSet & R = G[source(e, G)];
                V |= R;
            }

            for (const auto e : make_iterator_range(out_edges(i, G))) {
                BitSet & R = G[target(e, G)];
                R |= V;
            }

        }

    }

    G[0].reset();
    G[m - 1].set(nextRateId);

    std::vector<unsigned> partitionIds(m);

    auto convertUniqueNodeBitSetsToUniquePartitionIds = [&]() {
        PartitionMap partitionSets;
        unsigned nextPartitionId = 1;
        for (unsigned i = 0; i < m; ++i) {
            const auto u = sequence[i];
            const RelationshipNode & node = Relationships[u];
            if (node.Type == RelationshipNode::IsKernel) {
                BitSet & V = G[i];
                unsigned partitionId = 0;
                if (LLVM_LIKELY(V.any())) {
                    auto f = partitionSets.find(V);
                    if (f == partitionSets.end()) {
                        partitionId = nextPartitionId++;
                        partitionSets.emplace(V, partitionId);
                    } else {
                        partitionId = f->second;
                    }
                    assert (partitionId > 0);
                } else {
                    assert (node.Kernel == mPipelineKernel);
                }
                partitionIds[i] = partitionId;
            }
        }
        return nextPartitionId;
    };

    const auto synchronousPartitionCount = convertUniqueNodeBitSetsToUniquePartitionIds();

    assert (synchronousPartitionCount > 0);

    assert (Relationships[sequence[0]].Kernel == mPipelineKernel);
    assert (Relationships[sequence[m - 1]].Kernel == mPipelineKernel);


    // Stage 6: split (weakly) disconnected components within a partition into separate partitions

    std::vector<unsigned> componentId(m);
    std::iota(componentId.begin(), componentId.end(), 0);

    std::function<unsigned(unsigned)> find = [&](unsigned x) {
        assert (x < m);
        if (componentId[x] != x) {
            componentId[x] = find(componentId[x]);
        }
        return componentId[x];
    };

    auto union_find = [&](unsigned x, unsigned y) {
        assert (x < y);
        x = find(x);
        y = find(y);
        if (x != y) {
            componentId[y] = x;
        }
    };

    auto findIndex = [&](const unsigned vertex) {
        const auto s = std::find(sequence.begin(), sequence.end(), vertex);
        assert (s != sequence.end());
        const auto k = std::distance(sequence.begin(), s);
        assert (k < m);
        return k;
    };

    componentId[0] = 0;

    for (unsigned i = 1; i < m; ++i) {

        const auto producer = sequence[i];
        const RelationshipNode & node = Relationships[producer];

        if (node.Type == RelationshipNode::IsKernel) {
            const auto prodPartId = partitionIds[i];

            for (const auto e : make_iterator_range(out_edges(producer, Relationships))) {
                const auto output = target(e, Relationships);
                if (Relationships[output].Type == RelationshipNode::IsBinding) {
                    const auto f = first_out_edge(output, Relationships);
                    assert (Relationships[f].Reason != ReasonType::Reference);
                    const auto streamSet = target(f, Relationships);
                    assert (Relationships[streamSet].Type == RelationshipNode::IsRelationship);
                    assert (isa<StreamSet>(Relationships[streamSet].Relationship));
                    for (const auto g : make_iterator_range(out_edges(streamSet, Relationships))) {
                        assert (Relationships[g].Reason != ReasonType::Reference);
                        const auto input = target(g, Relationships);
                        assert (Relationships[input].Type == RelationshipNode::IsBinding);
                        const auto h = first_out_edge(input, Relationships);
                        assert (Relationships[h].Reason != ReasonType::Reference);
                        const auto consumer = target(h, Relationships);
                        assert (Relationships[consumer].Type == RelationshipNode::IsKernel);
                        const auto k = findIndex(consumer);
                        const auto consPartId = partitionIds[k];
                        assert (consPartId > 0);
                        if (prodPartId == consPartId) {
                            union_find(i, k);
                        }
                    }
                }
            }

            if (prodPartId == 0) {
                assert (node.Kernel == mPipelineKernel);
                union_find(0, i);
            }
        }
    }

    flat_set<unsigned> componentIds;
    componentIds.reserve(synchronousPartitionCount);

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {
            componentIds.insert(componentId[i]);
        }
    }

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {
            auto & c = componentId[i];
            const auto f = componentIds.find(c);
            const auto k = std::distance(componentIds.begin(), f);
            assert (c == 0 ^ k != 0);
            c = k;
        }
    }

    const auto partitionCount = componentIds.size();

    using RenumberingGraph = adjacency_list<vecS, vecS, bidirectionalS, no_property, unsigned>;

    // Stage 7: renumber the partition ids

    // To simplify processing later, renumber the partitions such that the partition id
    // of any predecessor of a kernel K is <= the partition id of K.

    RenumberingGraph T(partitionCount);

    for (unsigned i = 1; i < partitionCount; ++i) {
        add_edge(0, i, 0, T);
    }

    for (unsigned i = 1; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsRelationship) {
            const auto j = parent(i, G);
            const auto producer = sequence[j];
            assert (Relationships[producer].Type == RelationshipNode::IsKernel);
            const auto prodPartId = componentId[j];
            for (const auto e : make_iterator_range(out_edges(i, G))) {
                const auto k = target(e, G);
                const auto consumer = sequence[k];
                assert (Relationships[consumer].Type == RelationshipNode::IsKernel);
                const auto consPartId = componentId[k];
                if (prodPartId != consPartId) {
                    assert (consPartId > 0);
                    add_edge(prodPartId, consPartId, u, T);
                }
            }
        }
    }

    for (unsigned i = 1; i < (partitionCount - 1); ++i) {
        if (out_degree(i, T) == 0) {
            add_edge(i, partitionCount - 1, 0, T);
        }
    }

    std::vector<unsigned> renumberingSeq;
    renumberingSeq.reserve(partitionCount);

    if (LLVM_UNLIKELY(!lexical_ordering(T, renumberingSeq))) {
        report_fatal_error("Internal error: failed to generate acyclic partition graph");
    }

    assert (renumberingSeq[0] == 0);

    std::vector<unsigned> renumbered(partitionCount);

    for (unsigned i = 0; i < partitionCount; ++i) {
        const auto j = renumberingSeq[i];
        assert (j < partitionCount);
        renumbered[j] = i;
    }

    assert (renumbered[0] == 0);

    PartitionGraph P(partitionCount);

    for (unsigned i = 0; i < m; ++i) {
        const auto u = sequence[i];
        const RelationshipNode & node = Relationships[u];
        if (node.Type == RelationshipNode::IsKernel) {

            assert (partitionIds[i] < partitionCount);
            const auto j = renumbered[componentId[i]];
            assert (j < partitionCount);



            assert ((j > 0 && (j + 1) < partitionCount) ^ (node.Kernel == mPipelineKernel));

            P[j].Kernels.push_back(u);
            PartitionIds.emplace(u, j);
        }
    }

    #ifndef NDEBUG
    BEGIN_SCOPED_REGION
    flat_set<unsigned> included;
    included.reserve(numOfKernels);
    for (const auto u : P[0].Kernels) {
        assert ("kernel is in multiple partitions?" && included.insert(u).second);
        const auto & R = Relationships[u];
        assert (R.Type == RelationshipNode::IsKernel);
        assert (R.Kernel == mPipelineKernel);
    }
    auto numOfPartitionedKernels = P[0].Kernels.size();
    for (unsigned i = 1; i < partitionCount; ++i) {
        numOfPartitionedKernels += P[i].Kernels.size();
        for (const auto u : P[i].Kernels) {
            assert ("kernel is in multiple partitions?" && included.insert(u).second);
            const auto & R = Relationships[u];
            assert (R.Type == RelationshipNode::IsKernel);
        }
    }
    assert (numOfPartitionedKernels == numOfKernels);
    END_SCOPED_REGION
    #endif

    flat_set<std::pair<unsigned, unsigned>> duplicateFilter;

    for (unsigned i = 0; i < partitionCount; ++i) {
        assert (P[i].Kernels.size() > 0);
        const auto j = renumbered[i];
        assert (duplicateFilter.empty());
        for (const auto e : make_iterator_range(out_edges(i, T))) {
            const auto k = renumbered[target(e, T)];
            assert (k > j);
            const auto streamSet = T[e];
            if (LLVM_UNLIKELY(streamSet == 0)) continue;
            assert (streamSet < num_vertices(Relationships));
            assert (Relationships[streamSet].Type == RelationshipNode::IsRelationship);
            if (duplicateFilter.emplace(k, streamSet).second) {
                add_edge(j, k, streamSet, P);
            }
        }
        duplicateFilter.clear();
    }

    assert (partitionCount > 2);

    for (unsigned i = 1; i < (partitionCount - 1); ++i) {
        if (in_degree(i, P) == 0) {
            add_edge(0, i, 0, P);
        }
        if (out_degree(i, P) == 0) {
            add_edge(i, partitionCount - 1, 0, P);
        }
    }

    PartitionCount = partitionCount;

    return P;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief determinePartitionJumpIndices
 *
 * If a partition determines it has insufficient data to execute, identify which partition is the next one to test.
 * I.e., the one with input from some disjoint path. If none exists, we'll begin jump to "PartitionCount", which
 * marks the end of the processing loop.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::determinePartitionJumpIndices() {
     mPartitionJumpIndex.resize(PartitionCount);
#ifdef DISABLE_PARTITION_JUMPING
    for (unsigned i = 0; i < (PartitionCount - 1); ++i) {
        mPartitionJumpIndex[i] = i + 1;
    }
    mPartitionJumpIndex[(PartitionCount - 1)] = (PartitionCount - 1);
#else

    using BV = dynamic_bitset<>;
    using Graph = adjacency_list<hash_setS, vecS, bidirectionalS>;

    // Summarize the partitioning graph to only represent the existance of a dataflow relationship
    // between the partitions.

    Graph G(PartitionCount);

    for (auto producer = PipelineInput; producer < PipelineOutput; ++producer) {
        bool anyNonFixedOutput = false;
        for (const auto e : make_iterator_range(out_edges(producer, mBufferGraph))) {
            const BufferPort & port = mBufferGraph[e];
            const Binding & binding = port.Binding;
            if (!binding.getRate().isFixed()) {
                anyNonFixedOutput = true;
                break;
            }
        }

        const auto pid = KernelPartitionId[producer];
        assert (pid < PartitionCount);

        if (anyNonFixedOutput) {
            for (const auto e : make_iterator_range(out_edges(producer, mBufferGraph))) {
                const auto streamSet = target(e, mBufferGraph);
                for (const auto f : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                    const auto consumer = target(f, mBufferGraph);
                    const auto cid = KernelPartitionId[consumer];
                    assert (pid <= cid && cid < PartitionCount);
                    if (pid != cid) {
                        add_edge(pid, cid, G);
                    }
                }
            }
        } else {
            add_edge(pid, pid + 1U, G);
        }
    }

    const auto terminal = PartitionCount - 1U;

    for (auto partitionId = 0U; partitionId < terminal; ++partitionId) {
       if (out_degree(partitionId, G) == 0) {
           add_edge(partitionId, terminal, G);
       }
    }

    // Now compute the transitive reduction of the partition relationships
    BEGIN_SCOPED_REGION
    const reverse_traversal ordering(PartitionCount);
    assert (is_valid_topological_sorting(ordering, G));
    transitive_closure_dag(ordering, G);
    transitive_reduction_dag(ordering, G);
    END_SCOPED_REGION

    #ifndef NDEBUG
    for (unsigned i = 0; i < PartitionCount; ++i) {
        mPartitionJumpIndex[i] = -1U;
    }
    #endif

    std::vector<unsigned> rank(PartitionCount);
    for (unsigned i = 0; i < PartitionCount; ++i) { // forward topological ordering
        unsigned newRank = 0;
        for (const auto e : make_iterator_range(in_edges(i, G))) {
            newRank = std::max(newRank, rank[source(e, G)]);
        }
        rank[i] = newRank + 1;
    }

    std::vector<unsigned> occurences(PartitionCount);
    std::vector<unsigned> singleton(PartitionCount);

    std::vector<std::bitset<2>> ancestors(PartitionCount);

    std::vector<unsigned> reverseLCA(PartitionCount);
    for (unsigned i = 0; i < terminal; ++i) {
        reverseLCA[i] = i + 1;
    }

    for (unsigned i = 0; i < PartitionCount; ++i) {  // forward topological ordering
        const auto d = in_degree(i, G);

        if (d > 1) {

            Graph::in_edge_iterator begin, end;
            std::tie(begin, end) = in_edges(i, G);

            auto lca = i;
            for (auto ei = begin; (++ei) != end; ) {
                const auto x = source(*ei, G);
                for (auto ej = begin; ej != ei; ++ej) {
                    const auto y = source(*ej, G);
                    assert (x != y);

                    // Determine the common ancestors of each input to node_i
                    for (unsigned j = 0; j < lca; ++j) {
                        ancestors[j].reset();
                    }
                    ancestors[x].set(0);
                    ancestors[y].set(1);

                    std::fill_n(occurences.begin(), rank[i] - 1, 0);
                    for (auto j = i; j--; ) { // reverse topological ordering
                        for (const auto e : make_iterator_range(out_edges(j, G))) {
                            const auto v = target(e, G);
                            ancestors[j] |= ancestors[v];
                        }
                        if (ancestors[j].all()) {
                            const auto k = rank[j];
                            occurences[k]++;
                            singleton[k] = j;
                        }
                    }
                    // Now scan again through them to determine the single ancestor
                    // to the pair of inputs that is of highest rank.

                    for (auto j = rank[i] - 1; j--; ) {
                        if (occurences[j] == 1) {
                            lca = singleton[j];
                            break;
                        }
                    }
                }
            }
            assert (lca <= i);
            auto & val = reverseLCA[lca];
            val = std::max(val, i);
        }
    }
    reverseLCA[terminal] = terminal;

    for (auto partitionId = 1U; partitionId < terminal; ++partitionId) {
       add_edge(partitionId, partitionId + 1, G);
    }
    add_edge(terminal, terminal, G);

#if 0

    auto & out = errs();

    out << "digraph \"" << "J1" << "\" {\n";
    for (auto v : make_iterator_range(vertices(G))) {
        out << "v" << v << " [label=\"" << v << " : {";
        out << reverseLCA[v];
        out << "}\"];\n";
    }
    for (auto e : make_iterator_range(edges(G))) {
        const auto s = source(e, G);
        const auto t = target(e, G);
        out << "v" << s << " -> v" << t << ";\n";
    }

    out << "}\n\n";
    out.flush();

#endif

    for (unsigned i = 0; i < PartitionCount; ++i) {
        auto n = reverseLCA[i];
        assert (n < PartitionCount);
        while (in_degree(n, G) < 2) {
            const auto m = reverseLCA[n];
            assert (n != m);
            n = m;
            assert (n < PartitionCount);
        }
        assert (n > i || (i == (PartitionCount - 1)));
        mPartitionJumpIndex[i] = n;
    }

#endif
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makePartitionJumpGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::makePartitionJumpTree() {
    mPartitionJumpTree = PartitionJumpTree(PartitionCount);
    for (auto i = 0U; i < (PartitionCount - 1); ++i) {
        assert (mPartitionJumpIndex[i] >= i && mPartitionJumpIndex[i] < PartitionCount);
        add_edge(i, mPartitionJumpIndex[i], mPartitionJumpTree);
    }

#if 0

    auto & out = errs();

    out << "digraph \"" << "J2" << "\" {\n";
    for (auto v : make_iterator_range(vertices(mPartitionJumpTree))) {
        out << "v" << v << " [label=\"" << v << "\"];\n";
    }
    for (auto e : make_iterator_range(edges(mPartitionJumpTree))) {
        const auto s = source(e, mPartitionJumpTree);
        const auto t = target(e, mPartitionJumpTree);
        out << "v" << s << " -> v" << t << ";\n";
    }

    out << "}\n\n";
    out.flush();

#endif
}

} // end of namespace kernel

#endif // PARTITIONING_ANALYSIS_HPP
