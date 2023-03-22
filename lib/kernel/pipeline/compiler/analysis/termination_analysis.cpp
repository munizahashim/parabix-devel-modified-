#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"

namespace kernel {

#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyTerminationChecks
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyTerminationChecks() {

    using TerminationGraph = adjacency_list<hash_setS, vecS, bidirectionalS>;

    TerminationGraph G(PartitionCount);

    const auto terminal = KernelPartitionId[PipelineOutput];

    for (auto consumer = FirstKernel; consumer <= PipelineOutput; ++consumer) {
        const auto cid = KernelPartitionId[consumer];
        for (const auto e : make_iterator_range(in_edges(consumer, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isConstant())) continue;
            const auto producer = parent(streamSet, mBufferGraph);
            const auto pid = KernelPartitionId[producer];
            assert (pid <= cid);
            if (pid != cid) {
                add_edge(pid, cid, G);
            }
        }
    }

    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernelObj = getKernel(i); assert (kernelObj);
        if (LLVM_UNLIKELY(kernelObj->hasAttribute(AttrId::SideEffecting))) {
            const auto pid = KernelPartitionId[i];
            add_edge(pid, terminal, G);
        }
    }

    assert (FirstCall == (PipelineOutput + 1U));

    for (auto consumer = PipelineOutput; consumer <= LastCall; ++consumer) {
        for (const auto relationship : make_iterator_range(in_edges(consumer, mScalarGraph))) {
            const auto scalar = source(relationship, mScalarGraph);
            for (const auto production : make_iterator_range(in_edges(scalar, mScalarGraph))) {
                const auto producer = source(production, mScalarGraph);
                const auto partitionId = KernelPartitionId[producer];
                assert ("cannot occur" && partitionId != terminal);
                add_edge(partitionId, terminal, G);
            }
        }
    }

    assert ("program has no outputs? relationship construction should have reported this case." && in_degree(terminal, G) > 0);

    transitive_reduction_dag(G);

    mTerminationCheck.resize(PartitionCount, 0U);

    // we are only interested in the incoming edges of the pipeline output
    for (const auto e : make_iterator_range(in_edges(terminal, G))) {
        mTerminationCheck[source(e, G)] = TerminationCheckFlag::Soft;
    }

    // hard terminations
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(getKernel(i)->hasAttribute(AttrId::MayFatallyTerminate))) {
            mTerminationCheck[KernelPartitionId[i]] |= TerminationCheckFlag::Hard;
        }
    }

    mTerminationCheck[terminal] = 0;

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeTerminationPropagationGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::makeTerminationPropagationGraph() {

    // When a partition terminates, we want to inform any kernels that supply information to it that
    // one of their consumers has finished processing data. In a pipeline with a single output, this
    // isn't necessary but if a pipeline has multiple outputs, we could end up needlessly producing
    // data that will never be consumed.

    mTerminationPropagationGraph = TerminationPropagationGraph(LastKernel + 1U);

    HasTerminationSignal.resize(LastKernel + 1U);

    using TerminationGraph = adjacency_list<hash_setS, vecS, bidirectionalS, boost::dynamic_bitset<>>;

    using PropagationGraph = adjacency_list<hash_setS, vecS, bidirectionalS, bool>;

    unsigned newMark = 0;

    TerminationGraph T(PipelineOutput + 1);

    PropagationGraph G(PartitionCount);

    // reverse topological search

    for (auto kernel = LastKernel;  kernel >= FirstKernel; --kernel) {

        auto & marks = T[kernel];
        marks.resize(PipelineOutput + 1);
        assert (marks.none());

        const auto partId = KernelPartitionId[kernel];
        const auto root = FirstKernelInPartition[partId];

        bool addMark = false;
        bool mayTerminate = (kernel == root);

        const Kernel * const kernelObj = getKernel(kernel);
        for (const auto & attr : kernelObj->getAttributes()) {
            switch (attr.getKind()) {
                case AttrId::CanTerminateEarly:
                case AttrId::MustExplicitlyTerminate:
                    addMark = true;
                case AttrId::MayFatallyTerminate:
                    mayTerminate = true;
                default: break;
            }
        }

        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(e, mBufferGraph);
            for (const auto f : make_iterator_range(out_edges(streamSet, mBufferGraph))) {
                const auto consumer = target(f, mBufferGraph);
                const auto cid = KernelPartitionId[consumer];
                if (cid != partId) {
                    add_edge(cid, partId, G);
                }
                marks |= T[consumer];
            }
        }

        if (addMark) {
            add_edge(kernel, root, T);
            marks.set(newMark++);
        }
        if (mayTerminate) {
            HasTerminationSignal.set(kernel);
        }
    }

    printGraph(T, errs(), "T1");

    printGraph(G, errs(), "G1");

    const forward_traversal ordering(PartitionCount);
    assert (is_valid_topological_sorting(ordering, G));
    transitive_closure_dag(ordering, G);
    transitive_reduction_dag(ordering, G);

    printGraph(G, errs(), "G2");

    std::queue<PropagationGraph::vertex_descriptor> Q;
    flat_set<PropagationGraph::vertex_descriptor> V;
    for (auto pid = KernelPartitionId[FirstKernel]; pid < PartitionCount; ++pid) {
        const auto s = FirstKernelInPartition[pid];
        if (in_degree(s, T) != 0) {
            Q.push(pid);
            const auto & base = T[s];
            for (;;) {
                const auto u = Q.front();
                Q.pop();
                for (const auto e : make_iterator_range(out_edges(u, G))) {
                    const auto v = target(e, G);
                    const auto t = FirstKernelInPartition[v];
                    if (base.is_proper_subset_of(T[t])) {
                        add_edge(t, s, T);
                    } else if (V.emplace(v).second) {
                        assert (base.is_subset_of(T[t]));
                        Q.push(v);
                    }



                }




            }




        }
    }

    printGraph(T, errs(), "T2");

    if (newMark > 0) {
        exit(-1);
    }

//    for (auto kernel = FirstKernel;  kernel <= LastKernel; ++kernel) {

//        auto & marks = T[kernel];
//        marks.resize(PipelineOutput + 1);
//        assert (marks.empty());

//        bool addMark = false;
//        bool mayTerminate = false;

//        const Kernel * const kernelObj = getKernel(kernel);
//        for (const auto & attr : kernelObj->getAttributes()) {
//            switch (attr.getKind()) {
//                case AttrId::CanTerminateEarly:
//                case AttrId::MustExplicitlyTerminate:
//                    addMark = true;
//                case AttrId::MayFatallyTerminate:
//                    mayTerminate = true;
//                default: break;
//            }
//        }

//        if (LLVM_UNLIKELY(in_degree(kernel, mBufferGraph) == 0)) {
//            addMark = true;
//        } else {
//            for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
//                const auto streamSet = source(e, mBufferGraph);
//                if (in_degree(streamSet, mBufferGraph) != 0) {
//                    const auto producer = parent(streamSet, mBufferGraph);
//                    marks |= T[producer];
//                    add_edge(kernel, producer, T);
//                #ifndef NDEBUG
//                } else {
//                    assert (mBufferGraph[streamSet].isConstant());
//                #endif
//                }
//            }
//        }

//        if (addMark) {
//            marks.set(newMark++);
//        }
//        if (mayTerminate) {
//            HasTerminationSignal.set(kernel);
//        }
//    }

//    const reverse_traversal ordering(LastKernel + 1U);
//    assert (is_valid_topological_sorting(ordering, T));
//    transitive_closure_dag(ordering, T);
//    transitive_reduction_dag(ordering, T);

//    PropagationGraph G(PipelineOutput + 1);

//    for (auto kernel = LastKernel;  kernel >= FirstKernel; --kernel) {
//        for (const auto e : make_iterator_range(in_edges(kernel, T))) {
//            const auto prod = source(e, T);
//            const auto & P = T[kernel];
//            if (!P.is_subset_of(T[prod])) {
//                const auto partId = KernelPartitionId[kernel];
//                const auto first = FirstKernelInPartition[partId];
//                if (prod != first) {
//                    add_edge(kernel, first, G);
//                    T[first] |= P;
//                }
//            }
//        }
//    }









//    for (auto pid = KernelPartitionId[FirstKernel]; pid < PartitionCount; ++pid) {
//        const auto start = FirstKernelInPartition[pid];
//        const auto end = FirstKernelInPartition[pid + 1U];
//        assert (start <= end);
//        assert (end <= PipelineOutput);

//        for (const auto e : make_iterator_range(in_edges(start, mBufferGraph))) {
//            const auto streamSet = source(e, mBufferGraph);
//            const BufferNode & bn = mBufferGraph[streamSet];
//            if (LLVM_UNLIKELY(bn.isConstant())) continue;
//            const auto producer = parent(streamSet, mBufferGraph);
//            marks.set(KernelPartitionId[producer]);
//        }

//        const auto term = getKernel(start)->canSetTerminateSignal();
//        for (const auto j : marks.set_bits()) {
//            const auto k = FirstKernelInPartition[j];
//            assert (k < start);
//            add_edge(start, k, term, mTerminationPropagationGraph);
//        }
//        marks.reset();

//        // regardless of whether a partition root can terminate, every root has
//        // a terminated flag stored in the state that any kernel that cannot
//        // explicitly terminate shares.
//        HasTerminationSignal.set(start);
//        for (auto i = start + 1U; i < end; ++i) {
//            const Kernel * const kernelObj = getKernel(i); assert (kernelObj);
//            if (kernelObj->canSetTerminateSignal()) {
//                add_edge(i, start, true, mTerminationPropagationGraph);
//                HasTerminationSignal.set(i);
//            }
//        }
//    }

//    ReverseTopologicalOrdering ordering;
//    ordering.reserve(num_vertices(mTerminationPropagationGraph));
//    topological_sort(mTerminationPropagationGraph, std::back_inserter(ordering));

//    auto first = ordering.begin();
//    const auto end = ordering.end();

//    for (auto i = first; i != end; ++i) {
//        TerminationPropagationGraph::in_edge_iterator ei_begin, ei_end;
//        std::tie(ei_begin, ei_end) = in_edges(*i, mTerminationPropagationGraph);
//        bool anyPropagate = false;
//        bool allPropagate = true;
//        for (auto ei = ei_begin; ei != ei_end; ++ei) {
//            const auto p = mTerminationPropagationGraph[*ei];
//            anyPropagate |= p;
//            allPropagate &= p;
//        }
//        if (anyPropagate) {
//            const Kernel * const kernelObj = getKernel(*i); assert (kernelObj);
//            for (const auto & attr : kernelObj->getAttributes()) {
//                switch (attr.getKind()) {
//                    case AttrId::MustExplicitlyTerminate:
//                    case AttrId::SideEffecting:
//                        goto disable_propagated_signals;
//                    default:
//                        break;
//                }
//            }
//            if (allPropagate) {
//                for (auto e : make_iterator_range(out_edges(*i, mTerminationPropagationGraph))) {
//                    mTerminationPropagationGraph[e] = true;
//                }
//            } else {
//disable_propagated_signals:
//                for (auto ei = ei_begin; ei != ei_end; ++ei) {
//                    mTerminationPropagationGraph[*ei] = false;
//                }
//            }
//        }
//    }

//    remove_edge_if([&](const TerminationPropagationGraph::edge_descriptor e) {
//        return !mTerminationPropagationGraph[e];
//    }, mTerminationPropagationGraph);

//    transitive_closure_dag(ordering, mTerminationPropagationGraph);
//    transitive_reduction_dag(ordering, mTerminationPropagationGraph);

//    printGraph(mTerminationPropagationGraph, errs(), "T");

}

#endif

#if 1

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeTerminationPropagationGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::makeTerminationPropagationGraph() {

    // When a partition terminates, we want to inform any kernels that supply information to it that
    // one of their consumers has finished processing data. In a pipeline with a single output, this
    // isn't necessary but if a pipeline has multiple outputs, we could end up needlessly producing
    // data that will never be consumed.

    mTerminationPropagationGraph = TerminationPropagationGraph(LastKernel + 1U);

    HasTerminationSignal.resize(LastKernel + 1U);

    BitVector marks(PipelineOutput + 1U);

    for (auto pid = KernelPartitionId[FirstKernel]; pid < PartitionCount; ++pid) {
        const auto start = FirstKernelInPartition[pid];
        const auto end = FirstKernelInPartition[pid + 1U];
        assert (start <= end);
        assert (end <= PipelineOutput);

        for (const auto e : make_iterator_range(in_edges(start, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isConstant())) continue;
            const auto producer = parent(streamSet, mBufferGraph);
            marks.set(KernelPartitionId[producer]);
        }

        const auto term = getKernel(start)->canSetTerminateSignal();
        for (const auto j : marks.set_bits()) {
            const auto k = FirstKernelInPartition[j];
            assert (k < start);
            add_edge(start, k, term, mTerminationPropagationGraph);
        }
        marks.reset();

        // regardless of whether a partition root can terminate, every root has
        // a terminated flag stored in the state that any kernel that cannot
        // explicitly terminate shares.
        HasTerminationSignal.set(start);
        for (auto i = start + 1U; i < end; ++i) {
            const Kernel * const kernelObj = getKernel(i); assert (kernelObj);
            if (kernelObj->canSetTerminateSignal()) {
                add_edge(i, start, true, mTerminationPropagationGraph);
                HasTerminationSignal.set(i);
            }
        }
    }

    ReverseTopologicalOrdering ordering;
    ordering.reserve(num_vertices(mTerminationPropagationGraph));
    topological_sort(mTerminationPropagationGraph, std::back_inserter(ordering));

    auto first = ordering.begin();
    const auto end = ordering.end();

    for (auto i = first; i != end; ++i) {
        TerminationPropagationGraph::in_edge_iterator ei_begin, ei_end;
        std::tie(ei_begin, ei_end) = in_edges(*i, mTerminationPropagationGraph);
        bool anyPropagate = false;
        bool allPropagate = true;
        for (auto ei = ei_begin; ei != ei_end; ++ei) {
            const auto p = mTerminationPropagationGraph[*ei];
            anyPropagate |= p;
            allPropagate &= p;
        }
        if (anyPropagate) {
            const Kernel * const kernelObj = getKernel(*i); assert (kernelObj);
            for (const auto & attr : kernelObj->getAttributes()) {
                switch (attr.getKind()) {
                    case AttrId::MustExplicitlyTerminate:
                    case AttrId::SideEffecting:
                        goto disable_propagated_signals;
                    default:
                        break;
                }
            }
            if (allPropagate) {
                for (auto e : make_iterator_range(out_edges(*i, mTerminationPropagationGraph))) {
                    mTerminationPropagationGraph[e] = true;
                }
            } else {
disable_propagated_signals:
                for (auto ei = ei_begin; ei != ei_end; ++ei) {
                    mTerminationPropagationGraph[*ei] = false;
                }
            }
        }
    }

    remove_edge_if([&](const TerminationPropagationGraph::edge_descriptor e) {
        return !mTerminationPropagationGraph[e];
    }, mTerminationPropagationGraph);

    transitive_closure_dag(ordering, mTerminationPropagationGraph);
    transitive_reduction_dag(ordering, mTerminationPropagationGraph);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyTerminationChecks
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyTerminationChecks() {

    using TerminationGraph = adjacency_list<hash_setS, vecS, bidirectionalS>;

    TerminationGraph G(PartitionCount);

    // Although every kernel will eventually terminate, we only need to observe one kernel
    // in each partition terminating to know whether *all* of the kernels will terminate.
    // Since only the root of a partition could be a kernel that explicitly terminates,
    // we can "share" its termination state.

    const auto terminal = PartitionCount - 1U;

    for (auto consumer = FirstKernel; consumer <= PipelineOutput; ++consumer) {
        const auto cid = KernelPartitionId[consumer];
        for (const auto e : make_iterator_range(in_edges(consumer, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (LLVM_UNLIKELY(bn.isConstant())) continue;
            const auto producer = parent(streamSet, mBufferGraph);
            const auto pid = KernelPartitionId[producer];
            assert (pid <= cid);
            if (pid != cid) {
                add_edge(pid, cid, G);
            }
        }
    }

    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        const Kernel * const kernelObj = getKernel(i); assert (kernelObj);
        if (LLVM_UNLIKELY(kernelObj->hasAttribute(AttrId::SideEffecting))) {
            const auto pid = KernelPartitionId[i];
            add_edge(pid, terminal, G);
        }
    }

    assert (FirstCall == (PipelineOutput + 1U));

    for (auto consumer = PipelineOutput; consumer <= LastCall; ++consumer) {
        for (const auto relationship : make_iterator_range(in_edges(consumer, mScalarGraph))) {
            const auto scalar = source(relationship, mScalarGraph);
            for (const auto production : make_iterator_range(in_edges(scalar, mScalarGraph))) {
                const auto producer = source(production, mScalarGraph);
                const auto partitionId = KernelPartitionId[producer];
                assert ("cannot occur" && partitionId != terminal);
                add_edge(partitionId, terminal, G);
            }
        }
    }

    assert ("pipeline has no observable outputs?" && (in_degree(terminal, G) > 0));

    transitive_reduction_dag(G);

    assert (KernelPartitionId[LastKernel] == (PartitionCount - 2));

    for (auto i = KernelPartitionId[FirstKernel]; i < (PartitionCount - 1); ++i) {
        if (out_degree(i, G) == 0) {
            add_edge(i, terminal, G);
        }
    }

    if (in_degree(terminal, G) == 0) {
        report_fatal_error("Internal error: no termination signal propagated to pipeline end?");
    }

    mTerminationCheck.resize(PartitionCount, 0U);

    // we are only interested in the incoming edges of the pipeline output
    for (const auto e : make_iterator_range(in_edges(terminal, G))) {
        mTerminationCheck[source(e, G)] = TerminationCheckFlag::Soft;
    }

    // hard terminations
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(getKernel(i)->hasAttribute(AttrId::MayFatallyTerminate))) {
            mTerminationCheck[KernelPartitionId[i]] |= TerminationCheckFlag::Hard;
        }
    }

    mTerminationCheck[terminal] = 0;

}

#endif

#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeTerminationPropagationGraph
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::makeTerminationPropagationGraph() {

    // When a partition terminates, we want to inform any kernels that supply information to it that
    // one of their consumers has finished processing data. In a pipeline with a single output, this
    // isn't necessary but if a pipeline has multiple outputs, we could end up needlessly producing
    // data that will never be consumed.

    mTerminationPropagationGraph = TerminationPropagationGraph(PartitionCount);

    unsigned outputs = 0;
    for (unsigned i = 0; i < PartitionCount; ++i) {
        if (mTerminationCheck[i] & TerminationCheckFlag::Soft) {
            ++outputs;
        }
    }

    if (outputs < 2) {
        return;
    }

    BitVector inputs(PartitionCount);

    for (auto start = FirstKernel; start <= PipelineOutput; ) {
        const auto pid = KernelPartitionId[start];
        auto end = start + 1;
        for (; end <= PipelineOutput; ++end) {
            if (pid != KernelPartitionId[end]) {
                break;
            }
        }

        for (auto i = start; i < end; ++i) {
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto streamSet = source(e, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];
                if (LLVM_UNLIKELY(bn.isConstant())) continue;
                const auto producer = parent(streamSet, mBufferGraph);
                const auto partitionId = KernelPartitionId[producer];
                inputs.set(partitionId);
            }
        }
        inputs.reset(pid);
        for (const auto i : inputs.set_bits()) {
            add_edge(pid, i, mTerminationPropagationGraph);
        }
        inputs.reset();

        start = end;
    }

    transitive_reduction_dag(mTerminationPropagationGraph);

    for (auto end = LastKernel; end >= FirstKernel; ) {
        const auto pid = KernelPartitionId[end];
        auto start = end;
        for (; start > FirstKernel; --start) {
            if (pid != KernelPartitionId[start - 1U]) {
                break;
            }
        }

        const Kernel * const kernelObj = getKernel(start);

        auto prior = pid;
        if (kernelObj->canSetTerminateSignal()) {
            auto fork = pid;
            for (;;) {
                const auto n = out_degree(fork, mTerminationPropagationGraph);
                if (n != 1) {
                    break;
                }
                prior = fork;
                fork = child(fork, mTerminationPropagationGraph);
            }
        }

        clear_out_edges(pid, mTerminationPropagationGraph);
        if (prior != pid) {
            add_edge(pid, prior, mTerminationPropagationGraph);
        }

        end = start - 1U;
    }

}

#endif

}
