#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"

namespace kernel {

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

    HasTerminationSignal.resize(PipelineOutput + 1U);

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

}
