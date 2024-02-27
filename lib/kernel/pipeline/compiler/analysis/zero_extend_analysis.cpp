#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyZeroExtendedStreamSets
 *
 * Determine whether there are any zero extend attributes on any kernel and verify that every kernel with
 * zero extend attributes have at least one input that is not transitively dependent on a zero extended
 * input stream.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyZeroExtendedStreamSets() {

    #ifndef DISABLE_ZERO_EXTEND
    using Graph = adjacency_list<vecS, vecS, bidirectionalS>;

    for (unsigned kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        const RelationshipNode & rn = mStreamGraph[kernel];
        assert (rn.Type == RelationshipNode::IsKernel);
        const Kernel * const kernelObj = rn.Kernel; assert (kernelObj);
        const auto numOfInputs = in_degree(kernel, mStreamGraph);

        // First verify whether the ZeroExtend attributes are correct in the unmodified
        // system (to reduce the possibility of future programmer error.)

        Graph H(numOfInputs + 1);

        // enumerate the input relations
        for (const auto e : make_iterator_range(in_edges(kernel, mStreamGraph))) {
            const auto k = source(e, mStreamGraph);
            const RelationshipNode & rn = mStreamGraph[k];
            assert (rn.Type == RelationshipNode::IsBinding);
            const Binding & binding = rn.Binding;
            const RelationshipType & port = mStreamGraph[e];

            auto h =  first_in_edge(k, mStreamGraph);
            assert (mStreamGraph[h].Reason != ReasonType::Reference);
            const auto streamSet = source(h, mStreamGraph);
            assert (mStreamGraph[streamSet].Type == RelationshipNode::IsStreamSet);
            const auto isConstant = isa<RepeatingStreamSet>(mStreamGraph[streamSet].Relationship);
            const auto zeroExtend = binding.hasAttribute(AttrId::ZeroExtended);
            if (LLVM_UNLIKELY(isConstant && zeroExtend)) {
                SmallVector<char, 256> tmp;
                raw_svector_ostream msg(tmp);
                msg << kernelObj->getName() << '.' << binding.getName();
                msg << " cannot ZeroExtend a repeating streamset";
                report_fatal_error(StringRef(msg.str()));
            }
            if (LLVM_UNLIKELY(zeroExtend)) {
                add_edge(port.Number, numOfInputs, H);
                if (LLVM_UNLIKELY(binding.hasAttribute(AttrId::Principal))) {
                    SmallVector<char, 256> tmp;
                    raw_svector_ostream msg(tmp);
                    msg << kernelObj->getName() << '.' << binding.getName();
//                    if (zeroExtend) {
                        msg << " cannot have both ZeroExtend and Principal attributes";
//                    } else {
//                        msg << " cannot have a Principal repeating streamset";
//                    }
                    report_fatal_error(StringRef(msg.str()));
                }
            }

            if (LLVM_UNLIKELY(in_degree(k, mStreamGraph) != 1)) {
                graph_traits<RelationshipGraph>::in_edge_iterator ei, ei_end;
                std::tie(ei, ei_end) = in_edges(k, mStreamGraph);
                assert (std::distance(ei, ei_end) == 2);
                const auto f = *(ei + 1);
                const RelationshipType & ref = mStreamGraph[f];
                assert (ref.Reason == ReasonType::Reference);
                add_edge(ref.Number, port.Number, H);
            }
        }

        if (LLVM_LIKELY(in_degree(numOfInputs, H) == 0)) {
            continue;
        }

        if (LLVM_UNLIKELY(in_degree(numOfInputs, H) == numOfInputs)) {
            report_fatal_error(StringRef(kernelObj->getName()) + " requires at least one non-zero-extended input");
        }

        // Identify all transitive dependencies on zero-extended inputs
        transitive_closure_dag(H);

        if (LLVM_UNLIKELY(in_degree(numOfInputs, H) == numOfInputs)) {
            report_fatal_error(StringRef(kernelObj->getName()) + " requires at least one non-zero-extended input"
                                                   " that does not refer to a zero-extended input");
        }

        // A kernel input with a ZeroExtend attribute performs additional checks to determine whether it has
        // exhausted its actual data stream then switches to a "null stream" (assuming the producer is
        // finished.) This adds complexity to the overall pipeline but is unnecessary when we can guarantee
        // the ZeroExtend-ed stream cannot end before the other inputs. An easy test for this is to check
        // whether *any* of the inputs to a kernel are inputs to the partition. Any such input would not
        // dictate the dataflow within the partition until after this kernel has been executed.

        // TODO: once we can determine what inter-partition channels can bound the number of strides for
        // a partition, we can filter the ones that will never be zero-extended.

        for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            BufferPort & inputData = mBufferGraph[input];
            const auto streamSet = source(input, mBufferGraph);
            const BufferNode & bn = mBufferGraph[streamSet];
            if (bn.isNonThreadLocal() && !bn.isConstant()) {
                const Binding & binding = inputData.Binding;
                if (LLVM_UNLIKELY(binding.hasAttribute(AttrId::ZeroExtended))) {
                    inputData.Flags |= BufferPortType::IsZeroExtended;
                }
                HasZeroExtendedStream = true;
            }
        }

    }
    #endif
}

}
