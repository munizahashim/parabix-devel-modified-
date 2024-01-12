#include "pipeline_analysis.hpp"
#include "lexographic_ordering.hpp"
#include <z3.h>

#if Z3_VERSION_INTEGER >= LLVM_VERSION_CODE(4, 7, 0)
    typedef int64_t Z3_int64;
#else
    typedef long long int        Z3_int64;
#endif

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief simpleEstimateInterPartitionDataflow
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::simpleEstimateInterPartitionDataflow(PartitionGraph & P, pipeline_random_engine & rng) {

    using NonFixedTaintGraph = adjacency_list<hash_setS, vecS, bidirectionalS, bool, bool>;

    const auto cfg = Z3_mk_config();
    Z3_set_param_value(cfg, "model", "true");
    Z3_set_param_value(cfg, "proof", "false");
    const auto ctx = Z3_mk_context(cfg);
    Z3_del_config(cfg);
    const auto solver = Z3_mk_optimize(ctx);
    Z3_optimize_inc_ref(ctx, solver);

    auto hard_assert = [&](Z3_ast c) {
        Z3_optimize_assert(ctx, solver, c);
    };

    auto soft_assert = [&](Z3_ast c) {
        Z3_optimize_assert_soft(ctx, solver, c, "1", nullptr);
    };

    auto check = [&]() -> Z3_lbool {
        #if Z3_VERSION_INTEGER >= LLVM_VERSION_CODE(4, 5, 0)
        return Z3_optimize_check(ctx, solver, 0, nullptr);
        #else
        return Z3_optimize_check(ctx, solver);
        #endif
    };

    const auto varType = Z3_mk_real_sort(ctx);

    auto constant_real = [&](const Rational & value) {
        return Z3_mk_real(ctx, value.numerator(), value.denominator());
    };

    auto multiply =[&](Z3_ast X, const Rational & value) {
        if ((value.numerator() == 1) && (value.denominator() == 1)) {
            return X;
        }
        Z3_ast args[2] = { X, constant_real(value) };
        return Z3_mk_mul(ctx, 2, args);
    };

    const auto numOfPartitions = num_vertices(P);

    std::vector<Z3_ast> VarList(num_vertices(Relationships));

    const auto ONE = constant_real(1);

    for (unsigned partId = 0; partId < numOfPartitions; ++partId) {
        const PartitionData & N = P[partId];
        const auto & K = N.Kernels;
        auto rootVar = Z3_mk_fresh_const(ctx, nullptr, varType);
        hard_assert(Z3_mk_ge(ctx, rootVar, ONE));
        const auto m = K.size();
        for (unsigned i = 0; i < m; ++i) {
            VarList[K[i]] = multiply(rootVar, N.Repetitions[i]);
        }
    }

    NonFixedTaintGraph T(numOfPartitions);
    for (unsigned partId = 0; partId < numOfPartitions; ++partId) {
        T[partId] = false;
    }


    for (unsigned prodId = 0; prodId < numOfPartitions; ++prodId) {
        const PartitionData & N = P[prodId];
        const auto & K = N.Kernels;
        const auto m = K.size();
        for (unsigned i = 0; i < m; ++i) {
            const auto producer = K[i];
            assert (VarList[producer]);
            assert (Relationships[producer].Type == RelationshipNode::IsKernel);

            for (const auto e : make_iterator_range(out_edges(producer, Relationships))) {
                const auto binding = target(e, Relationships);
                if (Relationships[binding].Type == RelationshipNode::IsBinding) {
                    const auto f = first_out_edge(binding, Relationships);
                    assert (Relationships[f].Reason != ReasonType::Reference);
                    const auto streamSet = target(f, Relationships);
                    assert (Relationships[streamSet].Type == RelationshipNode::IsStreamSet);
                    const RelationshipNode & output = Relationships[binding];
                    assert (output.Type == RelationshipNode::IsBinding);
                    const Binding & outputBinding = output.Binding;
                    const ProcessingRate & oRate = outputBinding.getRate();


                    Z3_ast expOutRate = nullptr;

                    for (const auto e : make_iterator_range(out_edges(streamSet, Relationships))) {
                        const auto binding = target(e, Relationships);
                        const RelationshipNode & input = Relationships[binding];
                        if (input.Type == RelationshipNode::IsBinding) {
                            const Binding & inputBinding = input.Binding;
                            const ProcessingRate & iRate = inputBinding.getRate();

                            const auto f = first_out_edge(binding, Relationships);
                            assert (Relationships[f].Reason != ReasonType::Reference);
                            const unsigned consumer = target(f, Relationships);

                            const auto c = PartitionIds.find(consumer);
                            assert (c != PartitionIds.end());
                            const auto consumerPartitionId = c->second;
                            assert (prodId <= consumerPartitionId);

                            if (prodId != consumerPartitionId) {

                                // mark that there is non-fixed dataflow between these
                                const auto e = add_edge(prodId, consumerPartitionId, false, T).first;
                                if (!oRate.isFixed() || !iRate.isFixed()) {
                                    T[e] = true;
                                }

                                if (LLVM_UNLIKELY(oRate.isUnknown())) {
                                    continue;
                                }

                                if (expOutRate == nullptr) {
                                    const RelationshipNode & producerNode = Relationships[producer];
                                    assert (producerNode.Type == RelationshipNode::IsKernel);
                                    const auto s = producerNode.Kernel->getStride();
                                    const auto expectedOutput = (oRate.getLowerBound() + oRate.getUpperBound()) * Rational{s, 2};
                                    expOutRate = multiply(VarList[producer], expectedOutput);
                                }

                                Z3_ast expInRate = nullptr;

                                if (LLVM_UNLIKELY(iRate.isGreedy())) {
                                    expInRate = expOutRate;
                                } else {
                                    const RelationshipNode & consumerNode = Relationships[consumer];
                                    assert (consumerNode.Type == RelationshipNode::IsKernel);
                                    const auto s = consumerNode.Kernel->getStride();
                                    const auto expectedInput = (iRate.getLowerBound() + iRate.getUpperBound()) * Rational{s, 2};
                                    expInRate = multiply(VarList[consumer], expectedInput);
                                }

                                soft_assert(Z3_mk_eq(ctx, expOutRate, expInRate));
                            }
                        }
                    }

                }
            }
        }
    }

    #ifndef NDEBUG
    const reverse_traversal ordering(numOfPartitions);
    assert (is_valid_topological_sorting(ordering, T));
    #endif


    // iterate through the graph in topological order to determine what portions of
    // the program are not strictly fixed rate
    for (unsigned partId = 0; partId < numOfPartitions; ++partId) {
        for (const auto e : make_iterator_range(in_edges(partId, T))) {
            if (T[e]) {
                T[partId] = true;
                for (const auto f : make_iterator_range(out_edges(partId, T))) {
                    T[f] = true;
                }
                break;
            }
        }
    }

    if (LLVM_UNLIKELY(check() == Z3_L_FALSE)) {
        report_fatal_error("Z3 failed to find a solution to the maximum permitted dataflow problem");
    }

    const auto model = Z3_optimize_get_model(ctx, solver);
    Z3_model_inc_ref(ctx, model);

    size_t lcmOfDenom = 1UL;

    for (unsigned partId = 0; partId < numOfPartitions; ++partId) {
        PartitionData & N = P[partId];
        Z3_ast const stridesPerSegmentVar = VarList[N.Kernels[0]];
        Z3_ast value;
        if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, stridesPerSegmentVar, Z3_L_TRUE, &value) != Z3_L_TRUE)) {
            report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
        }

        Z3_int64 num, denom;
        if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &num, &denom) != Z3_L_TRUE)) {
            report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
        }


        assert (denom > 0);
        assert (num > 0);
        assert (N.Repetitions[0].numerator() > 0);
        assert (N.Repetitions[0].denominator() > 0);

        N.ExpectedStridesPerSegment = Rational{num, denom} / N.Repetitions[0];

        const auto m = N.ExpectedStridesPerSegment.denominator();
        if (m > 1) {
            lcmOfDenom = boost::lcm(lcmOfDenom, m);
        }

        N.StridesPerSegmentCoV = Rational{T[partId] ? 1U : 0U, 3U};
        N.LinkedGroupId = partId;
    }

    Z3_model_dec_ref(ctx, model);
    Z3_optimize_dec_ref(ctx, solver);
    Z3_del_context(ctx);
    Z3_reset_memory();

    for (unsigned partId = 0; partId < numOfPartitions; ++partId) {
        PartitionData & N = P[partId];
        N.ExpectedStridesPerSegment *= lcmOfDenom;
    }

}


}
