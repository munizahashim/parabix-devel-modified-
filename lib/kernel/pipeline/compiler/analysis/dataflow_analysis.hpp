#ifndef DATAFLOW_ANALYSIS_HPP
#define DATAFLOW_ANALYSIS_HPP

#include "pipeline_analysis.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeMinimumExpectedDataflow
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::computeMinimumExpectedDataflow(PartitionGraph & P) {

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

    const auto intType = Z3_mk_int_sort(ctx);

    auto constant_real = [&](const Rational value) {
        assert (value.numerator() > 0);
        return Z3_mk_real(ctx, value.numerator(), value.denominator());
    };

    auto multiply =[&](Z3_ast X, Z3_ast Y) {
        assert (X);
        assert (Y);
        Z3_ast args[2] = { X, Y };
        return Z3_mk_mul(ctx, 2, args);
    };

    const auto ONE = Z3_mk_int(ctx, 1, intType);

    const auto numOfPartitions = num_vertices(P);

    const auto m = num_vertices(Relationships);

    std::vector<Z3_ast> VarList(m);

    for (unsigned partition = 0; partition < numOfPartitions; ++partition) {
        const PartitionData & N = P[partition];
        for (const auto u : N.Kernels) {
            auto repVar = Z3_mk_fresh_const(ctx, nullptr, intType);
            hard_assert(Z3_mk_ge(ctx, repVar, ONE));
            VarList[u] = repVar; // multiply(rootVar, repVar);
        }
    }

    #warning Verify min dataflow considers multiple inputs of differing rates

    for (unsigned producerPartitionId = 1; producerPartitionId < numOfPartitions; ++producerPartitionId) {
        PartitionData & N = P[producerPartitionId];
        const auto & K = N.Kernels;

        for (const auto producer : K) {

            const RelationshipNode & producerNode = Relationships[producer];
            assert (producerNode.Type == RelationshipNode::IsKernel);

            const auto strideSize = producerNode.Kernel->getStride();

            assert (Relationships[producer].Type == RelationshipNode::IsKernel);
            for (const auto e : make_iterator_range(out_edges(producer, Relationships))) {
                const auto binding = target(e, Relationships);
                if (Relationships[binding].Type == RelationshipNode::IsBinding) {
                    const auto f = first_out_edge(binding, Relationships);
                    assert (Relationships[f].Reason != ReasonType::Reference);
                    const auto streamSet = target(f, Relationships);
                    assert (Relationships[streamSet].Type == RelationshipNode::IsRelationship);
                    assert (isa<StreamSet>(Relationships[streamSet].Relationship));

                    const RelationshipNode & output = Relationships[binding];
                    assert (output.Type == RelationshipNode::IsBinding);

                    const Binding & outputBinding = output.Binding;
                    const ProcessingRate & rate = outputBinding.getRate();
                    // ignore unknown output rates; we cannot reason about them here.
                    if (LLVM_LIKELY(rate.isUnknown())) {
                        continue;
                    }

                    const auto sum = rate.getLowerBound() + rate.getUpperBound();
                    const auto expectedOutput = sum * Rational{strideSize, 2};

                    assert (VarList[producer]);

                    const Z3_ast expOutRate = multiply(VarList[producer], constant_real(expectedOutput));

                    for (const auto e : make_iterator_range(out_edges(streamSet, Relationships))) {
                        const auto binding = target(e, Relationships);
                        const RelationshipNode & input = Relationships[binding];
                        if (LLVM_LIKELY(input.Type == RelationshipNode::IsBinding)) {
                            const auto f = first_out_edge(binding, Relationships);
                            assert (Relationships[f].Reason != ReasonType::Reference);
                            const unsigned consumer = target(f, Relationships);

                            const Binding & inputBinding = input.Binding;
                            const ProcessingRate & rate = inputBinding.getRate();

                            const auto c = PartitionIds.find(consumer);
                            assert (c != PartitionIds.end());
                            const auto consumerPartitionId = c->second;
                            assert (producerPartitionId <= consumerPartitionId);

                            const RelationshipNode & node = Relationships[consumer];
                            assert (node.Type == RelationshipNode::IsKernel);

                            if (LLVM_LIKELY(!rate.isGreedy())) {

                                const auto strideSize = node.Kernel->getStride();
                                const auto sum = rate.getLowerBound() + rate.getUpperBound();
                                const auto expectedInput = sum * Rational{strideSize, 2};
                                assert (VarList[consumer]);
                                const Z3_ast expInRate = multiply(VarList[consumer], constant_real(expectedInput));

                                if (producerPartitionId == consumerPartitionId && rate.isFixed()) {
                                    hard_assert(Z3_mk_eq(ctx, expOutRate, expInRate));
                                } else {
                                    hard_assert(Z3_mk_ge(ctx, expOutRate, expInRate));
                                    soft_assert(Z3_mk_eq(ctx, expOutRate, expInRate));
                                }
                            }
                        }
                    }

                }
            }
        }
    }

    #warning ADD LENGTH EQUALITY ASSERTIONS HERE

    if (LLVM_UNLIKELY(check() == Z3_L_FALSE)) {
        assert (false);
        report_fatal_error("Z3 failed to find a solution to minimum expected dataflow problem");
    }

    const auto model = Z3_optimize_get_model(ctx, solver);
    Z3_model_inc_ref(ctx, model);
    for (unsigned partition = 0; partition < numOfPartitions; ++partition) {
        PartitionData & N = P[partition];
        const auto & K = N.Kernels;
        const auto n = K.size();
        N.Repetitions.resize(n);
        for (unsigned i = 0; i < n; ++i) {
            Z3_ast value;
            if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, VarList[K[i]], Z3_L_TRUE, &value) != Z3_L_TRUE)) {
                report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
            }

            Z3_int64 num, denom;
            if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &num, &denom) != Z3_L_TRUE)) {
                report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
            }
            assert (num > 0 && denom == 1);
            N.Repetitions[i] = num;
        }


    }
    Z3_model_dec_ref(ctx, model);

    Z3_optimize_dec_ref(ctx, solver);
    Z3_del_context(ctx);

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeExpectedDataflow
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::computeExpectedDataflow() {

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

    const auto intType = Z3_mk_int_sort(ctx);

    const auto realType = Z3_mk_real_sort(ctx);

    auto constant_real = [&](const Rational & value) {
        return Z3_mk_real(ctx, value.numerator(), value.denominator());
    };

    auto maximum = [&](const BufferPort & rate) {
        return constant_real(rate.Maximum);
    };

    auto minimum = [&](const BufferPort & rate) {
        return constant_real(rate.Minimum);
    };

    auto multiply =[&](Z3_ast X, Z3_ast Y) {
        Z3_ast args[2] = { X, Y };
        return Z3_mk_mul(ctx, 2, args);
    };

    const auto firstKernel = out_degree(PipelineInput, mBufferGraph) == 0 ? FirstKernel : PipelineInput;
    const auto lastKernel = in_degree(PipelineOutput, mBufferGraph) == 0 ? LastKernel : PipelineOutput;

    std::vector<Z3_ast> VarList(LastStreamSet + 1, nullptr);

    bool useIntNumbers = true;

retry:

    Z3_optimize_push(ctx, solver);

    const auto ONE = constant_real(1);
    const auto TWO = constant_real(2);

    BEGIN_SCOPED_REGION

    auto make_partition_vars = [&](const unsigned first, const unsigned last) {
        auto rootVar = Z3_mk_fresh_const(ctx, nullptr, useIntNumbers ? intType : realType);
        hard_assert(Z3_mk_ge(ctx, rootVar, ONE));
        hard_assert(Z3_mk_le(ctx, rootVar, TWO));
        for (auto kernel = first; kernel <= last; ++kernel) {
            VarList[kernel] = multiply(rootVar, constant_real(MinimumNumOfStrides[kernel]));
        }
    };

    auto currentPartitionId = KernelPartitionId[firstKernel];
    auto firstKernelInPartition = firstKernel;
    for (auto kernel = (firstKernel + 1U); kernel < lastKernel; ++kernel) {
        const auto partitionId = KernelPartitionId[kernel];
        if (partitionId != currentPartitionId) {
            make_partition_vars(firstKernelInPartition, kernel - 1U);
            // set the first kernel for the next partition
            firstKernelInPartition = kernel;
            currentPartitionId = partitionId;
        }
    }
    make_partition_vars(firstKernelInPartition, lastKernel);

    END_SCOPED_REGION

    for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

        const auto partitionId = KernelPartitionId[kernel];

        const auto stridesPerSegmentVar = VarList[kernel];
        assert (stridesPerSegmentVar);

        unsigned numOfGreedyRates = 0;

        for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const auto streamSet = source(input, mBufferGraph);
            const auto producer = parent(streamSet, mBufferGraph);

            // we're only interested in inter-partition dataflow here
            if (KernelPartitionId[producer] == partitionId) {
                continue;
            }

            const BufferPort & inputRate = mBufferGraph[input];
            const Binding & binding = inputRate.Binding;
            const ProcessingRate & rate = binding.getRate();

            const auto producedRate = VarList[streamSet]; assert (producedRate);

            if (LLVM_UNLIKELY(rate.isGreedy())) {
                ++numOfGreedyRates;
                // ideally we want to always have enough data to execute
                soft_assert(Z3_mk_ge(ctx, producedRate, minimum(inputRate)));
            } else { // Fixed, Bounded or Partial Sum
                const auto consumedRate = multiply(stridesPerSegmentVar, minimum(inputRate));
                hard_assert(Z3_mk_ge(ctx, producedRate, consumedRate));
                Z3_ast args[2] = { producedRate, consumedRate };
                const auto diff = Z3_mk_sub(ctx, 2, args);
                Z3_optimize_minimize(ctx, solver, diff);
            }
        }

        // Any kernel with all greedy rates must exhaust its input in a single iteration.
        if (LLVM_UNLIKELY(numOfGreedyRates > 0 && numOfGreedyRates == in_degree(kernel, mBufferGraph))) {
            hard_assert(Z3_mk_eq(ctx, stridesPerSegmentVar, ONE));
        }

        for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const BufferPort & outputRate = mBufferGraph[output];
            const Binding & binding = outputRate.Binding;
            const ProcessingRate & rate = binding.getRate();
            const auto streamSet = target(output, mBufferGraph);

            assert (VarList[streamSet] == nullptr);

            if (LLVM_UNLIKELY(rate.isUnknown())) {
                // TODO: is there a better way to handle unknown outputs? This
                // free variable represents the ideal amount of data to transfer
                // to subsequent kernels but that isn't very meaningful here.
                auto v = Z3_mk_fresh_const(ctx, nullptr, intType);
                hard_assert(Z3_mk_ge(ctx, v, minimum(outputRate)));
                VarList[streamSet] = v;
            } else { // Fixed, Bounded or Partial Sum
                VarList[streamSet] = multiply(stridesPerSegmentVar, maximum(outputRate));
            }
        }
    }

    // Include any length equality assertions
    if (!mLengthAssertions.empty()) {
        flat_map<const StreamSet *, unsigned> M;

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            const auto output = in_edge(streamSet, mBufferGraph);
            const BufferPort & br = mBufferGraph[output];
            const Binding & binding = br.Binding;
            M.emplace(cast<StreamSet>(binding.getRelationship()), streamSet);
        }

        auto offset = [&](const StreamSet * streamSet) {
            const auto f = M.find(streamSet);
            assert (f != M.end());
            return f->second;
        };

        for (const LengthAssertion & la : mLengthAssertions) {
            const auto A = VarList[offset(la[0])];
            const auto B = VarList[offset(la[1])];
            soft_assert(Z3_mk_eq(ctx, A, B));
        }
    }

    if (LLVM_UNLIKELY(check() == Z3_L_FALSE)) {
        if (useIntNumbers) {
            // Earlier versions of Z3 seem to have an issue working out a solution to some problems
            // when using int-type variables. However, using real numbers generates an "infinite"
            // number of potential solutions and takes considerably longer to finish. Thus we only
            // fall back to using rational variables if this test fails.
            Z3_optimize_pop(ctx, solver);
            useIntNumbers = false;
            #ifndef NDEBUG
            std::fill(VarList.begin(), VarList.end(), nullptr);
            #endif
            goto retry;
        }
        report_fatal_error("Z3 failed to find a solution to the maximum permitted dataflow problem");
    }

    MaximumNumOfStrides.resize(PipelineOutput + 1);

    const auto model = Z3_optimize_get_model(ctx, solver);
    Z3_model_inc_ref(ctx, model);
    for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

        Z3_ast const stridesPerSegmentVar = VarList[kernel];
        Z3_ast value;
        if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, stridesPerSegmentVar, Z3_L_TRUE, &value) != Z3_L_TRUE)) {
            report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
        }

        Z3_int64 num, denom;
        if (LLVM_LIKELY(useIntNumbers)) {
            if (LLVM_UNLIKELY(Z3_get_numeral_int64(ctx, value, &num) != Z3_L_TRUE)) {
                report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
            }
            MaximumNumOfStrides[kernel] = num;
        } else {
            if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &num, &denom) != Z3_L_TRUE)) {
                report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
            }

            const auto x = MinimumNumOfStrides[kernel];
            const auto y = ceiling(Rational{num, denom * x});
            MaximumNumOfStrides[kernel]  = y * x;
        }


    }
    Z3_model_dec_ref(ctx, model);

    Z3_optimize_dec_ref(ctx, solver);
    Z3_del_context(ctx);
    Z3_reset_memory();

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeMaximumDataflow
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::computeMaximumDataflow(const bool expected) {

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

    const auto intType = Z3_mk_int_sort(ctx);

    const auto realType = Z3_mk_real_sort(ctx);

    auto constant_real = [&](const Rational & value) {
        return Z3_mk_real(ctx, value.numerator(), value.denominator());
    };

    auto maximum = [&](const BufferPort & rate) {
        return constant_real(rate.Maximum);
    };

    auto minimum = [&](const BufferPort & rate) {
        return constant_real(rate.Minimum);
    };

    auto minimum_non_zero = [&](const BufferPort & rate) {
        return constant_real(std::max(rate.Minimum, Rational{1}));
    };

    auto multiply =[&](Z3_ast X, Z3_ast Y) {
        Z3_ast args[2] = { X, Y };
        return Z3_mk_mul(ctx, 2, args);
    };

    const auto firstKernel = out_degree(PipelineInput, mBufferGraph) == 0 ? FirstKernel : PipelineInput;
    const auto lastKernel = in_degree(PipelineOutput, mBufferGraph) == 0 ? LastKernel : PipelineOutput;

    std::vector<Z3_ast> VarList(LastStreamSet + 1, nullptr);

    bool useIntNumbers = true;

retry:

    Z3_optimize_push(ctx, solver);

    const auto ONE = constant_real(1);
    const auto TWO = constant_real(2);

    BEGIN_SCOPED_REGION

    auto make_partition_vars = [&](const unsigned first, const unsigned last) {
        auto rootVar = Z3_mk_fresh_const(ctx, nullptr, useIntNumbers ? intType : realType);
        hard_assert(Z3_mk_ge(ctx, rootVar, ONE));
        if (expected) {
            hard_assert(Z3_mk_le(ctx, rootVar, TWO));
        }
        for (auto kernel = first; kernel <= last; ++kernel) {
            VarList[kernel] = multiply(rootVar, constant_real(MinimumNumOfStrides[kernel]));
        }
    };

    auto currentPartitionId = KernelPartitionId[firstKernel];
    auto firstKernelInPartition = firstKernel;
    for (auto kernel = (firstKernel + 1U); kernel < lastKernel; ++kernel) {
        const auto partitionId = KernelPartitionId[kernel];
        if (partitionId != currentPartitionId) {
            make_partition_vars(firstKernelInPartition, kernel - 1U);
            // set the first kernel for the next partition
            firstKernelInPartition = kernel;
            currentPartitionId = partitionId;
        }
    }
    make_partition_vars(firstKernelInPartition, lastKernel);

    END_SCOPED_REGION

    std::vector<Z3_ast> inputVars;

    for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

        const auto partitionId = KernelPartitionId[kernel];

        const auto stridesPerSegmentVar = VarList[kernel];
        assert (stridesPerSegmentVar);

        inputVars.clear();
        bool noNonGreedyRates = true;

        for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const auto streamSet = source(input, mBufferGraph);
            const auto producer = parent(streamSet, mBufferGraph);

            // we're only interested in inter-partition dataflow here
            if (KernelPartitionId[producer] == partitionId) {
                continue;
            }

            const BufferPort & inputRate = mBufferGraph[input];
            const Binding & binding = inputRate.Binding;
            const ProcessingRate & rate = binding.getRate();

            const auto producedRate = VarList[streamSet]; assert (producedRate);

            if (LLVM_UNLIKELY(rate.isGreedy())) {
                // ideally we want to always have enough data to execute
                soft_assert(Z3_mk_ge(ctx, producedRate, minimum(inputRate)));
            } else { // Fixed, Bounded or Partial Sum

                noNonGreedyRates = false;

                const auto consumedRate = multiply(stridesPerSegmentVar, minimum_non_zero(inputRate));
                hard_assert(Z3_mk_ge(ctx, producedRate, consumedRate));

                Z3_ast args[2] = { producedRate, consumedRate };
                const auto diff = Z3_mk_sub(ctx, 2, args);

                if (expected) {
                    Z3_optimize_minimize(ctx, solver, diff);
                } else {
                    soft_assert(Z3_mk_eq(ctx, producedRate, consumedRate));
                    inputVars.push_back(diff);
                }

            }
        }

        // Any kernel with all greedy rates must exhaust its input in a single iteration.
        if (LLVM_UNLIKELY(noNonGreedyRates)) {
            soft_assert(Z3_mk_eq(ctx, stridesPerSegmentVar, ONE));
        } else if (!expected) {
            const auto m = inputVars.size(); assert (m > 0);
            const auto sum = Z3_mk_add(ctx, m, inputVars.data());
            Z3_optimize_minimize(ctx, solver, sum);

//            const auto m = inputVars.size(); assert (m > 0);
//            auto minNumOfStrides = inputVars[0];
//            for (unsigned i = 1; i < m; ++i) {
//                auto cond = Z3_mk_lt(ctx, minNumOfStrides, inputVars[i]);
//                minNumOfStrides = Z3_mk_ite(ctx, cond, minNumOfStrides, inputVars[i]);
//            }
//            hard_assert(Z3_mk_ge(ctx, stridesPerSegmentVar, minNumOfStrides));
        }

        for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const BufferPort & outputRate = mBufferGraph[output];
            const Binding & binding = outputRate.Binding;
            const ProcessingRate & rate = binding.getRate();
            const auto streamSet = target(output, mBufferGraph);

            assert (VarList[streamSet] == nullptr);

            if (LLVM_UNLIKELY(rate.isUnknown())) {
                // TODO: is there a better way to handle unknown outputs? This
                // free variable represents the ideal amount of data to transfer
                // to subsequent kernels but that isn't very meaningful here.
                auto v = Z3_mk_fresh_const(ctx, nullptr, intType);
                hard_assert(Z3_mk_ge(ctx, v, minimum(outputRate)));
                VarList[streamSet] = v;
            } else { // Fixed, Bounded or Partial Sum
                VarList[streamSet] = multiply(stridesPerSegmentVar, maximum(outputRate));
            }
        }
    }

    // Include any length equality assertions
    if (!mLengthAssertions.empty()) {
        flat_map<const StreamSet *, unsigned> M;

        for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
            const auto output = in_edge(streamSet, mBufferGraph);
            const BufferPort & br = mBufferGraph[output];
            const Binding & binding = br.Binding;
            M.emplace(cast<StreamSet>(binding.getRelationship()), streamSet);
        }

        auto offset = [&](const StreamSet * streamSet) {
            const auto f = M.find(streamSet);
            assert (f != M.end());
            return f->second;
        };

        for (const LengthAssertion & la : mLengthAssertions) {
            const auto A = VarList[offset(la[0])];
            const auto B = VarList[offset(la[1])];
            soft_assert(Z3_mk_eq(ctx, A, B));
        }
    }

    errs() << "----------------\n";

    errs() << Z3_optimize_to_string(ctx, solver);

    errs() << "----------------\n";

    if (LLVM_UNLIKELY(check() == Z3_L_FALSE)) {
        if (useIntNumbers) {
            // Earlier versions of Z3 seem to have an issue working out a solution to some problems
            // when using int-type variables. However, using real numbers generates an "infinite"
            // number of potential solutions and takes considerably longer to finish. Thus we only
            // fall back to using rational variables if this test fails.
            Z3_optimize_pop(ctx, solver);
            useIntNumbers = false;
            #ifndef NDEBUG
            std::fill(VarList.begin(), VarList.end(), nullptr);
            #endif
            goto retry;
        }
        report_fatal_error("Z3 failed to find a solution to the maximum permitted dataflow problem");
    }

    auto updateStrideModel = [&](std::vector<unsigned> & numOfStrides) {
        numOfStrides.resize(PipelineOutput + 1);

        const auto model = Z3_optimize_get_model(ctx, solver);
        Z3_model_inc_ref(ctx, model);
        for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

            Z3_ast const stridesPerSegmentVar = VarList[kernel];
            Z3_ast value;
            if (LLVM_UNLIKELY(Z3_model_eval(ctx, model, stridesPerSegmentVar, Z3_L_TRUE, &value) != Z3_L_TRUE)) {
                report_fatal_error("Unexpected Z3 error when attempting to obtain value from model!");
            }

            Z3_int64 num, denom;
            if (LLVM_LIKELY(useIntNumbers)) {
                if (LLVM_UNLIKELY(Z3_get_numeral_int64(ctx, value, &num) != Z3_L_TRUE)) {
                    report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
                }
                numOfStrides[kernel] = num;
            } else {
                if (LLVM_UNLIKELY(Z3_get_numeral_rational_int64(ctx, value, &num, &denom) != Z3_L_TRUE)) {
                    report_fatal_error("Unexpected Z3 error when attempting to convert model value to number!");
                }

                const auto x = MinimumNumOfStrides[kernel];
                const auto y = ceiling(Rational{num, denom * x});
                numOfStrides[kernel]  = y * x;
            }


        }
        Z3_model_dec_ref(ctx, model);
    };

    if (expected) {
        updateStrideModel(ExpectedNumOfStrides);
    } else {
        updateStrideModel(MaximumNumOfStrides);
    }

    Z3_optimize_dec_ref(ctx, solver);
    Z3_del_context(ctx);
    Z3_reset_memory();

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyInterPartitionSymbolicRates
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyInterPartitionSymbolicRates() {

    using BitSet = dynamic_bitset<>;

    const auto firstKernel = out_degree(PipelineInput, mBufferGraph) == 0 ? FirstKernel : PipelineInput;
    const auto lastKernel = in_degree(PipelineOutput, mBufferGraph) == 0 ? LastKernel : PipelineOutput;

    const auto m = num_edges(mBufferGraph);

    std::vector<BitSet> portRateSet(m + LastStreamSet - FirstStreamSet + 1U);

    unsigned portNum = 0;
    unsigned nextRateId = PartitionCount;

    for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {
        auto updateEdgeRate = [&](const BufferGraph::edge_descriptor & e, const size_t streamSet) {
            BufferPort & port = mBufferGraph[e];
            assert (portNum < m);
            port.SymbolicRateId = portNum++;
            const BufferNode & bn = mBufferGraph[streamSet];
            if (bn.isNonThreadLocal()) {
                const Binding & binding = port.Binding;
                if (isNonSynchronousRate(binding)) {
                    BitSet & bs = portRateSet[port.SymbolicRateId];
                    const auto rateId = nextRateId++;
                    bs.resize(nextRateId);
                    bs.set(rateId);
                }
            }
            if (!bn.IsLinear) {
                // TODO: this is overly strict but we'd need to prove that for all
                // mutually reachable I/O states of both the producer and consumer,
                // they would be at the same mod'ed position.
                const auto k = m + streamSet - FirstStreamSet;
                assert (k < portRateSet.size());
                BitSet & bs = portRateSet[k];
                const auto rateId = nextRateId++;
                bs.resize(nextRateId);
                bs.set(rateId);
            }
        };

        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            updateEdgeRate(e, source(e, mBufferGraph));
        }
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            updateEdgeRate(e, target(e, mBufferGraph));
        }
    }

    BitSet accum(nextRateId);

    for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

        accum.reset();

        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            const auto streamSet = source(e, mBufferGraph);
            const BufferPort & port = mBufferGraph[e];
            BitSet & bs = portRateSet[port.SymbolicRateId];
            bs.resize(nextRateId);
            bs.set(KernelPartitionId[kernel]);
            const auto k = m + streamSet - FirstStreamSet;
            assert (k < portRateSet.size());
            const BitSet & src = portRateSet[m + streamSet - FirstStreamSet];
            assert (src.size() == bs.size());
            bs |= src;
            accum |= bs;
        }

        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(e, mBufferGraph);
            const BufferPort & port = mBufferGraph[e];
            BitSet & bs = portRateSet[port.SymbolicRateId];
            bs.resize(nextRateId);
            const auto k = m + streamSet - FirstStreamSet;
            assert (k < portRateSet.size());
            BitSet & dst = portRateSet[m + streamSet - FirstStreamSet];
            dst.resize(nextRateId);
            bs |= accum;
            dst |= bs;
        }
    }

    std::map<BitSet, unsigned> rateMap;

    for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

        auto updateEdgeRate = [&](const BufferGraph::edge_descriptor & e) {
            BufferPort & port = mBufferGraph[e];
            BitSet & bs = portRateSet[port.SymbolicRateId];


            assert (bs.size() == nextRateId);
            const auto f = rateMap.find(bs);
            unsigned symRateId = 0;
            if (f == rateMap.end()) {
                symRateId = (unsigned)rateMap.size() + 1U;
                rateMap.emplace(std::move(bs), symRateId);
            } else {
                symRateId = f->second;
            }
            port.SymbolicRateId = symRateId;
        };

        for (const auto e : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            updateEdgeRate(e);
        }
        for (const auto e : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            updateEdgeRate(e);
        }
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeMinimumStrideLengthForConsistentDataflow
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::computeMinimumStrideLengthForConsistentDataflow() {

    // TODO: we already do this when scheduling. Organize the logic better to only do it once if this
    // ends up being necessary for performance.

    const auto firstKernel = out_degree(PipelineInput, mBufferGraph) == 0 ? FirstKernel : PipelineInput;
    const auto lastKernel = in_degree(PipelineOutput, mBufferGraph) == 0 ? LastKernel : PipelineOutput;

    StrideStepLength.resize(PipelineOutput + 1);

    auto make_partition_vars = [&](const unsigned first, const unsigned last) {
        auto gcd = MaximumNumOfStrides[first];
        for (auto i = first + 1; i <= last; ++i) {
            gcd = boost::gcd(gcd, MaximumNumOfStrides[i]);
        }
        for (auto i = first; i <= last; ++i) {
            StrideStepLength[i] = (MaximumNumOfStrides[i] / gcd);
        }
    };

    auto currentPartitionId = KernelPartitionId[firstKernel];
    auto firstKernelInPartition = firstKernel;
    for (auto kernel = (firstKernel + 1U); kernel <= lastKernel; ++kernel) {
        const auto partitionId = KernelPartitionId[kernel];
        if (partitionId != currentPartitionId) {
            make_partition_vars(firstKernelInPartition, kernel - 1U);
            // set the first kernel for the next partition
            firstKernelInPartition = kernel;
            currentPartitionId = partitionId;
        }
    }
    if (firstKernelInPartition <= lastKernel) {
        make_partition_vars(firstKernelInPartition, lastKernel);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief calculatePartialSumStepFactors
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::calculatePartialSumStepFactors() {

    PartialSumStepFactorGraph G(LastStreamSet + 1);

    for (auto kernel = LastKernel; kernel >= FirstKernel; --kernel) {

        auto checkForPopCountRef = [&](const BufferGraph::edge_descriptor io) {
            const BufferPort & port = mBufferGraph[io];
            const Binding & binding = port.Binding;
            const ProcessingRate & rate = binding.getRate();
            if (LLVM_UNLIKELY(rate.isPartialSum())) {
                const auto inputRefPort = getReference(kernel, port.Port);
                const auto streamSet = getInputBufferVertex(kernel, inputRefPort);
                assert (streamSet > LastKernel);
                const auto refOuput = in_edge(streamSet, mBufferGraph);
                const BufferPort & refOutputRate = mBufferGraph[refOuput];
                const BufferPort & refInputRate = getInputPort(kernel, inputRefPort);
                const auto ratio = refInputRate.Minimum / refOutputRate.Minimum;
                assert (ratio.denominator() == 1 && ratio.numerator() > 0);
                assert (!edge(streamSet, kernel, G).second);
                add_edge(streamSet, kernel, ratio.numerator(), G);
            }
        };

        for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
            checkForPopCountRef(input);
        }
        for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            checkForPopCountRef(output);
        }

        for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
            const auto streamSet = target(output, mBufferGraph);
            if (out_degree(streamSet, G) != 0) {
                unsigned maxStepFactor = 0;
                for (const auto e : make_iterator_range(out_edges(streamSet, G))) {
                    maxStepFactor = std::max(maxStepFactor, G[e]);
                }
                assert (maxStepFactor > 0);
                if (maxStepFactor > 1) {
                    add_edge(kernel, streamSet, maxStepFactor, G);
                }
            }
        }
    }

    mPartialSumStepFactorGraph = G;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifyKernelsOnHybridThread
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifyKernelsOnHybridThread() {
    KernelOnHybridThread.resize(PipelineOutput + 1U);
    KernelOnHybridThread.reset();
    PartitionOnHybridThread.resize(PartitionCount);
    PartitionOnHybridThread.reset();
    if (mPipelineKernel->getNumOfThreads() > 1) {
        for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
            if (LLVM_UNLIKELY(getKernel(i)->hasAttribute(AttrId::IsolateOnHybridThread))) {
                KernelOnHybridThread.set(i);
            }
        }
        for (unsigned k : KernelOnHybridThread.set_bits()) {
            assert (k <= LastKernel);
            const auto p = KernelPartitionId[k];
            assert (p < PartitionCount);
            PartitionOnHybridThread.set(p);
        }
    }
}

} // end of kernel namespace

#endif // DATAFLOW_ANALYSIS_HPP
