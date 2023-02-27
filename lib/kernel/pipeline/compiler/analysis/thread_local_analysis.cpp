#include "pipeline_analysis.hpp"
#include "evolutionary_algorithm.hpp"
#include <boost/icl/interval_set.hpp>

using boost::icl::interval_set;

namespace kernel {

// TODO: nested pipeline kernels could report how much internal memory they require
// and reason about that here (and in the scheduling phase)

constexpr static unsigned BUFFER_SIZE_INIT_POPULATION_SIZE = 15;

constexpr static unsigned BUFFER_SIZE_GA_MAX_INIT_TIME_SECONDS = 2;

constexpr static unsigned BUFFER_SIZE_POPULATION_SIZE = 30;

constexpr static unsigned BUFFER_SIZE_GA_MAX_TIME_SECONDS = 15;

constexpr static unsigned BUFFER_SIZE_GA_STALLS = 50;

// Intel spatial prefetcher pulls cache line pairs, aligned to 128 bytes.
constexpr static unsigned SPATIAL_PREFETCHER_ALIGNMENT = 128;

constexpr static unsigned NON_HUGE_PAGE_SIZE = 4096;

using IntervalGraph = adjacency_list<hash_setS, vecS, undirectedS>;

using IntervalSet = interval_set<unsigned>;

using Interval = IntervalSet::interval_type; // std::pair<unsigned, unsigned>;


struct BufferLayoutOptimizerWorker final : public PermutationBasedEvolutionaryAlgorithmWorker {

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief repair
     ** ------------------------------------------------------------------------------------------------------------- */
    void repair(Candidate & /* candidate */, pipeline_random_engine & rng) final { }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief fitness
     ** ------------------------------------------------------------------------------------------------------------- */
    size_t fitness(const Candidate & candidate, pipeline_random_engine & rng) final {

        const auto candidateLength = candidate.size();

        size_t max_colours = 0;
        for (unsigned i = 0; i < candidateLength; ++i) {
            const auto a = candidate[i];
            assert (a < candidateLength);
            size_t w = weight[a];

            assert (GC_IntervalSet.empty());

            for (unsigned j = 0; j != i; ++j) {
                assert (j < candidateLength);
                const auto b = candidate[j];
                assert (b < candidateLength);
                if (edge(a, b, I).second) {
                    const auto & interval = GC_Intervals[b];
                    auto l = interval.lower();
                    auto r = interval.upper();
                    GC_IntervalSet.insert(Interval::right_open(l, r));
                }
            }

            size_t start = 0;
            auto end = w;
            if (!GC_IntervalSet.empty()) {
//                auto d = w;
                for (const auto & interval : GC_IntervalSet) {
                    if (end < interval.lower()) {
                        break;
                    } else {
//                        const auto l = interval.lower();
                        const auto r = interval.upper();
                        // We want memory to be laid out s.t. when we expand it at run time,
                        // we're guaranteed that we won't overlap another buffer and ideally
                        // optimize to a solution that won't require a huge amount of
                        // additional space. To do so, we increase the weight (bytes required)
                        // so that the size of each placement in sequence is non-decreasing.

                        // NOTE: this is not the final size of the placement.

                        // TODO: is max sufficient? do we need a LCM?
//                        const auto m = r - l;
//                        if (d < m) {
//                            d = m;
//                        }
                        start = r;
                        end = r + w;
                    }
                }
                GC_IntervalSet.clear();
            }
            assert (a < candidateLength);
//            const auto end = start + w;
            GC_Intervals[a] = Interval::right_open(start, end);
            max_colours = std::max(max_colours, end);
        }

        return max_colours;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getIntervals
     ** ------------------------------------------------------------------------------------------------------------- */
    const std::vector<Interval> & getIntervals(const OrderingDAWG & O, const unsigned candidateLength, pipeline_random_engine & rng) {
        Candidate chosen;
        chosen.reserve(candidateLength);
        Vertex u = 0;
        while (out_degree(u, O) != 0) {
            const auto e = first_out_edge(u, O);
            const auto k = O[e];
            chosen.push_back(k);
            u = target(e, O);
        }
        fitness(chosen, rng);
        return GC_Intervals;
    }

    BufferLayoutOptimizerWorker(const IntervalGraph & I, const std::vector<unsigned> & weight,
                                const unsigned candidateLength, pipeline_random_engine & rng)
    : I(I), weight(weight), GC_Intervals(candidateLength) {
        assert (num_vertices(I) == candidateLength);
        assert (weight.size() >= candidateLength);
    }

private:
    const IntervalGraph & I;
    const std::vector<unsigned> & weight;

    IntervalSet GC_IntervalSet;
    std::vector<Interval> GC_Intervals;
};

struct BufferLayoutOptimizer final : public PermutationBasedEvolutionaryAlgorithm {

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getIntervals
     ** ------------------------------------------------------------------------------------------------------------- */
    const std::vector<Interval> & getIntervals(const OrderingDAWG & O, pipeline_random_engine & rng) {
        auto w = (BufferLayoutOptimizerWorker *)mainWorker.get();
        return w->getIntervals(O, candidateLength, rng);
    }

    std::unique_ptr<PermutationBasedEvolutionaryAlgorithmWorker> makeWorker(pipeline_random_engine & rng) final {
        return std::make_unique<BufferLayoutOptimizerWorker>(I, weight, candidateLength, rng);
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief constructor
     ** ------------------------------------------------------------------------------------------------------------- */
    BufferLayoutOptimizer(const unsigned numOfLocalStreamSets
                         , IntervalGraph && I
                         , std::vector<unsigned> && weight
                         , pipeline_random_engine & srcRng)
    : PermutationBasedEvolutionaryAlgorithm (numOfLocalStreamSets,
                                             BUFFER_SIZE_GA_MAX_INIT_TIME_SECONDS,
                                             BUFFER_SIZE_INIT_POPULATION_SIZE,
                                             BUFFER_SIZE_GA_MAX_TIME_SECONDS,                                             
                                             BUFFER_SIZE_POPULATION_SIZE,
                                             BUFFER_SIZE_GA_STALLS,
                                             std::max(codegen::SegmentThreads, codegen::TaskThreads),
                                             srcRng)
    , I(std::move(I))
    , weight(weight) {

    }


private:

    const IntervalGraph I;
    const std::vector<unsigned> weight;

};

#if 1


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief determineInitialThreadLocalBufferLayout
 *
 * Given our buffer graph, we want to identify the best placement to maximize sequential prefetching behavior with
 * the minimal total memory required. Although this assumes that the memory-aware scheduling algorithm was first
 * called, it does not actually use any data from it. The reason for this disconnection is to enable us to explore
 * the impact of static memory allocation independent of the chosen scheduling algorithm.
 *
 * Because the Intel L2 streamer prefetcher has one forward and one reverse monitor per page, a streamset will
 * only be placed in a page in which no other streamset accesses it during the same kernel invocation.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::determineInitialThreadLocalBufferLayout(BuilderRef b, pipeline_random_engine & rng) {

    // This process serves two purposes: (1) generate the initial memory layout for our thread-local
    // streamsets. (2) determine how many the number of pages to assign each streamset based on the
    // number of strides executed by the parition root.

    const auto n = LastStreamSet - FirstStreamSet + 1U;

    // TODO: can we insert a zero-extension region rather than having a secondary buffer?

    std::vector<unsigned> mapping(n, -1U);

    RequiredThreadLocalStreamSetMemory = 0;

    PartitionRootStridesPerThreadLocalPage.resize(PartitionCount);

    NumOfPartialOverflowStridesPerPartitionRootStride.resize(PartitionCount);

    unsigned numOfThreadLocalStreamSets = 0U;

    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isThreadLocal()) {
            mapping[streamSet - FirstStreamSet] = numOfThreadLocalStreamSets;
            ++numOfThreadLocalStreamSets;
        }
    }

    if (LLVM_UNLIKELY(numOfThreadLocalStreamSets == 0)) {
        return;
    }

    DataLayout DL(b->getModule());

    const auto blockWidth = b->getBitBlockWidth();

    const size_t pageSize = b->getPageSize();

    IntervalGraph I(numOfThreadLocalStreamSets);

    std::vector<unsigned> weight(numOfThreadLocalStreamSets, 0);
    std::vector<int> remaining(numOfThreadLocalStreamSets, 0); // NOTE: signed int type is necessary here
    std::vector<Rational> streamSetFactor(numOfThreadLocalStreamSets);

    for (unsigned partitionId = 0; partitionId < PartitionCount; ++partitionId) {
        const auto firstKernel = FirstKernelInPartition[partitionId];
        const auto firstKernelOfNextPartition = FirstKernelInPartition[partitionId + 1];


        Rational minVal{std::numeric_limits<size_t>::max()};

        size_t maxOverflow{0};

        const auto firstStrideLength = getKernel(firstKernel)->getStride();

        const auto BW = firstStrideLength * StrideRepetitionVector[firstKernel];

        bool hasThreadLocal = false;

        for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {

            const auto strideLength = getKernel(kernel)->getStride();

            // Because data is layed out in a "strip mined" format within streamsets, the type of
            // each "chunk" will be blockwidth items in length.

            const Rational W{strideLength * StrideRepetitionVector[kernel], BW };

            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];

                if (bn.isThreadLocal()) {
                    // determine the number of bytes this streamset requires per *root kernel* stride
                    const BufferPort & producerRate = mBufferGraph[output];
                    const Binding & outputRate = producerRate.Binding;
                    Type * const type = StreamSetBuffer::resolveType(b, outputRate.getType());
                    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
                    const auto typeSize = DL.getTypeAllocSize(type);
                    #else
                    const auto typeSize = DL.getTypeAllocSize(type).getFixedSize();
                    #endif
                    const ProcessingRate & rate = outputRate.getRate();
                    assert (rate.isFixed() || rate.isPopCount());
                    const auto S = typeSize * W * rate.getUpperBound();

                    assert (S.numerator() > 0);

                    if (S < minVal) {
                        minVal = S;
                    }

                    const auto j = mapping[streamSet - FirstStreamSet];
                    assert (j != -1U);
                    streamSetFactor[j] = S;

                    assert (bn.UnderflowCapacity == 0);

                    const size_t overflowBytes = bn.OverflowCapacity * typeSize;
                    maxOverflow = std::max(maxOverflow, overflowBytes);

                    hasThreadLocal = true;
                }
            }
        }

        if (hasThreadLocal) {

            const auto M = Rational{pageSize} / minVal;

            PartitionRootStridesPerThreadLocalPage[partitionId] = M;

            NumOfPartialOverflowStridesPerPartitionRootStride[partitionId] = Rational{maxOverflow} / BW;

            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {
                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j != -1U);
                        const auto S = streamSetFactor[j] * M;
                        assert (S.denominator() == 1);
                        const auto size = S.numerator();
                        assert (size > 0);
                        weight[j] = size;
                        bn.RequiredCapacity = size;
                        // record how many consumers exist before the streamset memory can be reused
                        // (NOTE: the +1 is to indicate this kernel requires each output streamset
                        // to be distinct even if one or more of the outputs is not used later.)
                        remaining[j] = out_degree(streamSet, mBufferGraph) + 1U;
                    }
                }
            }

            // Mark any overlapping allocations in our interval graph.
            for (unsigned i = 0; i != numOfThreadLocalStreamSets; ++i) {
                if (remaining[i] > 0) {
                    for (unsigned j = 0; j != i; ++j) {
                        if (remaining[j] > 0) {
                            add_edge(j, i, I);
                        }
                    }
                }
            }


            // Determine which streamsets are no longer alive

            for (auto kernel = firstKernel; kernel < firstKernelOfNextPartition; ++kernel) {

                for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                    const auto streamSet = target(output, mBufferGraph);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j != -1U);
                        assert (remaining[j] > 0);
                        remaining[j]--;
                    }
                }
                for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                    const auto streamSet = source(input, mBufferGraph);
                    const BufferNode & bn = mBufferGraph[streamSet];
                    if (bn.isThreadLocal()) {
                        const auto j = mapping[streamSet - FirstStreamSet];
                        assert (j != -1U);
                        assert (remaining[j] > 0);
                        remaining[j]--;
                    }
                }

            }

        }
    }

    BufferLayoutOptimizer BA(numOfThreadLocalStreamSets, std::move(I), std::move(weight), rng);
    BA.runGA();

    auto requiredMemory = BA.getBestFitnessValue();
    assert (requiredMemory > 0);
    auto O = BA.getResult();

//    errs() << "requiredMemory=" << requiredMemory << "\n";

    // TODO: apart from total memory, when would one layout be better than another?
    // Can we quantify it based on the buffer graph order? Currently, we just take
    // the first one.
    const auto intervals = BA.getIntervals(O, rng);
    for (auto streamSet = FirstStreamSet; streamSet <= LastStreamSet; ++streamSet) {
        const auto i = streamSet - FirstStreamSet;
        BufferNode & bn = mBufferGraph[streamSet];
        if (bn.isThreadLocal()) {
            const auto j = mapping[i];
            const auto & interval = intervals[j];
            bn.BufferStart = interval.lower();
            bn.BufferEnd = interval.upper();
            assert (bn.BufferEnd <= requiredMemory);

            const auto producer = parent(streamSet, mBufferGraph);
            const auto partitionId = KernelPartitionId[producer];
            const auto root = FirstKernelInPartition[partitionId];
            const auto K = (MaximumNumOfStrides[root] / PartitionRootStridesPerThreadLocalPage[partitionId]);
            const auto L = K * bn.BufferEnd;
            assert (L.denominator() == 1);

            requiredMemory = std::max(requiredMemory, L.numerator());
        }
    }

//    Rational S{1};
//    for (auto partId = KernelPartitionId[FirstKernel]; partId < PartitionCount; ++partId ) {
//        const auto firstKernel = FirstKernelInPartition[partId];
//        if (in_degree(firstKernel, mBufferGraph) == 0) {
//            const size_t K = StrideRepetitionVector[partId];
//            const size_t L = MaximumNumOfStrides[firstKernel];
//            S = std::max(S, Rational{L, K});
//        }
//    }

//    const auto M = requiredMemory * ceiling(S);

//    errs() << "requiredMemory'=" << requiredMemory << "\n";

//    assert (M >= requiredMemory);
    RequiredThreadLocalStreamSetMemory = requiredMemory;

}


#else


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief determineInitialThreadLocalBufferLayout
 *
 * Given our buffer graph, we want to identify the best placement to maximize sequential prefetching behavior with
 * the minimal total memory required. Although this assumes that the memory-aware scheduling algorithm was first
 * called, it does not actually use any data from it. The reason for this disconnection is to enable us to explore
 * the impact of static memory allocation independent of the chosen scheduling algorithm.
 *
 * Because the Intel L2 streamer prefetcher has one forward and one reverse monitor per page, a streamset will
 * only be placed in a page in which no other streamset accesses it during the same kernel invocation.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::determineInitialThreadLocalBufferLayout(BuilderRef b, pipeline_random_engine & rng) {

    // This process serves two purposes: (1) generate the initial memory layout for our thread-local
    // streamsets. (2) determine how many the number of pages to assign each streamset based on the
    // number of strides executed by the parition root.

    const auto n = LastStreamSet - FirstStreamSet + 1U;

    // TODO: can we insert a zero-extension region rather than having a secondary buffer?

    std::vector<unsigned> weight(n, 0);
    std::vector<int> remaining(n, 0); // NOTE: signed int type is necessary here
    std::vector<unsigned> mapping(n, -1U);

    RequiredThreadLocalStreamSetMemory = 0;

    PartitionRootStridesPerThreadLocalPage.resize(PartitionCount);

    PartitionOverflowStrides.resize(PartitionCount);

    BEGIN_SCOPED_REGION

    // The buffer graph is constructed in order of how the compiler will structure the pipeline program
    // (i.e., the invocation order of its kernels.)

    std::vector<Rational> streamSetFactor(n);

    DataLayout DL(b->getModule());

    auto optimizeThreadLocalBufferLayout = [&](const unsigned firstKernel, const unsigned lastKernel) {

        unsigned count = 0;

        #ifndef NDEBUG
        std::fill_n(mapping.begin(), n, -1U);
        #endif

        const auto blockWidth = b->getBitBlockWidth();

        const auto baseRV = StrideRepetitionVector[firstKernel];

        // Calculate the memory requirements of each thread local streamset for this partition as
        // generated by a single stride of work from the root kernel.

        // NOTE: the repetition value of a root may differ from that of the other kernels

        const size_t pageSize = b->getPageSize();

        Rational minVal{pageSize};

        Rational requiredOverflowStrides{0};

        for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

            const auto strideLength = getKernel(kernel)->getStride();

            // Because data is layed out in a "strip mined" format within streamsets, the type of
            // each "chunk" will be blockwidth items in length.

            const Rational weightScale{strideLength * StrideRepetitionVector[kernel], blockWidth * baseRV };

            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                BufferNode & bn = mBufferGraph[streamSet];

                if (bn.isThreadLocal()) {
                    // determine the number of bytes this streamset requires per *root kernel* stride
                    const BufferPort & producerRate = mBufferGraph[output];
                    const Binding & outputRate = producerRate.Binding;
                    Type * const type = StreamSetBuffer::resolveType(b, outputRate.getType());
                    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
                    const auto typeSize = DL.getTypeAllocSize(type);
                    #else
                    const auto typeSize = DL.getTypeAllocSize(type).getFixedSize();
                    #endif
                    const ProcessingRate & rate = outputRate.getRate();
                    assert (rate.isFixed());
                    const auto S = (typeSize * weightScale) * rate.getLowerBound();

                    if (S < minVal) {
                        minVal = S;
                    }

                    assert (bn.UnderflowCapacity == 0);

                    if (bn.OverflowCapacity) {

                        const auto k = bn.OverflowCapacity / weightScale;
                        requiredOverflowStrides = std::max(requiredOverflowStrides, k);
                    }

                    streamSetFactor[streamSet - FirstStreamSet] = S;
                }
            }
        }




//        if (maxVal.numerator() == 0) return;

        // We now know the size of the largest unit of work resulting from a single partition root stride.
        //


        // Scale the layout s.t. the effective size of a placement is at least as large as the prior ones?
        // This would permit a easier rescaling option at runtime.


        // Every M strides requires will cause all of the streamsets to completely fill their

        const auto M = Rational{pageSize} / minVal;

        assert (M.numerator() > 0);

        #ifndef USE_DYNAMIC_SEGMENT_LENGTH_SLIDING_WINDOW
        // ROUNDUP(x,a/b) = a * CEILING((b*x)/a))/b
        const auto a = M.numerator();
        const auto b = M.denominator();
        const auto c = MaximumNumOfStrides[firstKernel] * b;
        const auto d = (c + a - 1U) / a;
        assert ((a * d) % b == 0);
        const auto V = ((a * d) / b);
        #else
        const auto & V = M;
        #endif

        for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {
            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                BufferNode & bn = mBufferGraph[streamSet];
                if (bn.isThreadLocal()) {
                    const auto i = streamSet - FirstStreamSet;
                    assert (i < n);
                    const auto j = count++;
                    assert (mapping[i] == -1U);
                    mapping[i] = j;
                    const auto S = streamSetFactor[i] * V;
                    assert (S.denominator() == 1);
                    weight[j] = S.numerator();
                    bn.RequiredCapacity = S.numerator();
                }
            }
        }

        if (count == 0) return;

        // Construct the weighted interval graph for our local streamsets

        IntervalGraph I(count); // live memory interval graph

        #ifdef PREVENT_THREAD_LOCAL_BUFFERS_FROM_SHARING_MEMORY
        for (unsigned i = 1; i < count; ++i) {
            for (unsigned j = 0; j < i; ++j) {
                add_edge(j, i, I);
            }
        }
        #else
        for (auto kernel = firstKernel, m = 0U; kernel <= lastKernel; ++kernel) {

            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const auto i = streamSet - FirstStreamSet;
                const auto j = mapping[i];
                if (j != -1U) {
                    // record how many consumers exist before the streamset memory can be reused
                    // (NOTE: the +1 is to indicate this kernel requires each output streamset
                    // to be distinct even if one or more of the outputs is not used later.)
                    assert (j == m);
                    assert (j < count);
                    remaining[j] = out_degree(streamSet, mBufferGraph) + 1U;
                    m = j + 1;
                }
            }

            // Mark any overlapping allocations in our interval graph.
            for (unsigned i = 0; i != m; ++i) {
                if (remaining[i] > 0) {
                    for (unsigned j = 0; j != i; ++j) {
                        if (remaining[j] > 0) {
                            add_edge(j, i, I);
                        }
                    }
                }
            }

            auto markFinishedStreamSets = [&](const unsigned streamSet) {
                const auto i = streamSet - FirstStreamSet;
                assert (FirstStreamSet <= streamSet && streamSet <= LastStreamSet);
                assert (i < n);
                const auto j = mapping[i];
                if (j != -1U) {
                    assert (j < m);
                    assert (remaining[j] > 0);
                    remaining[j]--;
                }
            };

            // Determine which streamsets are no longer alive
            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                markFinishedStreamSets(target(output, mBufferGraph));
            }
            for (const auto input : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                markFinishedStreamSets(source(input, mBufferGraph));
            }
        }

        #endif

        BufferLayoutOptimizer BA(count, std::move(I), std::move(weight), rng);
        BA.runGA();

        const auto requiredMemory = BA.getBestFitnessValue();

        assert (requiredMemory > 0);

        auto O = BA.getResult();

        // If we know memory was laid out in such a way that we can trust the scaling of
        // buffers to not overlap, how do we transform the



        // TODO: apart from total memory, when would one layout be better than another?
        // Can we quantify it based on the buffer graph order? Currently, we just take
        // the first one.
        const auto intervals = BA.getIntervals(O, rng);

        size_t maxEnd = 0;

        for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {
            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const auto i = streamSet - FirstStreamSet;
                const auto j = mapping[i];
                if (j == -1U) {
                    assert (mBufferGraph[streamSet].isNonThreadLocal());
                } else {
                    BufferNode & bn = mBufferGraph[streamSet];
                    const auto & interval = intervals[j];
                    bn.BufferStart = interval.lower();
                    bn.BufferEnd = interval.upper();
                    maxEnd = std::max<size_t>(maxEnd, bn.BufferEnd);
                    assert (bn.BufferEnd <= requiredMemory);
                }
            }
        }

        assert (maxEnd == requiredMemory);

        const auto partId = KernelPartitionId[firstKernel];

        PartitionRootStridesPerThreadLocalPage[partId] = M;
        PartitionOverflowStrides[partId] = requiredOverflowStrides;


        #ifdef USE_DYNAMIC_SEGMENT_LENGTH_SLIDING_WINDOW
        const auto & P = PartitionRootStridesPerThreadLocalPage[partId];

        auto K = requiredMemory;
        if (P < MaximumNumOfStrides[firstKernel]) {
            Rational X(requiredMemory * MaximumNumOfStrides[firstKernel] * P.denominator(), P.numerator());
            K = (X.numerator() + X.denominator() - 1U) / X.denominator();
        }
        RequiredThreadLocalStreamSetMemory = std::max(RequiredThreadLocalStreamSetMemory, K);
        #else
        RequiredThreadLocalStreamSetMemory = std::max(RequiredThreadLocalStreamSetMemory, requiredMemory);
        #endif

        assert (MaximumNumOfStrides[firstKernel] > 0);

        assert (K >= requiredMemory);

    };

    auto currentPartitionId = KernelPartitionId[FirstKernel];
    auto firstKernelInPartition = FirstKernel;
    for (auto kernel = FirstKernel; kernel <= LastKernel; ++kernel) {
        const auto partitionId = KernelPartitionId[kernel];
        if (partitionId != currentPartitionId) {
            optimizeThreadLocalBufferLayout(firstKernelInPartition, kernel - 1U);
            // set the first kernel for the next partition
            firstKernelInPartition = kernel;
            currentPartitionId = partitionId;
        }
    }
    optimizeThreadLocalBufferLayout(firstKernelInPartition, LastKernel);
}

}

#endif

} // end of kernel namespace
