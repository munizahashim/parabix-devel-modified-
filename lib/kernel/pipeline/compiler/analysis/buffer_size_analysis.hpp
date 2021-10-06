#ifndef BUFFER_SIZE_ANALYSIS_HPP
#define BUFFER_SIZE_ANALYSIS_HPP

#include "pipeline_analysis.hpp"
#include <boost/icl/interval_set.hpp>

using boost::icl::interval_set;

namespace kernel {

// TODO: nested pipeline kernels could report how much internal memory they require
// and reason about that here (and in the scheduling phase)

namespace { // anonymous namespace

constexpr static unsigned BUFFER_LAYOUT_INITIAL_CANDIDATES = 30;

constexpr static unsigned BUFFER_LAYOUT_INITIAL_CANDIDATE_ATTEMPTS = 200;

constexpr static unsigned BUFFER_SIZE_POPULATION_SIZE = 30;

constexpr static unsigned BUFFER_SIZE_GA_ROUNDS = 1000;

constexpr static unsigned BUFFER_SIZE_GA_STALLS = 50;

// Intel spatial prefetcher pulls cache line pairs, aligned to 128 bytes.
constexpr static unsigned SPATIAL_PREFETCHER_ALIGNMENT = 128;

constexpr static unsigned NON_HUGE_PAGE_SIZE = 4096;

using IntervalGraph = adjacency_list<hash_setS, vecS, undirectedS>;

using IntervalSet = interval_set<unsigned>;

using Interval = IntervalSet::interval_type; // std::pair<unsigned, unsigned>;

struct BufferLayoutOptimizer final : public PermutationBasedEvolutionaryAlgorithm {

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief initGA
     ** ------------------------------------------------------------------------------------------------------------- */
    bool initGA(Population & initialPopulation) override {

        for (unsigned r = 0; r < BUFFER_LAYOUT_INITIAL_CANDIDATE_ATTEMPTS; ++r) {
            Candidate C(candidateLength);
            std::iota(C.begin(), C.end(), 0);
            std::shuffle(C.begin(), C.end(), rng);

            if (insertCandidate(std::move(C), initialPopulation)) {
                if (initialPopulation.size() >= BUFFER_LAYOUT_INITIAL_CANDIDATES) {
                    return false;
                }
            }

        }

        return true;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief repair
     ** ------------------------------------------------------------------------------------------------------------- */
    void repairCandidate(Candidate & /* candidate */) override { }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief fitness
     ** ------------------------------------------------------------------------------------------------------------- */
    size_t fitness(const Candidate & candidate) override {

        assert (candidate.size() == candidateLength);

        unsigned max_colours = 0;
        for (unsigned i = 0; i < candidateLength; ++i) {
            const auto a = candidate[i];
            assert (a < candidateLength);
            const auto w = weight[a];

            assert (GC_IntervalSet.empty());
            for (unsigned j = 0; j != i; ++j) {
                const auto b = candidate[j];
                if (edge(a, b, I).second) {
                    const auto & interval = GC_Intervals[b];
                    auto l = interval.lower();
                    auto r = interval.upper();
                    if (edge(a, b, C).second) {
                        l = round_down_to(l, NON_HUGE_PAGE_SIZE);
                        r = round_up_to(r, NON_HUGE_PAGE_SIZE);
                    }
                    GC_IntervalSet.insert(Interval::right_open(l, r));
                }
            }

            unsigned start = 0;
            unsigned end = w;

            if (!GC_IntervalSet.empty()) {
                for (const auto & interval : GC_IntervalSet) {
                    if (end < interval.lower()) {
                        break;
                    } else {
                        start = round_up_to(interval.upper(), SPATIAL_PREFETCHER_ALIGNMENT);
                        end = start + w;
                    }
                }
                GC_IntervalSet.clear();
            }
            GC_Intervals[a] = Interval::right_open(start, end);
            max_colours = std::max(max_colours, end);
        }

        return max_colours;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getIntervals
     ** ------------------------------------------------------------------------------------------------------------- */
    const std::vector<Interval> & getIntervals(const OrderingDAWG & O) {
        Candidate chosen;
        chosen.reserve(candidateLength);
        Vertex u = 0;
        while (out_degree(u, O) != 0) {
            const auto e = first_out_edge(u, O);
            const auto k = O[e];
            chosen.push_back(k);
            u = target(e, O);
        }
        assert (chosen.size() == candidateLength);
        fitness(chosen);
        return GC_Intervals;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief constructor
     ** ------------------------------------------------------------------------------------------------------------- */
    BufferLayoutOptimizer(const unsigned numOfLocalStreamSets
                         , IntervalGraph && I, IntervalGraph && C
                         , const std::vector<unsigned> & weight
                         , random_engine & srcRng)
    : PermutationBasedEvolutionaryAlgorithm (numOfLocalStreamSets,
                                             BUFFER_SIZE_GA_ROUNDS, BUFFER_SIZE_GA_STALLS, BUFFER_SIZE_POPULATION_SIZE, srcRng)
    , I(std::move(I))
    , C(std::move(C))
    , weight(weight)
    , GC_Intervals(numOfLocalStreamSets) {

    }


private:

    const IntervalGraph I;
    const IntervalGraph C;
    const std::vector<unsigned> & weight;
    IntervalSet GC_IntervalSet;

    std::vector<Interval> GC_Intervals;

};

} // end of anonymous namespace

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief determineBufferLayout
 *
 * Given our buffer graph, we want to identify the best placement to maximize sequential prefetching behavior with
 * the minimal total memory required. Although this assumes that the memory-aware scheduling algorithm was first
 * called, it does not actually use any data from it. The reason for this disconnection is to enable us to explore
 * the impact of static memory allocation independent of the chosen scheduling algorithm.
 *
 * Because the Intel L2 streamer prefetcher has one forward and one reverse monitor per page, a streamset will
 * only be placed in a page in which no other streamset accesses it during the same kernel invocation.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::determineBufferLayout(BuilderRef b, random_engine & rng) {

    // Construct the weighted interval graph for our local streamsets

    const auto n = LastStreamSet - FirstStreamSet + 1U;

    #warning TODO: can we insert a zero-extension region rather than having a secondary buffer?

    std::vector<unsigned> weight(n, 0);
    std::vector<int> remaining(n, 0); // NOTE: signed int type is necessary here
    std::vector<unsigned> mapping(n, -1U);

    RequiredThreadLocalStreamSetMemory = 0;

    BEGIN_SCOPED_REGION

    // The buffer graph is constructed in order of how the compiler will structure the pipeline program
    // (i.e., the invocation order of its kernels.)

    DataLayout DL(b->getModule());

    auto optimizeThreadLocalBufferLayout = [&](const unsigned firstKernel, const unsigned lastKernel) {

        unsigned count = 0;

        #ifndef NDEBUG
        std::fill_n(mapping.begin(), n, -1U);
        #endif

        for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const BufferNode & bn = mBufferGraph[streamSet];

                if (bn.Locality == BufferLocality::ThreadLocal) {
                    // determine the number of bytes this streamset requires
                    const BufferPort & producerRate = mBufferGraph[output];
                    const Binding & outputRate = producerRate.Binding;

                    Type * const type = StreamSetBuffer::resolveType(b, outputRate.getType());
                    #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
                    const auto typeSize = DL.getTypeAllocSize(type);
                    #else
                    const auto typeSize = DL.getTypeAllocSize(type).getFixedSize();
                    #endif
                    assert (typeSize > 0);
                    const auto c = bn.UnderflowCapacity + bn.RequiredCapacity + bn.OverflowCapacity;

                    assert (c > 0);
                    const auto w = c * typeSize;
                    assert (w > 0);
                    const auto i = streamSet - FirstStreamSet;
                    assert (i < n);
                    const auto j = count++;
                    assert (mapping[i] == -1U);
                    mapping[i] = j;
                    weight[j] = w * THREAD_LOCAL_BUFFER_OVERSIZE_FACTOR;
                }
            }
        }

        if (LLVM_UNLIKELY(count == 0)) {
            return;
        }

        IntervalGraph I(count); // live memory interval graph

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

        IntervalGraph C(count); // co-used interval graph

        flat_set<unsigned> coused;

        for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

            assert (coused.empty());

            for (const auto output : make_iterator_range(in_edges(kernel, mBufferGraph))) {
                const auto streamSet = source(output, mBufferGraph);
                const auto i = streamSet - FirstStreamSet;
                const auto j = mapping[i];
                if (j != -1U) {
                    assert (j < count);
                    coused.insert(j);
                }
            }

            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const auto i = streamSet - FirstStreamSet;
                const auto j = mapping[i];
                if (j != -1U) {
                    assert (j < count);
                    coused.insert(j);
                }
            }

            const auto n = coused.size();
            if (n > 1) {
                auto begin = coused.begin();
                const auto end = coused.end();
                for (auto i = begin; ++i != end; ) {
                    const auto a = *i;
                    assert (a < count);
                    for (auto j = begin; j != i; ++j) {
                        const auto b = *j;
                        assert (b < a);
                        add_edge(b, a, C);
                    }
                }
            }
            coused.clear();
        }

        BufferLayoutOptimizer BA(count, std::move(I), std::move(C), weight, rng);
        BA.runGA();

        const auto requiredMemory = BA.getBestFitnessValue();

        assert (requiredMemory > 0);

        RequiredThreadLocalStreamSetMemory = std::max(RequiredThreadLocalStreamSetMemory, requiredMemory);

        auto O = BA.getResult();

        // TODO: apart from total memory, when would one layout be better than another?
        // Can we quantify it based on the buffer graph order? Currently, we just take
        // the first one.

        const auto intervals = BA.getIntervals(O);

        for (auto kernel = firstKernel; kernel <= lastKernel; ++kernel) {

            for (const auto output : make_iterator_range(out_edges(kernel, mBufferGraph))) {
                const auto streamSet = target(output, mBufferGraph);
                const auto i = streamSet - FirstStreamSet;
                const auto j = mapping[i];
                if (j == -1U) {
                    assert (mBufferGraph[streamSet].Locality != BufferLocality::ThreadLocal);
                } else {
                    BufferNode & bn = mBufferGraph[streamSet];
                    const auto & interval = intervals[j];
                    bn.BufferStart = interval.lower();
                    bn.BufferEnd = interval.upper();
                }
            }
        }

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

} // end of kernel namespace

#endif // BUFFER_SIZE_ANALYSIS_HPP
