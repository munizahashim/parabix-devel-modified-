#ifndef BUFFER_SIZE_ANALYSIS_HPP
#define BUFFER_SIZE_ANALYSIS_HPP

#include "pipeline_analysis.hpp"

namespace kernel {

// TODO: nested pipeline kernels could report how much internal memory they require
// and reason about that here (and in the scheduling phase)

namespace { // anonymous namespace

constexpr static unsigned BUFFER_LAYOUT_INITIAL_CANDIDATES = 30;

constexpr static unsigned BUFFER_LAYOUT_INITIAL_CANDIDATE_ATTEMPTS = 200;

constexpr static unsigned BUFFER_SIZE_POPULATION_SIZE = 30;

constexpr static unsigned BUFFER_SIZE_GA_ROUNDS = 100;

constexpr static unsigned BUFFER_SIZE_GA_STALLS = 30;

constexpr static unsigned NON_HUGE_PAGE_SIZE = 4096;

using IntervalGraph = adjacency_list<hash_setS, vecS, undirectedS, no_property, no_property>;

using Interval = std::pair<unsigned, unsigned>;

struct BufferLayoutOptimizer final : public PermutationBasedEvolutionaryAlgorithm {

    using ColourLine = flat_set<Interval>;

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
    void repairCandidate(Candidate & /* candidate */) override { };

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief fitness
     ** ------------------------------------------------------------------------------------------------------------- */
    size_t fitness(const Candidate & candidate) override {

        assert (candidate.size() == candidateLength);

        std::fill_n(remaining.begin(), candidateLength, 0);

        GC_CL.clear();

        for (unsigned i = 0; i < candidateLength; ++i) {
            GC_ordering[candidate[i]] = i;
        }

        size_t max_colours = 0;
        for (unsigned i = 0; i < candidateLength; ++i) {
            const auto u = candidate[i];
            assert (u < candidateLength);
            const auto w = weight[u];
            assert (w > 0);
            const auto pageWidth = round_up_to(w, NON_HUGE_PAGE_SIZE);

            remaining[u] = out_degree(u, I);
            unsigned first = 0;
            for (const auto & interval : GC_CL) {
                const auto last = interval.first;
                assert (first <= last);
                if ((first + pageWidth) < last) {
                    break;
                }
                first = interval.second;
            }
            const auto last = first + w;
            assert (first <= last);
            if (last > max_colours) {
                max_colours = last;
            }

            GC_Intervals[u] = std::make_pair(first, last);

            GC_CL.emplace(first, last);

            for (const auto e : make_iterator_range(out_edges(u, I))) {
                const auto j = target(e, I);
                if (GC_ordering[j] < i) {
                    assert (remaining[j] > 0);
                    remaining[j]--;
                }
            }

            for (unsigned j = 0; j <= i; ++j) {
                const auto v = candidate[i];
                assert (v < candidateLength);
                if (remaining[v] == 0) {
                    const auto f = GC_CL.find(GC_Intervals[v]);
                    assert (f != GC_CL.end());
                    GC_CL.erase(f);
                    remaining[v] = -1;
                }
            }

        }

        return max_colours;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief getIntervals
     ** ------------------------------------------------------------------------------------------------------------- */
    std::vector<Interval> getIntervals(const OrderingDAWG & O) {
        Candidate chosen;
        chosen.reserve(candidateLength);
        Vertex u = 0;
        while (out_degree(u, O) != 0) {
            const auto e = first_out_edge(u, O);
            const auto k = O[e];
            chosen.push_back(k);
            u = target(e, O);
        }
        fitness(chosen);
        return GC_Intervals;
    }

    /** ------------------------------------------------------------------------------------------------------------- *
     * @brief constructor
     ** ------------------------------------------------------------------------------------------------------------- */
    BufferLayoutOptimizer(const unsigned numOfLocalStreamSets
                         , IntervalGraph && I
                         , std::vector<size_t> && weight
                         , std::vector<int> && remaining
                         , random_engine & rng)
    : PermutationBasedEvolutionaryAlgorithm (numOfLocalStreamSets,
                                             BUFFER_SIZE_GA_ROUNDS, BUFFER_SIZE_GA_STALLS, BUFFER_SIZE_POPULATION_SIZE, rng)
    , I(std::move(I))
    , weight(std::move(weight))
    , remaining(std::move(remaining))
    , GC_Intervals(numOfLocalStreamSets)
    , GC_ordering(numOfLocalStreamSets) {

    }


private:

    const IntervalGraph I;
    const std::vector<size_t> weight;

    std::vector<int> remaining;
    std::vector<Interval> GC_Intervals;
    std::vector<unsigned> GC_ordering;

    ColourLine GC_CL;

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

    const auto firstKernel = out_degree(PipelineInput, mBufferGraph) == 0 ? FirstKernel : PipelineInput;
    const auto lastKernel = in_degree(PipelineOutput, mBufferGraph) == 0 ? LastKernel : PipelineOutput;

    #warning TODO: can we insert a zero-extension region rather than having a secondary buffer?

    unsigned numOfLocalStreamSets = 0;
    IntervalGraph I(n);
    std::vector<size_t> weight(n, 0);
    std::vector<int> remaining(n, 0); // NOTE: signed int type is necessary here


    std::vector<unsigned> mapping(n, -1U);
    const auto alignment = b->getCacheAlignment() * 2; // Intel L2 spatial prefetcher pulls cache block pairs

    BEGIN_SCOPED_REGION

    // The buffer graph is constructed in order of how the compiler will structure the pipeline program
    // (i.e., the invocation order of its kernels.)

    DataLayout DL(b->getModule());

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
                const auto w = round_up_to(c * typeSize, alignment);
                assert (w > 0);
                assert ((w % alignment) == 0);

                const auto i = streamSet - FirstStreamSet;
                assert (i < n);

                const auto j = numOfLocalStreamSets++;
                mapping[i] = j;

                weight[j] = w;

                // record how many consumers exist before the streamset memory can be reused
                // (NOTE: the +1 is to indicate this kernel requires each output streamset
                // to be distinct even if one or more of the outputs is not used later.)
                remaining[j] = out_degree(streamSet, mBufferGraph) + 1U;
            }
        }

        // Mark any overlapping allocations in our interval graph.

        for (unsigned i = 0; i < numOfLocalStreamSets; ++i) {
            if (remaining[i] > 0) {
                for (unsigned j = 0; j < i; ++j) {
                    if (remaining[j] > 0) {
                        add_edge(i, j, I);
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
                assert (j < numOfLocalStreamSets);
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

    END_SCOPED_REGION

    if (LLVM_UNLIKELY(numOfLocalStreamSets == 0)) {
        return;
    }

    BufferLayoutOptimizer BA(numOfLocalStreamSets,
                             std::move(I), std::move(weight), std::move(remaining), rng);

    BA.runGA();

    RequiredThreadLocalStreamSetMemory = BA.getBestFitnessValue();

    auto O = BA.getResult();

    // TODO: apart from total memory, when would one layout be better than another?
    // Can we quantify it based on the buffer graph order? Currently, we just take
    // the first one.

    const auto intervals = BA.getIntervals(O);

    for (unsigned i = 0; i < n; ++i) {
        const auto j = mapping[i];
        if (j != -1U) {
            BufferNode & bn = mBufferGraph[FirstStreamSet + i];
            const auto & interval = intervals[j];
            bn.BufferStart = interval.first;
            assert ((bn.BufferStart % alignment) == 0);
        }
    }

    assert (RequiredThreadLocalStreamSetMemory > 0);

}

}

#endif // BUFFER_SIZE_ANALYSIS_HPP
