#include "evolutionary_algorithm.hpp"
#include <chrono>

using namespace std::chrono;

using TimePoint = time_point<system_clock, seconds>;

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runGA
 ** ------------------------------------------------------------------------------------------------------------- */
const PermutationBasedEvolutionaryAlgorithm & PermutationBasedEvolutionaryAlgorithm::runGA() {

    population.reserve(3 * maxCandidates);
    assert (population.empty());

    const auto enumeratedAll = initGA(population);

    if (LLVM_UNLIKELY(population.empty())) {
        report_fatal_error("Initial GA candidate set is empty");
    }

    if (LLVM_UNLIKELY(enumeratedAll)) {
        goto enumerated_entire_search_space;
    }

    BEGIN_SCOPED_REGION

    assert (candidateLength > 1);

    permutation_bitset bitString(candidateLength);

    BitVector uncopied(candidateLength);

    std::uniform_real_distribution<double> zeroToOneReal(0.0, 1.0);

    Population nextGeneration;
    nextGeneration.reserve(3 * maxCandidates);

    constexpr auto minFitVal = std::numeric_limits<FitnessValueType>::min();
    constexpr auto maxFitVal = std::numeric_limits<FitnessValueType>::max();
    constexpr auto worstFitnessValue = FitnessValueEvaluator::eval(minFitVal, maxFitVal) ? maxFitVal : minFitVal;

    unsigned averageStallCount = 0;
    unsigned bestStallCount = 0;

    double priorAverageFitness = worstFitnessValue;
    FitnessValueType priorBestFitness = worstFitnessValue;

    std::vector<double> weights;

    flat_set<unsigned> chosen;
    chosen.reserve(maxCandidates);

    const auto limit = system_clock::now() + seconds(maxTime);

    for (unsigned g = 0; system_clock::now() < limit; ++g) {

        const auto populationSize = population.size();
        assert (populationSize > 1);

        const auto c = maxStallGenerations - std::max(averageStallCount, bestStallCount);
        const auto d = std::min(maxTime - g, c);
        assert (d >= 1);
        const double currentMutationRate = (double)(d) / (double)(maxStallGenerations) + 0.03;
        const double currentCrossoverRate = 1.0 - currentMutationRate;

        // CROSSOVER:

        for (unsigned i = 1; i < populationSize; ++i) {
            for (unsigned j = 0; j < i; ++j) {
                if (zeroToOneReal(rng) <= currentCrossoverRate) {

                    const Candidate & A = population[i]->first;
                    const Candidate & B = population[j]->first;

                    // generate a random bit string
                    bitString.randomize(rng);

                    auto crossover = [&](const Candidate & A, const Candidate & B, const bool selector) {

                        Candidate C(candidateLength);

                        assert (candidateLength > 1);
                        assert (C.size() == candidateLength);

                        uncopied.reset();

                        #ifndef NDEBUG
                        unsigned count = 0;
                        #endif

                        for (unsigned k = 0; k < candidateLength; ++k) {
                            if (bitString.test(k) == selector) {
                                const auto v = A[k];
                                assert (v < candidateLength);
                                assert ("candidate contains duplicate values?" && !uncopied.test(v));
                                uncopied.set(v);
                                #ifndef NDEBUG
                                ++count;
                                #endif
                            } else {
                                C[k] = A[k];
                            }
                        }

                        for (unsigned k = 0U, p = -1U; k < candidateLength; ++k) {
                            const auto t = bitString.test(k);
                            if (t == selector) {
                                // V contains 1-bits for every entry we did not
                                // directly copy from A into C. We now insert them
                                // into C in the same order as they are in B.
                                #ifndef NDEBUG
                                assert (count-- > 0);
                                #endif
                                for (;;){
                                    ++p;
                                    assert (p < candidateLength);
                                    const auto v = B[p];
                                    assert (v < candidateLength);
                                    if (uncopied.test(v)) {
                                        break;
                                    }
                                }
                                C[k] = B[p];
                            }
                        }

                        assert (count == 0);

                        repairCandidate(C);
                        insertCandidate(std::move(C), population, true);
                    };

                    crossover(A, B, true);

                    crossover(B, A, false);

                }
            }

        }

        // MUTATION:

        for (unsigned i = 0; i < populationSize; ++i) {
            if (zeroToOneReal(rng) <= currentMutationRate) {

                auto & A = population[i];

                Candidate C{A->first};

                const auto a = std::uniform_int_distribution<unsigned>{0, candidateLength - 2}(rng);
                const auto b = std::uniform_int_distribution<unsigned>{a + 1, candidateLength - 1}(rng);
                std::shuffle(C.begin() + a, C.begin() + b, rng);

                repairCandidate(C);
                insertCandidate(std::move(C), population, true);
            }
        }

        const auto newPopulationSize = population.size();

        FitnessValueType sumOfGenerationalFitness = 0.0;
        auto minFitness = maxFitVal;
        auto maxFitness = minFitVal;

        for (const auto & I : population) {
            const auto fitness = I->second;
            sumOfGenerationalFitness += fitness;
            if (minFitness > fitness) {
                minFitness = fitness;
            }
            if (maxFitness < fitness) {
                maxFitness = fitness;
            }
        }

        const double averageGenerationFitness = ((double)sumOfGenerationalFitness) / ((double)newPopulationSize);

        FitnessValueType bestGenerationalFitness = maxFitness;
        if (FitnessValueEvaluator::eval(minFitness, maxFitness)) {
            bestGenerationalFitness = minFitness;
        }

        if (LLVM_UNLIKELY(newPopulationSize == populationSize)) {
            if (++averageStallCount == maxStallGenerations) {
                break;
            }
            if (++bestStallCount == maxStallGenerations) {
                break;
            }
            continue;
        }

        if (abs_subtract(averageGenerationFitness, priorAverageFitness) <= static_cast<double>(averageStallThreshold)) {
            if (++averageStallCount == maxStallGenerations) {
                break;
            }
        } else {
            averageStallCount = 0;
        }
        assert (averageStallCount < maxStallGenerations);



        if (abs_subtract(bestGenerationalFitness, priorBestFitness) <= maxStallThreshold) {
            if (++bestStallCount == maxStallGenerations) {
                break;
            }
        } else {
            bestStallCount = 0;
        }
        assert (bestStallCount < maxStallGenerations);

        // BOLTZMANN SELECTION:
        if (newPopulationSize > maxCandidates) {

            assert (nextGeneration.empty());

            if (LLVM_UNLIKELY(minFitness == maxFitness)) {



                std::shuffle(population.begin(), population.end(), rng);
                for (unsigned i = 0; i < maxCandidates; ++i) {
                    nextGeneration.emplace_back(population[i]);
                }
            } else {

                // Calculate the variance for the annealing factor

                double sumDiffOfSquares = 0.0;
                for (unsigned i = 0; i < newPopulationSize; ++i) {
                    const auto w = population[i]->second;
                    const auto d = w - averageGenerationFitness;
                    sumDiffOfSquares += d * d;
                }

//                    constexpr double beta = 4.0;

                double beta;
                if (LLVM_LIKELY(sumDiffOfSquares == 0)) {
                    beta = 4.0;
                } else {
                    beta = std::sqrt(newPopulationSize / sumDiffOfSquares);
                }

                if (weights.size() < newPopulationSize) {
                    weights.resize(newPopulationSize);
                }

                const auto weights_end = weights.begin() + newPopulationSize;

                assert (chosen.empty());

                auto sumX = 0.0;
                unsigned fittestIndividual = 0;
                const double r = beta / (double)(maxFitness - minFitness);
                for (unsigned i = 0; i < newPopulationSize; ++i) {
                    const auto itr = population[i];
                    assert (itr->first.size() == candidateLength);
                    const auto w = itr->second;
                    assert (w >= bestGenerationalFitness);
                    if (w == bestGenerationalFitness) {
                        fittestIndividual = i;
                    }
                    const double x = std::exp((double)(w - minFitness) * r);
                    const auto y = std::max(x, std::numeric_limits<double>::min());
                    sumX += y;
                    weights[i] = sumX;
                }
                // ELITISM: always keep the fittest candidate for the next generation
                chosen.insert(fittestIndividual);
                std::uniform_real_distribution<double> selector(0, sumX);
                while (chosen.size() < maxCandidates) {
                    const auto d = selector(rng);
                    assert (d < sumX);
                    const auto f = std::upper_bound(weights.begin(), weights_end, d);
                    assert (f != weights_end);
                    const unsigned j = std::distance(weights.begin(), f);
                    assert (j < newPopulationSize);
                    chosen.insert(j);
                }
                for (unsigned i : chosen) {
                    assert (i < newPopulationSize);
                    const auto itr = population[i];
                    assert (itr->first.size() == candidateLength);
                    nextGeneration.push_back(itr);
                }
                chosen.clear();
            }

            population.swap(nextGeneration);
            nextGeneration.clear();
        }

        // errs() << "averageGenerationFitness=" << averageGenerationFitness << "\n";
        // errs() << "bestGenerationalFitness=" << bestGenerationalFitness << "\n";

        priorAverageFitness = averageGenerationFitness;
        priorBestFitness = bestGenerationalFitness;
    }

    END_SCOPED_REGION

enumerated_entire_search_space:

    std::sort(population.begin(), population.end(), FitnessComparator{});

    return *this;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief insertCandidate
 ** ------------------------------------------------------------------------------------------------------------- */
bool PermutationBasedEvolutionaryAlgorithm::insertCandidate(Candidate && candidate, Population & population, const bool alwaysAddToPopulation) {
    assert (candidate.size() == candidateLength);
    #ifndef NDEBUG
    BitVector check(candidateLength);
    for (unsigned i = 0; i != candidateLength; ++i) {
        const auto v = candidate[i];
        assert ("invalid candidate #" && v < candidateLength);
        check.set(v);
    }
    assert ("duplicate candidate #" && (check.count() == candidateLength));
    #endif
    // NOTE: do not erase candidates or switch the std::map to something else without
    // verifying whether the population iterators are being invalidated.
    const auto f = candidates.emplace(std::move(candidate), 0);
    if (LLVM_LIKELY(f.second)) {
        const auto value = fitness(f.first->first);
        f.first->second = value;
    }
    assert (f.first != candidates.end());
    if (alwaysAddToPopulation || f.second) {
        population.emplace_back(f.first);
    }
    return f.second;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getResult
 ** ------------------------------------------------------------------------------------------------------------- */
OrderingDAWG PermutationBasedEvolutionaryAlgorithm::getResult() const {
    assert (std::is_sorted(population.begin(), population.end(), FitnessComparator{}));

    // Construct a trie of all possible best (lowest) orderings of this partition

    auto i = population.begin();
    const auto end = population.end();
    OrderingDAWG result(1);
    const auto bestWeight = (*i)->second;
    do {
        make_trie((*i)->first, result);
    } while ((++i != end) && (bestWeight == (*i)->second));

    return result;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief make_trie
 ** ------------------------------------------------------------------------------------------------------------- */
void PermutationBasedEvolutionaryAlgorithm::make_trie(const Candidate & C, OrderingDAWG & O) const {
    assert (num_vertices(O) > 0);
    assert (C.size() == candidateLength);
    auto u = 0;

    for (unsigned i = 0; i != candidateLength; ) {
        const auto j = C[i];
        assert (j < candidateLength);
        for (const auto e : make_iterator_range(out_edges(u, O))) {
            if (O[e] == j) {
                u = target(e, O);
                goto in_trie;
            }
        }
        BEGIN_SCOPED_REGION
        const auto v = add_vertex(O);
        add_edge(u, v, j, O);
        u = v;
        END_SCOPED_REGION
in_trie:    ++i;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief runHarmonySearch
 ** ------------------------------------------------------------------------------------------------------------- */
const BitStringBasedHarmonySearch & BitStringBasedHarmonySearch::runHarmonySearch() {

    assert (candidateLength > 1);

    population.reserve(maxCandidates);

    if (LLVM_UNLIKELY(initialize(population))) {
        goto enumerated_entire_search_space;
    }

    BEGIN_SCOPED_REGION

    std::uniform_real_distribution<double> zeroToOneReal(0.0, 1.0);

    std::uniform_int_distribution<unsigned> zeroOrOneInt(0, 1);

    FitnessValueType sumOfGenerationalFitness = 0.0;

    for (const auto & I : population) {
        const auto fitness = I->second;
        sumOfGenerationalFitness += fitness;
    }


    constexpr auto minFitVal = std::numeric_limits<FitnessValueType>::min();
    constexpr auto maxFitVal = std::numeric_limits<FitnessValueType>::max();
    constexpr auto worstFitnessValue = FitnessValueEvaluator::eval(minFitVal, maxFitVal) ? maxFitVal : minFitVal;

    unsigned averageStallCount = 0;

    double priorAverageFitness = worstFitnessValue;


    Population nextGeneration;
    nextGeneration.reserve(maxCandidates);

    Candidate newCandidate(candidateLength);

    for (unsigned round = 0; round < maxRounds; ++round) {

        const auto populationSize = population.size();

        assert (populationSize <= maxCandidates);

        auto considerationRate = CosAmplitude * std::cos(CosAngularFrequency * (double)round) + CosShift;

        std::uniform_int_distribution<unsigned> upToN(0, populationSize - 1);

        for (unsigned j = 0; j < candidateLength; ++j) {
            if (zeroToOneReal(rng) < considerationRate) {
                const auto k = upToN(rng);
                const bool v = population[k]->first.test(j);
                newCandidate.set(j, v);
            } else {
                newCandidate.set(j, zeroOrOneInt(rng));
            }
        }

        repairCandidate(newCandidate);

        const auto f = candidates.insert(std::make_pair(newCandidate, 0));
        if (LLVM_LIKELY(f.second)) {
            const auto val = fitness(f.first->first);
            if (val >= population.front()->second) {
                sumOfGenerationalFitness += val;
                f.first->second = val;
                bestResult = std::max(bestResult, val);
                std::pop_heap(population.begin(), population.end(), FitnessComparator{});
                const auto & worst = population.back();
                sumOfGenerationalFitness -= worst->second;
                population.pop_back();
                population.emplace_back(f.first);
                std::push_heap(population.begin(), population.end(), FitnessComparator{});
            }
        }

        const auto n = population.size();

        const double averageGenerationFitness = ((double)sumOfGenerationalFitness) / ((double)n);

        if (abs_subtract(averageGenerationFitness, priorAverageFitness) <= static_cast<double>(averageStallThreshold)) {
            if (++averageStallCount == maxStallGenerations) {
                break;
            }
        } else {
            averageStallCount = 0;
        }
        assert (averageStallCount < maxStallGenerations);

    }

    END_SCOPED_REGION

enumerated_entire_search_space:

    std::sort_heap(population.begin(), population.end(), FitnessComparator{});

    return *this;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief removeLeastFitCandidates
 ** ------------------------------------------------------------------------------------------------------------- */
void BitStringBasedHarmonySearch::updatePopulation(Population & nextGeneration) {
    for (const auto & I : nextGeneration) {
        assert (population.size() <= maxCandidates);
        if (population.size() == maxCandidates) {
            if (I->second >= population.front()->second) {
                std::pop_heap(population.begin(), population.end(), FitnessComparator{});
                population.pop_back();
            } else {
                // New item exceeds the weight of the heaviest candiate
                // in the population.
                continue;
            }
        }
        population.emplace_back(I);
        std::push_heap(population.begin(), population.end(), FitnessComparator{});
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief insertCandidate
 ** ------------------------------------------------------------------------------------------------------------- */
bool BitStringBasedHarmonySearch::insertCandidate(const Candidate & candidate, Population & population) {
    const auto f = candidates.insert(std::make_pair(candidate, 0));
    if (LLVM_LIKELY(f.second)) {
        const auto val = fitness(f.first->first);
        f.first->second = val;
        bestResult = std::max(bestResult, val);
        population.emplace_back(f.first);
        std::push_heap(population.begin(), population.end(), FitnessComparator{});
        return true;
    }
    return false;
}

}
