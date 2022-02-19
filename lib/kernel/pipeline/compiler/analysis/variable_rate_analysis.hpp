#ifndef VARIABLE_RATE_ANALYSIS_HPP
#define VARIABLE_RATE_ANALYSIS_HPP

#include "pipeline_analysis.hpp"

#ifdef USE_EXPERIMENTAL_SIMULATION_BASED_VARIABLE_RATE_ANALYSIS

#include <util/slab_allocator.h>

namespace kernel {

namespace {

using SimulationAllocator = SlabAllocator<uint8_t>;

struct SimulationPort {
    uint32_t QueueLength = 0U; // use negative to indicate an unsatisfied demand?

    virtual bool consume(uint32_t & pending, random_engine & rng) = 0;

    virtual void produce(random_engine & rng) = 0;

    virtual void commit(const uint32_t pending) {
        assert (pending <= QueueLength);
        QueueLength -= pending;
    }

    SimulationPort() = default;

    void * operator new (std::size_t size, SimulationAllocator & allocator) noexcept {
        return allocator.allocate<uint8_t>(size);
    }
};

struct FixedPort final : public SimulationPort {

    FixedPort(const uint32_t amount)
    : SimulationPort()
    ,  mAmount(amount) { }

    bool consume(uint32_t & pending, random_engine & /* rng */) override {
        pending = mAmount;
        if (QueueLength < mAmount) {
            return false;
        }
        // pending = mAmount;
        return true;
    }

    void produce(random_engine & /* rng */) override {
        QueueLength += mAmount;
    }

private:
    const uint32_t mAmount;
};


struct UniformBoundedPort final : public SimulationPort {

    UniformBoundedPort(const unsigned min, const unsigned max)
    : SimulationPort()
    ,  mMin(min), mMax(max), LastVal(mMax + 1U) { }

    bool consume(uint32_t & pending, random_engine & rng) override {
        if (LastVal > mMax) {
            std::uniform_int_distribution<uint32_t> dst(mMin, mMax);
            LastVal = dst(rng);
        }
        pending = LastVal;
        if (QueueLength < mMax) {
            return false;
        }
        LastVal = mMax + 1U;
        return true;
    }

    void produce(random_engine & rng) override {
        std::uniform_int_distribution<uint32_t> dst(mMin, mMax);
        const auto k = dst(rng);
        QueueLength += k;
    }

private:
    const unsigned mMin;
    const unsigned mMax;
    uint32_t LastVal;
};

struct PartialSumGenerator {

    uint64_t readStepValue(const uint64_t start, const uint64_t end, random_engine & rng) {

        // Since PartialSum rates can have multiple ports referring to the same reference streamset, we store the
        // history of partial sum values in a circular buffer but silently drop entries after every user has read
        // the value.

        // TODO: make sure that we fully populate the array at initialization

        assert (end > start);
        assert (start >= HeadPosition);

        const auto required = (end - HeadPosition);

        if (LLVM_UNLIKELY(required >= Capacity)) {

            const auto r = (required + Capacity * 2 - 1);
            const auto newCapacity = r - (r % required);
            assert (newCapacity >= Capacity * 2);
            uint64_t * const newHistory = Allocator.allocate<uint64_t>(newCapacity);

            size_t k = 0;
            for (;;) {
                const auto l = (HeadOffset + k) % Capacity;
                newHistory[k] = History[l];
                if (l == TailOffset) break;
                ++k;
                assert (k < Capacity);
            }
            Allocator.deallocate(History);
            auto partialSum = newHistory[k];
            while (++k < newCapacity) {
                partialSum += generateStepValue(rng);
                newHistory[k] = partialSum;
            }
            HeadOffset = 0;
            TailOffset = newCapacity - 1;
            History = newHistory;
            Capacity = newCapacity;

        } else {
            assert ((HeadOffset < Capacity) && (TailOffset < Capacity));
            auto t = (TailOffset + 1) % Capacity;
            while (t != HeadOffset) {
                History[t] = History[TailOffset] + generateStepValue(rng);
                TailOffset = t;
                t = (t + 1) % Capacity;
            }
        }

        const auto i = ((start - HeadPosition) + HeadOffset) % Capacity;
        const auto j = ((end - HeadPosition) + HeadOffset) % Capacity;
        const auto a = History[i];
        const auto b = History[j];
        assert (a <= b);
        const auto c = b - a;
        assert (c <= (MaxStepSize * (end - start)));
        return c;
    }

    void updateReadPosition(const unsigned userId, const uint64_t position) {
        assert (userId < Users);
        UserReadPosition[userId] = position;
        auto min = position;
        for (unsigned i = 0; i < Users; ++i) {
            min = std::min(min, UserReadPosition[i]);
        }
     //   errs() << " -- update" << userId << " position=" << position << " -> " << min << "\n";
        assert (HeadPosition <= min);
        const auto k = (min - HeadPosition);
        HeadOffset = (HeadOffset + k) % Capacity;
        HeadPosition = min;
    }

    PartialSumGenerator(const unsigned users, const unsigned historyLength, const unsigned maxSize, SimulationAllocator & allocator)
    : MaxStepSize(maxSize)
    , Users(users)
    , HeadOffset(0)
    , HeadPosition(0)
    , TailOffset(0)
    , Capacity(std::max<unsigned>(historyLength * 2, 32))
    , History(allocator.allocate<uint64_t>(Capacity))
    , UserReadPosition(allocator.allocate<uint64_t>(users))
    , Allocator(allocator) {
        assert (historyLength > 0);
    }

    void initializeGenerator(random_engine & rng) {
        uint64_t partialSum = 0;
        History[0] = 0;
        for (unsigned i = 1; i < Capacity; ++i) {
            partialSum += generateStepValue(rng);
            History[i] = partialSum;
        }
        TailOffset = Capacity - 1;
        for (unsigned i = 0; i < Users; ++i) {
            UserReadPosition[i] = 0;
        }
    }

    void * operator new (std::size_t size, SimulationAllocator & allocator) noexcept {
        return allocator.allocate<uint8_t>(size);
    }

protected:

    virtual uint32_t generateStepValue(random_engine & rng) const = 0;

    const uint32_t MaxStepSize;

private:
    const unsigned Users;
    unsigned HeadOffset;
    uint64_t HeadPosition;
    unsigned TailOffset;
    unsigned Capacity;

    uint64_t * History;
    uint64_t * const UserReadPosition;

    SimulationAllocator & Allocator;
};

struct UniformDistributionPartialSumGenerator : public PartialSumGenerator {

    UniformDistributionPartialSumGenerator(const uint32_t users,
                                           const uint32_t maxValuePerStep,
                                           const unsigned historyLength,
                                           SimulationAllocator & allocator)
    : PartialSumGenerator(users, historyLength, maxValuePerStep, allocator) {
        assert (maxValuePerStep > 0);
    }

protected:

    uint32_t generateStepValue(random_engine & rng) const override {
        std::uniform_int_distribution<uint32_t> dst(0U, MaxStepSize);
        const auto r = dst(rng);
        assert (r <= MaxStepSize);
        return r;
    }

};

struct PartialSumPort final : public SimulationPort {

    PartialSumPort(PartialSumGenerator & generator, const unsigned userId, const unsigned step)
    : SimulationPort()
    , Generator(generator), UserId(userId), Step(step), Index(0)
    #ifndef NDEBUG
    , PreviousValue(-1U) // temporary sanity test value
    #endif
    {
        assert (step > 0);
    }

    bool consume(uint32_t & pending, random_engine & rng) override {
        const auto m = Generator.readStepValue(Index, Index + Step, rng);
        assert (m == PreviousValue || PreviousValue == -1U);
        pending = m;
        #ifndef NDEBUG
        PreviousValue = m;
        #endif
        return (QueueLength >= m);
    }

    void commit(const uint32_t pending) override {
        assert (pending <= QueueLength);
        QueueLength -= pending;
        Index += Step;
        PreviousValue = -1U;
        Generator.updateReadPosition(UserId, Index);
    }

    void produce(random_engine & rng) override {
        const auto m = Generator.readStepValue(Index, Index + Step, rng);
        QueueLength += m;
        Index += Step;
        Generator.updateReadPosition(UserId, Index);
    }

private:
    PartialSumGenerator & Generator;
    const unsigned UserId;
    const unsigned Step;
    unsigned Index;
    #ifndef NDEBUG
    unsigned PreviousValue;
    #endif
};


struct RelativePort final : public SimulationPort {

    RelativePort(const uint32_t & baseRateValue)
    : SimulationPort()
    , BaseRateValue(baseRateValue){ }

    bool consume(uint32_t & pending, random_engine & rng) override {
        const auto k = BaseRateValue;
        pending = k;
        return (QueueLength >= k);
    }

    void produce(random_engine & rng) override {
        const auto k = BaseRateValue;
        QueueLength += k;
    }

private:
    const uint32_t & BaseRateValue;
};

struct GreedyPort final : public SimulationPort {

    GreedyPort(const uint32_t min)
    : SimulationPort()
    , LowerBound(min){ }

    bool consume(uint32_t & pending, random_engine & /* rng */) override {
        pending = QueueLength;
        if (QueueLength < LowerBound || QueueLength == 0) {
            return false;
        }
        return true;
    }

    void produce(random_engine & rng) override {
        llvm_unreachable("uncaught program error? greedy rate cannot be an output rate");
    }

private:
    const uint32_t LowerBound;
};

struct BlockSizedPort final : public SimulationPort {

    BlockSizedPort(const unsigned blockSize)
    : SimulationPort()
    , BaseRate(nullptr), BlockSize(blockSize) { }

    bool consume(uint32_t & pending, random_engine & rng) override {
        const auto ql = BaseRate->QueueLength;
        const auto remainder = (ql % BlockSize);
        BaseRate->QueueLength -= remainder;
        if (BaseRate->consume(pending, rng)) {
            QueueLength = BaseRate->QueueLength;
            BaseRate->QueueLength += remainder;
            return true;
        } else {
            BaseRate->QueueLength = ql;
            return false;
        }
    }

    void produce(random_engine & rng) override {
        BaseRate->produce(rng);
        const auto ql = BaseRate->QueueLength;
        const auto remainder = (ql % BlockSize);
        QueueLength = BaseRate->QueueLength - remainder;
    }

    SimulationPort * BaseRate;
private:
    const unsigned BlockSize;
};


struct SimulationNode {
    SimulationPort ** const Input;
    SimulationPort ** const Output;
    const unsigned Inputs;
    const unsigned Outputs;


    SimulationNode(const unsigned inputs, const unsigned outputs, SimulationAllocator & allocator)
    : Input(inputs ? allocator.allocate<SimulationPort *>(inputs) : nullptr),
      Output(outputs ? allocator.allocate<SimulationPort *>(outputs) : nullptr),
      Inputs(inputs), Outputs(outputs) {

    }

    virtual unsigned fire(uint32_t * const pendingArray, random_engine & rng) = 0;

    void * operator new (std::size_t size, SimulationAllocator & allocator) noexcept {
        return allocator.allocate<uint8_t>(size);
    }
};

// we use a fork for both streamsets and relative rates
struct SimulationFork final : public SimulationNode {

    SimulationFork(const unsigned inputs, const unsigned outputs, SimulationAllocator & allocator)
    : SimulationNode(inputs, outputs, allocator) {

    }

    unsigned fire(uint32_t * const /* endingArray */, random_engine & /* rng */) override {
        assert (Inputs == 1);
        SimulationPort * const I = Input[0];
        const auto ql = I->QueueLength;
        I->QueueLength = 0;
        for (unsigned i = 0; i < Outputs; ++i) {
            Output[i]->QueueLength += ql;
        }
        return ql;
    }
};

struct SimulationActor : public SimulationNode {

    SimulationActor(const unsigned inputs, const unsigned outputs, SimulationAllocator & allocator)
    : SimulationNode(inputs, outputs, allocator)
    , SumOfStrides(0)
    , SumOfStridesSquared(0) {

    }

    unsigned fire(uint32_t * const pendingArray, random_engine & rng) override {
        uint64_t strides = 0;
        for (;;) {
            // can't remove any items until we determine we can execute a full stride
            for (unsigned i = 0; i < Inputs; ++i) {
                SimulationPort * const I = Input[i];
                pendingArray[i] = 0;
                if (!I->consume(pendingArray[i], rng)) {
                    SumOfStrides += strides;
                    SumOfStridesSquared += (strides * strides);
                    return strides;
                }
            }
            for (unsigned i = 0; i < Inputs; ++i) {
                Input[i]->commit(pendingArray[i]);
            }
            for (unsigned i = 0; i < Outputs; ++i) {
                Output[i]->produce(rng);
            }
            ++strides;
        }
    }

    uint64_t SumOfStrides;
    uint64_t SumOfStridesSquared;
};

// TODO: how can we correctly handle programs with multiple source kernels where they
// need to execute a different number of strides? We could first try running a
// demand-driven simulation but in cases where the output is rare, this could result
// in a huge segment lengths. We could scale the smallest source segment length is
// one but what if they have a large standard deviation?

struct SimulationSourceActor final : public SimulationActor {

    SimulationSourceActor(const unsigned outputs, SimulationAllocator & allocator)
    : SimulationActor(0, outputs, allocator) {

    }

    unsigned fire(uint32_t * const pendingArray, random_engine & rng) override {
        for (unsigned i = 0; i < Outputs; ++i) {
            Output[i]->produce(rng);
        }
        SumOfStrides += 1;
        SumOfStridesSquared += 1;
        return 1;
    }
};

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief computeExpectedVariableRateDataflow
 *
 * This algorithm uses simulation to try and determine the expected number of strides per segment and standard
 * deviation. It executes a data-driven simulation to converge upon a solution.
 *
 * Since we're only interested in modelling the steady state with an infinite input stream, we ignore attributes
 * such as Add, Delay, LookAhead, and ZeroExtend but do consider BlockSize.
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::estimateInterPartitionDataflow(PartitionGraph & P, random_engine & rng) {

    SimulationAllocator allocator;

    struct PortNode {
        unsigned Binding;
        unsigned StreamSet;
        PortNode() = default;
        PortNode(const unsigned binding, const unsigned streamSet)
        : Binding(binding)
        , StreamSet(streamSet) {

        }
    };

    using PortGraph = adjacency_list<vecS, vecS, bidirectionalS, PortNode>;

    struct PartitionPort {
        Rational Repetitions;
        BindingRef Binding;
        unsigned Reference;
        unsigned PartialSumStepLength;
        PartitionPort() = default;
        PartitionPort(const Rational & repetitions, const BindingRef & binding,
                      const unsigned refId, const unsigned partialSumStepLength)
        : Repetitions(repetitions), Binding(binding)
        , Reference(refId), PartialSumStepLength(partialSumStepLength) {

        }
    };

    using Graph = adjacency_list<vecS, vecS, bidirectionalS, no_property, PartitionPort>;

    // scan through the graph and build up a temporary graph first so we can hopefully lay the
    // memory out for the simulation graph in a more prefetch friendly way.

    // Review Chintana system tomorrow.

    // Cycle counter reporting oddly on colours=always?


    const auto numOfPartitions = num_vertices(P);

    Graph G(numOfPartitions);

    flat_map<unsigned, unsigned> streamSetMap;

    struct PartialSumData {
        unsigned StepSize;
        unsigned RequiredCapacity;
        unsigned GCD;
        unsigned Count;
        unsigned Index;


        PartialSumData(const unsigned stepSize)
        : StepSize(stepSize), RequiredCapacity(1), Count(0), Index(0) {

        }
    };

    flat_map<unsigned, PartialSumData> partialSumMap;

    std::vector<unsigned> ordering;

    for (unsigned partitionId = 1; partitionId < numOfPartitions; ++partitionId) {
        const PartitionData & N = P[partitionId];
        const auto n = N.Kernels.size();
        for (unsigned i = 0; i < n; ++i) {
            const auto kernelId = N.Kernels[i];
            assert (Relationships[kernelId].Type == RelationshipNode::IsKernel);
            const RelationshipNode & producerNode = Relationships[kernelId];
            const Kernel * const kernelObj = producerNode.Kernel;
            const auto strideSize = kernelObj->getStride();
            const auto reps = N.Repetitions[i] * strideSize;

            if (LLVM_UNLIKELY(isa<PopCountKernel>(kernelObj))) {
                const Binding & input = cast<PopCountKernel>(kernelObj)->getInputStreamSetBinding(0);
                const ProcessingRate & rate = input.getRate();
                assert (rate.isFixed());
                const auto stepSize = rate.getRate() * reps;
                assert (stepSize.denominator() == 1);
                const auto output = child(kernelId, Relationships);
                assert (Relationships[output].Type == RelationshipNode::IsBinding);
                const auto streamSet = child(output, Relationships);
                assert (Relationships[streamSet].Type == RelationshipNode::IsRelationship);
                assert (isa<StreamSet>(Relationships[streamSet].Relationship));
                const unsigned k = stepSize.numerator();
                partialSumMap.emplace(streamSet, PartialSumData{k});
            }

            // We cannot assume that the ports for this kernel ensure that a referred port
            // occurs prior to the referee.

            PortGraph P;

            for (const auto e : make_iterator_range(in_edges(kernelId, Relationships))) {
                const auto input = source(e, Relationships);
                if (Relationships[input].Type == RelationshipNode::IsBinding) {
                    const auto f = first_in_edge(input, Relationships);
                    assert (Relationships[f].Reason != ReasonType::Reference);
                    const auto streamSet = source(f, Relationships);
                    assert (Relationships[streamSet].Type == RelationshipNode::IsRelationship);
                    assert (isa<StreamSet>(Relationships[streamSet].Relationship));
                    const auto g = first_in_edge(streamSet, Relationships);
                    assert (Relationships[g].Reason != ReasonType::Reference);
                    const auto output = source(g, Relationships);
                    assert (Relationships[output].Type == RelationshipNode::IsBinding);
                    const auto h = first_in_edge(output, Relationships);
                    assert (Relationships[h].Reason != ReasonType::Reference);
                    const auto producer = source(h, Relationships);
                    assert (Relationships[producer].Type == RelationshipNode::IsKernel);
                    const auto c = PartitionIds.find(producer);
                    assert (c != PartitionIds.end());
                    const auto producerPartitionId = c->second;
                    assert (producerPartitionId <= partitionId);
                    if (producerPartitionId != partitionId) {
                        add_vertex(PortNode{static_cast<unsigned>(input), static_cast<unsigned>(streamSet)}, P);
                    }
                }
            }

            const auto numOfInputs = num_vertices(P);

            for (const auto e : make_iterator_range(out_edges(kernelId, Relationships))) {
                const auto output = target(e, Relationships);
                if (Relationships[output].Type == RelationshipNode::IsBinding) {
                    const auto f = first_out_edge(output, Relationships);
                    assert (Relationships[f].Reason != ReasonType::Reference);
                    const auto streamSet = target(f, Relationships);
                    assert (Relationships[streamSet].Type == RelationshipNode::IsRelationship);
                    assert (isa<StreamSet>(Relationships[streamSet].Relationship));
                    for (const auto e : make_iterator_range(out_edges(streamSet, Relationships))) {
                        const auto input = target(e, Relationships);
                        const RelationshipNode & inputNode = Relationships[input];
                        assert (inputNode.Type == RelationshipNode::IsBinding);
                        const auto f = first_out_edge(input, Relationships);
                        assert (Relationships[f].Reason != ReasonType::Reference);
                        const unsigned consumer = target(f, Relationships);
                        const auto c = PartitionIds.find(consumer);
                        assert (c != PartitionIds.end());
                        const auto consumerPartitionId = c->second;
                        assert (partitionId <= consumerPartitionId);
                        if (consumerPartitionId != partitionId) {
                            add_vertex(PortNode{static_cast<unsigned>(output), static_cast<unsigned>(streamSet)}, P);
                            break;
                        }
                    }
                }
            }

            const auto numOfPorts = num_vertices(P);

            if (numOfPorts > 0) {
                for (unsigned i = 0; i < numOfPorts; ++i) {
                    const auto & portNode = P[i];
                    const RelationshipNode & node = Relationships[portNode.Binding];
                    assert (node.Type == RelationshipNode::IsBinding);
                    const Binding & binding = node.Binding;
                    const ProcessingRate & rate = binding.getRate();
                    if (LLVM_UNLIKELY(rate.isRelative() || rate.isPartialSum())) {
                        RelationshipGraph::in_edge_iterator ei, ei_end;
                        std::tie(ei, ei_end) = in_edges(portNode.Binding, Relationships);
                        assert (in_degree(portNode.Binding, Relationships) > 1);
                        while (++ei != ei_end) {
                            if (LLVM_LIKELY(Relationships[*ei].Reason == ReasonType::Reference)) {
                                const auto ref = source(*ei, Relationships);
                                assert (Relationships[ref].Type == RelationshipNode::IsBinding);
                                assert (ref != portNode.Binding);
                                if (LLVM_LIKELY(rate.isPartialSum())) {
                                    const Binding & refBinding = Relationships[ref].Binding;
                                    const ProcessingRate & refRate = refBinding.getRate();
                                    assert (refRate.isFixed());
                                    const auto g = in_edge(ref, Relationships);
                                    assert (Relationships[g].Reason == ReasonType::ImplicitPopCount);
                                    const auto R = refRate.getRate() * reps;
                                    assert (R.denominator() == 1);
                                    const auto cap = R.numerator();
                                    const auto partialSumStreamSet = source(g, Relationships);
                                    const auto p = partialSumMap.find(partialSumStreamSet);
                                    assert (p != partialSumMap.end());
                                    PartialSumData & P = p->second;
                                    if (P.Count == 0) {
                                        P.RequiredCapacity = cap;
                                        P.GCD = cap;
                                        P.Count = 1;
                                    } else {
                                        P.RequiredCapacity = boost::lcm<unsigned>(P.RequiredCapacity, cap);
                                        P.GCD = boost::gcd<unsigned>(P.GCD, cap);
                                        P.Count++;
                                    }
                                }
                                for (unsigned j = 0; j < numOfPorts; ++j) {
                                    if (P[j].Binding == ref) {
                                        add_edge(i, j, P);
                                        break;
                                    }
                                }
                                goto found_output_ref;
                            }
                        }
                    }
found_output_ref:   continue;
                }
                assert (ordering.empty());
                lexical_ordering(P, ordering);
                assert (ordering.size() == numOfPorts);

                // A relative rate can be either relative to an input or an output rate. Only an output port can be
                // relative to an output port and output base ports can be handled easily by contracting the output
                // streamsets.

                // When a port is relative to an input rate, we need to produce or consume data at an equivalent
                // rate. If the base rate is a PartialSum, we could subsitute the Relative rate with the PartialSum
                // rate but if its a Bounded rate, we need to reuse the same random number. Because its a Bounded
                // rate, we know that the partitioning algorithm must place the producer and consumer(s) of the
                // Bounded rate into separate partitions so the base rate will exist somewhere in the simulation
                // graph. Since this is computationally cheaper than using a PartialSum look-up, we use the
                // RelativePort for PartialSums whenever possible.

                // TODO: this still assumes ports relative to a PartialSum will have the PartialSum in the graph.
                // We need to make it so that the relative port is considered a PartialSum itself.

                for (unsigned i = 0; i < numOfPorts; ++i) {
                    const auto j = ordering[i];
                    const auto & portNode = P[j];
                    const RelationshipNode & node = Relationships[portNode.Binding];
                    assert (node.Type == RelationshipNode::IsBinding);
                    const Binding & binding = node.Binding;
                    const ProcessingRate & rate = binding.getRate();

                    unsigned streamSet = 0;
                    unsigned refId = 0;
                    unsigned stepLength = 0;

                    auto getRelativeRefId = [&](const unsigned k) {
                        unsigned r = 0;
                        for (; r < i; ++r) {
                            if (ordering[r] == k) {
                                return r;
                            }
                        }
                        llvm_unreachable("cannot find relative ref port?");
                    };

                    auto getPartialSumRefId = [&](const unsigned binding) {
                        RelationshipGraph::in_edge_iterator ei, ei_end;
                        std::tie(ei, ei_end) = in_edges(portNode.Binding, Relationships);
                        assert (in_degree(portNode.Binding, Relationships) > 1);
                        while (++ei != ei_end) {
                            if (LLVM_LIKELY(Relationships[*ei].Reason == ReasonType::Reference)) {
                                const auto ref = source(*ei, Relationships);
                                assert (Relationships[ref].Type == RelationshipNode::IsBinding);
                                const Binding & refBinding = Relationships[ref].Binding;
                                const ProcessingRate & refRate = refBinding.getRate();
                                assert (refRate.getKind() == RateId::Fixed);
                                const auto r = refRate.getRate() * reps;
                                assert (r.denominator() == 1);
                                stepLength = r.numerator();
                                const auto id = parent(ref, Relationships);
                                assert (partialSumMap.count(id) != 0);
                                return id;
                            }
                        }
                        llvm_unreachable("cannot find partialsum ref port?");
                    };

                    if (j < numOfInputs) {
                        const auto itr = streamSetMap.find(portNode.StreamSet);
                        assert (itr != streamSetMap.end());
                        streamSet = itr->second;
                        if (rate.isRelative()) {
                            const auto k = parent(j, P);
                            assert (k < numOfInputs);
                            refId = getRelativeRefId(k);
                        } else if (rate.isPartialSum()) {
                            refId = getPartialSumRefId(portNode.Binding);
                        }
                        // if we already have a matching fixed rate, use that intead.
                        bool addEdge = true;
                        if (rate.isFixed()) {
                            for (const auto e : make_iterator_range(in_edges(partitionId, G))) {
                                if (LLVM_UNLIKELY(source(e, G) == streamSet)) {
                                    const PartitionPort & P = G[e];
                                    const Binding & b = P.Binding;
                                    const ProcessingRate & r = b.getRate();
                                    if ((reps * rate.getRate()) == (P.Repetitions * r.getRate())) {
                                        addEdge = false;
                                        break;
                                    }
                                }
                            }
                        }
                        if (addEdge) {
                            add_edge(streamSet, partitionId, PartitionPort(reps, binding, refId, stepLength), G);
                        }
                    } else { // is an output
                        assert (streamSetMap.find(portNode.StreamSet) == streamSetMap.end());
                        if (LLVM_UNLIKELY(rate.isRelative())) {
                            const auto k = parent(j, P);
                            if (k >= numOfInputs) {
                                const auto itr = streamSetMap.find(P[k].StreamSet);
                                assert (itr != streamSetMap.end());
                                streamSet = itr->second;
                                goto fuse_existing_streamset;
                            }
                            refId = getRelativeRefId(k);
                        } else {
                            if (rate.isFixed()) {
                                // if we already have a fixed rate output with the same outgoing rate,
                                // fuse the output streamsets to simplify the simulator
                                for (const auto e : make_iterator_range(out_edges(partitionId, G))) {
                                    const PartitionPort & P = G[e];
                                    const Binding & b = P.Binding;
                                    const ProcessingRate & r = b.getRate();
                                    if (r.isFixed()) {
                                        if ((reps * rate.getRate()) == (P.Repetitions * r.getRate())) {
                                            streamSet = target(e, G);
                                            goto fuse_existing_streamset;
                                        }
                                    }
                                }
                            } else if (rate.isPartialSum()) {
                                refId = getPartialSumRefId(portNode.Binding);
                            }
                        }
                        streamSet = add_vertex(G);
                        add_edge(partitionId, streamSet, PartitionPort(reps, binding, refId, stepLength), G);
fuse_existing_streamset:
                        streamSetMap.emplace(std::make_pair(portNode.StreamSet, streamSet));
                    }
                }
                ordering.clear();
            }
        }
    }

    assert (ordering.empty());
    const auto nodeCount = num_vertices(G);
    ordering.reserve(nodeCount);
    topological_sort(G, std::back_inserter(ordering));
    assert (ordering.size() == nodeCount);
    unsigned maxInDegree = 0;
    for (unsigned u = 0; u < nodeCount; ++u) {
        maxInDegree = std::max<unsigned>(maxInDegree, in_degree(u, G));
    }
    uint32_t * const pendingArray = allocator.allocate<uint32_t>(maxInDegree);

    SimulationNode ** const nodes = allocator.allocate<SimulationNode *>(nodeCount);

    #ifdef NDEBUG
    for (unsigned i = 0; i < nodeCount; ++i) {
        nodes[i] = nullptr;
    }
    #endif

    flat_map<unsigned, PartialSumGenerator *> partialSumGeneratorMap;

    flat_map<Graph::edge_descriptor, SimulationPort *> portMap;

#if 0

    BEGIN_SCOPED_REGION

    auto & out = errs();

    std::array<char, RateId::__Count> C;
    C[RateId::Fixed] = 'F';
    C[RateId::PopCount] = 'P';
    C[RateId::NegatedPopCount] = 'N';
    C[RateId::PartialSum] = 'S';
    C[RateId::Relative] = 'R';
    C[RateId::Bounded] = 'B';
    C[RateId::Greedy] = 'G';
    C[RateId::Unknown] = 'U';

    out << "digraph \"G\" {\n";
    for (auto v : make_iterator_range(vertices(G))) {
        out << "v" << v << " [label=\"" << v << "\"];\n";
    }

    auto write_rational = [&](const Rational & v) {
        if (LLVM_LIKELY(v.denominator() == 1)) {
            out << v.numerator();
        } else {
            out << '(' << v.numerator() << '/' << v.denominator() << ')';
        }
    };

    for (unsigned v = 0; v < numOfPartitions; ++v) {
        auto write_edge = [&](const Graph::edge_descriptor e, const unsigned idx) {
            const auto s = source(e, G);
            const auto t = target(e, G);
            out << "v" << s << " -> v" << t << " [label=\"" << idx << ": ";
            const PartitionPort & p = G[e];
            const Binding & b = p.Binding;
            const auto bs = b.hasAttribute(AttrId::BlockSize);
            if (bs) {
                out << "[BS" << b.findAttribute(AttrId::BlockSize).amount() << " : ";
            }

            const ProcessingRate & r = b.getRate();
            switch (r.getKind()) {
                case RateId::Fixed:
                case RateId::Greedy:
                case RateId::Unknown:
                    out << C[r.getKind()];
                    write_rational(r.getLowerBound() * p.Repetitions);
                    break;
                case RateId::Bounded:
                    out << C[RateId::Bounded];
                    write_rational(r.getLowerBound() * p.Repetitions);
                    out << '-';
                    write_rational(r.getUpperBound() * p.Repetitions);
                    break;
                case RateId::PartialSum:
                    BEGIN_SCOPED_REGION
                    out << C[RateId::PartialSum];
                    const auto f = partialSumMap.find(p.Reference);
                    assert (f != partialSumMap.end());
                    PartialSumData & data = f->second;
                    out << data.StepSize << "x" << data.GCD;
                    // write_rational(r.getUpperBound() * p.Repetitions);
                    END_SCOPED_REGION
                    break;
                case RateId::__Count:
                    llvm_unreachable("ProcessingRate __Count should not be used.");
            }
            if (p.Reference) {
                out << " ref=" << p.Reference;
            }
            if (bs) {
                out << "]";
            }
            out << "\"];\n";
        };
        unsigned inputIdx = 0;
        for (const auto e : make_iterator_range(in_edges(v, G))) {
            write_edge(e, inputIdx++);
        }
        unsigned outputIdx = 0;
        for (const auto e : make_iterator_range(out_edges(v, G))) {
            write_edge(e, outputIdx++);
        }
    }

    out << "}\n\n";
    out.flush();

    END_SCOPED_REGION
#endif

    unsigned simulationNodes = 0;

    std::vector<SimulationNode *> partitionNodes(numOfPartitions);

    for (unsigned i = nodeCount; i-- > 0; ) {
        const auto u = ordering[i];

        const auto inputs = in_degree(u, G);
        const auto outputs = out_degree(u, G);

        if (LLVM_UNLIKELY(inputs == 0 && outputs == 0)) {
            partitionNodes[u] = nullptr;
            continue;
        }

        SimulationNode * sn = nullptr;
        if (u < numOfPartitions) {
            if (inputs == 0) {
                sn = new (allocator) SimulationSourceActor(outputs, allocator);
            } else {
                sn = new (allocator) SimulationActor(inputs, outputs, allocator);
            }
            partitionNodes[u] = sn;
        } else {
            sn = new (allocator) SimulationFork(inputs, outputs, allocator);
        }
        assert (simulationNodes < nodeCount);
        nodes[simulationNodes] = sn;
        ++simulationNodes;

        unsigned inputIdx = 0;
        for (const auto e : make_iterator_range(in_edges(u, G))) {
            assert (inputIdx < inputs);
            const auto f = portMap.find(e);
            assert (f != portMap.end());
            sn->Input[inputIdx++] = f->second;
        }
        assert (inputIdx == inputs);

        unsigned outputIdx = 0;
        for (const auto e : make_iterator_range(out_edges(u, G))) {
            assert (outputIdx < outputs);
            const PartitionPort & p = G[e];
            const Binding & binding = p.Binding;
            // TODO: block size ports aren't correct. ignored for now.

//            BlockSizedPort * blockSize = nullptr;
//            if (LLVM_UNLIKELY(binding.hasAttribute(AttrId::BlockSize))) {
//                const Attribute & attr = binding.findAttribute(AttrId::BlockSize);
//                blockSize = new (allocator) BlockSizedPort(attr.amount());
//            }
            const ProcessingRate & rate = binding.getRate();
            SimulationPort * port = nullptr;
            switch (rate.getKind()) {
                case RateId::Fixed:
                    BEGIN_SCOPED_REGION
                    const auto r = rate.getRate() * p.Repetitions;
                    assert (r.denominator() == 1);
                    port = new (allocator) FixedPort(r.numerator());
                    END_SCOPED_REGION
                    break;
                case RateId::Bounded:
                    BEGIN_SCOPED_REGION
                    const auto l = rate.getLowerBound() * p.Repetitions;
                    const auto u = rate.getUpperBound() * p.Repetitions;
                    assert (l.denominator() == 1);
                    assert (u.denominator() == 1);
                    port = new (allocator) UniformBoundedPort(l.numerator(), u.numerator());
                    END_SCOPED_REGION
                    break;
                case RateId::PartialSum:
                    BEGIN_SCOPED_REGION
                    const auto f = partialSumMap.find(p.Reference);
                    assert (f != partialSumMap.end());
                    PartialSumData & data = f->second;
                    const auto g = partialSumGeneratorMap.find(p.Reference);
                    PartialSumGenerator * gen = nullptr;
                    if (LLVM_LIKELY(g == partialSumGeneratorMap.end())) {                        
                        assert (Relationships[p.Reference].Type == RelationshipNode::IsRelationship);
                        assert ((data.RequiredCapacity % data.GCD) == 0);
                        gen = new (allocator) UniformDistributionPartialSumGenerator(
                                    data.Count, data.StepSize * data.GCD, data.RequiredCapacity / data.GCD, allocator);
                        gen->initializeGenerator(rng);
                        partialSumGeneratorMap.emplace(p.Reference, gen);
                    } else {
                        gen = g->second;
                    }
                    assert (data.Count > 0);
                    const auto userId = data.Index++;
                    assert (userId < data.Count);                   
//                    errs() << p.Reference << ": reqCapacity=" << data.RequiredCapacity << ", stepSize=" << data.StepSize << "\n";
//                    errs() << " -- " << p.PartialSumStepLength << "\n";
                    assert (p.PartialSumStepLength <= data.RequiredCapacity);
                    assert ((p.PartialSumStepLength % data.GCD) == 0);
                    port = new (allocator) PartialSumPort(*gen, userId, p.PartialSumStepLength / data.GCD);
                    END_SCOPED_REGION
                    break;
                case kernel::ProcessingRate::Relative:
                    BEGIN_SCOPED_REGION
                    port = new (allocator) RelativePort(pendingArray[p.Reference]);
                    END_SCOPED_REGION
                    break;
                case kernel::ProcessingRate::Greedy:
                    BEGIN_SCOPED_REGION
                    assert (rate.getRate().denominator() == 1);
                    port = new (allocator) GreedyPort(rate.getRate().numerator());
                    END_SCOPED_REGION
                    break;
                default: llvm_unreachable("unhandled processing rate");
            }
//            if (blockSize) {
//                blockSize->BaseRate = port;
//                port = blockSize;
//            }
            assert (portMap.count(e) == 0);
            portMap.emplace(std::make_pair(e, port));
            sn->Output[outputIdx++] = port;
        }
        assert (outputIdx == outputs);

    }

// run the simulation

// TODO: run this for K seconds instead of a fixed number of iterations

    const uint64_t ITERATIONS = 100000;

    for (uint64_t r = 0; r < ITERATIONS; ++r) {
        for (unsigned i = 0; i < simulationNodes; ++i) {
            assert (nodes[i]);
            nodes[i]->fire(pendingArray, rng);
        }
    }


    const long double fITERATIONS = ITERATIONS;

// now calculate the expected dataflow from the simulation

    for (unsigned i = 0; i < numOfPartitions; ++i) {
        PartitionData & D = P[i];
        if (partitionNodes[i]) {
            const SimulationActor * const sn = reinterpret_cast<SimulationActor *>(partitionNodes[i]);
            const uint64_t SQS = sn->SumOfStrides;
            const uint64_t SSQ = sn->SumOfStridesSquared;
            D.ExpectedStridesPerSegment = Rational{SQS, ITERATIONS};
            const auto a = (ITERATIONS * SSQ);
            const auto b = (SQS * SQS);
            assert (a >= b);
            D.StdDevStridesPerSegment = std::sqrt((long double)(a - b)) / fITERATIONS;
        } else {
            D.ExpectedStridesPerSegment = Rational{0, 1};
            D.StdDevStridesPerSegment = 0.0;
        }


            errs() << "P_" << i << ".ExpectedStridesPerSegment="
                    << D.ExpectedStridesPerSegment.numerator() << "/"
                    << D.ExpectedStridesPerSegment.denominator() << "\n";
            errs() << "P_" << i << ".StdDevStridesPerSegment=" << D.StdDevStridesPerSegment << "\n";
    }

}

}

#endif // VARIABLE_RATE_ANALYSIS_HPP

#endif
