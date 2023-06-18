#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/pipeline/optimizationbranch.h>
#include <kernel/core/kernel_builder.h>
#include <boost/container/flat_map.hpp>
#include <boost/function_output_iterator.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/Timer.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Constants.h>
#include <toolchain/toolchain.h>

// TODO: the builders should detect if there is only one kernel in a pipeline / both branches are equivalent and return the single kernel. Modify addOrDeclareMainFunction.

// TODO: make a templated compile method to automatically validate and cast the main function to the correct type

using namespace llvm;
using namespace boost;
using namespace boost::container;

namespace kernel {

using Scalars = PipelineKernel::Scalars;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief compile()
 ** ------------------------------------------------------------------------------------------------------------- */
void * ProgramBuilder::compile() {
    Kernel * const kernel = makeKernel();
    if (LLVM_UNLIKELY(kernel == nullptr)) {
        report_fatal_error("Main pipeline contains no kernels nor function calls.");
    }
    void * finalObj;
    {
#if LLVM_VERSION_INTEGER >= LLVM_VERSION_CODE(4, 0, 0)
        NamedRegionTimer T(kernel->getSignature(), kernel->getName(),
                           "pipeline", "Pipeline Compilation",
                           codegen::TimeKernelsIsEnabled);
#else
        NamedRegionTimer T(kernel->getName(), "Pipeline Compilation",
                           codegen::TimeKernelsIsEnabled);
#endif
        finalObj = compileKernel(kernel);
    }
    return finalObj;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief compileKernel
 ** ------------------------------------------------------------------------------------------------------------- */
void * ProgramBuilder::compileKernel(Kernel * const kernel) {
    mDriver.addKernel(kernel);
    mDriver.generateUncachedKernels();
    return mDriver.finalizeObject(kernel);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeKernel
 ** ------------------------------------------------------------------------------------------------------------- */
Kernel * PipelineBuilder::initializeKernel(Kernel * const kernel, const unsigned flags) {
    mDriver.addKernel(kernel);
    mKernels.emplace_back(kernel, flags);
    return kernel;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializePipeline
 ** ------------------------------------------------------------------------------------------------------------- */
PipelineKernel * PipelineBuilder::initializePipeline(PipelineKernel * const pk, const unsigned flags) {
    // TODO: this isn't a very good way of doing this but if I want to allow users to always use a builder,
    // this gives me a safe workaround for the problem.
    PipelineBuilder nested(Internal{}, mDriver, pk->mInputStreamSets, pk->mOutputStreamSets, pk->mInputScalars, pk->mOutputScalars);
    std::unique_ptr<PipelineBuilder> tmp(&nested);
    pk->instantiateInternalKernels(tmp);
    tmp.release();
    initializeKernel(pk, flags);
    pk->mKernels.swap(nested.mKernels);
    pk->mCallBindings.swap(nested.mCallBindings);
    pk->mLengthAssertions.swap(nested.mLengthAssertions);
    return pk;
}

StreamSet * PipelineBuilder::CreateRepeatingBixNum(unsigned bixNumBits, pattern_t nums, bool isDynamic) {
    std::vector<std::vector<uint64_t>> templatePattern;
    templatePattern.resize(bixNumBits);
    for (unsigned i = 0; i < bixNumBits; i++) {
        templatePattern[i].resize(nums.size());
    }
    for (unsigned j = 0; j < nums.size(); j++) {
        for (unsigned i = 0; i < bixNumBits; i++) {
            templatePattern[i][j] = static_cast<uint64_t>((nums[j] >> i) & 1U);
        }
    }
    return CreateRepeatingStreamSet(1, templatePattern, isDynamic);
}

using Kernels = PipelineBuilder::Kernels;

enum class VertexType { Kernel, StreamSet, Scalar };

using AttrId = Attribute::KindId;

using TypeId = Relationship::ClassTypeId;

using Graph = adjacency_list<hash_setS, vecS, bidirectionalS, const Relationship *, unsigned>;

using Vertex = Graph::vertex_descriptor;
using Map = flat_map<const Relationship *, Vertex>;

template <typename Graph>
inline typename graph_traits<Graph>::edge_descriptor out_edge(const typename graph_traits<Graph>::vertex_descriptor u, const Graph & G) {
    assert (out_degree(u, G) == 1);
    return *out_edges(u, G).first;
}



/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addAttributesFrom
 *
 * Add any attributes from a set of kernels
 ** ------------------------------------------------------------------------------------------------------------- */
void addKernelProperties(const Kernels & kernels, Kernel * const output) {
    unsigned mustTerminate = 0;
    bool canTerminate = false;
    bool sideEffecting = false;
    bool fatalTermination = false;
    unsigned stride = 0;
    for (const auto & K : kernels) {
        Kernel * kernel = K.Object;
        for (const Attribute & attr : kernel->getAttributes()) {
            switch (attr.getKind()) {
                case AttrId::MustExplicitlyTerminate:
                    mustTerminate++;
                    break;
                case AttrId::MayFatallyTerminate:
                    fatalTermination = true;
                    break;
                case AttrId::CanTerminateEarly:
                    canTerminate = true;
                    break;
                case AttrId::SideEffecting:
                    sideEffecting = true;
                    break;
                default: continue;
            }
        }
        assert (kernel->getStride());
        if (stride) {
            stride = boost::lcm(stride, kernel->getStride());
        } else {
            stride = kernel->getStride();
        }
    }

    if (fatalTermination) {
        output->addAttribute(MayFatallyTerminate());
    }
    if (LLVM_UNLIKELY(mustTerminate == kernels.size())) {
        output->addAttribute(MustExplicitlyTerminate());
    } else if (canTerminate && !fatalTermination) {
        output->addAttribute(CanTerminateEarly());
    }
    if (sideEffecting) {
        output->addAttribute(SideEffecting());
    }
    output->setStride(stride);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeKernel
 ** ------------------------------------------------------------------------------------------------------------- */
Kernel * PipelineBuilder::makeKernel() {

    mDriver.generateUncachedKernels();

    const auto numOfKernels = mKernels.size();
    const auto numOfCalls = mCallBindings.size();

    // TODO: optimization must be able to synchronize non-InternallySynchronized kernels to
    // allow the following.

//    if (LLVM_UNLIKELY(numOfKernels <= 1 && numOfCalls == 0 && !mRequiresPipeline)) {
//        if (numOfKernels == 0) {
//            return nullptr;
//        } else {
//            return mKernels.back();
//        }
//    }

    std::string signature;
    signature.reserve(1024);
    raw_string_ostream out(signature);

    out << 'P' << mNumOfThreads << 'B' << codegen::BufferSegments;
    if (mExternallySynchronized) {
        out << 'E';
    }

    switch (codegen::PipelineCompilationMode) {
        case codegen::PipelineCompilationModeOptions::DefaultFast:
            out << 'F';
            break;
        case codegen::PipelineCompilationModeOptions::Expensive:
            out << 'X';
            break;
    }

    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::EnableAnonymousMMapedDynamicLinearBuffers))) {
        out << "+AML";
    }

    if (LLVM_UNLIKELY(codegen::AnyDebugOptionIsSet())) {
        if (DebugOptionIsSet(codegen::EnableCycleCounter)) {
            out << "+CYC";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::EnableBlockingIOCounter))) {
            out << "+BIC";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceDynamicBuffers))) {
            out << "+DB";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceStridesPerSegment))) {
            out << "+SS";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::GenerateTransferredItemCountHistogram))) {
            out << "+GTH";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::GenerateDeferredItemCountHistogram))) {
            out << "+GDH";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::DisableThreadLocalStreamSets))) {
            out << "-TL";
        }
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::EnableAnonymousMMapedDynamicLinearBuffers))) {
            out << "+AML";
        }
    }
    #ifdef ENABLE_PAPI
    if (LLVM_UNLIKELY(codegen::PapiCounterOptions != codegen::OmittedOption)) {
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::DisplayPAPICounterThreadTotalsOnly))) {
            out << "+PPO:" << codegen::PapiCounterOptions;
        } else {
            out << "+PPA:" << codegen::PapiCounterOptions;
        }
    }
    #endif

    bool hasRepeatingStreamSet = false;
    bool containsKernelFamilyCalls = false;

    if (mUniqueName.empty()) {

        constexpr auto pipelineInput = 0U;
        constexpr auto firstKernel = 1U;
        const auto firstCall = firstKernel + numOfKernels;
        const auto pipelineOutput = firstCall + numOfCalls;

        Graph G(pipelineOutput + 1);
        Map M;

        auto enumerateProducerBindings = [&](const Vertex producer, const Bindings & bindings) {
            const auto n = bindings.size();
            for (unsigned i = 0; i < n; ++i) {
                Relationship * const rel = bindings[i].getRelationship();
                const auto f = M.find(rel);
                if (LLVM_UNLIKELY(f != M.end())) {
                    SmallVector<char, 256> tmp;
                    raw_svector_ostream out(tmp);
                    const auto existingProducer = target(out_edge(f->second, G), G);
                    out << bindings[i].getName() << " is ";
                    if (LLVM_UNLIKELY(existingProducer == pipelineInput)) {
                        out << "an input to the pipeline";
                    } else {
                        const auto & K = mKernels[existingProducer - firstKernel];
                        out << "produced by " << K.Object->getName();
                    }
                    out << " and ";
                    if (LLVM_UNLIKELY(producer == pipelineOutput)) {
                        out << "an output of the pipeline";
                    } else {
                        const auto & K = mKernels[producer - firstKernel];
                        out << "produced by " << K.Object->getName();
                    }
                    out << ".";
                    report_fatal_error(out.str());
                }
                const auto bufferVertex = add_vertex(rel, G);
                M.emplace(rel, bufferVertex);
                add_edge(bufferVertex, producer, i, G); // buffer -> producer ordering
            }
        };

        enumerateProducerBindings(pipelineInput, mInputScalars);
        enumerateProducerBindings(pipelineInput, mInputStreamSets);
        for (unsigned i = 0; i < numOfKernels; ++i) {
            const auto & k = mKernels[i].Object;
            k->ensureLoaded();
            enumerateProducerBindings(firstKernel + i, k->getOutputScalarBindings());
            enumerateProducerBindings(firstKernel + i, k->getOutputStreamSetBindings());
        }

        struct RelationshipVector {
            size_t size() const {
                if (LLVM_LIKELY(mUseBindings)) {
                    return mBindings->size();
                } else {
                    return mArgs->size();
                }
            }

            Relationship * getRelationship(unsigned i) const {
                if (LLVM_LIKELY(mUseBindings)) {
                    return (*mBindings)[i].getRelationship();
                } else {
                    return (*mArgs)[i];
                }
            }

            RelationshipVector(const Bindings & bindings)
            : mUseBindings(true)
            , mBindings(&bindings) {

            }

            RelationshipVector(const Scalars & args)
            : mUseBindings(false)
            , mArgs(&args) {

            }

        private:
            const bool mUseBindings;
            union {
            const Bindings * const mBindings;
            const Scalars * const mArgs;
            };
        };

        auto enumerateConsumerBindings = [&](const Vertex consumerVertex, const RelationshipVector array) {
            const auto n = array.size();
            for (unsigned i = 0; i < n; ++i) {
                Relationship * const rel = array.getRelationship(i);
                assert ("relationship cannot be null!" && rel);
                auto f = M.find(rel);
                if (LLVM_UNLIKELY(f == M.end())) {
                    // TODO: should we record the consumers of a repeating streamset or is knowing
                    // their count sufficient?
                    if (LLVM_UNLIKELY(isa<RepeatingStreamSet>(rel) || isa<ScalarConstant>(rel))) {
                        const auto bufferVertex = add_vertex(rel, G);
                        f = M.emplace(rel, bufferVertex).first;
                    } else {
                        SmallVector<char, 256> tmp;
                        raw_svector_ostream out(tmp);
                        if (consumerVertex < firstCall) {
                            const auto & K = mKernels[consumerVertex - firstKernel];
                            const Kernel * const consumer = K.Object;
                            const Binding & input = ((rel->getClassTypeId() == TypeId::Scalar)
                                                       ? consumer->getInputScalarBinding(i)
                                                       : consumer->getInputStreamSetBinding(i));
                            out << "input " << i << " (" << input.getName() << ") of ";
                            out << "kernel " << consumer->getName();
                        } else { // TODO: function calls should retain name
                            out << "argument " << i << " of ";
                            out << "function call " << (consumerVertex - mKernels.size() + 1);
                        }
                        out << " is not a constant, produced by a kernel or an input to the pipeline";
                        report_fatal_error(out.str());
                    }
                }
                const auto bufferVertex = f->second;
                assert (bufferVertex < num_vertices(G));
                add_edge(consumerVertex, bufferVertex, i, G); // consumer -> buffer ordering
            }
        };

        for (unsigned i = 0; i < numOfKernels; ++i) {
            Kernel * const k = mKernels[i].Object;
            enumerateConsumerBindings(firstKernel + i, k->getInputScalarBindings());
            enumerateConsumerBindings(firstKernel + i, k->getInputStreamSetBindings());
        }
        for (unsigned i = 0; i < numOfCalls; ++i) {
            enumerateConsumerBindings(firstCall + i, mCallBindings[i].Args);
        }
        enumerateConsumerBindings(pipelineOutput, mOutputScalars);
        enumerateConsumerBindings(pipelineOutput, mOutputStreamSets);

        for (unsigned i = 0; i < numOfKernels; ++i) {
            out << "_K";
            const auto & K = mKernels[i];
            if (K.isFamilyCall()) {
                out << K.Object->getFamilyName();
                containsKernelFamilyCalls = true;
            } else {
                out << K.Object->getName();
                if (K.Object->containsKernelFamilyCalls()) {
                    containsKernelFamilyCalls = true;
                }
            }
        }
        for (unsigned i = 0; i < numOfCalls; ++i) {
            out << "_C" << mCallBindings[i].Name;
        }
        out << '@';

        const auto firstRelationship = pipelineOutput + 1;
        const auto lastRelationship = num_vertices(G);

        for (auto i = firstRelationship; i != lastRelationship; ++i) {
            const Relationship * const r = G[i]; assert (r);
            if (LLVM_UNLIKELY(isa<RepeatingStreamSet>(r))) {
                const RepeatingStreamSet * rs = cast<RepeatingStreamSet>(r);
                const auto numElements = rs->getNumElements();
                const auto fieldWidth = rs->getNumElements();
                out << 'R' << numElements << 'x' << fieldWidth;
                if (rs->isUnaligned()) {
                    out << 'U';
                }
                if (rs->isDynamic()) {
                    hasRepeatingStreamSet = true;
                } else {
                    const auto width = fieldWidth / 4UL;
                    SmallVector<char, 16> tmp(width + 1);
                    for (unsigned i = 0;;) {
                        assert (i < numElements);
                        const auto & vec = rs->getPattern(i);
                        out << ':';
                        // write to hex code
                        for (auto v : vec) {
                            unsigned j = 0;
                            while (v) {
                                const auto c = (v & 15);
                                v >>= 4;
                                if (c < 10) {
                                    tmp[j] = '0' + c;
                                } else {
                                    tmp[j] = 'A' + (c - 10U);
                                }
                                ++j;
                            }
                            for (auto k = j; k < width; ++k) {
                                out << '0';
                            }
                            while (j) {
                                out << tmp[--j];
                            }
                        }
                        if ((++i) == numElements) {
                            break;
                        }
                    }
                }
            } else if (LLVM_UNLIKELY(isa<TruncatedStreamSet>(r))) {
                auto f = M.find(cast<TruncatedStreamSet>(r)->getData());
                if (LLVM_UNLIKELY(f == M.end())) {
                    report_fatal_error("Truncated streamset data has no producer");
                }
                out << 'T' << f->second;
            } else {
                char typeCode;
                switch (r->getClassTypeId()) {
                    case TypeId::StreamSet:
                        typeCode = 'S';
                        break;
                    case TypeId::ScalarConstant:
                        typeCode = 'C';
                        break;
                    case TypeId::Scalar:
                        typeCode = 'V';
                        break;
                    default: llvm_unreachable("unknown relationship type");
                }
                out << typeCode;
            }

            if (LLVM_LIKELY(out_degree(i, G) != 0)) {
                const auto e = out_edge(i, G);
                const auto j = target(e, G);
                out << j << '.' << G[e];
            }

            for (const auto e : make_iterator_range(in_edges(i, G))) {
                const auto k = source(e, G);
                out << '_' << k << '.' << G[e];
            }
        }
    } else { // the programmer provided a unique name
        out << mUniqueName;
        for (unsigned i = 0; i < numOfKernels; ++i) {
            const auto & K = mKernels[i];
            if (K.isFamilyCall()) {
                containsKernelFamilyCalls = true;
            }
            Kernel * const k = K.Object;
            k->ensureLoaded();
            if (k->generatesDynamicRepeatingStreamSets()) {
                hasRepeatingStreamSet = true;
            }
            if (k->containsKernelFamilyCalls()) {
                containsKernelFamilyCalls = true;
            }
            for (unsigned i = 0; i < k->getNumOfStreamInputs(); ++i) {
                const StreamSet * const in = k->getInputStreamSet(i);
                if (LLVM_UNLIKELY(isa<RepeatingStreamSet>(in))) {
                    if (cast<RepeatingStreamSet>(in)->isDynamic()) {
                        hasRepeatingStreamSet = true;
                        break;
                    }
                }
            }
        }
    }

    out.flush();

    PipelineKernel * const pipeline =
        new PipelineKernel(mDriver.getBuilder(), std::move(signature),
                           mNumOfThreads,
                           containsKernelFamilyCalls, hasRepeatingStreamSet,
                           std::move(mKernels), std::move(mCallBindings),
                           std::move(mInputStreamSets), std::move(mOutputStreamSets),
                           std::move(mInputScalars), std::move(mOutputScalars),
                           std::move(mLengthAssertions));
    if (mExternallySynchronized) {
        pipeline->addAttribute(InternallySynchronized());
    }

    addKernelProperties(pipeline->getKernels(), pipeline);

    return pipeline;
}


using AttributeCombineSet = flat_map<AttrId, unsigned>;

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief combineAttributes
 ** ------------------------------------------------------------------------------------------------------------- */
void combineAttributes(const Binding & S, AttributeCombineSet & C) {
    for (const Attribute & s : S) {
        auto f = C.find(s.getKind());
        if (LLVM_LIKELY(f == C.end())) {
            C.emplace(s.getKind(), s.amount());
        } else {
            // TODO: we'll need some form of attribute combination function
            f->second = std::max(f->second, s.amount());
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief makeKernel
 ** ------------------------------------------------------------------------------------------------------------- */
Kernel * OptimizationBranchBuilder::makeKernel() {

    // TODO: the rates of the optimization branches should be determined by
    // the actual kernels within the branches.

    mNonZeroBranch->setExternallySynchronized(true);
    Kernel * const nonZero = mNonZeroBranch->makeKernel();
    if (nonZero) mDriver.addKernel(nonZero);

    mAllZeroBranch->setExternallySynchronized(true);
    Kernel * const allZero = mAllZeroBranch->makeKernel();
    if (allZero) mDriver.addKernel(allZero);

    std::string name;
    raw_string_ostream out(name);

    out << "OB:";
    if (LLVM_LIKELY(isa<StreamSet>(mCondition))) {
        StreamSet * const streamSet = cast<StreamSet>(mCondition);
        out << streamSet->getFieldWidth() << "x" << streamSet->getNumElements();
    } else {
        Scalar * const scalar = cast<Scalar>(mCondition);
        out << scalar->getFieldWidth();
    }
    out << ";Z=\"";
    if (nonZero) out << nonZero->getFamilyName();
    out << "\";N=\"";
    if (allZero) out << allZero->getFamilyName();
    out << "\"";
    out.flush();

    OptimizationBranch * const br =
            new OptimizationBranch(mDriver.getBuilder(), std::move(name),
                                   mCondition, nonZero, allZero,
                                   std::move(mInputStreamSets), std::move(mOutputStreamSets),
                                   std::move(mInputScalars), std::move(mOutputScalars));
    addKernelProperties({{nonZero, PipelineKernel::Family}, {allZero, PipelineKernel::Family}}, br);
    br->addAttribute(InternallySynchronized());
    return br;
}

Scalar * PipelineBuilder::getInputScalar(const StringRef name) {
    for (Binding & input : mInputScalars) {
        if (name.equals(input.getName())) {
            if (input.getRelationship() == nullptr) {
                input.setRelationship(mDriver.CreateScalar(input.getType()));
            }
            return cast<Scalar>(input.getRelationship());
        }
    }
    report_fatal_error("no scalar named " + name);
}

void PipelineBuilder::setInputScalar(const StringRef name, Scalar * value) {
    for (Binding & input : mInputScalars) {
        if (name.equals(input.getName())) {
            input.setRelationship(value);
            return;
        }
    }
    report_fatal_error("no scalar named " + name);
}

Scalar * PipelineBuilder::getOutputScalar(const StringRef name) {
    for (Binding & output : mOutputScalars) {
        if (name.equals(output.getName())) {
            if (output.getRelationship() == nullptr) {
                output.setRelationship(mDriver.CreateScalar(output.getType()));
            }
            return cast<Scalar>(output.getRelationship());
        }
    }
    report_fatal_error("no scalar named " + name);
}

void PipelineBuilder::setOutputScalar(const StringRef name, Scalar * value) {
    for (Binding & output : mOutputScalars) {
        if (name.equals(output.getName())) {
            output.setRelationship(value);
            return;
        }
    }
    report_fatal_error("no scalar named " + name);
}


PipelineBuilder::PipelineBuilder(BaseDriver & driver,
    Bindings && stream_inputs, Bindings && stream_outputs,
    Bindings && scalar_inputs, Bindings && scalar_outputs,
    const unsigned numOfThreads)
: mDriver(driver)
, mNumOfThreads(numOfThreads)
, mInputStreamSets(stream_inputs)
, mOutputStreamSets(stream_outputs)
, mInputScalars(scalar_inputs)
, mOutputScalars(scalar_outputs) {

    for (unsigned i = 0; i < mInputScalars.size(); i++) {
        Binding & input = mInputScalars[i];
        if (input.getRelationship() == nullptr) {
            input.setRelationship(driver.CreateScalar(input.getType()));
        }
    }
    for (unsigned i = 0; i < mInputStreamSets.size(); i++) {
        Binding & input = mInputStreamSets[i];
        if (LLVM_UNLIKELY(input.getRelationship() == nullptr)) {
            report_fatal_error(input.getName() + " must be set upon construction");
        }
    }
    for (unsigned i = 0; i < mOutputStreamSets.size(); i++) {
        Binding & output = mOutputStreamSets[i];
        if (LLVM_UNLIKELY(output.getRelationship() == nullptr)) {
            report_fatal_error(output.getName() + " must be set upon construction");
        }
    }
    for (unsigned i = 0; i < mOutputScalars.size(); i++) {
        Binding & output = mOutputScalars[i];
        if (output.getRelationship() == nullptr) {
            output.setRelationship(driver.CreateScalar(output.getType()));
        }
    }

}

PipelineBuilder::PipelineBuilder(Internal, BaseDriver & driver,
    Bindings stream_inputs, Bindings stream_outputs,
    Bindings scalar_inputs, Bindings scalar_outputs)
: PipelineBuilder(driver,
                  std::move(stream_inputs), std::move(stream_outputs),
                  std::move(scalar_inputs), std::move(scalar_outputs),
                  1) {

}

ProgramBuilder::ProgramBuilder(
    BaseDriver & driver,
    Bindings && stream_inputs, Bindings && stream_outputs,
    Bindings && scalar_inputs, Bindings && scalar_outputs)
: PipelineBuilder(
      driver,
      std::move(stream_inputs), std::move(stream_outputs),
      std::move(scalar_inputs), std::move(scalar_outputs),
      codegen::SegmentThreads) {

}

Bindings replaceManagedWithSharedManagedBuffers(const Bindings & bindings) {
    Bindings replaced;
    replaced.reserve(bindings.size());
    for (const Binding & binding : bindings) {        
        Binding newBinding(binding.getName(), binding.getRelationship(), binding.getRate());
        for (const Attribute & attr : binding.getAttributes()) {
            if (attr.getKind() == Attribute::KindId::ManagedBuffer) {
                newBinding.push_back(SharedManagedBuffer());
            } else {
                newBinding.push_back(attr);
            }
        }
        replaced.emplace_back(std::move(newBinding));
    }
    return replaced;
}

OptimizationBranchBuilder::OptimizationBranchBuilder(
      BaseDriver & driver,
      Relationship * const condition,
      Bindings && stream_inputs, Bindings && stream_outputs,
      Bindings && scalar_inputs, Bindings && scalar_outputs)
: PipelineBuilder(PipelineBuilder::Internal{},
      driver,
      std::move(stream_inputs), std::move(stream_outputs),
      std::move(scalar_inputs), std::move(scalar_outputs))
, mCondition(condition)
, mNonZeroBranch(std::unique_ptr<PipelineBuilder>(
                    new PipelineBuilder(
                    PipelineBuilder::Internal{}, mDriver,
                    mInputStreamSets, replaceManagedWithSharedManagedBuffers(mOutputStreamSets),
                    mInputScalars, mOutputScalars)))
, mAllZeroBranch(std::unique_ptr<PipelineBuilder>(
                    new PipelineBuilder(
                    PipelineBuilder::Internal{}, mDriver,
                    mInputStreamSets, replaceManagedWithSharedManagedBuffers(mOutputStreamSets),
                    mInputScalars, mOutputScalars))) {

}

OptimizationBranchBuilder::~OptimizationBranchBuilder() {

}

}
