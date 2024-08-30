#include <grep/nested_grep_engine.h>
#include <grep/regex_passes.h>
#include <re/unicode/casing.h>
#include <re/transforms/exclude_CC.h>
#include <re/transforms/to_utf8.h>
#include <re/unicode/re_name_resolve.h>
#include <kernel/io/source_kernel.h>
#include <kernel/basis/s2p_kernel.h>
#include <re/cc/cc_kernel.h>
#include <kernel/scan/scanmatchgen.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/core/kernel_builder.h>
#include <llvm/Support/raw_ostream.h>
#include <grep/grep_toolchain.h>

#include <re/printer/re_printer.h>

using namespace kernel;
using namespace llvm;

namespace grep {

class CopyBreaksToMatches final : public MultiBlockKernel {
public:

    CopyBreaksToMatches(LLVMTypeSystemInterface & ts,
               StreamSet * const BasisBits,
               StreamSet * const u8index,
               StreamSet * const breaks,
               StreamSet * const matches)
    : MultiBlockKernel(ts
                       , "gitignoreC"
                       // inputs
                       , {{"BasisBits", BasisBits}, {"u8index", u8index}, {"breaks", breaks}}
                       // outputs
                       , {{"matches", matches, FixedRate(), Add1()}}
                       // scalars
                       , {}, {}, {}) {

    }

protected:
    void generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) override {
        PointerType * const int8PtrTy = b.getInt8PtrTy();
        Value * const processed = b.getProcessedItemCount("breaks");
        Value * const source = b.CreatePointerCast(b.getRawInputPointer("breaks", processed), int8PtrTy);
        Value * const produced = b.getProducedItemCount("matches");
        Value * const target = b.CreatePointerCast(b.getRawOutputPointer("matches", produced), int8PtrTy);
        Value * const toCopy = b.CreateMul(numOfStrides, b.getSize(getStride()));
        b.CreateMemCpy(target, source, toCopy, b.getBitBlockWidth() / 8);
    }

};

NestedInternalSearchEngine::NestedInternalSearchEngine(BaseDriver & driver)
: mGrepRecordBreak(GrepRecordBreakKind::LF)
, mCaseInsensitive(false)
, mGrepDriver(driver)
, mBreakCC(nullptr)
, mNested(1, nullptr) {

}

void NestedInternalSearchEngine::push(const re::PatternVector & patterns) {
    // If we have no patterns and this is the "root" pattern,
    // we'll still need an empty gitignore kernel even if it
    // just returns the record break stream for input.
    // Otherwise just reuse the parent kernel.

    assert (mBreakCC && mBasisBits && mU8index && mBreaks && mMatches);

    Kernel * kernel = nullptr;
    const auto preserve = mGrepDriver.getPreservesKernels();
    mGrepDriver.setPreserveKernels(true);
    if (LLVM_UNLIKELY(patterns.empty())) {
        if (LLVM_LIKELY(mNested.size() > 1)) {
            mNested.push_back(mNested.back());
            return;
        } else {
            kernel = new CopyBreaksToMatches(mGrepDriver,
                                             mBasisBits, mU8index, mBreaks,
                                             mMatches);
        }
    } else {

        auto E = CreatePipeline(mGrepDriver,
            Input<streamset_t>{"basis", mBasisBits}, Input<streamset_t>{"u8index", mU8index}, Input<streamset_t>{"breaks", mBreaks},
            Output<streamset_t>{"matches", mMatches, Add1(), ManagedBuffer()},
            InternallySynchronized());

        StreamSet * resultSoFar = mBreaks;

        Kernel * const outerKernel = mNested.back();

        std::string tmp;
        raw_string_ostream name(tmp);
        name << "gitignore";

        auto addKernelCode = [&](Kernel * const K) {
            char flags = '0';
            if (LLVM_LIKELY(K->isStateful())) {
                flags |= 1;
            }
            if (LLVM_UNLIKELY(K->hasThreadLocal())) {
                flags |= 2;
            }
            if (LLVM_UNLIKELY(K->allocatesInternalStreamSets())) {
                flags |= 4;
            }
            name << flags;
        };

        if (outerKernel) {
            Kernel * const chained = E.AddKernelCall(outerKernel, PipelineKernel::Family);
            addKernelCode(chained);
            resultSoFar = chained->getOutputStreamSet(0); assert (resultSoFar);
        }

        const auto n = patterns.size();

        for (unsigned i = 0; i != n; ++i) {
            StreamSet * MatchResults = nullptr;
            if (LLVM_UNLIKELY(i == (n - 1UL))) {
                MatchResults = E.getOutputStreamSet(0);
            } else {
                MatchResults = E.CreateStreamSet();
            }

            auto options = std::make_unique<GrepKernelOptions>();

            auto r = resolveCaseInsensitiveMode(patterns[i].second, mCaseInsensitive);
            r = regular_expression_passes(r);
            r = re::exclude_CC(r, mBreakCC);
            r = resolveAnchors(r, mBreakCC);
            r = toUTF8(r);

            options->setRE(r);
            options->setBarrier(mBreaks);
            options->addAlphabet(&cc::UTF8, mBasisBits);
            options->setResults(MatchResults);
            // check if we need to combine the current result with the new set of matches
            const bool exclude = (patterns[i].first == re::PatternKind::Exclude);
            if (i || outerKernel || exclude) {
                options->setCombiningStream(exclude ? GrepCombiningType::Exclude : GrepCombiningType::Include, resultSoFar);
            }
            options->addExternal("UTF8_index", mU8index);
            addKernelCode(E.CreateKernelFamilyCall<ICGrepKernel>(std::move(options)));
            resultSoFar = MatchResults;

        }
        assert (resultSoFar == E.getOutputStreamSet(0));

        name.flush();

        E.setUniqueName(name.str());

        kernel = E.makeKernel();
    }

    mGrepDriver.setPreserveKernels(preserve);
    mNested.push_back(kernel);
}

void NestedInternalSearchEngine::pop() {
    assert (mNested.size() > 1);
    mNested.pop_back();
    assert (mMainMethod.size() > 0);
    mMainMethod.pop_back();
    assert (mMainMethod.size() + 1 == mNested.size());
}

void NestedInternalSearchEngine::init() {

    if (mGrepRecordBreak == GrepRecordBreakKind::Null) {
        mBreakCC = re::makeByte(0x0);
    } else {// if (mGrepRecordBreak == GrepRecordBreakKind::LF)
        mBreakCC = re::makeByte(0x0A);
    }

    mBasisBits = mGrepDriver.CreateStreamSet(8);
    mU8index = mGrepDriver.CreateStreamSet();
    mBreaks = mGrepDriver.CreateStreamSet();
    mMatches = mGrepDriver.CreateStreamSet();

}

void NestedInternalSearchEngine::grepCodeGen() {

    // TODO: we should be able to avoid constructing the main pipeline if there is a way to
    // pass the information for the nested kernel address in through the "main" function.

    assert (mBreakCC && mBasisBits && mU8index && mBreaks && mMatches);

    const auto preserve = mGrepDriver.getPreservesKernels();
    mGrepDriver.setPreserveKernels(true);

    auto E = CreatePipeline(mGrepDriver, Input<const char *>{"buffer"}, Input<size_t>{"length"}, Input<MatchAccumulator &>{"accumulator"} );

    Scalar * const buffer = E.getInputScalar("buffer");
    Scalar * const length = E.getInputScalar("length");
    Scalar * const accumulator = E.getInputScalar("accumulator");

    StreamSet * const ByteStream = E.CreateStreamSet(1, 8);
    E.CreateKernelCall<MemorySourceKernel>(buffer, length, ByteStream);
    E.CreateKernelCall<S2PKernel>(ByteStream, mBasisBits);
    E.CreateKernelCall<CharacterClassKernelBuilder>(std::vector<re::CC *>{mBreakCC}, mBasisBits, mBreaks);
    E.CreateKernelCall<UTF8_index>(mBasisBits, mU8index);

    assert (mNested.size() > 1 && mNested.back());
    Kernel * const outer = mNested.back();
    E.AddKernelCall(outer, PipelineKernel::KernelBindingFlag::Family);
    if (MatchCoordinateBlocks > 0) {
        StreamSet * const MatchCoords = E.CreateStreamSet(3, sizeof(size_t) * 8);
        E.CreateKernelCall<MatchCoordinatesKernel>(mMatches, mBreaks, MatchCoords, MatchCoordinateBlocks);
        Kernel * const matchK = E.CreateKernelCall<MatchReporter>(ByteStream, MatchCoords, accumulator);
        matchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        matchK->link("finalize_match_wrapper", finalize_match_wrapper);
    } else {
        Kernel * const scanMatchK = E.CreateKernelCall<ScanMatchKernel>(mMatches, mBreaks, ByteStream, accumulator, ScanMatchBlocks);
        scanMatchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        scanMatchK->link("finalize_match_wrapper", finalize_match_wrapper);
    }

    mMainMethod.push_back(E.compile());
    assert (mMainMethod.size() + 1 == mNested.size());
    mGrepDriver.setPreserveKernels(preserve);
}

void NestedInternalSearchEngine::doGrep(const char * search_buffer, size_t bufferLength, MatchAccumulator & accum) {
    auto f = mMainMethod.back();
    f(search_buffer, bufferLength, accum);
}

NestedInternalSearchEngine::~NestedInternalSearchEngine() { }


}
