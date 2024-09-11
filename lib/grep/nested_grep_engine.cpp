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
, mNested(1, nullptr) {


}

void NestedInternalSearchEngine::push(const re::PatternVector & patterns) {
    // If we have no patterns and this is the "root" pattern,
    // we'll still need an empty gitignore kernel even if it
    // just returns the record break stream for input.
    // Otherwise just reuse the parent kernel.

    const auto preserve = mGrepDriver.getPreservesKernels();
    mGrepDriver.setPreserveKernels(true);

    auto P = CreatePipeline(mGrepDriver, Input<const char *>{"buffer"}, Input<size_t>{"length"}, Input<MatchAccumulator &>{"accumulator"} );

    Scalar * const buffer = P.getInputScalar("buffer");
    Scalar * const length = P.getInputScalar("length");
    Scalar * const accumulator = P.getInputScalar("accumulator");



    StreamSet * const ByteStream = P.CreateStreamSet(1, 8);
    P.CreateKernelCall<MemorySourceKernel>(buffer, length, ByteStream);
    StreamSet * const mBasisBits = P.CreateStreamSet(8);
    P.CreateKernelCall<S2PKernel>(ByteStream, mBasisBits);
    StreamSet * const mBreaks = P.CreateStreamSet();

    re::CC * mBreakCC = nullptr;

    if (mGrepRecordBreak == GrepRecordBreakKind::Null) {
        mBreakCC = re::makeByte(0x0);
    } else {// if (mGrepRecordBreak == GrepRecordBreakKind::LF)
        mBreakCC = re::makeByte(0x0A);
    }


    P.CreateKernelCall<CharacterClassKernelBuilder>(std::vector<re::CC *>{mBreakCC}, mBasisBits, mBreaks);
    StreamSet * const mU8index = P.CreateStreamSet();
    P.CreateKernelCall<UTF8_index>(mBasisBits, mU8index);

    StreamSet * const mMatches = P.CreateStreamSet();

    assert (mNested.size() > 0 && mNested[0] == nullptr);
    assert (mNested.size() == 1 || mNested[1] != nullptr);

    Kernel * kernel = nullptr;

    if (LLVM_UNLIKELY(patterns.empty())) {

        if (LLVM_LIKELY(mNested.size() > 1)) {
            kernel = mNested.back();
            mNested.push_back(kernel);
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
            Kernel * const chained = E.AddKernelFamilyCall(outerKernel);
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

    P.AddKernelFamilyCall(kernel);
    if (MatchCoordinateBlocks > 0) {
        StreamSet * const MatchCoords = P.CreateStreamSet(3, sizeof(size_t) * 8);
        P.CreateKernelCall<MatchCoordinatesKernel>(mMatches, mBreaks, MatchCoords, MatchCoordinateBlocks);
        Kernel * const matchK = P.CreateKernelCall<MatchReporter>(ByteStream, MatchCoords, accumulator);
        matchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        matchK->link("finalize_match_wrapper", finalize_match_wrapper);
    } else {
        Kernel * const scanMatchK = P.CreateKernelCall<ScanMatchKernel>(mMatches, mBreaks, ByteStream, accumulator, ScanMatchBlocks);
        scanMatchK->link("accumulate_match_wrapper", accumulate_match_wrapper);
        scanMatchK->link("finalize_match_wrapper", finalize_match_wrapper);
    }

    mGrepDriver.setPreserveKernels(preserve);
    mNested.push_back(kernel);

    mMainMethod.push_back(P.compile());
    assert (mMainMethod.size() + 1 == mNested.size());

    mGrepDriver.setPreserveKernels(preserve);
}

void NestedInternalSearchEngine::pop() {
    assert (mNested.size() > 1);
    mNested.pop_back();
    assert (mMainMethod.size() > 0);
    mMainMethod.pop_back();
    assert (mMainMethod.size() + 1 == mNested.size());
}

void NestedInternalSearchEngine::doGrep(const char * search_buffer, size_t bufferLength, MatchAccumulator & accum) {
    assert (mMainMethod.size() > 0);
    auto f = mMainMethod.back(); assert (f);
    f(search_buffer, bufferLength, accum);
}

NestedInternalSearchEngine::~NestedInternalSearchEngine() { }


}
