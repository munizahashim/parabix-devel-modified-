/*
 *  Copyright (c) 2018 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include <kernel/core/idisa_target.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/pipeline/pipeline_kernel.h>
#include <kernel/pipeline/driver/cpudriver.h>
#include <kernel/pipeline/program_builder.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/deletion.h>
#include <testing/assert.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <boost/integer/common_factor_rt.hpp>
#include <llvm/IR/Constant.h>
#include <testing/assert.h>
#include <random>
#include <util/aligned_allocator.h>
#include <limits>
#include <boost/core/bit.hpp>

using namespace kernel;
using namespace llvm;
using namespace testing;
using namespace boost::integer;

enum Mode {
    Filter,
    Spread,
    Any
};

static cl::opt<unsigned> optTestLength("test-length", cl::desc("Source length of each test"), cl::init(0));

static cl::opt<unsigned> optFieldWidth("field-width", cl::desc("Field width of pattern elements"), cl::init(0));

static cl::opt<Mode>
optMode("mode", cl::init(Mode::Any), cl::desc("Set the front-end optimization level:"),
                  cl::values(clEnumValN(Mode::Filter, "filter", "no optimizations (default)"),
                             clEnumValN(Mode::Spread, "spread", "trivial optimizations"),
                             clEnumValN(Mode::Any, "any", "standard optimizations")));


static cl::opt<bool> optVerbose("v", cl::desc("Print verbose output"), cl::init(false));

template<size_t FieldWidth>
uint32_t runTestCase(CPUDriver & driver, const size_t testLength, const Mode mode, std::default_random_engine & rng) {

    using datatype_t = typename boost::uint_t<FieldWidth>::exact;

    using Allocator = AlignedAllocator<uint8_t, (512 / 8)>;

    Allocator alloc;

    datatype_t * const source = (datatype_t*)alloc.allocate(testLength * sizeof(datatype_t));

    std::uniform_int_distribution<datatype_t> dataDist(0ULL, std::numeric_limits<datatype_t>::max());
    for (size_t i = 0; i < testLength; ++i) {
        source[i] = dataDist(rng);
    }

    std::uniform_int_distribution<uint64_t> markDist(0ULL, std::numeric_limits<uint64_t>::max());

    uint64_t * markers = nullptr;

    size_t markerLength = 0;

    datatype_t * result = nullptr;

    size_t resultLength = 0;


    if (mode == Mode::Filter) {

        markerLength = (testLength + 63) / 64;

        markers = (uint64_t*)alloc.allocate(markerLength * sizeof(uint64_t));

        size_t popCount = 0;

        for (size_t i = 0; i < markerLength; ++i) {
            uint64_t v = markDist(rng);

            while (LLVM_UNLIKELY(popCount > testLength)) {
                // remove rightmost bit
                assert (v);
                v &= (v - 1);
                --popCount;
            }
            assert (popCount <= testLength);
            markers[i] = v;
            popCount += boost::core::popcount(v);
        }

        resultLength = (popCount + 511ULL) & ~511ULL;

        result = (datatype_t*)alloc.allocate(resultLength * sizeof(datatype_t));

        size_t out = 0;

        for (size_t i = 0; i < markerLength; ++i) {

            auto v = markers[i];
            while (v) {
                const auto p = boost::core::countr_zero(v);
                assert ((v & (1ULL << p)) != 0);
                v ^= (1ULL << p);
                assert (out <= resultLength);
                result[out++] = source[(i * 64) | p];
            }

        }

        assert (out == popCount);

        for (; out < resultLength; ++out) {
            result[out] = 0;
        }

        markerLength = testLength;

    } else if (mode == Mode::Spread) {

        size_t popCount = 0;

        size_t capacity = (testLength * 4  + 511) / 64;

        markers = (uint64_t*)alloc.allocate(capacity * sizeof(uint64_t));

        size_t total = 0;

        while (popCount < testLength) {

            uint64_t v = markDist(rng);
            popCount += boost::core::popcount(v);

            if (LLVM_UNLIKELY(total >= capacity)) {
                // resize
                const auto newCapacity = (capacity * 2);
                uint64_t * const newMarkers = (uint64_t*)alloc.allocate(newCapacity * sizeof(uint64_t));
                std::memcpy(newMarkers, markers, capacity * sizeof(uint64_t));
                markers = newMarkers;
                capacity = newCapacity;
            }

            while (LLVM_UNLIKELY(popCount > testLength)) {
                // remove rightmost bit
                assert (v);
                v &= (v - 1);
                --popCount;
            }
            markers[total++] = v;
        }

        assert (popCount == testLength);

        markerLength = (total * 64);
        resultLength = (markerLength + 511) & ~511;
        result = (datatype_t*)alloc.allocate(resultLength * sizeof(datatype_t));

        size_t in = 0;
        size_t out = 0;

        for (size_t i = 0; i < total; ++i) {
            uint64_t v = markers[i];
            while (v) {
                const auto p = boost::core::countr_zero(v);
                assert ((v & (1ULL << p)) != 0);
                v ^= (1ULL << p);
                const size_t nextOut = (i * 64) | p;
                assert (nextOut < resultLength);
                for (; out < nextOut; ++out) {
                    assert (out < resultLength);
                    result[out] = 0;
                }
                assert (in < testLength);
                assert (out < resultLength);
                result[out++] = source[in++];
            }
        }
        assert (in == testLength);

        for (; out < resultLength; ++out) {
            result[out] = 0;
        }
    }

    auto P = CreatePipeline(driver,
                            Input<streamset_t>{"source", 1, FieldWidth}, Input<streamset_t>{"markers", 1, 1},
                            Input<streamset_t>{"result", 1, FieldWidth},
                            Input<uint32_t &>{"output"});

    StreamSet * generated = P.CreateStreamSet(1, FieldWidth);

    if (mode == Mode::Filter) {
        P.CreateKernelCall<ByteFilterByMaskKernel>(P.getInputStreamSet(0), P.getInputStreamSet(1), generated);
    } else if (mode == Mode::Spread) {
        P.CreateKernelCall<ByteSpreadByMaskKernel>(P.getInputStreamSet(0), P.getInputStreamSet(1), generated);
    }

    Scalar * invalid = P.getInputScalar(0); assert (invalid);

    StreamSet * toMatch = P.getInputStreamSet(2); assert (toMatch);

    P.CreateKernelCall<StreamEquivalenceKernel>(StreamEquivalenceKernel::Mode::EQ, generated, toMatch, invalid);

    auto func = P.compile();

    StreamSetPtr pSource{source, testLength};
    StreamSetPtr pMarkers{markers, markerLength};
    StreamSetPtr pResult{result, resultLength};

    uint32_t retVal = 0;

    func(pSource, pMarkers, pResult, retVal);

    return retVal;
}


bool runTestCase(CPUDriver & driver, const size_t testLength, const size_t fieldWidth, const Mode mode, std::default_random_engine & rng) {

    uint32_t result = 0;

    try {
        if (fieldWidth == 8) {
            result = runTestCase<8>(driver, testLength, mode, rng);
        } else if (fieldWidth == 16) {
            result = runTestCase<16>(driver, testLength, mode, rng);
        } else if (fieldWidth == 32) {
            result = runTestCase<32>(driver, testLength, mode, rng);
        } else if (fieldWidth == 64) {
            result = runTestCase<64>(driver, testLength, mode, rng);
        } else {
            llvm::report_fatal_error("Unexpected field width");
        }
    }  catch (...) {
        result = 1;
    }

    if (result != 0 || optVerbose) {
        if (mode == Mode::Filter) {
            llvm::errs() << "FILTER";
        } else if (mode == Mode::Spread) {
            llvm::errs() << "SPREAD";
        }

        llvm::errs() << " TEST: " << testLength << 'x' << fieldWidth << " -- ";
        if (result == 0) {
            llvm::errs() << "success";
        } else {
            llvm::errs() << "failed";
        }
        llvm::errs() << '\n';
    }

    return (result != 0);
}

bool runTestCase(CPUDriver & driver, std::default_random_engine & rng) {

    size_t testLength = 0;
    if (optTestLength.getNumOccurrences()) {
        testLength = optTestLength.getValue();
    } else {
        std::uniform_int_distribution<size_t> tlDist(1000ULL, 10000ULL);
        testLength = tlDist(rng);
    }

    size_t fieldWidth = 0;
    if (optFieldWidth.getNumOccurrences()) {
        fieldWidth = optFieldWidth.getValue();
    } else {
        std::uniform_int_distribution<size_t> fwDist(3ULL, 6ULL);
        fieldWidth = 1ULL << (fwDist(rng));
    }

    Mode mode;
    if (optMode.getValue() == Mode::Any) {
        std::uniform_int_distribution<unsigned> modeDist(Mode::Filter, Mode::Spread);
        mode = (Mode)modeDist(rng);
    } else {
        mode = optMode.getValue();
    }

    return runTestCase(driver, testLength, fieldWidth, mode, rng);
}


int main(int argc, char *argv[]) {
    codegen::ParseCommandLineOptions(argc, argv, {});
    CPUDriver driver("test");
    std::random_device rd;
    std::default_random_engine rng(rd());

    bool testResult = false;
    for (unsigned rounds = 0; rounds < 100; ++rounds) {
        testResult |= runTestCase(driver, rng);
    }
    return testResult ? -1 : 0;
}
