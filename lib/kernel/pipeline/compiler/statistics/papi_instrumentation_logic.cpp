#ifdef ENABLE_PAPI

#include "../pipeline_compiler.hpp"
#include <papi.h>
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>
#include <codegen/TypeBuilder.h>

// TODO: merge cycle counter with papi?

namespace kernel {

using papi_counter_t = unsigned long_long;

namespace {
constexpr auto add_perf_events_failure_message =
   "Check papi_avail for available options or enter sysctl -w kernel.perf_event_paranoid=0\n"
   "to reenable cpu event tracing at the kernel level.";
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getPAPIEventCounterType
 ** ------------------------------------------------------------------------------------------------------------- */
ArrayType * PipelineCompiler::getPAPIEventCounterType(KernelBuilder & b) const {
    IntegerType * const papiCounterTy = TypeBuilder<papi_counter_t, false>::get(b.getContext());
//    const auto pw = papiCounterTy->getIntegerBitWidth();
//    const auto bw = b.getBitBlockWidth();
//    ArrayType * papiEventCounterTotalTy = nullptr;
//    if (NumOfPAPIEvents == 1 || (pw > bw)) {
//        return ArrayType::get(papiCounterTy, NumOfPAPIEvents);
//    } else {
//        const auto k = bw / pw; assert (k > 0);
//        const auto n = (NumOfPAPIEvents + k - 1) / k;
//        return ArrayType::get(kernel::FixedVectorType::get(papiCounterTy, k), n);
//    }


    return ArrayType::get(papiCounterTy, NumOfPAPIEvents);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addPAPIEventCounterPipelineProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addPAPIEventCounterPipelineProperties(KernelBuilder & b) {
    if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {

        // TODO: make a better method than this for accumulating the final thread local counts.
        // We can't share a global scalar that isn't guarded by synchronization but could pass
        // the state of thread 0 to all other threads during other users.

        // NOTE: it might actually be beneficial to guard the exit point for the pipeline but
        // this should be tested in the context of nested pipelines before committing to such
        // a design. The following choice to use a global accumulator in the thread local
        // destructor can easily be converted to doing this.

        auto accumRule = ThreadLocalScalarAccumulationRule::Sum;
        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::DisplayPAPICounterThreadTotalsOnly))) {
            accumRule = ThreadLocalScalarAccumulationRule::DoNothing;
        }

        ArrayType * const papiDataTy = getPAPIEventCounterType(b);
        mTarget->addThreadLocalScalar(papiDataTy, STATISTICS_PAPI_TOTAL_COUNT_ARRAY, PipelineOutput, accumRule);

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addPAPIEventCounterKernelProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addPAPIEventCounterKernelProperties(KernelBuilder & b, const unsigned kernel, const bool /* isRoot */) {
    if (LLVM_UNLIKELY(NumOfPAPIEvents)) {

        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::DisplayPAPICounterThreadTotalsOnly))) {
            return;
        }

        ArrayType * const papiDataTy = ArrayType::get(getPAPIEventCounterType(b), NUM_OF_PAPI_KERNEL_COUNTERS);
        const auto prefix = makeKernelName(kernel) + STATISTICS_PAPI_COUNT_ARRAY_SUFFIX;
        const auto groupId = getCacheLineGroupId(kernel);
        if (mIsStatelessKernel.test(kernel)) {
            mTarget->addThreadLocalScalar(papiDataTy, prefix, groupId, ThreadLocalScalarAccumulationRule::Sum);
        } else {
            mTarget->addInternalScalar(papiDataTy, prefix, groupId);
        }

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getPAPIEventSet
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::getPAPIEventSet(KernelBuilder & b) {
    if (LLVM_UNLIKELY(NumOfPAPIEvents)) {
        PAPIEventSetVal = b.getScalarField(STATISTICS_PAPI_EVENT_SET_CODE);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief registerPAPIThread
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::registerPAPIThread(KernelBuilder & /* b */) const {
    // PAPI documentation indicates this and unregister thread are not necessary.
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief createPAPIMeasurementArrays
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::createPAPIMeasurementArrays(KernelBuilder & b) {
    if (LLVM_UNLIKELY(NumOfPAPIEvents)) {
        // PAPI_start starts counting all of the hardware events contained in the previously defined EventSet.
        // All counters are implicitly set to zero before counting.
        ArrayType * const papiCounterArrayTy = getPAPIEventCounterType(b);
        Constant * nil = Constant::getNullValue(papiCounterArrayTy);
        for (unsigned i = 0; i < NUM_OF_PAPI_COUNTERS; ++i) {
            Value * ptr = b.CreateAllocaAtEntryPoint(papiCounterArrayTy);
            b.CreateStore(nil, ptr);
            PAPIEventCounterArray[i] = ptr;
        }
        PAPITempMeasurementArray = b.CreateAllocaAtEntryPoint(papiCounterArrayTy);
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief readPAPIEventsIntoArray
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::readPAPIMeasurement(KernelBuilder & b, Value * const measurementArray) const {
    assert (NumOfPAPIEvents > 0);
    assert (measurementArray);

    Module * const m = b.getModule();
    Function * const PAPIReadFn = m->getFunction("PAPI_read"); assert (PAPIReadFn);
    FixedArray<Value *, 2> args;
    args[0] = PAPIEventSetVal; assert (PAPIEventSetVal);
    PointerType * const papiCounterPtrTy = TypeBuilder<papi_counter_t, false>::get(b.getContext())->getPointerTo();
    args[1] = b.CreatePointerCast(measurementArray, papiCounterPtrTy); assert (measurementArray);
    // TODO: should probably check the error code here but if we do get an error,
    // what can we avoid contaminating the results but also inform the user something
    // went wrong?

    Value * const retVal = b.CreateCall(PAPIReadFn->getFunctionType(), PAPIReadFn, args);
    if (LLVM_UNLIKELY(CheckAssertions)) {
        Value * valid = b.CreateICmpEQ(retVal, ConstantInt::get(retVal->getType(), PAPI_OK));
        b.CreateAssert (valid, "PAPI_read failed");
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief startPAPIMeasurement
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::startPAPIMeasurement(KernelBuilder & b, const PAPIKernelCounter measurementType) const {
    if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {
        readPAPIMeasurement(b, PAPIEventCounterArray[measurementType]);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief startPAPIMeasurement
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::startPAPIMeasurement(KernelBuilder & b, const std::initializer_list<PAPIKernelCounter> measurementType) const {
    if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {
        auto counter = measurementType.begin();
        const auto first = *counter;
        startPAPIMeasurement(b, first);
        ArrayType * const papiCounterArrayTy = getPAPIEventCounterType(b);
        while (++counter != measurementType.end()) {
            for (unsigned i = 0; i < NumOfPAPIEvents; ++i) {
                b.CreateStore(b.CreateLoad(papiCounterArrayTy, PAPIEventCounterArray[first]), PAPIEventCounterArray[*counter]);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief accumPAPIMeasurementWithoutReset
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::accumPAPIMeasurementWithoutReset(KernelBuilder & b, const size_t kernelId, const PAPIKernelCounter measurementType) const {
    if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {

        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::DisplayPAPICounterThreadTotalsOnly))) {
            return;
        }

        readPAPIMeasurement(b, PAPITempMeasurementArray);

        const auto prefix = makeKernelName(kernelId);
        Value * eventCounterSumArray; Type * eventCounterSumty;

        std::tie(eventCounterSumArray, eventCounterSumty) = b.getScalarFieldPtr(prefix + STATISTICS_PAPI_COUNT_ARRAY_SUFFIX);
        Type * const counterArrayTy = eventCounterSumty->getArrayElementType();
        assert (counterArrayTy->isArrayTy());
        Type * const counterTy = counterArrayTy->getArrayElementType();

        Constant * i32_ZERO = b.getInt32(0);

        FixedArray<Value *, 2> from;
        from[0] = i32_ZERO;

        FixedArray<Value *, 3> update;
        update[0] = i32_ZERO;
        update[1] = b.getInt32((unsigned)measurementType);

        for (unsigned i = 0; i < NumOfPAPIEvents; ++i) {
            from[1] = b.getInt32(i);
            Value * const beforeVal = b.CreateLoad(counterTy, b.CreateGEP(counterArrayTy, PAPIEventCounterArray[measurementType], from));
            Value * const afterVal = b.CreateLoad(counterTy, b.CreateGEP(counterArrayTy, PAPITempMeasurementArray, from));
            Value * const diff = b.CreateSaturatingSub(afterVal, beforeVal);
            update[2] = from[1];
            Value * const ptr = b.CreateGEP(eventCounterSumty, eventCounterSumArray, update);
            Value * const curr = b.CreateLoad(counterTy, ptr);
            assert (curr->getType() == diff->getType());
            Value * const updatedVal = b.CreateAdd(curr, diff);
            b.CreateStore(updatedVal, ptr);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief accumPAPIMeasurementWithoutReset
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::accumPAPIMeasurementWithoutReset(KernelBuilder & b, const size_t kernelId, Value * const cond, const PAPIKernelCounter ifTrue, const PAPIKernelCounter ifFalse) const {
    if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {

        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::DisplayPAPICounterThreadTotalsOnly))) {
            return;
        }

        readPAPIMeasurement(b, PAPITempMeasurementArray);

        const auto prefix = makeKernelName(kernelId);
        Value * eventCounterSumArray; Type * eventCounterSumty;

        std::tie(eventCounterSumArray, eventCounterSumty) = b.getScalarFieldPtr(prefix + STATISTICS_PAPI_COUNT_ARRAY_SUFFIX);
        Type * const counterArrayTy = eventCounterSumty->getArrayElementType();
        assert (counterArrayTy->isArrayTy());
        Type * const counterTy = counterArrayTy->getArrayElementType();

        Constant * i32_ZERO = b.getInt32(0);

        FixedArray<Value *, 2> from;
        from[0] = i32_ZERO;

        FixedArray<Value *, 3> update;
        update[0] = i32_ZERO;
        update[1] = b.CreateSelect(cond, b.getInt32(ifTrue), b.getInt32(ifFalse));

        Value * array = b.CreateSelect(cond, PAPIEventCounterArray[ifTrue], PAPIEventCounterArray[ifFalse]);

        for (unsigned i = 0; i < NumOfPAPIEvents; ++i) {
            from[1] = b.getInt32(i);
            Value * const beforeVal = b.CreateLoad(counterTy, b.CreateGEP(counterArrayTy, array, from));
            Value * const afterVal = b.CreateLoad(counterTy, b.CreateGEP(counterArrayTy, PAPITempMeasurementArray, from));
            Value * const diff = b.CreateSaturatingSub(afterVal, beforeVal);
            update[2] = from[1];
            Value * const ptr = b.CreateGEP(eventCounterSumty, eventCounterSumArray, update);
            Value * const curr = b.CreateLoad(counterTy, ptr);
            assert (curr->getType() == diff->getType());
            Value * const updatedVal = b.CreateAdd(curr, diff);
            b.CreateStore(updatedVal, ptr);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordTotalPAPIMeasurement
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordTotalPAPIMeasurement(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(NumOfPAPIEvents > 0)) {

        readPAPIMeasurement(b, PAPITempMeasurementArray);

        Value * eventCounterSumArray; Type * counterArrayTy;
        std::tie(eventCounterSumArray, counterArrayTy) = b.getScalarFieldPtr(STATISTICS_PAPI_TOTAL_COUNT_ARRAY);

        Type * const counterTy = counterArrayTy->getArrayElementType();
        assert (counterTy->isIntegerTy());

        Constant * const i32_ZERO = b.getInt32(0);

        FixedArray<Value *, 2> from;
        from[0] = i32_ZERO;

        for (unsigned i = 0; i < NumOfPAPIEvents; ++i) {
            from[1] = b.getInt32(i);
            Value * const beforeVal = b.CreateLoad(counterTy, b.CreateGEP(counterArrayTy, PAPIEventCounterArray[PAPIKernelCounter::PAPI_FULL_PIPELINE_TIME], from));
            Value * const afterVal = b.CreateLoad(counterTy, b.CreateGEP(counterArrayTy, PAPITempMeasurementArray, from));
            Value * const diff = b.CreateSaturatingSub(afterVal, beforeVal);
            Value * const ptr = b.CreateGEP(counterArrayTy, eventCounterSumArray, from);
            Value * const curr = b.CreateLoad(counterTy, ptr);
            assert (curr->getType() == diff->getType());
            Value * const updatedVal = b.CreateAdd(curr, diff);
            b.CreateStore(updatedVal, ptr);
        }
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief unregisterPAPIThread
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::unregisterPAPIThread(KernelBuilder & /* b */) const {
    // PAPI documentation indicates this and register thread are not necessary.
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief startPAPI
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::startPAPI(KernelBuilder & b) {
//    if (LLVM_UNLIKELY(NumOfPAPIEvents)) {
//        getPAPIEventSet(b);
//        readPAPIMeasurement(b, PipelineOutput, PAPIReadKernelStartMeasurementArray);
//    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief stopPAPIAndDestroyEventSet
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::stopPAPI(KernelBuilder & b) {
//    if (LLVM_UNLIKELY(NumOfPAPIEvents)) {
//        accumPAPIMeasurementWithoutReset(b, PAPIReadKernelStartMeasurementArray, PipelineOutput, PAPI_FULL_PIPELINE_TIME);
//    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief checkPAPIRetVal
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::checkPAPIRetValAndExitOnError(KernelBuilder & b, StringRef source, const int expected, Value * const retVal) const {

    IntegerType * const intTy = TypeBuilder<int, false>::get(b.getContext());
    ConstantInt * const papiOk = ConstantInt::get(intTy, expected);

    assert (NumOfPAPIEvents > 0);

    BasicBlock * const current = b.GetInsertBlock();
    Function * const function = current->getParent();
    Module * const m = function->getParent();
    BasicBlock * onError = BasicBlock::Create(b.getContext(), source + "_onError", function, current->getNextNode());
    BasicBlock * onSuccess = BasicBlock::Create(b.getContext(), source + "_onSuccess", function, onError);

    Value * const isOk = b.CreateICmpEQ(retVal, papiOk);
    b.CreateLikelyCondBr(isOk, onSuccess, onError);

    b.SetInsertPoint(onError);
    Function * PAPI_strerrFn = m->getFunction("PAPI_strerror");
    if (PAPI_strerrFn == nullptr) {
        PointerType * const int8PtrTy = b.getInt8PtrTy();
        FunctionType * PAPI_strerrFnTy = FunctionType::get(int8PtrTy, {intTy}, false);
        PAPI_strerrFn = Function::Create(PAPI_strerrFnTy, Function::ExternalLinkage, "PAPI_strerror", m);
    }
    FixedArray<Value *, 1> strerrArgs;
    strerrArgs[0] = retVal;
    Value * const strerr = b.CreateCall(PAPI_strerrFn->getFunctionType(), PAPI_strerrFn, strerrArgs);


    FixedArray<Value *, 4> args;
    args[0] = b.getInt32(STDERR_FILENO);
    args[1] = b.GetString("Error: %s returned %s\n");
    args[2] = b.GetString(source);
    args[3] = strerr;
    Function * const Dprintf = b.GetDprintf();
    b.CreateCall(Dprintf->getFunctionType(), Dprintf, args);
    b.CreateExit(-1);
    b.CreateBr(onSuccess);

    b.SetInsertPoint(onSuccess);
}

template<typename T>
auto constexpr cpow(T base, T exponent) -> T {
    static_assert(std::is_integral<T>(), "exponent must be integral");
    return exponent == 0 ? 1 : base * cpow<T>(base, exponent - 1);
}

inline size_t ceil_log10(size_t n) {
    size_t k = 1;
    constexpr auto E16 = cpow<size_t>(10ULL, 16ULL);
    errs() << "E16: " << E16 << "\n";
    while (n >= E16) {
        n /= E16;
        k += 16;
    }
    constexpr auto E8 = cpow<size_t>(10ULL, 8ULL);
    if (n >= E8) {
        n /= E8;
        k += 8;
    }
    constexpr auto E4 = cpow<size_t>(10ULL, 4ULL);
    if (n >= E4) {
        n /= E4;
        k += 4;
    }
    while (n) {
        n /= 10;
        k++;
    }
    return k;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __print_pipeline_PAPI_report
 ** ------------------------------------------------------------------------------------------------------------- */
namespace {
extern "C"
BOOST_NOINLINE
void __print_pipeline_PAPI_report(const unsigned numOfKernels, const char ** kernelNames,
                                  const unsigned numOfEvents, const int * const eventCode,
                                  const papi_counter_t ** const values,
                                  const papi_counter_t * const totals) {

    auto & out = errs();

    size_t maxNameLength = 4;
    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto len = std::strlen(kernelNames[i]);
        maxNameLength = std::max(maxNameLength, len);
    }

    size_t maxEventLength = 8;
    char eventName[PAPI_MAX_STR_LEN + 1];
    for (unsigned i = 0; i != numOfEvents; i++) {
        const auto rval = PAPI_event_code_to_name(eventCode[i], eventName);
        if (LLVM_LIKELY(rval == PAPI_OK)) {
            maxEventLength = std::max(maxEventLength, std::strlen(eventName));
        }
    }

    papi_counter_t maxCounter = 5;
    for (unsigned i = 0; i != numOfEvents; i++) {
        maxCounter = std::max(maxCounter, totals[i]);
    }
    const auto maxCounterLength = std::max(ceil_log10(maxCounter), 7UL);

    out << "PAPI REPORT\n";
    out.indent(4 + maxNameLength + 1 + maxEventLength + 3 + (4 * 7) + 2);
    out << "INS+\n";
    out << "  # "  // kernel #
           "NAME"; // kernel Name
    assert (maxNameLength > 4);
    out.indent(maxNameLength - 3);

    out << "EVENT";
    assert (maxEventLength > 5);
    out.indent(maxEventLength - 4);

    out << "|   SYNC " // kernel synchronization %,
           "  PART " // partition synchronization %,
           "  EXPD " // buffer expansion %,
           "  COPY " // look ahead + copy back + look behind %,
           "  PIPE " // pipeline overhead %,
           "  EXEC " // execution %,
           "|";
    out.indent(maxCounterLength - 7);
    out << "SUBTOTAL"; // total kernel value.
    out.indent(7);
    out << "%\n";

    std::string kernelNameFmt;
    BEGIN_SCOPED_REGION
    raw_string_ostream formatter(kernelNameFmt);
    formatter << "%3" PRIu32 " " // kernel #
            "%-" << maxNameLength << "s" // kernel name
            " ";
    END_SCOPED_REGION

    std::string eventNameFmt;
    BEGIN_SCOPED_REGION
    raw_string_ostream formatter(eventNameFmt);
    formatter << "%-" << maxEventLength << "s" // event name
                 " |";
    END_SCOPED_REGION


    std::string valueFmt;
    BEGIN_SCOPED_REGION
    raw_string_ostream formatter(valueFmt);
    formatter << " | %" << maxCounterLength << PRIuMAX "  %6.2f\n";
    END_SCOPED_REGION

    // data is in [kernels][counters][events] ordering

    // ArrayType::get(ArrayType::get(papiCounterTy, NumOfPAPIEvents), NUM_OF_PAPI_KERNEL_COUNTERS);

    #define GET_VALUE(kernel, event, counter) \
        values[kernel][((counter) * numOfEvents + event)]

    SmallVector<papi_counter_t, 4> other_subtotals(numOfEvents, 0);

    for (unsigned i = 0; i < numOfKernels; ++i) {
        for (unsigned j = 0; j < numOfEvents; ++j) {
            if (j == 0) {
                out << llvm::format(kernelNameFmt.data(), i + 1, kernelNames[i]);
            } else {
                out.indent(maxNameLength + 5);
            }

            const auto rval = PAPI_event_code_to_name(eventCode[j], eventName);
            if (LLVM_LIKELY(rval == PAPI_OK)) {
                out << llvm::format(eventNameFmt.data(), (char*)eventName);
            } else {
                out.write_hex(eventCode[j]);
            }

            const auto subtotal = GET_VALUE(i, j, PAPI_KERNEL_TOTAL);
            const long double fsubtotal = subtotal;
            assert (subtotal <= totals[j]);

            for (unsigned k = 0; k < PAPI_KERNEL_EXECUTION; ++k) {
                const auto v = GET_VALUE(i, j, k);
                const long double val = v;
                assert (v <= subtotal);
                const auto r = (val / fsubtotal);
                out << llvm::format(" %6.2f", ((double)(r)) * 100.0);
            }

            BEGIN_SCOPED_REGION
            papi_counter_t sum = 0;
            for (unsigned k = 0; k < PAPI_KERNEL_TOTAL; ++k) {
                sum += GET_VALUE(i, j, k);
            }
            assert (sum <= subtotal);

            const papi_counter_t other = (sum < subtotal) ? (subtotal - sum) : 0ULL;
            assert (other < subtotal);

            assert (other_subtotals[j] <= totals[j]);
            assert (other_subtotals[j] <= (totals[j] - other));
            other_subtotals[j] = other_subtotals[j] + other;
            assert (other_subtotals[j] <= totals[j]);
            const long double fother = other;
            const auto r = (fother / fsubtotal);
      //      assert (0.0L <= r && r <= 1.0L);
            out << llvm::format(" %6.2f", (double)(r) * 100.0);
            END_SCOPED_REGION

            BEGIN_SCOPED_REGION
            const auto v = GET_VALUE(i, j, PAPI_KERNEL_EXECUTION);
            const long double val = v;
            const auto r = (val / fsubtotal);
        //    assert (0.0L <= r && r <= 1.0L);
            out << llvm::format(" %6.2f", (double)(r) * 100.0);
            END_SCOPED_REGION

            BEGIN_SCOPED_REGION
            const long double ftotal = totals[j];
            const auto r = (fsubtotal / ftotal);
          //  assert (0.0L <= r && r <= 1.0L);
            const double ratio = (double)(r) * 100.0;
            out << llvm::format(valueFmt.data(), subtotal, ratio);
            END_SCOPED_REGION

        }
    }



    for (unsigned j = 0; j < numOfEvents; ++j) {

        out << '\n';

        if (j == 0) {
            out.indent(4);
            out << "TOTAL";
            out.indent(maxNameLength - 5 + 1);
        } else {
            out.indent(4 + maxNameLength + 1);
        }

        const auto rval = PAPI_event_code_to_name(eventCode[j], eventName);
        if (LLVM_LIKELY(rval == PAPI_OK)) {
            out << llvm::format(eventNameFmt.data(), (char*)eventName);
        } else {
            out.write_hex(eventCode[j]);
        }

        const long double ftotal = totals[j];
        for (unsigned k = 0; k < PAPI_KERNEL_EXECUTION; ++k) {
            papi_counter_t subtotal = 0;
            for (unsigned i = 0; i < numOfKernels; ++i) {
                subtotal += GET_VALUE(i, j, k);
            }
            const long double fsubtotal = subtotal;
            const auto r = (fsubtotal / ftotal);
          //  assert (0.0L <= r && r <= 1.0L);
            out << llvm::format(" %6.2f", (double)(r) * 100.0);
        }

        BEGIN_SCOPED_REGION
        const long double fsubtotal = other_subtotals[j];
        const auto r = (fsubtotal / ftotal);
      //  assert (0.0L <= r && r <= 1.0L);
        out << llvm::format(" %6.2f", (double)(r) * 100.0);
        END_SCOPED_REGION

        BEGIN_SCOPED_REGION
        papi_counter_t subtotal = 0;
        for (unsigned i = 0; i < numOfKernels; ++i) {
            subtotal += GET_VALUE(i, j, PAPI_KERNEL_EXECUTION);
        }
        const long double fsubtotal = subtotal;
        const auto r = (fsubtotal / ftotal);
     //   assert (0.0L <= r && r <= 1.0L);
        out << llvm::format(" %6.2f", (double)(r) * 100.0);
        END_SCOPED_REGION

        BEGIN_SCOPED_REGION
        papi_counter_t subtotal = 0;
        for (unsigned i = 0; i < numOfKernels; ++i) {
            subtotal += GET_VALUE(i, j, PAPI_KERNEL_TOTAL);
        }
        const long double fsubtotal = subtotal;
        const auto r = (fsubtotal / ftotal);
     //   assert (0.0L <= r && r <= 1.0L);
        out << llvm::format(valueFmt.data(), subtotal, (double)(r) * 100.0);
        END_SCOPED_REGION
    }

    #undef GET_POS

    out << '\n';

}

extern "C"
BOOST_NOINLINE
void __print_pipeline_totals_PAPI_report(const unsigned numOfThreads,
                                         const unsigned numOfEvents, const int * const eventCode,
                                         const papi_counter_t * const totals) {

                         // totals contains the final event counts for the program;
                         // values has numOfKernels * numOfEvents * numOfMeasurements
                         // event counts in that order.

    auto & out = errs();

    char eventName[PAPI_MAX_STR_LEN + 1];

    for (unsigned i = 0, idx = 0; i < numOfEvents; ++i) {

        const auto rval = PAPI_event_code_to_name(eventCode[i], eventName);
        if (LLVM_LIKELY(rval == PAPI_OK)) {
            out << (char*)eventName;
        } else {
            out.write_hex(eventCode[i]);
        }

        for (unsigned j = 0; j < numOfThreads; ++j) {
            out << ',' << totals[idx++];
        }

        out << '\n';

    }

}

}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printPAPIReport
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printPAPIReportIfRequested(KernelBuilder & b) {
    if (LLVM_UNLIKELY(NumOfPAPIEvents)) {

        IntegerType * const papiCounterTy = TypeBuilder<papi_counter_t, false>::get(b.getContext());
        PointerType * const counterPtrTy = papiCounterTy->getPointerTo();

        IntegerType * const intTy = TypeBuilder<unsigned, false>::get(b.getContext());

        ConstantInt * const ZERO = b.getInt32(0);

        auto toGlobal = [&](ArrayRef<Constant *> array, Type * const type, size_t size) {
            ArrayType * const arTy = ArrayType::get(type, size);
            Constant * const ar = ConstantArray::get(arTy, array);
            Module & mod = *b.getModule();
            GlobalVariable * const gv = new GlobalVariable(mod, ar->getType(), true, GlobalValue::PrivateLinkage, ar);
            FixedArray<Value *, 2> tmp;
            tmp[0] = ZERO;
            tmp[1] = ZERO;
            return b.CreateInBoundsGEP(arTy, gv, tmp);
        };

        Value * const arrayOfEventCodes = b.getScalarField(STATISTICS_PAPI_EVENT_SET_LIST);

        Value * totals; Type * totalsTy;

        std::tie(totals, totalsTy) = b.getScalarFieldPtr(STATISTICS_PAPI_TOTAL_COUNT_ARRAY);

        if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::DisplayPAPICounterThreadTotalsOnly))) {

            FixedArray<Value *, 5> args;
            args[0] = mMaximumNumOfThreads;
            args[1] = ConstantInt::get(intTy, NumOfPAPIEvents);
            args[2] = arrayOfEventCodes;
            args[3] = b.CreatePointerCast(totals, counterPtrTy);

            Function * const reportPrinter = b.getModule()->getFunction("__print_pipeline_totals_PAPI_report");
            assert (reportPrinter);
            b.CreateCall(reportPrinter->getFunctionType(), reportPrinter, args);

        } else {

            DataLayout dl(b.getModule());

            std::vector<Constant *> kernelNames;
            for (auto i = FirstKernel; i <= LastKernel; ++i) {
                const Kernel * const kernel = getKernel(i);
                kernelNames.push_back(b.GetString(kernel->getName()));
            }

            const auto numOfKernels = LastKernel - FirstKernel + 1U;

            PointerType * const int8PtrTy = b.getInt8PtrTy();

            Value * const arrayOfKernelNames = toGlobal(kernelNames, int8PtrTy, numOfKernels);

            ArrayType * const arTy = ArrayType::get(counterPtrTy, LastKernel - FirstKernel + 1);

            Value * const pointerArray = b.CreateAlloca(arTy);

            FixedArray<Value *, 2> indices;
            indices[0] = b.getInt32(0);
            for (size_t i = FirstKernel; i <= LastKernel; ++i) {
                Value * base; Type * ty;
                std::tie(base, ty) = b.getScalarFieldPtr(makeKernelName(i) + STATISTICS_PAPI_COUNT_ARRAY_SUFFIX);
                indices[1] = b.getInt32(i - FirstKernel);
                b.CreateStore(b.CreatePointerCast(base, counterPtrTy), b.CreateGEP(arTy, pointerArray, indices));

            }

            Function * const reportPrinter = b.getModule()->getFunction("__print_pipeline_PAPI_report");
            assert (reportPrinter);

            FixedArray<Value *, 6> args;
            args[0] = ConstantInt::get(intTy, numOfKernels);
            args[1] = arrayOfKernelNames;
            args[2] = ConstantInt::get(intTy, NumOfPAPIEvents);
            args[3] = arrayOfEventCodes;
            args[4] = b.CreatePointerCast(pointerArray, counterPtrTy->getPointerTo()); // values
            args[5] = b.CreatePointerCast(totals, counterPtrTy);


            b.CreateCall(reportPrinter->getFunctionType(), reportPrinter, args);

        }

    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkPAPILibrary
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::linkPAPILibrary(KernelBuilder & b) {
//    b.LinkFunction("PAPI_library_init", PAPI_library_init);
//    b.LinkFunction("PAPI_thread_init", PAPI_thread_init);
//    b.LinkFunction("PAPI_create_eventset", PAPI_create_eventset);
//    b.LinkFunction("PAPI_add_events", PAPI_add_events);
    b.LinkFunction("PAPI_start", PAPI_start);
    b.LinkFunction("PAPI_read", PAPI_read);
//    b.LinkFunction("PAPI_accum", PAPI_accum);
    b.LinkFunction("PAPI_stop", PAPI_stop);
    b.LinkFunction("PAPI_cleanup_eventset", PAPI_cleanup_eventset);
    b.LinkFunction("PAPI_destroy_eventset", PAPI_destroy_eventset);
    b.LinkFunction("PAPI_shutdown", PAPI_shutdown);
    b.LinkFunction("PAPI_strerror", PAPI_strerror);
    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::DisplayPAPICounterThreadTotalsOnly))) {
        b.LinkFunction("__print_pipeline_totals_PAPI_report", __print_pipeline_totals_PAPI_report);
    } else {
        b.LinkFunction("__print_pipeline_PAPI_report", __print_pipeline_PAPI_report);
    }
}

}

#endif // #ifdef ENABLE_PAPI
