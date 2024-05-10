#include "../pipeline_compiler.hpp"
#include <boost/format.hpp>
#include <llvm/Support/Format.h>

// TODO: Print Total CPU Cycles

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addInternalKernelCycleCountProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addCycleCounterProperties(KernelBuilder & b, const unsigned kernelId, const bool isRoot) {

    const auto groupId = getCacheLineGroupId(kernelId);

    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        // TODO: make these thread local to prevent false sharing and enable
        // analysis of thread distributions?
        Type * const int64Ty = b.getInt64Ty();
        Type * rootPropertyIntTy = int64Ty;
        if (isRoot) {
            rootPropertyIntTy = int64Ty;
        } else {
            rootPropertyIntTy = StructType::get(b.getContext());
        }

        FixedArray<Type *, NUM_OF_KERNEL_CYCLE_COUNTERS> fields;
        fields[KERNEL_SYNCHRONIZATION] = int64Ty;
        fields[PARTITION_JUMP_SYNCHRONIZATION] = rootPropertyIntTy;
        fields[BUFFER_EXPANSION] = int64Ty;
        fields[BUFFER_COPY] = int64Ty;
        fields[KERNEL_EXECUTION] = int64Ty;
        fields[TOTAL_TIME] = int64Ty;
        fields[SQ_SUM_TOTAL_TIME] = int64Ty;
        fields[NUM_OF_INVOCATIONS] = int64Ty;

        StructType * const cycleCounterTy = StructType::get(b.getContext(), fields);
        const auto prefix = makeKernelName(kernelId) + STATISTICS_CYCLE_COUNT_SUFFIX;
        mTarget->addInternalScalar(cycleCounterTy, prefix, groupId);
    }

    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::EnableBlockingIOCounter))) {

        const auto prefix = makeKernelName(kernelId);
        IntegerType * const int64Ty = b.getInt64Ty();

        // # of blocked I/O channel attempts in which no strides
        // were possible (i.e., blocked on first iteration)

        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const auto port = mBufferGraph[e].Port;
            const auto prefix = makeBufferName(kernelId, port);
            mTarget->addInternalScalar(int64Ty, prefix + STATISTICS_BLOCKING_IO_SUFFIX, groupId);
        }
        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            // TODO: ignore dynamic buffers
            const auto port = mBufferGraph[e].Port;
            const auto prefix = makeBufferName(kernelId, port);
            mTarget->addInternalScalar(int64Ty, prefix + STATISTICS_BLOCKING_IO_SUFFIX, groupId);
        }
    }

    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceBlockedIO))) {

        FixedArray<Type *, 3> fields;
        IntegerType * const sizeTy = b.getSizeTy();
        fields[0] = sizeTy->getPointerTo();
        fields[1] = sizeTy;
        fields[2] = sizeTy;
        StructType * const historyTy = StructType::get(b.getContext(), fields);

        // # of blocked I/O channel attempts in which no strides
        // were possible (i.e., blocked on first iteration)
        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const auto & bp = mBufferGraph[e];
            if (bp.canModifySegmentLength()) {
                const auto prefix = makeBufferName(kernelId, bp.Port);
                mTarget->addInternalScalar(historyTy, prefix + STATISTICS_BLOCKING_IO_HISTORY_SUFFIX, groupId);
            }
        }
        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const auto & bp = mBufferGraph[e];
            if (bp.canModifySegmentLength()) {
                const auto prefix = makeBufferName(kernelId, bp.Port);
                mTarget->addInternalScalar(historyTy, prefix + STATISTICS_BLOCKING_IO_HISTORY_SUFFIX, groupId);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief trackCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
inline bool PipelineCompiler::trackCycleCounter(const CycleCounter type) const {
    if (LLVM_UNLIKELY(EnableCycleCounter)) {
        return true;
    }
    if (mUseDynamicMultithreading) {
        switch (type) {
            case KERNEL_SYNCHRONIZATION:
            case PARTITION_JUMP_SYNCHRONIZATION:
//            case TOTAL_TIME:
                return true;
            default: break;
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief startCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::startCycleCounter(KernelBuilder & b, const CycleCounter type) {
    if (trackCycleCounter(type)) {
        Value * const counter = b.CreateReadCycleCounter();
        mCycleCounters[(unsigned)type] = counter;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief startCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::startCycleCounter(KernelBuilder & b, const std::initializer_list<CycleCounter> types) {
    Value * counter = nullptr;
    for (auto type : types) {
        if (trackCycleCounter(type)) {
            if (counter == nullptr) {
                counter = b.CreateReadCycleCounter();
            }
            mCycleCounters[(unsigned)type] = counter;
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateOptionalCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateCycleCounter(KernelBuilder & b, const unsigned kernelId, const CycleCounter type) const {
    if (trackCycleCounter(type)) {
        assert (FirstKernel <= kernelId && kernelId <= LastKernel);
        Value * const end = b.CreateReadCycleCounter();
        Value * const start = mCycleCounters[(unsigned)type]; assert (start);
        Value * const duration = b.CreateSub(end, start);

        IntegerType * sizeTy = b.getSizeTy();

        if (mUseDynamicMultithreading) {
            Value * const cur = b.CreateLoad(sizeTy, mAccumulatedSynchronizationTimePtr);
            Value * const accum = b.CreateAdd(cur, duration);
            b.CreateStore(accum, mAccumulatedSynchronizationTimePtr);
        }

        if (EnableCycleCounter) {

            const auto prefix = makeKernelName(kernelId);
            Value * ptr; Type * ty;
            std::tie(ptr, ty) = b.getScalarFieldPtr(prefix  + STATISTICS_CYCLE_COUNT_SUFFIX);
            FixedArray<Value *, 2> index;
            index[0] = b.getInt32(0);
            index[1] = b.getInt32(type);
            Value * const sumCounterPtr = b.CreateGEP(ty, ptr, index);
            Value * const sumRunningCount = b.CreateLoad(sizeTy, sumCounterPtr);
            Value * const sumUpdatedCount = b.CreateAdd(sumRunningCount, duration);
            b.CreateStore(sumUpdatedCount, sumCounterPtr);

            if (type == CycleCounter::TOTAL_TIME) {
                index[1] = b.getInt32(SQ_SUM_TOTAL_TIME);
                Value * const sqSumCounterPtr = b.CreateGEP(ty, ptr, index);
                Value * const sqSumRunningCount = b.CreateLoad(sizeTy, sqSumCounterPtr);
                Value * sqDuration = b.CreateZExt(duration, sqSumRunningCount->getType());
                sqDuration = b.CreateMul(sqDuration, sqDuration);
                Value * const sqSumUpdatedCount = b.CreateAdd(sqSumRunningCount, sqDuration);
                b.CreateStore(sqSumUpdatedCount, sqSumCounterPtr);
                if (mIsPartitionRoot) {
                    index[1] = b.getInt32(NUM_OF_INVOCATIONS);
                    Value * const invokePtr = b.CreateGEP(ty, ptr, index);
                    Value * const invoked = b.CreateLoad(sizeTy, invokePtr);
                    Value * const invoked2 = b.CreateAdd(invoked, b.getSize(1));
                    b.CreateStore(invoked2, invokePtr);
                }
            }

        }

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateOptionalCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateCycleCounter(KernelBuilder & b, const unsigned kernelId, Value * const cond, const CycleCounter ifTrue, const CycleCounter ifFalse) const {
    if (EnableCycleCounter) {
        Value * const end = b.CreateReadCycleCounter();
        Value * const start = mCycleCounters[(unsigned)ifTrue]; assert (start);
        assert (mCycleCounters[(unsigned)ifFalse] == start);
        Value * const duration = b.CreateSub(end, start);
        const auto prefix = makeKernelName(kernelId);
        Value * ptr; Type * ty;
        std::tie(ptr, ty) = b.getScalarFieldPtr(prefix  + STATISTICS_CYCLE_COUNT_SUFFIX);
        assert (ty->isStructTy());
        FixedArray<Value *, 2> index;
        index[0] = b.getInt32(0);
        index[1] = b.getInt32(ifTrue);
        Value * const sumCounterPtrA = b.CreateGEP(ty, ptr, index);
        index[1] = b.getInt32(ifFalse);
        Value * const sumCounterPtrB = b.CreateGEP(ty, ptr, index);
        Value * const sumCounterPtr = b.CreateSelect(cond, sumCounterPtrA, sumCounterPtrB);

        Value * const sumRunningCount = b.CreateLoad(b.getSizeTy(), sumCounterPtr);
        Value * const sumUpdatedCount = b.CreateAdd(sumRunningCount, duration);
        b.CreateStore(sumUpdatedCount, sumCounterPtr);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateOptionalCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateTotalCycleCounterTime(KernelBuilder & b) const {
    if (trackCycleCounter(FULL_PIPELINE_TIME)) {
        Value * const end = b.CreateReadCycleCounter();
        Value * const start = mCycleCounters[(unsigned)FULL_PIPELINE_TIME];
        if (LLVM_UNLIKELY(CheckAssertions)) {
            b.CreateAssert (b.CreateICmpULT(start, end), "???");
        }


        Value * const duration = b.CreateSub(end, mCycleCounters[(unsigned)FULL_PIPELINE_TIME]);
        // total is thread local but gets summed at the end; no need to worry about
        // multiple threads updating it.
        Value * const ptr = getScalarFieldPtr(b, STATISTICS_CYCLE_COUNT_TOTAL).first;
        Value * const updated = b.CreateAdd(b.CreateLoad(b.getSizeTy(), ptr), duration);
        b.CreateStore(updated, ptr);
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief selectPrincipleCycleCountBinding
 ** ------------------------------------------------------------------------------------------------------------- */
StreamSetPort PipelineCompiler::selectPrincipleCycleCountBinding(const unsigned kernel) const {
    const auto numOfInputs = in_degree(kernel, mBufferGraph);
    assert (degree(kernel, mBufferGraph));
    if (numOfInputs == 0) {
        return StreamSetPort{PortType::Output, 0};
    } else {
        for (unsigned i = 0; i < numOfInputs; ++i) {
            const auto inputPort = StreamSetPort{PortType::Input, i};
            const Binding & input = getInputBinding(kernel, inputPort);
            if (LLVM_UNLIKELY(input.hasAttribute(AttrId::Principal))) {
                return inputPort;
            }
        }
        return StreamSetPort{PortType::Input, 0};
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __print_pipeline_cycle_counter_report
 ** ------------------------------------------------------------------------------------------------------------- */
namespace {
extern "C"
BOOST_NOINLINE
void __print_pipeline_cycle_counter_report(const uint64_t numOfKernels,
                                           const char ** kernelNames,
                                           const uint64_t * const values,
                                           const uint64_t totalCycles,
                                           const uint64_t baseItemCount) {

                         // totals contains the final event counts for the program;
                         // values has numOfKernels * numOfEvents * numOfMeasurements
                         // event counts in that order.

    unsigned maxNameLength = 4;
    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto len = std::strlen(kernelNames[i]);
        maxNameLength = std::max<unsigned>(maxNameLength, len);
    }

    const auto maxKernelIdLength = ((unsigned)std::ceil(std::log10(numOfKernels))) + 1U;

    // TODO: cycle counter reporting oddly on colours=always?

    uint64_t maxItemCount = 0;
    uint64_t maxCycleCount = 0;

    long double baseCyclesPerItem = 0L;
    if (baseItemCount) {
        baseCyclesPerItem = ((long double)(totalCycles) / (long double)(baseItemCount));
    }
    auto maxCyclesPerItem = baseCyclesPerItem;

    #ifndef NDEBUG
    const auto REQ_INTEGERS = numOfKernels * (NUM_OF_KERNEL_CYCLE_COUNTERS + 1);
    #endif

    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto k = i * (NUM_OF_KERNEL_CYCLE_COUNTERS + 1);
        assert ((k + TOTAL_TIME) < numOfKernels * (NUM_OF_KERNEL_CYCLE_COUNTERS + 1));
        const uint64_t itemCount = values[k];
        maxItemCount = std::max(maxItemCount, itemCount);
        const auto cycleCount = values[k + TOTAL_TIME + 1];
        if (itemCount > 0) {
            const auto cyclesPerItem = ((long double)(cycleCount) / (long double)(itemCount));
            maxCyclesPerItem = std::max(cyclesPerItem, maxCyclesPerItem);
        }
        assert (cycleCount <= totalCycles);
        maxCycleCount = std::max(maxCycleCount, cycleCount);
    }

    maxCyclesPerItem += std::numeric_limits<long double>::epsilon();

    const auto maxItemCountLength = std::max<unsigned>(std::ceil(std::log10(maxItemCount)), 5);
    const auto maxCycleCountLength = std::max<unsigned>(std::ceil(std::log10(maxCycleCount)), 6);
    const auto maxCyclesPerItemLength = std::max<unsigned>(std::ceil(std::log10(maxCyclesPerItem)) + 2, 4);

    auto & out = errs();

    out << "CYCLE COUNTER:\n\n";

    out << right_justify("#", maxKernelIdLength); // kernel #
    out << " NAME"; // kernel Name
    assert (maxNameLength >= 4);
    assert (maxItemCountLength >= 5);
    out.indent((maxNameLength - 4) + (maxItemCountLength - 5) + 1);
    out << "ITEMS"; // items processed
    assert (maxCycleCountLength >= 6);
    out.indent((maxCycleCountLength - 6) + 1);
    out << "CYCLES"; // CPU cycles
    assert (maxCyclesPerItemLength >= 4);
    out.indent((maxCyclesPerItemLength - 4));
    out << " RATE" // cycles per item,
           "  SYNC" // kernel synchronization %,
           "  PART" // partition synchronization %,
           "  EXPD" // buffer expansion %,
           "  COPY" // look ahead + copy back + look behind %,
           "  PIPE" // pipeline overhead %,
           "  EXEC" // execution %,
           "     %\n"; // % of Total CPU Cycles.


    unsigned k = 0;

    const long double fTotal = totalCycles;

    boost::format percfmt("%5.1f");
    boost::format fixedfmt("%.0f");
    boost::format ratefmt("%.1f");
    boost::format covfmt(" +- %4.1f\n");

    std::array<uint64_t, KERNEL_EXECUTION + 3> subtotals;
    std::fill_n(subtotals.begin(), KERNEL_EXECUTION + 3, 0);

    for (unsigned i = 0; i < numOfKernels; ++i) {

        assert ((k + TOTAL_TIME + 1) < REQ_INTEGERS);
        const uint64_t intItemCount = values[k++];
        assert (intItemCount <= maxItemCount);
        const uint64_t intSubTotal = values[k + TOTAL_TIME];
        assert (intSubTotal <= maxCycleCount);

        out << right_justify(std::to_string(i + 1), maxKernelIdLength)
            << ' '
            << left_justify(StringRef{kernelNames[i], std::strlen(kernelNames[i])}, maxNameLength)
            << ' '
            << right_justify(std::to_string(intItemCount), maxItemCountLength)
            << ' '
            << right_justify(std::to_string(intSubTotal), maxCycleCountLength)
            << ' ';

        const long double fSubTotal = intSubTotal;

        if (intItemCount > 0) {
            const auto cyclesPerItem = (fSubTotal / (long double)(intItemCount));
            assert (cyclesPerItem <= maxCyclesPerItem);
            out << right_justify((ratefmt % cyclesPerItem).str(), maxCyclesPerItemLength) << ' ';
        } else {
            out << right_justify("--", maxCyclesPerItemLength + 1);
        }

        uint64_t knownOverheads = 0;

        for (unsigned j = KERNEL_SYNCHRONIZATION; j < KERNEL_EXECUTION; ++j) {
            assert (k < REQ_INTEGERS);
            const auto v = values[k++];
            subtotals[j] += v;
            knownOverheads += v;
            const double cycPerc = (((long double)v) * 100.0L) / fSubTotal;
            out << (percfmt % cycPerc).str() << ' ';
        }
        assert (k < REQ_INTEGERS);
        const auto intExecTime = values[k++];
        knownOverheads += intExecTime;
        assert (knownOverheads <= intSubTotal);

        const auto intOverhead = (knownOverheads > intSubTotal) ? 0UL : (intSubTotal - knownOverheads);

        subtotals[KERNEL_EXECUTION] += intOverhead;
        subtotals[KERNEL_EXECUTION + 1] += intExecTime;

        const double overheadPerc = (((long double)intOverhead) * 100.0L) / fSubTotal;
        out << (percfmt % overheadPerc).str() << ' ';

        const double execTimePerc = (((long double)intExecTime) * 100.0L) / fSubTotal;
        out << (percfmt % execTimePerc).str() << ' ';
        assert (values[k] == intSubTotal);
        ++k;

        subtotals[KERNEL_EXECUTION + 2] += intSubTotal;

        const auto perc = (fSubTotal * 100.0L) / fTotal;
        out << (percfmt % perc).str();
        assert (k < REQ_INTEGERS);
        const long double fSqrTotalCycles = values[k++];
        assert (k < REQ_INTEGERS);
        const uint64_t n = values[k++];
        const long double N = n;
        const auto x = std::max((fSqrTotalCycles * N) - (fSubTotal * fSubTotal), 0.0L);
        const auto avgStddev = (std::sqrt(x) * 100.0L) / (N * fTotal);
        out << (covfmt % avgStddev).str();

    }
    assert (k == REQ_INTEGERS);

    out << "\n";
    out.indent(maxKernelIdLength + maxNameLength + maxItemCountLength + maxCycleCountLength + maxCyclesPerItemLength - 2);
    out << "TOTAL:";

    for (unsigned j = KERNEL_SYNCHRONIZATION; j < (KERNEL_EXECUTION + 3); ++j) {
        const auto v = subtotals[j];
        const double cycPerc = (((long double)(v * 100)) / fTotal);
        out << ' ' << (percfmt % cycPerc).str();
    }
    out << '\n';

}
} // end of anonymous namespace

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalCycleCounter
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printOptionalCycleCounter(KernelBuilder & b) {
    if (LLVM_UNLIKELY(EnableCycleCounter)) {

        ConstantInt * const ZERO = b.getInt32(0);

        auto toGlobal = [&](ArrayRef<Constant *> array, Type * const type, size_t size) {
            ArrayType * const arTy = ArrayType::get(type, size);
            Constant * const ar = ConstantArray::get(arTy, array);
            Module & mod = *b.getModule();
            GlobalVariable * const gv = new GlobalVariable(mod, arTy, true, GlobalValue::PrivateLinkage, ar);
            FixedArray<Value *, 2> tmp;
            tmp[0] = ZERO;
            tmp[1] = ZERO;
            return b.CreateInBoundsGEP(arTy, gv, tmp);
        };

        std::vector<Constant *> kernelNames;
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            const Kernel * const kernel = getKernel(i);
            kernelNames.push_back(b.GetString(kernel->getName()));
        }

        assert (LastKernel < PipelineOutput);

        const auto numOfKernels = LastKernel - FirstKernel + 1U;

        PointerType * const int8PtrTy = b.getInt8PtrTy();

        Value * const arrayOfKernelNames = toGlobal(kernelNames, int8PtrTy, numOfKernels);

        const auto REQ_INTEGERS = numOfKernels * (NUM_OF_KERNEL_CYCLE_COUNTERS + 1);

        Constant * const requiredSpace =  b.getSize(sizeof(uint64_t) * REQ_INTEGERS);

        IntegerType * int64Ty = b.getInt64Ty();

        PointerType * const int64PtrTy = int64Ty->getPointerTo();

        Value * const values = b.CreatePointerCast(b.CreateAlignedMalloc(requiredSpace, sizeof(uint64_t)), int64PtrTy);

        auto currentPartitionId = -1U;



        Constant * const INT64_ZERO = b.getInt64(0);

        unsigned k = 0;

        Value * currentN = nullptr;

        Value * baseItemCount = nullptr;


        for (unsigned i = 0; i < numOfKernels; ++i) {
            assert (FirstKernel + i <= LastKernel);
            const auto prefix = makeKernelName(FirstKernel + i);

            const auto partitionId = KernelPartitionId[FirstKernel + i];
            const bool isRoot = (partitionId != currentPartitionId);
            currentPartitionId = partitionId;

            const auto binding = selectPrincipleCycleCountBinding(FirstKernel + i);
            Value * const items = b.getScalarField(makeBufferName(FirstKernel + i, binding) + ITEM_COUNT_SUFFIX);
            if (in_degree(FirstKernel + i, mBufferGraph) == 0) {
                if (baseItemCount == nullptr) {
                    baseItemCount = items;
                } else {
                    Value * const eq = b.CreateICmpEQ(baseItemCount, items);
                    baseItemCount = b.CreateSelect(eq, baseItemCount, INT64_ZERO);
                }
            }

            assert (k < REQ_INTEGERS);
            Value * const itemsPtr = b.CreateGEP(b.getInt64Ty(), values, b.getInt32(k++));
            b.CreateStore(items, itemsPtr);
            Value * cycleCountPtr; Type * cycleCountTy;

            std::tie(cycleCountPtr, cycleCountTy) = b.getScalarFieldPtr(prefix + STATISTICS_CYCLE_COUNT_SUFFIX);

            FixedArray<Value *, 2> index;
            index[0] = b.getInt32(0);
            for (unsigned j = 0; j < NUM_OF_INVOCATIONS; ++j) {
                Value * sumCycles = INT64_ZERO;
                if (isRoot || j != PARTITION_JUMP_SYNCHRONIZATION) {
                    index[1] = b.getInt32(j);
                    sumCycles = b.CreateLoad(int64Ty, b.CreateGEP(cycleCountTy, cycleCountPtr, index));
                } else {
                    assert (cycleCountTy->getStructElementType(j)->isEmptyTy());
                }
                assert (k < REQ_INTEGERS);
                b.CreateStore(sumCycles, b.CreateGEP(int64Ty, values, b.getInt32(k++)));
            }

            if (isRoot) {
                index[1] = b.getInt32(NUM_OF_INVOCATIONS);
                currentN = b.CreateLoad(b.getInt64Ty(), b.CreateGEP(cycleCountTy, cycleCountPtr, index));
            }
            assert (currentN);
            assert (k < REQ_INTEGERS);
            b.CreateStore(currentN, b.CreateGEP(int64Ty, values, b.getInt32(k++)));
        }
        assert (k == REQ_INTEGERS);

        FixedArray<Value *, 5> args;
        args[0] = ConstantInt::get(int64Ty, numOfKernels);
        args[1] = arrayOfKernelNames;
        args[2] = values;
        args[3] = b.getScalarField(STATISTICS_CYCLE_COUNT_TOTAL);
        args[4] = (baseItemCount == nullptr) ? INT64_ZERO : baseItemCount;

        Function * const reportPrinter = b.getModule()->getFunction("__print_pipeline_cycle_counter_report");
        assert (reportPrinter);
        b.CreateCall(reportPrinter->getFunctionType(), reportPrinter, args);

        b.CreateFree(values);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordBlockingIO
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordBlockingIO(KernelBuilder & b, const StreamSetPort port) const {
    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::EnableBlockingIOCounter))) {
        const auto prefix = makeBufferName(mKernelId, port);
        Value * counterPtr; Type * ty;
        std::tie(counterPtr, ty) = b.getScalarFieldPtr(prefix + STATISTICS_BLOCKING_IO_SUFFIX);
        Value * const runningCount = b.CreateLoad(ty, counterPtr);
        Value * const updatedCount = b.CreateAdd(runningCount, b.getSize(1));
        b.CreateStore(updatedCount, counterPtr);
    }
    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceBlockedIO))) {

        const auto prefix = makeBufferName(mKernelId, port);
        Value * historyPtr; Type * ty;
        std::tie(historyPtr, ty) = b.getScalarFieldPtr(prefix + STATISTICS_BLOCKING_IO_HISTORY_SUFFIX);

        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);
        Constant * const TWO = b.getInt32(2);

        IntegerType * sizeTy = b.getSizeTy();

        Value * const traceLogArrayField = b.CreateGEP(ty, historyPtr, {ZERO, ZERO});
        Value * const traceLogArray = b.CreateLoad(sizeTy->getPointerTo(), traceLogArrayField);

        Value * const traceLogCountField = b.CreateGEP(ty, historyPtr, {ZERO, ONE});
        Value * const traceLogCount = b.CreateLoad(sizeTy, traceLogCountField);

        Value * const traceLogCapacityField = b.CreateGEP(ty, historyPtr, {ZERO, TWO});
        Value * const traceLogCapacity = b.CreateLoad(sizeTy, traceLogCapacityField);

        BasicBlock * const expandHistory = b.CreateBasicBlock(prefix + "_expandHistory", mKernelLoopCall);
        BasicBlock * const recordExpansion = b.CreateBasicBlock(prefix + "_recordExpansion", mKernelLoopCall);

        Value * const hasEnough = b.CreateICmpULT(traceLogCount, traceLogCapacity);

        BasicBlock * const entryBlock = b.GetInsertBlock();
        b.CreateLikelyCondBr(hasEnough, recordExpansion, expandHistory);

        b.SetInsertPoint(expandHistory);
        Constant * const SZ_ONE = b.getSize(1);
        Constant * const SZ_MIN_SIZE = b.getSize(32);
        Value * const newCapacity = b.CreateUMax(b.CreateShl(traceLogCapacity, SZ_ONE), SZ_MIN_SIZE);
        Value * const expandedLogArray = b.CreateRealloc(b.getSizeTy(), traceLogArray, newCapacity);
        assert (expandedLogArray->getType() == traceLogArray->getType());
        b.CreateStore(expandedLogArray, traceLogArrayField);
        b.CreateStore(newCapacity, traceLogCapacityField);
        BasicBlock * const branchExitBlock = b.GetInsertBlock();
        b.CreateBr(recordExpansion);

        b.SetInsertPoint(recordExpansion);
        PHINode * const logArray = b.CreatePHI(traceLogArray->getType(), 2);
        logArray->addIncoming(traceLogArray, entryBlock);
        logArray->addIncoming(expandedLogArray, branchExitBlock);
        b.CreateStore(b.CreateAdd(traceLogCount, b.getSize(1)), traceLogCountField);
        b.CreateStore(mSegNo, b.CreateGEP(sizeTy, logArray, traceLogCount));
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalBlockingIOStatistics
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printOptionalBlockingIOStatistics(KernelBuilder & b) {
    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::EnableBlockingIOCounter))) {

        // Print the title line
        Function * Dprintf = b.GetDprintf();
        FunctionType * fTy = Dprintf->getFunctionType();

        size_t maxKernelLength = 0;
        size_t maxBindingLength = 0;

        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            const Kernel * const kernel = getKernel(i);
            maxKernelLength = std::max(maxKernelLength, kernel->getName().size());
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                const Binding & ref = binding.Binding;
                maxBindingLength = std::max(maxBindingLength, ref.getName().length());
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                const Binding & ref = binding.Binding;
                maxBindingLength = std::max(maxBindingLength, ref.getName().length());
            }
        }
        maxKernelLength += 4;
        maxBindingLength += 4;

        SmallVector<char, 100> buffer;
        raw_svector_ostream line(buffer);

        line << "BLOCKING I/O STATISTICS:\n\n"
                "  # "  // kernel ID #  (only shown for first)
                 "KERNEL"; // kernel Name (only shown for first)
        line.indent(maxKernelLength - 6);
        line << "PORT";
        line.indent(5 + maxBindingLength - 4); // I/O Type (e.g., input port 3 = I3), Port Name
        line << " BUFFER " // buffer ID #
                "   BLOCKED "
                 "     %%\n"; // % of blocking attempts

        Constant * const STDERR = b.getInt32(STDERR_FILENO);

        FixedArray<Value *, 2> titleArgs;
        titleArgs[0] = STDERR;
        titleArgs[1] = b.GetString(line.str());
        b.CreateCall(fTy, Dprintf, titleArgs);

        // Print each kernel line

        // generate first line format string
        buffer.clear();
        line << "%3" PRIu32 " " // kernel #
                "%-" << maxKernelLength << "s" // kernel name
                "%c%-3" PRIu32 " " // I/O type
                "%-" << maxBindingLength << "s" // port name
                "%7" PRIu32 // buffer ID #
                "%11" PRIu64 " " // # of blocked attempts
                "%6.2f\n"; // % of blocking attempts
        Value * const firstLine = b.GetString(line.str());

        // generate remaining lines format string
        buffer.clear();
        line.indent(maxKernelLength + 4); // spaces for kernel #, kernel name
        line << "%c%-3" PRIu32 " " // I/O type
                "%-" << maxBindingLength << "s" // port name
                "%7" PRIu32 // buffer ID #
                "%11" PRIu64 " " // # of blocked attempts
                "%6.2f\n"; // % of blocking attempts
        Value * const remainingLines = b.GetString(line.str());

        SmallVector<Value *, 8> args;

        Type * const doubleTy = b.getDoubleTy();
        Constant * const fOneHundred = ConstantFP::get(doubleTy, 100.0);

        Constant * const I = b.getInt8('I');
        Constant * const O = b.getInt8('O');

        for (auto i = FirstKernel; i <= LastKernel; ++i) {

            const auto prefix = makeKernelName(i);
            // total # of segments processed by kernel
            const auto & lockType = LOGICAL_SEGMENT_SUFFIX[isDataParallel(i) ? SYNC_LOCK_PRE_INVOCATION : SYNC_LOCK_FULL];
            Value * const totalNumberOfSegments = b.getScalarField(prefix + lockType);
            Value * const fTotalNumberOfSegments = b.CreateUIToFP(totalNumberOfSegments, doubleTy);

            const Kernel * const kernel = getKernel(i);
            Constant * kernelName = b.GetString(kernel->getName());

            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {

                const BufferPort & binding = mBufferGraph[e];
                if (binding.canModifySegmentLength()) {
                    args.push_back(STDERR);
                    if (kernelName) {
                        args.push_back(firstLine);
                        args.push_back(b.getInt32(i));
                        args.push_back(kernelName);
                        kernelName = nullptr;
                    } else {
                        args.push_back(remainingLines);
                    }
                    args.push_back(I);
                    const auto inputPort = binding.Port.Number;
                    args.push_back(b.getInt32(inputPort));
                    const Binding & ref = binding.Binding;

                    args.push_back(b.GetString(ref.getName()));

                    args.push_back(b.getInt32(source(e, mBufferGraph)));

                    const auto prefix = makeBufferName(i, binding.Port);
                    Value * const blockedCount = b.getScalarField(prefix + STATISTICS_BLOCKING_IO_SUFFIX);
                    args.push_back(blockedCount);

                    Value * const fBlockedCount = b.CreateUIToFP(blockedCount, doubleTy);
                    Value * const fBlockedCount100 = b.CreateFMul(fBlockedCount, fOneHundred);
                    args.push_back(b.CreateFDiv(fBlockedCount100, fTotalNumberOfSegments));

                    b.CreateCall(fTy, Dprintf, args);

                    args.clear();
                }
            }

            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {

                const BufferPort & binding = mBufferGraph[e];
                if (binding.canModifySegmentLength()) {
                    args.push_back(STDERR);
                    if (kernelName) {
                        args.push_back(firstLine);
                        args.push_back(b.getInt32(i));
                        args.push_back(kernelName);
                        kernelName = nullptr;
                    } else {
                        args.push_back(remainingLines);
                    }
                    args.push_back(O);
                    const auto outputPort = binding.Port.Number;
                    args.push_back(b.getInt32(outputPort));
                    const Binding & ref = binding.Binding;
                    args.push_back(b.GetString(ref.getName()));

                    args.push_back(b.getInt32(target(e, mBufferGraph)));

                    const auto prefix = makeBufferName(i, binding.Port);
                    Value * const blockedCount = b.getScalarField(prefix + STATISTICS_BLOCKING_IO_SUFFIX);
                    args.push_back(blockedCount);

                    Value * const fBlockedCount = b.CreateUIToFP(blockedCount, doubleTy);
                    Value * const fBlockedCount100 = b.CreateFMul(fBlockedCount, fOneHundred);
                    args.push_back(b.CreateFDiv(fBlockedCount100, fTotalNumberOfSegments));

                    b.CreateCall(fTy, Dprintf, args);

                    args.clear();
                }
            }

        }
        // print final new line
        FixedArray<Value *, 2> finalArgs;
        finalArgs[0] = STDERR;
        finalArgs[1] = b.GetString("\n");
        b.CreateCall(fTy, Dprintf, finalArgs);
    }
}


/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printOptionalBlockedIOPerSegment(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceBlockedIO))) {

        IntegerType * const sizeTy = b.getSizeTy();
        PointerType * const sizePtrTy = sizeTy->getPointerTo();

        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);

        ConstantInt * const sz_ZERO = b.getSize(0);
        ConstantInt * const sz_ONE = b.getSize(1);

        Function * Dprintf = b.GetDprintf();
        FunctionType * fTy = Dprintf->getFunctionType();

        // Print the first title line
        SmallVector<char, 100> buffer;
        raw_svector_ostream format(buffer);
        format << "BLOCKED I/O PER SEGMENT:\n\n";
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            unsigned count = 0;
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                if (mBufferGraph[e].canModifySegmentLength()) {
                    ++count;
                }
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                if (mBufferGraph[e].canModifySegmentLength()) {
                    ++count;
                }
            }
            if (count) {
                const Kernel * const kernel = getKernel(i);
                std::string temp = kernel->getName();
                boost::replace_all(temp, "\"", "\\\"");
                format << ",\"" << i << " " << temp << "\"";
                for (unsigned j = 1; j < count; ++j) {
                    format.write(',');
                }
            }
        }
        format << "\n";

        Constant * const STDERR = b.getInt32(STDERR_FILENO);

        FixedArray<Value *, 2> titleArgs;
        titleArgs[0] = STDERR;
        titleArgs[1] = b.GetString(format.str());
        b.CreateCall(fTy, Dprintf, titleArgs);

        // Print the second title line
        buffer.clear();
        format << "SEG #";
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                if (mBufferGraph[e].canModifySegmentLength()) {
                    goto has_ports;
                }
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                if (mBufferGraph[e].canModifySegmentLength()) {
                    goto has_ports;
                }
            }
            continue;
has_ports:
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto & bp = mBufferGraph[e];
                if (bp.canModifySegmentLength()) {
                    format << ",I" << bp.Port.Number << ':' << source(e, mBufferGraph);
                }
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const auto & bp = mBufferGraph[e];
                if (bp.canModifySegmentLength()) {
                    format << ",O" << bp.Port.Number << ':' << source(e, mBufferGraph);
                }
            }
        }
        format << '\n';
        titleArgs[1] = b.GetString(format.str());
        b.CreateCall(fTy, Dprintf, titleArgs);

        // Generate line format string
        buffer.clear();
        format << "%" PRIu64; // seg #
        unsigned fieldCount = 0;
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            unsigned count = 0;
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                if (mBufferGraph[e].canModifySegmentLength()) {
                    ++count;
                }
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                if (mBufferGraph[e].canModifySegmentLength()) {
                    ++count;
                }
            }
            fieldCount += count;
            for (unsigned j = 0; j < count; ++j) {
                format << ",%" PRIu64; // strides
            }
        }
        format << "\n";

        // Print each kernel line
//        SmallVector<Value *, 64> lastLineArgs(fieldCount + 3);
//        lastLineArgs[0] = STDERR;
//        lastLineArgs[1] = b.GetString(format.str());

//        SmallVector<Value *, 64> currentLineArgs(fieldCount + 3);
//        currentLineArgs[0] = STDERR;
//        currentLineArgs[1] = b.GetString(format.str());

        // Pre load all of pointers to the the trace log out of the pipeline state.

        SmallVector<Value *, 64> traceLogArray(fieldCount);
        SmallVector<Value *, 64> traceLengthArray(fieldCount);

        for (unsigned i = FirstKernel, j = 0; i <= LastKernel; ++i) {

            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const auto & bp = mBufferGraph[e];
                if (bp.canModifySegmentLength()) {
                    const auto prefix = makeBufferName(i, bp.Port);
                    Value * historyPtr; Type * historyTy;
                    std::tie(historyPtr, historyTy) = b.getScalarFieldPtr(prefix + STATISTICS_BLOCKING_IO_HISTORY_SUFFIX);
                    assert (j < fieldCount);
                    traceLogArray[j] = b.CreateLoad(sizePtrTy, b.CreateGEP(historyTy, historyPtr, {ZERO, ZERO}));
                    traceLengthArray[j] = b.CreateLoad(sizeTy, b.CreateGEP(historyTy, historyPtr, {ZERO, ONE}));
                    j++;
                }
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const auto & bp = mBufferGraph[e];
                if (bp.canModifySegmentLength()) {
                    const auto prefix = makeBufferName(i, bp.Port);
                    Value * historyPtr; Type * historyTy;
                    std::tie(historyPtr, historyTy) = b.getScalarFieldPtr(prefix + STATISTICS_BLOCKING_IO_HISTORY_SUFFIX);
                    assert (j < fieldCount);
                    traceLogArray[j] = b.CreateLoad(sizePtrTy, b.CreateGEP(historyTy, historyPtr, {ZERO, ZERO}));
                    traceLengthArray[j] = b.CreateLoad(sizeTy, b.CreateGEP(historyTy, historyPtr, {ZERO, ONE}));
                    j++;
                }
            }
        }

        // Start printing the data lines

        // To only print out when the line changes, we store the values for the last line
        // we printed as well as its line number. If the last line was not printed,
        // we print it and the new line out; otherwise we only print out the new line.

        SmallVector<PHINode *, 64> currentIndex(fieldCount);
        SmallVector<Value *, 64> updatedIndex(fieldCount);
        SmallVector<PHINode *, 64> lastPrintedValue(fieldCount);

        BasicBlock * const loopEntry = b.GetInsertBlock();
        BasicBlock * const loopStart = b.CreateBasicBlock("reportStridesPerSegment");
        BasicBlock * const doWriteLine = b.CreateBasicBlock("writeLine");

        BasicBlock * const printLastLine = b.CreateBasicBlock("printLastLine");
        BasicBlock * const printCurrentLine = b.CreateBasicBlock("printCurrentLine");

        BasicBlock * const checkNext = b.CreateBasicBlock("checkNext");
        b.CreateBr(loopStart);

        b.SetInsertPoint(loopStart);
        PHINode * const segNo = b.CreatePHI(sizeTy, 2);
        segNo->addIncoming(sz_ZERO, loopEntry);
        PHINode * const lastPrintedSegNo = b.CreatePHI(sizeTy, 2);
        lastPrintedSegNo->addIncoming(sz_ZERO, loopEntry);

        for (unsigned i = 0; i < fieldCount; ++i) {
            PHINode * const index = b.CreatePHI(sizeTy, 2);
            index->addIncoming(sz_ZERO, loopEntry);
            currentIndex[i] = index;
            PHINode * const val = b.CreatePHI(sizeTy, 2);
            val->addIncoming(sz_ONE, loopEntry);
            lastPrintedValue[i] = val;
        }

        SmallVector<Value *, 64> currentVal(fieldCount + 3);

        Value * writeLine = b.CreateICmpEQ(segNo, mSegNo);
        for (unsigned i = 0; i < fieldCount; ++i) {

            BasicBlock * const entry = b.GetInsertBlock();
            BasicBlock * const check = b.CreateBasicBlock();
            BasicBlock * const update = b.CreateBasicBlock();
            BasicBlock * const next = b.CreateBasicBlock();
            Value * const notEndOfTrace = b.CreateICmpNE(currentIndex[i], traceLengthArray[i]);
            b.CreateLikelyCondBr(notEndOfTrace, check, next);

            b.SetInsertPoint(check);
            Value * const nextSegNo = b.CreateLoad(sizeTy, b.CreateGEP(sizeTy, traceLogArray[i], currentIndex[i]));
            b.CreateCondBr(b.CreateICmpEQ(segNo, nextSegNo), update, next);

            b.SetInsertPoint(update);
            Value * const currentIndex1 = b.CreateAdd(currentIndex[i], sz_ONE);
            b.CreateBr(next);

            b.SetInsertPoint(next);
            PHINode * const nextIndex = b.CreatePHI(sizeTy, 3);
            nextIndex->addIncoming(currentIndex[i], entry);
            nextIndex->addIncoming(currentIndex[i], check);
            nextIndex->addIncoming(currentIndex1, update);
            updatedIndex[i] = nextIndex;

            PHINode * const val = b.CreatePHI(sizeTy, 3);
            val->addIncoming(sz_ZERO, entry);
            val->addIncoming(sz_ZERO, check);
            val->addIncoming(sz_ONE, update);

            currentVal[i] = val;

            lastPrintedValue[i]->addIncoming(val, checkNext);

            Value * const changed = b.CreateICmpNE(val, lastPrintedValue[i]);
            writeLine = b.CreateOr(writeLine, changed);
        }

        Value * const nextSegNo = b.CreateAdd(segNo, sz_ONE);

        BasicBlock * const blockedValueListExit = b.GetInsertBlock();
        b.CreateCondBr(writeLine, doWriteLine, checkNext);

        b.SetInsertPoint(doWriteLine);
        Value * const writeLast = b.CreateICmpNE(lastPrintedSegNo, segNo);
        b.CreateCondBr(writeLast, printLastLine, printCurrentLine);

        b.SetInsertPoint(printLastLine);
        SmallVector<Value *, 64> args(fieldCount + 3);
        args[0] = STDERR;
        args[1] = b.GetString(format.str());
        args[2] = b.CreateSub(segNo, sz_ONE);
        for (unsigned i = 0; i < fieldCount; ++i) {
            args[i + 3] = lastPrintedValue[i];
        }
        b.CreateCall(fTy, Dprintf, args);
        b.CreateBr(printCurrentLine);

        b.SetInsertPoint(printCurrentLine);
        args[2] = segNo;
        for (unsigned i = 0; i < fieldCount; ++i) {
            args[i + 3] = currentVal[i];
        }
        b.CreateCall(fTy, Dprintf, args);
        b.CreateBr(checkNext);

        b.SetInsertPoint(checkNext);
        BasicBlock * const loopExit = b.CreateBasicBlock("reportStridesPerSegmentExit");
        BasicBlock * const loopEnd = b.GetInsertBlock();
        for (unsigned i = 0; i < fieldCount; ++i) {
            currentIndex[i]->addIncoming(updatedIndex[i], loopEnd);
        }
        PHINode * const printedSegNo = b.CreatePHI(sizeTy, 2);
        printedSegNo->addIncoming(nextSegNo, printCurrentLine);
        printedSegNo->addIncoming(lastPrintedSegNo, blockedValueListExit);
        lastPrintedSegNo->addIncoming(printedSegNo, loopEnd);

        segNo->addIncoming(nextSegNo, loopEnd);

        Value * const notDone = b.CreateICmpNE(segNo, mSegNo);
        b.CreateLikelyCondBr(notDone, loopStart, loopExit);

        b.SetInsertPoint(loopExit);
        // print final new line
        FixedArray<Value *, 2> finalArgs;
        finalArgs[0] = STDERR;
        finalArgs[1] = b.GetString("\n");
        b.CreateCall(fTy, Dprintf, finalArgs);
        for (unsigned i = 0; i < fieldCount; ++i) {
            b.CreateFree(traceLogArray[i]);
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeBufferExpansionHistory
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeBufferExpansionHistory(KernelBuilder & b) const {

    if (LLVM_UNLIKELY(mTraceDynamicBuffers)) {

        const auto firstBuffer = PipelineOutput + 1;
        const auto lastBuffer = num_vertices(mBufferGraph);

        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);
        Constant * const TWO = b.getInt32(2);
        Constant * const SZ_ZERO = b.getSize(0);
        Constant * const SZ_ONE = b.getSize(1);



        for (unsigned i = firstBuffer; i < lastBuffer; ++i) {
            const BufferNode & bn = mBufferGraph[i];

            const StreamSetBuffer * const buffer = bn.Buffer; assert (buffer);

            if (buffer->isDynamic()) {
                const auto pe = in_edge(i, mBufferGraph);
                const auto p = source(pe, mBufferGraph);
                const BufferPort & rd = mBufferGraph[pe];
                const auto prefix = makeBufferName(p, rd.Port);

                Value * traceData; Type * traceTy;
                std::tie(traceData, traceTy) = b.getScalarFieldPtr(prefix + STATISTICS_BUFFER_EXPANSION_SUFFIX);

                const auto numOfConsumers = std::max(out_degree(i, mConsumerGraph), 1UL);
                const auto n = numOfConsumers + 3;
                Type * const entryTy = ArrayType::get(b.getSizeTy(), n);

                Value * const entryData = b.CreatePageAlignedMalloc(entryTy, SZ_ONE);
                // fill in the struct
                b.CreateStore(entryData, b.CreateGEP(traceTy, traceData, {ZERO, ZERO}));
                b.CreateStore(SZ_ONE, b.CreateGEP(traceTy, traceData, {ZERO, ONE}));
                // then the initial record
                b.CreateStore(SZ_ZERO, b.CreateGEP(entryTy, entryData, {ZERO, ZERO}));
                b.CreateStore(buffer->getInternalCapacity(b), b.CreateGEP(entryTy, entryData, {ZERO, ONE}));

                unsigned sizeTyWidth = b.getSizeTy()->getIntegerBitWidth() / 8;
                Constant * const length = b.getSize(sizeTyWidth * (n - 2));
                b.CreateMemZero(b.CreateGEP(entryTy, entryData, {ZERO, TWO}), length, sizeTyWidth);

            }
        }

    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordBufferExpansionHistory
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordBufferExpansionHistory(KernelBuilder & b,
                                                    const unsigned streamSet,
                                                    const BufferNode & bn,
                                                    const BufferPort & port,
                                                    const StreamSetBuffer * const buffer) const {

    assert (mTraceDynamicBuffers && buffer->isDynamic());
    const StreamSetPort outputPort = port.Port;
    const auto prefix = makeBufferName(mKernelId, outputPort);

    Value * traceData; Type * traceDataTy;
    std::tie(traceData, traceDataTy) = b.getScalarFieldPtr(prefix + STATISTICS_BUFFER_EXPANSION_SUFFIX);


    Constant * const ZERO = b.getInt32(0);
    Constant * const ONE = b.getInt32(1);
    Constant * const TWO = b.getInt32(2);
    Constant * const THREE = b.getInt32(3);

    IntegerType * const sizeTy = b.getSizeTy();
    const auto numOfConsumers = std::max(out_degree(streamSet, mConsumerGraph), 1UL);
    const auto n = numOfConsumers + 3;
    Type * const entryTy = ArrayType::get(sizeTy, numOfConsumers + 3);

    Value * const traceLogArrayField = b.CreateGEP(traceDataTy, traceData, {ZERO, ZERO});
    Value * entryArray = b.CreateLoad(entryTy->getPointerTo(), traceLogArrayField);

    Value * const traceLogCountField = b.CreateGEP(traceDataTy, traceData, {ZERO, ONE});
    Value * const traceIndex = b.CreateLoad(sizeTy, traceLogCountField);
    Value * const traceCount = b.CreateAdd(traceIndex, b.getSize(1));

    entryArray = b.CreateRealloc(entryTy, entryArray, traceCount);
    b.CreateStore(entryArray, traceLogArrayField);
    b.CreateStore(traceCount, traceLogCountField);

    FixedArray<Value *, 2> indices;
    indices[0] = traceIndex;

    // segment num  0
    indices[1] = ZERO;
    b.CreateStore(mSegNo, b.CreateGEP(entryTy, entryArray, indices));
    // new capacity 1
    indices[1] = ONE;
    b.CreateStore(buffer->getInternalCapacity(b), b.CreateGEP(entryTy, entryArray, indices));
    // produced item count 2
    indices[1] = TWO;
    Value * const produced = mCurrentProducedItemCountPhi[outputPort];
    b.CreateStore(produced, b.CreateGEP(entryTy, entryArray, indices));

    // consumer processed item count [3,n)
    if (LLVM_LIKELY(!bn.isReturned())) {
        const auto id = getTruncatedStreamSetSourceId(streamSet);

        Value * consumerDataPtr; Type * consumerTy;
        std::tie(consumerDataPtr, consumerTy) = b.getScalarFieldPtr(CONSUMED_ITEM_COUNT_PREFIX + std::to_string(id));
        Value * const processedPtr = b.CreateGEP(consumerTy, consumerDataPtr, { ZERO, ONE });
        Value * const logPtr = b.CreateGEP(entryTy, entryArray, {traceIndex, THREE});
        unsigned sizeTyWidth = b.getSizeTy()->getIntegerBitWidth() / 8;
        Constant * const length = b.getSize(sizeTyWidth * (n - 3));
        b.CreateMemCpy(logPtr, processedPtr, length, sizeTyWidth);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalBufferExpansionHistory
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printOptionalBufferExpansionHistory(KernelBuilder & b) {

    if (LLVM_UNLIKELY(mTraceDynamicBuffers)) {

        // Print the title line
        Function * Dprintf = b.GetDprintf();
        FunctionType * fTy = Dprintf->getFunctionType();

        size_t maxKernelLength = 0;
        size_t maxBindingLength = 0;
        for (auto i = FirstKernel; i <= LastKernel; ++i) {
            const Kernel * const kernel = getKernel(i);
            maxKernelLength = std::max(maxKernelLength, kernel->getName().size());
            for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                const Binding & ref = binding.Binding;
                maxBindingLength = std::max(maxBindingLength, ref.getName().length());
            }
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const BufferPort & binding = mBufferGraph[e];
                const Binding & ref = binding.Binding;
                maxBindingLength = std::max(maxBindingLength, ref.getName().length());
            }
        }
        maxKernelLength += 4;
        maxBindingLength += 4;

        SmallVector<char, 160> buffer;
        raw_svector_ostream format(buffer);

        // TODO: if expanding buffers are supported again, we need another field here for streamset size

        format << "BUFFER EXPANSION HISTORY:\n\n"
                  "  # "  // kernel ID #  (only shown for first)
                  "KERNEL"; // kernel Name (only shown for first)
        format.indent(maxKernelLength - 6);
        format << "PORT";
        format.indent(5 + maxBindingLength - 4); // I/O Type (e.g., input port 3 = I3), Port Name
        format << " BUFFER " // buffer ID #
                  "        SEG # "
                  "     ITEM CAPACITY\n";

        Constant * const STDERR = b.getInt32(STDERR_FILENO);
        FixedArray<Value *, 2> constantArgs;
        constantArgs[0] = STDERR;
        constantArgs[1] = b.GetString(format.str());
        b.CreateCall(fTy, Dprintf, constantArgs);

        const auto totalLength = 4 + maxKernelLength + 4 + maxBindingLength + 7 + 15 + 18 + 2;

        // generate a single-line (-) bar
        buffer.clear();
        for (unsigned i = 0; i < totalLength; ++i) {
            format.write('-');
        }
        format.write('\n');
        Constant * const singleBar = b.GetString(format.str());
        constantArgs[1] = singleBar;
        b.CreateCall(fTy, Dprintf, constantArgs);


        // generate the produced/processed title line
        buffer.clear();
        format.indent(totalLength - 19);
        format << "PRODUCED/PROCESSED\n";
        constantArgs[1] = b.GetString(format.str());
        b.CreateCall(fTy, Dprintf, constantArgs);

        // generate a double-line (=) bar
        buffer.clear();
        for (unsigned i = 0; i < totalLength; ++i) {
            format.write('=');
        }
        format.write('\n');
        Constant * const doubleBar = b.GetString(format.str());
        constantArgs[1] = doubleBar;
        b.CreateCall(fTy, Dprintf, constantArgs);

        // Generate expansion line format string
        buffer.clear();
        format << "%3" PRIu32 " " // kernel #
                  "%-" << maxKernelLength << "s" // kernel name
                  "O%-3" PRIu32 " " // I/O type
                  "%-" << maxBindingLength << "s" // port name
                  "%7" PRIu32 // buffer ID #
                  "%14" PRIu64 " " // segment #
                  "%18" PRIu64 "\n"; // item capacity
        Constant * const expansionFormat = b.GetString(format.str());

        // Generate the item count history format string
        buffer.clear();
        format << "%3" PRIu32 " " // kernel #
                  "%-" << maxKernelLength << "s" // kernel name
                  "%c%-3" PRIu32 " " // I/O type
                  "%-" << maxBindingLength << "s" // port name
                  "%40" PRIu64 "\n"; // produced/processed item count
        Constant * const itemCountFormat = b.GetString(format.str());

        // Print each kernel line
        FixedArray<Value *, 9> expansionArgs;
        expansionArgs[0] = STDERR;
        expansionArgs[1] = expansionFormat;

        FixedArray<Value *, 8> itemCountArgs;
        itemCountArgs[0] = STDERR;
        itemCountArgs[1] = itemCountFormat;

        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);
        Constant * const TWO = b.getInt32(2);

        Constant * const SZ_ZERO = b.getSize(0);
        Constant * const SZ_ONE = b.getSize(1);

        IntegerType * sizeTy = b.getSizeTy();

        for (auto i = FirstKernel; i <= LastKernel; ++i) {

            for (const auto output : make_iterator_range(out_edges(i, mBufferGraph))) {
                const BufferPort & br = mBufferGraph[output];
                const auto buffer = target(output, mBufferGraph);
                const BufferNode & bn = mBufferGraph[buffer];
                if (bn.Buffer->isDynamic()) {

                    //  # KERNEL                      PORT                      BUFFER         SEG #      ITEM CAPACITY

                    expansionArgs[2] = b.getInt32(i);
                    expansionArgs[3] = b.GetString(getKernel(i)->getName());
                    const auto outputPort = br.Port.Number;
                    expansionArgs[4] = b.getInt32(outputPort);
                    const Binding & binding = br.Binding;
                    expansionArgs[5] = b.GetString(binding.getName());
                    expansionArgs[6] = b.getInt32(buffer);

                    const auto prefix = makeBufferName(i, br.Port);

                    Value * traceData; Type * traceTy;
                    std::tie(traceData, traceTy) = b.getScalarFieldPtr(prefix + STATISTICS_BUFFER_EXPANSION_SUFFIX);

                    Value * const traceArrayField = b.CreateGEP(traceTy, traceData, {ZERO, ZERO});
                    const auto numOfConsumers = std::max(out_degree(buffer, mConsumerGraph), 1UL);

                    Type * const arrayTy = ArrayType::get(sizeTy, numOfConsumers + 3);
                    Value * const entryArray = b.CreateLoad(arrayTy->getPointerTo(), traceArrayField);

                    Value * const traceCountField = b.CreateGEP(traceTy, traceData, {ZERO, ONE});
                    Value * const traceCount = b.CreateLoad(sizeTy, traceCountField);

                    BasicBlock * const outputEntry = b.GetInsertBlock();
                    BasicBlock * const outputLoop = b.CreateBasicBlock(prefix + "_bufferExpansionReportLoop");
                    BasicBlock * const writeItemCount = b.CreateBasicBlock(prefix + "_bufferWriteItemCountLoop");

                    BasicBlock * const nextEntry = b.CreateBasicBlock(prefix + "_bufferWriteItemCountLoop");

                    BasicBlock * const outputExit = b.CreateBasicBlock(prefix + "_bufferExpansionReportExit");

                    b.CreateBr(outputLoop);

                    b.SetInsertPoint(outputLoop);
                    PHINode * const index = b.CreatePHI(b.getSizeTy(), 2);
                    index->addIncoming(SZ_ZERO, outputEntry);

                    Value * const isFirst = b.CreateICmpEQ(index, SZ_ZERO);
                    Value * const nextIndex = b.CreateAdd(index, SZ_ONE);
                    Value * const isLast = b.CreateICmpEQ(nextIndex, traceCount);
                    Value * const onlyEntry = b.CreateICmpEQ(traceCount, SZ_ONE);

                    Value * const openingBar = b.CreateSelect(onlyEntry, doubleBar, singleBar);

                    Value * const segmentNumField = b.CreateGEP(arrayTy, entryArray, {index, ZERO});
                    Value * const segmentNum = b.CreateLoad(sizeTy, segmentNumField);
                    expansionArgs[7] = segmentNum;

                    Value * const newBufferSizeField = b.CreateGEP(arrayTy, entryArray, {index, ONE});
                    Value * const newBufferSize = b.CreateLoad(sizeTy, newBufferSizeField);
                    expansionArgs[8] = newBufferSize;

                    b.CreateCall(fTy, Dprintf, expansionArgs);

                    constantArgs[1] = openingBar;

                    b.CreateCall(fTy, Dprintf, constantArgs);

                    b.CreateCondBr(isFirst, nextEntry, writeItemCount);

                    // Do not write processed/produced item counts for the first entry as they
                    // are guaranteed to be 0 and only add noise to the log output.
                    b.SetInsertPoint(writeItemCount);

                    // --------------------------------------------------------------------------------------------------

                    //  # KERNEL                      PORT                                           PRODUCED/PROCESSED

                    itemCountArgs[2] = expansionArgs[2];
                    itemCountArgs[3] = expansionArgs[3];
                    itemCountArgs[4] = b.getInt8('O');
                    itemCountArgs[5] = expansionArgs[4];
                    itemCountArgs[6] = expansionArgs[5];
                    Value * const producedField = b.CreateGEP(arrayTy, entryArray, {index, TWO});
                    itemCountArgs[7] = b.CreateLoad(sizeTy, producedField);

                    b.CreateCall(fTy, Dprintf, itemCountArgs);
                    itemCountArgs[4] = b.getInt8('I');
                    for (const auto e : make_iterator_range(out_edges(buffer, mConsumerGraph))) {
                        const ConsumerEdge & c = mConsumerGraph[e];
                        const auto consumer = target(e, mConsumerGraph);
                        itemCountArgs[2] = b.getInt32(consumer);
                        itemCountArgs[3] = b.GetString(getKernel(consumer)->getName());
                        itemCountArgs[5] = b.getInt32(c.Port);
                        const Binding & binding = getBinding(consumer, StreamSetPort{PortType::Input, c.Port});
                        itemCountArgs[6] = b.GetString(binding.getName());
                        const auto k = c.Index + 2; assert (k > 2);
                        Value * const processedField = b.CreateGEP(arrayTy, entryArray, {index, b.getInt32(k)});
                        itemCountArgs[7] = b.CreateLoad(sizeTy, processedField);
                        b.CreateCall(fTy, Dprintf, itemCountArgs);
                    }

                    Value * const closingBar = b.CreateSelect(isLast, doubleBar, singleBar);
                    constantArgs[1] = closingBar;
                    b.CreateCall(fTy, Dprintf, constantArgs);

                    b.CreateBr(nextEntry);

                    b.SetInsertPoint(nextEntry);
                    index->addIncoming(nextIndex, nextEntry);
                    b.CreateCondBr(isLast, outputExit, outputLoop);

                    b.SetInsertPoint(outputExit);
                    b.CreateFree(entryArray);
                }
            }
        }
        // print final new line
        constantArgs[1] = doubleBar;
        b.CreateCall(fTy, Dprintf, constantArgs);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief initializeStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::initializeStridesPerSegment(KernelBuilder & b) const {

    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceStridesPerSegment))) {

        const auto prefix = makeKernelName(mKernelId);

        Value * traceData; Type * traceDataTy;

        std::tie(traceData, traceDataTy) = b.getScalarFieldPtr(prefix + STATISTICS_STRIDES_PER_SEGMENT_SUFFIX);
        Type * const traceLogTy =  ArrayType::get(b.getSizeTy(), 2);

        Constant * const SZ_ZERO = b.getSize(0);
        Constant * const SZ_DEFAULT_CAPACITY = b.getSize(4096 / (sizeof(size_t) * 2));
        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);
        Constant * const TWO = b.getInt32(2);
        Constant * const THREE = b.getInt32(3);
        Constant * const MAX_INT = ConstantInt::getAllOnesValue(b.getSizeTy());

        Value * const traceDataArray = b.CreatePageAlignedMalloc(traceLogTy, SZ_DEFAULT_CAPACITY);

        // fill in the struct
        b.CreateStore(MAX_INT, b.CreateGEP(traceDataTy, traceData, {ZERO, ZERO})); // "last" num of strides
        b.CreateStore(traceDataArray, b.CreateGEP(traceDataTy, traceData, {ZERO, ONE})); // trace log
        b.CreateStore(SZ_ZERO, b.CreateGEP(traceDataTy, traceData, {ZERO, TWO})); // trace length
        b.CreateStore(SZ_DEFAULT_CAPACITY, b.CreateGEP(traceDataTy, traceData, {ZERO, THREE})); // trace capacity
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordStridesPerSegment(KernelBuilder & b, const unsigned kernelId, Value * const totalStrides) const {

    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceStridesPerSegment))) {
        // NOTE: this records only the change to attempt to reduce the memory usage of this log.
        assert (KernelPartitionId[kernelId - 1] != KernelPartitionId[kernelId]);
        const auto prefix = makeKernelName(kernelId);

        Value * toUpdate; Type * traceTy;

        std::tie(toUpdate, traceTy) = b.getScalarFieldPtr(prefix + STATISTICS_STRIDES_PER_SEGMENT_SUFFIX);

        Module * const m = b.getModule();
        Function * updateSegmentsPerStrideTrace = m->getFunction("$trace_strides_per_segment");
        if (LLVM_UNLIKELY(updateSegmentsPerStrideTrace == nullptr)) {
            auto ip = b.saveIP();
            LLVMContext & C = b.getContext();
            Type * const sizeTy = b.getSizeTy();
            Type * const tracePtrTy = traceTy->getPointerTo();
            FunctionType * fty = FunctionType::get(b.getVoidTy(), { tracePtrTy, sizeTy, sizeTy }, false);
            updateSegmentsPerStrideTrace = Function::Create(fty, Function::PrivateLinkage, "$trace_strides_per_segment", m);

            BasicBlock * const entry = BasicBlock::Create(C, "entry", updateSegmentsPerStrideTrace);
            BasicBlock * const update = BasicBlock::Create(C, "update", updateSegmentsPerStrideTrace);
            BasicBlock * const expand = BasicBlock::Create(C, "expand", updateSegmentsPerStrideTrace);
            BasicBlock * const write = BasicBlock::Create(C, "write", updateSegmentsPerStrideTrace);
            BasicBlock * const exit = BasicBlock::Create(C, "exit", updateSegmentsPerStrideTrace);

            b.SetInsertPoint(entry);

            auto arg = updateSegmentsPerStrideTrace->arg_begin();
            arg->setName("traceData");
            Value * traceData = &*arg++;
            arg->setName("segNo");
            Value * const segNo = &*arg++;
            arg->setName("numOfStrides");
            Value * const numOfStrides = &*arg++;
            assert (arg == updateSegmentsPerStrideTrace->arg_end());

//            Type * const recordStructTy = ArrayType::get(sizeTy, 2);

//            FixedArray<Type *, 4> traceStruct;
//            traceStruct[0] = sizeTy; // last num of strides (to avoid unnecessary loads of the trace
//                                     // log and simplify the logic for first stride)
//            traceStruct[1] = recordStructTy->getPointerTo(); // pointer to trace log
//            traceStruct[2] = sizeTy; // trace length
//            traceStruct[3] = sizeTy; // trace capacity (for realloc)

            Constant * const ZERO = b.getInt32(0);
            Constant * const ONE = b.getInt32(1);
            Constant * const TWO = b.getInt32(2);
            Constant * const THREE = b.getInt32(3);

            Value * const lastNumOfStridesField = b.CreateGEP(traceTy, traceData, {ZERO, ZERO});
            Value * const lastNumOfStrides = b.CreateLoad(sizeTy, lastNumOfStridesField);
            Value * const changed = b.CreateICmpNE(lastNumOfStrides, numOfStrides);
            b.CreateCondBr(changed, update, exit);

            b.SetInsertPoint(update);
            Value * const traceLogField = b.CreateGEP(traceTy, traceData, {ZERO, ONE});
            assert (traceLogField->getType()->isPointerTy());
            Type * const recordStructTy = ArrayType::get(sizeTy, 2);
            Type * const recordStructPtrTy = recordStructTy->getPointerTo();
            Value * const traceLog = b.CreateLoad(recordStructPtrTy, traceLogField);
            Value * const traceLengthField = b.CreateGEP(traceTy, traceData, {ZERO, TWO});
            Value * const traceLength = b.CreateLoad(sizeTy, traceLengthField);
            Value * const traceCapacityField = b.CreateGEP(traceTy, traceData, {ZERO, THREE});
            assert (traceCapacityField->getType()->isPointerTy());
            Value * const traceCapacity = b.CreateLoad(sizeTy, traceCapacityField);
            Value * const hasSpace = b.CreateICmpNE(traceLength, traceCapacity);
            b.CreateLikelyCondBr(hasSpace, write, expand);

            b.SetInsertPoint(expand);
            Value * const nextTraceCapacity = b.CreateShl(traceCapacity, 1);
            assert (traceLog->getType()->isPointerTy());
            Value * const expandedtraceLog = b.CreateRealloc(recordStructTy, traceLog, nextTraceCapacity);
            assert (expandedtraceLog->getType() == traceLog->getType());
            b.CreateStore(expandedtraceLog, traceLogField);
            b.CreateStore(nextTraceCapacity, traceCapacityField);
            b.CreateBr(write);

            b.SetInsertPoint(write);
            PHINode * const traceLogPhi = b.CreatePHI(recordStructPtrTy, 2);
            traceLogPhi->addIncoming(traceLog, update);
            traceLogPhi->addIncoming(expandedtraceLog, expand);
            b.CreateStore(segNo, b.CreateGEP(recordStructTy, traceLogPhi, {traceLength , ZERO}));
            b.CreateStore(numOfStrides, b.CreateGEP(recordStructTy, traceLogPhi, {traceLength , ONE}));
            b.CreateStore(numOfStrides, lastNumOfStridesField);
            Value * const newTraceLength = b.CreateAdd(traceLength, b.getSize(1));
            b.CreateStore(newTraceLength, traceLengthField);

            b.CreateBr(exit);

            b.SetInsertPoint(exit);
            b.CreateRetVoid();

            b.restoreIP(ip);
        }


        FixedArray<Value *, 3> args;
        args[0] = toUpdate;
        args[1] = mSegNo;
        args[2] = totalStrides;
        b.CreateCall(updateSegmentsPerStrideTrace, args);

    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief concludeStridesPerSegmentRecording
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::concludeStridesPerSegmentRecording(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceStridesPerSegment))) {
        auto currentPartitionId = KernelPartitionId[PipelineInput];
        Constant * const sz_ZERO = b.getSize(0);
        for (auto kernelId = FirstKernel; kernelId <= LastKernel; ++kernelId) {
            const auto partitionId = KernelPartitionId[kernelId];
            if (partitionId != currentPartitionId) {
                currentPartitionId = partitionId;
                recordStridesPerSegment(b, kernelId, sz_ZERO);
            }
        }
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printOptionalStridesPerSegment(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(DebugOptionIsSet(codegen::TraceStridesPerSegment))) {

        IntegerType * const sizeTy = b.getSizeTy();

        Constant * const ZERO = b.getInt32(0);
        Constant * const ONE = b.getInt32(1);
        Constant * const TWO = b.getInt32(2);

        ConstantInt * const SZ_ZERO = b.getSize(0);
        ConstantInt * const SZ_ONE = b.getSize(1);

        // Print the title line
        Function * Dprintf = b.GetDprintf();
        FunctionType * fTy = Dprintf->getFunctionType();

        SmallVector<char, 1024> buffer;
        raw_svector_ostream format(buffer);
        format << "STRIDES PER SEGMENT:\n\n"
                  "SEG #";

        SmallVector<unsigned, 64> partitionRootIds;
        partitionRootIds.reserve(PartitionCount);
        for (unsigned kernel = FirstKernel, currentPartitionId = 0; kernel <= LastKernel; ++kernel) {
            const auto partitionId = KernelPartitionId[kernel];
            if (currentPartitionId != partitionId) {
                currentPartitionId = partitionId;
                partitionRootIds.push_back(kernel);
            }
        }
        assert (partitionRootIds.size() == (PartitionCount - 2));

        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            const Kernel * const kernel = getKernel(partitionRootIds[i]);
            std::string temp = kernel->getName();
            boost::replace_all(temp, "\"", "\\\"");
            format << ",\"" << partitionRootIds[i] << "." << temp << "\"";
        }
        format << "\n";


        Constant * const STDERR = b.getInt32(STDERR_FILENO);

        FixedArray<Value *, 2> titleArgs;
        titleArgs[0] = STDERR;
        titleArgs[1] = b.GetString(format.str());
        b.CreateCall(fTy, Dprintf, titleArgs);

        // Generate line format string
        buffer.clear();

        format << "%" PRIu64; // seg #
        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            format << ",%" PRIu64; // strides
        }
        format << "\n";

        // Print each kernel line
        SmallVector<Value *, 64> args((PartitionCount - 2) + 3);
        args[0] = STDERR;
        args[1] = b.GetString(format.str());

        // Pre load all of pointers to the the trace log out of the pipeline state.

        SmallVector<Value *, 64> traceLogArray(PartitionCount - 1);
        SmallVector<Value *, 64> traceLengthArray(PartitionCount - 1);

        Type * const recordStructTy = ArrayType::get(sizeTy, 2);
        Type * const recordStructPtrTy = recordStructTy->getPointerTo();

        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            const auto prefix = makeKernelName(partitionRootIds[i]);
            Value * traceData; Type * traceTy;
            std::tie(traceData, traceTy) = b.getScalarFieldPtr(prefix + STATISTICS_STRIDES_PER_SEGMENT_SUFFIX);
            traceLogArray[i] = b.CreateLoad(recordStructPtrTy, b.CreateGEP(traceTy, traceData, {ZERO, ONE}));
            traceLengthArray[i] = b.CreateLoad(sizeTy, b.CreateGEP(traceTy, traceData, {ZERO, TWO}));
        }

        // Start printing the data lines

        BasicBlock * const loopEntry = b.GetInsertBlock();
        BasicBlock * const loopStart = b.CreateBasicBlock("reportStridesPerSegment");
        b.CreateBr(loopStart);

        b.SetInsertPoint(loopStart);
        PHINode * const segNo = b.CreatePHI(sizeTy, 2);
        segNo->addIncoming(SZ_ZERO, loopEntry);
        SmallVector<PHINode *, 64> currentIndex(PartitionCount - 1);
        SmallVector<PHINode *, 64> updatedIndex(PartitionCount - 1);
        SmallVector<PHINode *, 64> currentValue(PartitionCount - 1);
        SmallVector<PHINode *, 64> updatedValue(PartitionCount - 1);

        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            currentIndex[i] = b.CreatePHI(sizeTy, 2);
            currentIndex[i]->addIncoming(SZ_ZERO, loopEntry);
            currentValue[i] = b.CreatePHI(sizeTy, 2);
            currentValue[i]->addIncoming(SZ_ZERO, loopEntry);
        }

        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            const auto prefix = makeKernelName(partitionRootIds[i]);

            BasicBlock * const entry = b.GetInsertBlock();
            BasicBlock * const check = b.CreateBasicBlock();
            BasicBlock * const update = b.CreateBasicBlock();
            BasicBlock * const next = b.CreateBasicBlock();
            Value * const notEndOfTrace = b.CreateICmpNE(currentIndex[i], traceLengthArray[i]);
            b.CreateLikelyCondBr(notEndOfTrace, check, next);

            b.SetInsertPoint(check);
            Value * const nextSegNo = b.CreateLoad(sizeTy, b.CreateGEP(recordStructTy, traceLogArray[i], { currentIndex[i], ZERO }));
            b.CreateCondBr(b.CreateICmpEQ(segNo, nextSegNo), update, next);

            b.SetInsertPoint(update);
            Value * const numOfStrides = b.CreateLoad(sizeTy, b.CreateGEP(recordStructTy, traceLogArray[i], { currentIndex[i], ONE }));
            Value * const currentIndex1 = b.CreateAdd(currentIndex[i], SZ_ONE);
            b.CreateBr(next);

            b.SetInsertPoint(next);
            PHINode * const nextValue = b.CreatePHI(sizeTy, 3);
            nextValue->addIncoming(SZ_ZERO, entry);
            nextValue->addIncoming(currentValue[i], check);
            nextValue->addIncoming(numOfStrides, update);
            updatedValue[i] = nextValue;

            PHINode * const nextIndex = b.CreatePHI(sizeTy, 3);
            nextIndex->addIncoming(currentIndex[i], entry);
            nextIndex->addIncoming(currentIndex[i], check);
            nextIndex->addIncoming(currentIndex1, update);
            updatedIndex[i] = nextIndex;
        }

        args[2] = segNo;
        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            args[i + 3] = updatedValue[i];
        }

        b.CreateCall(fTy, Dprintf, args);

        BasicBlock * const loopExit = b.CreateBasicBlock("reportStridesPerSegmentExit");
        BasicBlock * const loopEnd = b.GetInsertBlock();
        segNo->addIncoming(b.CreateAdd(segNo, SZ_ONE), loopEnd);

        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            currentIndex[i]->addIncoming(updatedIndex[i], loopEnd);
            currentValue[i]->addIncoming(updatedValue[i], loopEnd);
        }

        Value * const notDone = b.CreateICmpULT(segNo, mSegNo);

        b.CreateLikelyCondBr(notDone, loopStart, loopExit);

        b.SetInsertPoint(loopExit);
        // print final new line
        FixedArray<Value *, 2> finalArgs;
        finalArgs[0] = STDERR;
        finalArgs[1] = b.GetString("\n");
        b.CreateCall(fTy, Dprintf, finalArgs);
        for (unsigned i = 0; i < (PartitionCount - 2); ++i) {
            b.CreateFree(traceLogArray[i]);
        }
    }
}

#define ITEM_COUNT_DELTA_CHUNK_LENGTH (64 * 1024)

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addProducedItemCountDeltaProperties(KernelBuilder & b, unsigned kernel) const {
    if (LLVM_UNLIKELY(TraceProducedItemCounts)) {
        addItemCountDeltaProperties(b, kernel, STATISTICS_PRODUCED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordProducedItemCountDeltas(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(TraceProducedItemCounts)) {
        const auto n = out_degree(mKernelId, mBufferGraph);
        if (LLVM_UNLIKELY(n == 0)) {
            return;
        }
        Vec<Value *> current(n);
        Vec<Value *> prior(n);
        for (unsigned i = 0; i != n; ++i) {
            const StreamSetPort port(PortType::Output, i);
            const auto streamSet = getOutputBufferVertex(port);
            current[i] = mFullyProducedItemCount[port];
            prior[i] = mInitiallyProducedItemCount[streamSet];
        }
        recordItemCountDeltas(b, current, prior, STATISTICS_PRODUCED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printProducedItemCountDeltas(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(TraceProducedItemCounts)) {
        printItemCountDeltas(b, "PRODUCED ITEM COUNT DELTAS PER SEGMENT:", STATISTICS_PRODUCED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addUnconsumedItemCountProperties(KernelBuilder & b, unsigned kernel) const {
    if (LLVM_UNLIKELY(TraceUnconsumedItemCounts)) {
        addItemCountDeltaProperties(b, kernel, STATISTICS_UNCONSUMED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordUnconsumedItemCounts(KernelBuilder & b) {
    if (LLVM_UNLIKELY(TraceUnconsumedItemCounts)) {
        const auto n = out_degree(mKernelId, mBufferGraph);
        if (LLVM_UNLIKELY(n == 0)) {
            return;
        }
        Vec<Value *> current(n);
        Vec<Value *> prior(n);
        for (unsigned i = 0; i != n; ++i) {
            const StreamSetPort port(PortType::Output, i);
            const auto streamSet = getOutputBufferVertex(port);
            current[i] = mInitiallyProducedItemCount[streamSet];
            prior[i] = readConsumedItemCount(b, streamSet);
        }
        recordItemCountDeltas(b, current, prior, STATISTICS_UNCONSUMED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printOptionalStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printUnconsumedItemCounts(KernelBuilder & b) const {
    if (LLVM_UNLIKELY(TraceUnconsumedItemCounts)) {
        printItemCountDeltas(b, "UNCONSUMED ITEM COUNTS PER SEGMENT:", STATISTICS_UNCONSUMED_ITEM_COUNT_SUFFIX);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordStridesPerSegment
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::recordItemCountDeltas(KernelBuilder & b,
                                             const Vec<Value *> & current,
                                             const Vec<Value *> & prior,
                                             const StringRef suffix) const {
    const auto kernelName = makeKernelName(mKernelId);
    const auto fieldName = (kernelName + suffix).str();



    Value * trace = b.getScalarFieldPtr(fieldName).first;

    auto & C = b.getContext();
    IntegerType * const sizeTy = b.getSizeTy();
    PointerType * const voidPtrTy = b.getVoidPtrTy();
    const auto n = out_degree(mKernelId, mBufferGraph);
    ArrayType * const logTy = ArrayType::get(ArrayType::get(sizeTy, n), ITEM_COUNT_DELTA_CHUNK_LENGTH);
    StructType * const logChunkTy = StructType::get(C, { logTy, voidPtrTy } );
    PointerType * const traceTy = logChunkTy->getPointerTo();

    BasicBlock * const expand = b.CreateBasicBlock(kernelName + "_expandItemCountDeltArray");
    BasicBlock * const record = b.CreateBasicBlock(kernelName + "_recordItemCountDelta");

    Value * const offset = b.CreateURem(mSegNo, b.getSize(ITEM_COUNT_DELTA_CHUNK_LENGTH));
    Constant * const ZERO = b.getInt32(0);
    Constant * const ONE = b.getInt32(1);

    Value * const currentLog = b.CreateLoad(traceTy, trace);
    BasicBlock * const entry = b.GetInsertBlock();
    b.CreateCondBr(b.CreateIsNull(offset), expand, record);

    b.SetInsertPoint(expand);
    Value * const newLog = b.CreatePageAlignedMalloc(logChunkTy);
    b.CreateStore(b.CreatePointerCast(currentLog, voidPtrTy), b.CreateGEP(logChunkTy, newLog, { ZERO, ONE}));
    b.CreateStore(newLog, trace);
    BasicBlock * const expandExit = b.GetInsertBlock();
    b.CreateBr(record);

    b.SetInsertPoint(record);
    PHINode * const log = b.CreatePHI(logTy->getPointerTo(), 2);
    log->addIncoming(currentLog, entry);
    log->addIncoming(newLog, expandExit);

    FixedArray<Value *, 4> indices;
    indices[0] = ZERO;
    indices[1] = ZERO;
    indices[2] = offset;

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        const BufferPort & out = mBufferGraph[e];
        const auto i = out.Port.Number;
        Value * const delta = b.CreateSub(current[i], prior[i]);
        indices[3] = b.getInt32(i);
        b.CreateStore(delta, b.CreateGEP(logChunkTy, log, indices));
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addItemCountDeltaProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addItemCountDeltaProperties(KernelBuilder & b, const unsigned kernel, const StringRef suffix) const {
    const auto n = out_degree(kernel, mBufferGraph);
    LLVMContext & C = b.getContext();
    IntegerType * const sizeTy = b.getSizeTy();
    PointerType * const voidPtrTy = b.getVoidPtrTy();
    ArrayType * const logTy = ArrayType::get(ArrayType::get(sizeTy, n), ITEM_COUNT_DELTA_CHUNK_LENGTH);
    StructType * const logChunkTy = StructType::get(C, { logTy, voidPtrTy } );
    PointerType * const traceTy = logChunkTy->getPointerTo();
    const auto fieldName = (makeKernelName(kernel) + suffix).str();
    const auto groupId = getCacheLineGroupId(kernel);
    mTarget->addInternalScalar(traceTy, fieldName, groupId);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printItemCountDeltas
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printItemCountDeltas(KernelBuilder & b, const StringRef title, const StringRef suffix) const {
    IntegerType * const sizeTy = b.getSizeTy();

    Constant * const ZERO = b.getInt32(0);
    Constant * const ONE = b.getInt32(1);

    ConstantInt * const SZ_ZERO = b.getSize(0);
    ConstantInt * const SZ_ONE = b.getSize(1);


    // Print the title line
    Function * Dprintf = b.GetDprintf();
    FunctionType * fTy = Dprintf->getFunctionType();

    SmallVector<char, 100> buffer;
    raw_svector_ostream format(buffer);
    format << title << "\n\n"
              "SEG #";

    for (auto i = FirstKernel; i <= LastKernel; ++i) {

        const Kernel * const kernel = getKernel(i);
        std::string kernelName = kernel->getName();
        boost::replace_all(kernelName, "\"", "\\\"");

        for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & out = br.Binding;
            format << "," << target(e, mBufferGraph)
                   << " " << kernelName
                   << '.' << out.getName();
        }
    }

    format << "\n";


    Constant * const STDERR = b.getInt32(STDERR_FILENO);

    FixedArray<Value *, 2> titleArgs;
    titleArgs[0] = STDERR;
    titleArgs[1] = b.GetString(format.str());
    b.CreateCall(fTy, Dprintf, titleArgs);

    // Generate line format string
    buffer.clear();

    const auto inputStreamSets = out_degree(PipelineInput, mBufferGraph);

    format << "%" PRIu64; // seg #
    for (auto i = FirstStreamSet + inputStreamSets; i <= LastStreamSet; ++i) {
        format << ",%" PRIu64; // processed item count delta
    }
    format << "\n";

    // Print each kernel line
    SmallVector<Value *, 64> args(LastStreamSet - (FirstStreamSet + inputStreamSets) + 4);
    args[0] = STDERR;
    args[1] = b.GetString(format.str());

    SmallVector<Value *, 64> traceLogArray(LastKernel + 1);
    SmallVector<StructType *, 64> traceLogType(LastKernel + 1);

    LLVMContext & C = b.getContext();
    PointerType * const voidPtrTy = b.getVoidPtrTy();

    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        const auto n = out_degree(i, mBufferGraph);
        if (LLVM_UNLIKELY(n == 0)) {
            traceLogArray[i] = nullptr;
            traceLogType[i] = nullptr;
        } else {
            const auto fieldName = (makeKernelName(i) + suffix).str();
            traceLogArray[i] = b.getScalarFieldPtr(fieldName).first;
            ArrayType * const logTy = ArrayType::get(ArrayType::get(sizeTy, n), ITEM_COUNT_DELTA_CHUNK_LENGTH);
            traceLogType[i] = StructType::get(C, { logTy, voidPtrTy } );
        }
    }

    Constant * const CHUNK_LENGTH = b.getSize(ITEM_COUNT_DELTA_CHUNK_LENGTH);
    Value * const chunkCount = b.CreateCeilUDiv(mSegNo, CHUNK_LENGTH);
    if (LLVM_UNLIKELY(CheckAssertions)) {
        b.CreateAssert(mSegNo, "final segment number cannot be zero");
    }

    // Start printing the data lines
    BasicBlock * const loopEntry = b.GetInsertBlock();
    BasicBlock * const loopStart = b.CreateBasicBlock("reportStridesPerSegment");
    BasicBlock * const getNextLogChunk = b.CreateBasicBlock("getNextLogChunk");
    BasicBlock * const selectLogChunk = b.CreateBasicBlock("getNextLogChunk");
    BasicBlock * const printLogEntry = b.CreateBasicBlock("getNextLogChunk");
    b.CreateBr(loopStart);

    b.SetInsertPoint(loopStart);
    PHINode * const segNo = b.CreatePHI(sizeTy, 2);
    segNo->addIncoming(SZ_ZERO, loopEntry);
    PHINode * const chunkIndex = b.CreatePHI(sizeTy, 2);
    chunkIndex->addIncoming(chunkCount, loopEntry);
    SmallVector<PHINode *, 64> currentChunk(LastKernel + 1);
    for (auto i = PipelineInput; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(traceLogArray[i] == nullptr)) continue;
        PointerType * logChunkPtrTy = traceLogType[i]->getPointerTo();
        currentChunk[i] = b.CreatePHI(logChunkPtrTy, 2);
        currentChunk[i]->addIncoming(ConstantPointerNull::get(logChunkPtrTy), loopEntry);
    }

    args[2] = segNo;

    Value * const offset = b.CreateURem(segNo, CHUNK_LENGTH);
    b.CreateCondBr(b.CreateIsNull(offset), getNextLogChunk, printLogEntry);

    b.SetInsertPoint(getNextLogChunk);
    PHINode * const nextChunkIndexPhi = b.CreatePHI(sizeTy, 2);
    nextChunkIndexPhi->addIncoming(SZ_ZERO, loopStart);
    SmallVector<PHINode *, 64> subsequentChunk(LastKernel + 1);
    SmallVector<Value *, 64> nextChunk(LastKernel + 1);
    for (auto i = PipelineInput; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(traceLogArray[i] == nullptr)) continue;
        subsequentChunk[i] = b.CreatePHI(traceLogArray[i]->getType(), 2);
        subsequentChunk[i]->addIncoming(traceLogArray[i], loopStart);
    }


    FixedArray<Value *, 2> nextChunkIndices;
    nextChunkIndices[0] = ZERO;
    nextChunkIndices[1] = ONE;
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(subsequentChunk[i] == nullptr)) continue;
        nextChunk[i] = b.CreateLoad(voidPtrTy, subsequentChunk[i]);
        Value * subsequentChunk2 = b.CreateGEP(traceLogType[i], nextChunk[i], nextChunkIndices);
        subsequentChunk2 = b.CreatePointerCast(subsequentChunk2, subsequentChunk[i]->getType());
        subsequentChunk[i]->addIncoming(subsequentChunk2, getNextLogChunk);
    }
    Value * const nextChunkIndex = b.CreateAdd(nextChunkIndexPhi, SZ_ONE);
    nextChunkIndexPhi->addIncoming(nextChunkIndex, getNextLogChunk);

    Value * const foundCorrectChunk = b.CreateICmpEQ(nextChunkIndex, chunkIndex);
    b.CreateCondBr(foundCorrectChunk, selectLogChunk, getNextLogChunk);

    b.SetInsertPoint(selectLogChunk);
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(nextChunk[i] == nullptr)) continue;
        Value * spentLog = b.CreateLoad(voidPtrTy, b.CreateGEP(traceLogType[i], nextChunk[i], nextChunkIndices));
        b.CreateFree(spentLog);
    }
    b.CreateBr(printLogEntry);

    b.SetInsertPoint(printLogEntry);
    PHINode * const nextChunkIndexPhi2 = b.CreatePHI(sizeTy, 2);
    nextChunkIndexPhi2->addIncoming(chunkIndex, loopStart);
    nextChunkIndexPhi2->addIncoming(nextChunkIndexPhi, selectLogChunk);
    SmallVector<PHINode *, 64> currentChunkPhi(LastKernel + 1);
    for (auto i = PipelineInput; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(nextChunk[i] == nullptr)) continue;
        Type * const ty = nextChunk[i]->getType();
        currentChunkPhi[i] = b.CreatePHI(ty, 2);
        currentChunkPhi[i]->addIncoming(currentChunk[i], loopStart);
        currentChunkPhi[i]->addIncoming(nextChunk[i], selectLogChunk);
    }

    FixedArray<Value *, 4> indices;
    indices[0] = ZERO;
    indices[1] = ZERO;
    indices[2] = offset;
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(currentChunkPhi[i] == nullptr)) {
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const auto streamSet = target(e, mBufferGraph);
                assert (streamSet >= FirstStreamSet && streamSet <= LastStreamSet);
                const auto idx = (streamSet - (FirstStreamSet + inputStreamSets)) + 3;
                assert (args[idx] == nullptr);
                args[idx] = SZ_ZERO;
            }
        } else {
            for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
                const auto streamSet = target(e, mBufferGraph);
                assert (streamSet >= FirstStreamSet && streamSet <= LastStreamSet);
                const auto idx = (streamSet - (FirstStreamSet + inputStreamSets)) + 3;
                const BufferPort & out = mBufferGraph[e];
                indices[3] = b.getInt32(out.Port.Number);
                assert (args[idx] == nullptr);
                args[idx] = b.CreateLoad(sizeTy, b.CreateGEP(traceLogType[i], currentChunkPhi[i], indices));
            }
        }
    }

    b.CreateCall(fTy, Dprintf, args);

    BasicBlock * const loopExit = b.CreateBasicBlock("reportStridesPerSegmentExit");
    BasicBlock * const loopEnd = b.GetInsertBlock();
    Value * const nextSegNo = b.CreateAdd(segNo, SZ_ONE);
    segNo->addIncoming(nextSegNo, loopEnd);
    chunkIndex->addIncoming(nextChunkIndexPhi2, loopEnd);
    for (auto i = FirstKernel; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(currentChunk[i] == nullptr)) continue;
        currentChunk[i]->addIncoming(currentChunkPhi[i], loopEnd);
    }

    Value * const notDone = b.CreateICmpNE(nextSegNo, mSegNo);
    b.CreateLikelyCondBr(notDone, loopStart, loopExit);

    b.SetInsertPoint(loopExit);
    // print final new line
    FixedArray<Value *, 2> finalArgs;
    finalArgs[0] = STDERR;
    finalArgs[1] = b.GetString("\n");
    b.CreateCall(fTy, Dprintf, finalArgs);
    for (auto i = PipelineInput; i <= LastKernel; ++i) {
        if (LLVM_UNLIKELY(currentChunkPhi[i] == nullptr)) continue;
        b.CreateFree(currentChunkPhi[i]);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkInstrumentationFunctions
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::linkInstrumentationFunctions(KernelBuilder & b) {
    b.LinkFunction("__print_pipeline_cycle_counter_report", __print_pipeline_cycle_counter_report);
}

}
