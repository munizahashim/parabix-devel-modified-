#ifndef HISTOGRAM_GENERATION_LOGIC_HPP
#define HISTOGRAM_GENERATION_LOGIC_HPP

#include "pipeline_compiler.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordsAnyHistogramData
 ** ------------------------------------------------------------------------------------------------------------- */
inline bool PipelineCompiler::recordsAnyHistogramData() const {
    if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram)) {
        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bd = br.Binding;
            const ProcessingRate & pr = bd.getRate();
            if (!pr.isFixed() || bd.hasAttribute(AttrId::Deferred)) {
                return true;
            }
        }
        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bd = br.Binding;
            const ProcessingRate & pr = bd.getRate();
            if (!pr.isFixed() || bd.hasAttribute(AttrId::Deferred)) {
                return true;
            }
        }
    }
    return false;
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief addHistogramProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::addHistogramProperties(BuilderRef b, const size_t kernelId, const size_t groupId) {

    assert (mGenerateTransferredItemCountHistogram);

    IntegerType * const i64Ty = b->getInt64Ty();

    FixedArray<Type *, 3> fields;
    fields[0] = i64Ty;
    fields[1] = i64Ty;
    fields[2] = b->getVoidPtrTy();
    StructType * const listTy = StructType::get(b->getContext(), fields);

    const auto anyGreedy = hasAnyGreedyInput(kernelId);

    auto addProperties = [&](const BufferPort & br) {
        const Binding & bd = br.Binding;
        const ProcessingRate & pr = bd.getRate();
        if (LLVM_UNLIKELY(bd.hasAttribute(AttrId::Deferred))) {
            const auto prefix = makeBufferName(kernelId, br.Port);
            mTarget->addInternalScalar(listTy, prefix + STATISTICS_TRANSFERRED_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX, groupId);
        }
        // fixed rate doesn't need to be tracked as the only one that wouldn't be the exact rate would be
        // the final partial one but that isn't a very interesting value to model.
        if (!anyGreedy && pr.isFixed()) {
            return;
        }
        const auto prefix = makeBufferName(kernelId, br.Port);
        Type * histTy = nullptr;
        if (LLVM_UNLIKELY(anyGreedy || pr.isUnknown())) {
            // we do not know how many items will be transferred with any deferred, greedy or unknown
            // rate; keep a linked list of entries. The first one will always refer to a 0-item entry
            // since it's simpler than trying to rearrange the initial entry.
            histTy = listTy;
        } else {
            histTy = ArrayType::get(i64Ty, ceiling(br.Maximum) + 1);
        }


        mTarget->addInternalScalar(histTy, prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX, groupId);

    };

    for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
        addProperties(mBufferGraph[e]);
    }

    for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
        addProperties(mBufferGraph[e]);
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateTransferredItemsForHistogramData
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateTransferredItemsForHistogramData(BuilderRef b) {

    assert (mGenerateTransferredItemCountHistogram);

    ConstantInt * const sz_ZERO = b->getSize(0);
    ConstantInt * const sz_ONE = b->getSize(1);

    const auto anyGreedy = hasAnyGreedyInput(mKernelId);

    auto recordDynamicEntry = [&](Value * const initialEntry, Value * const position) {

        Module * const m = b->getModule();

        Function * func = m->getFunction("updateHistogramList");
        if (func == nullptr) {

            PointerType * const entryPtrTy = cast<PointerType>(initialEntry->getType());
            PointerType * const voidPtrTy = b->getVoidPtrTy();
            IntegerType * const sizeTy = b->getSizeTy();

            FunctionType * funcTy = FunctionType::get(b->getVoidTy(), {entryPtrTy, sizeTy}, false);

            ConstantInt * const i32_ZERO = b->getInt32(0);
            ConstantInt * const i32_ONE = b->getInt32(1);
            ConstantInt * const i32_TWO = b->getInt32(2);
            ConstantInt * const sz_ONE = b->getSize(1);

            const auto ip = b->saveIP();

            LLVMContext & C = m->getContext();
            func = Function::Create(funcTy, Function::InternalLinkage, "updateHistogramList", m);

            BasicBlock * const entry = BasicBlock::Create(C, "entry", func);
            BasicBlock * scanLoop = BasicBlock::Create(C, "scanLoop", func);
            BasicBlock * checkInsert = BasicBlock::Create(C, "checkInsert", func);
            BasicBlock * updateOrInsertEntry = BasicBlock::Create(C, "updateOrInsertEntry", func);
            BasicBlock * insertNewEntry = BasicBlock::Create(C, "insertNewEntry", func);
            BasicBlock * updateEntry = BasicBlock::Create(C, "updateEntry", func);

            b->SetInsertPoint(entry);

            auto arg = func->arg_begin();
            auto nextArg = [&]() {
                assert (arg != func->arg_end());
                Value * const v = &*arg;
                std::advance(arg, 1);
                return v;
            };

            Value * const rootEntry = nextArg();
            rootEntry->setName("initialEntry");
            Value * const position = nextArg();
            position->setName("position");
            assert (arg == func->arg_end());

            b->CreateUnlikelyCondBr(b->CreateICmpEQ(position, sz_ZERO), updateEntry, scanLoop);

            b->SetInsertPoint(scanLoop);
            PHINode * const currentEntry = b->CreatePHI(entryPtrTy, 2, "lastEntry");
            currentEntry->addIncoming(rootEntry, entry);
            PHINode * const lastPosition = b->CreatePHI(b->getSizeTy(), 2, "lastPosition");
            lastPosition->addIncoming(sz_ZERO, entry);

            FixedArray<Value *, 2> offset;
            offset[0] = i32_ZERO;
            offset[1] = i32_TWO;

            Value * const nextEntryPtr = b->CreatePointerCast(b->CreateGEP(currentEntry, offset), entryPtrTy->getPointerTo());
            Value * const nextEntry = b->CreateLoad(nextEntryPtr);
            Value * const noMore = b->CreateICmpEQ(nextEntry, ConstantPointerNull::get(entryPtrTy));
            b->CreateCondBr(noMore, insertNewEntry, checkInsert);

            b->SetInsertPoint(checkInsert);
            offset[1] = i32_ZERO;
            Value * const curPos = b->CreateLoad(b->CreateGEP(nextEntry, offset));
            if (LLVM_UNLIKELY(CheckAssertions)) {
                Value * const valid = b->CreateICmpULT(lastPosition, curPos);
                b->CreateAssert(valid, "Histogram history error: last position %" PRIu64 " >= current position %" PRIu64, lastPosition, curPos);
            }
            currentEntry->addIncoming(nextEntry, checkInsert);
            lastPosition->addIncoming(curPos, checkInsert);
            b->CreateCondBr(b->CreateICmpULT(curPos, position), scanLoop, updateOrInsertEntry);

            b->SetInsertPoint(updateOrInsertEntry);
            b->CreateCondBr(b->CreateICmpEQ(curPos, position), updateEntry, insertNewEntry);

            b->SetInsertPoint(insertNewEntry);
            Value * const size = ConstantExpr::getSizeOf(entryPtrTy->getPointerElementType());
            Value * const newEntry = b->CreatePointerCast(b->CreateAlignedMalloc(size, sizeof(uint64_t)), entryPtrTy);
            b->CreateStore(position, b->CreateGEP(newEntry, offset));
            offset[1] = i32_ONE;
            b->CreateStore(sz_ONE, b->CreateGEP(newEntry, offset));
            offset[1] = i32_TWO;
            b->CreateStore(b->CreatePointerCast(nextEntry, voidPtrTy), b->CreateGEP(newEntry, offset));
            b->CreateStore(b->CreatePointerCast(newEntry, voidPtrTy), b->CreateGEP(currentEntry, offset));
            b->CreateRetVoid();

            b->SetInsertPoint(updateEntry);
            PHINode * const entryToUpdate = b->CreatePHI(entryPtrTy, 2);
            entryToUpdate->addIncoming(currentEntry, updateOrInsertEntry);
            entryToUpdate->addIncoming(rootEntry, entry);

            offset[1] = i32_ONE;
            Value * const ptr = b->CreateGEP(entryToUpdate, offset);
            b->CreateStore(b->CreateAdd(b->CreateLoad(ptr), sz_ONE), ptr);
            b->CreateRetVoid();

            b->restoreIP(ip);
        }

        FixedArray<Value *, 2> args;
        args[0] = initialEntry;
        args[1] = position;

        b->CreateCall(func->getFunctionType(), func, args);

    };

    auto recordPort = [&](const BufferPort & br) {
        const Binding & bd = br.Binding;
        const ProcessingRate & pr = bd.getRate();
        if (LLVM_UNLIKELY(bd.hasAttribute(AttrId::Deferred))) {
            const auto prefix = makeBufferName(mKernelId, br.Port);
            Value * const base = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
            Value * diff = nullptr;
            if (br.Port.Type == PortType::Input) {
                diff = b->CreateSub(mCurrentProcessedItemCountPhi[br.Port], mCurrentProcessedDeferredItemCountPhi[br.Port]);
            } else {
                diff = b->CreateSub(mCurrentProducedItemCountPhi[br.Port], mCurrentProducedDeferredItemCountPhi[br.Port]);
            }
            recordDynamicEntry(base, diff);
        }
        // fixed rate doesn't need to be tracked as the only one that wouldn't be the exact rate would be
        // the final partial one but that isn't a very interesting value to model.
        if (!anyGreedy && pr.isFixed()) {
            return;
        }
        const auto prefix = makeBufferName(mKernelId, br.Port);
        Value * const base = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
        Value * diff = nullptr;
        if (br.Port.Type == PortType::Input) {
            diff = b->CreateSub(mProcessedItemCount[br.Port], mCurrentProcessedItemCountPhi[br.Port]);
        } else {
            diff = b->CreateSub(mProducedItemCount[br.Port], mCurrentProducedItemCountPhi[br.Port]);
        }

        if (LLVM_UNLIKELY(anyGreedy || pr.isUnknown())) {
            recordDynamicEntry(base, diff);
        } else {
            if (LLVM_UNLIKELY(CheckAssertions)) {
                ArrayType * ty = cast<ArrayType>(base->getType()->getPointerElementType());
                Value * const maxSize = b->getSize(ty->getArrayNumElements() - 1);
                Value * const valid = b->CreateICmpULE(diff, maxSize);
                Constant * const bindingName = b->GetString(br.Binding.get().getName());
                b->CreateAssert(valid, "%s.%s: attempting to update %" PRIu64 "-th value of histogram data "
                                       "but internal array can only support up to %" PRIu64 " elements",
                                        mCurrentKernelName, bindingName, diff, maxSize);
            }
            FixedArray<Value *, 2> args;
            args[0] = sz_ZERO;
            args[1] = diff;
            Value * const toInc = b->CreateGEP(base, args);
            b->CreateStore(b->CreateAdd(b->CreateLoad(toInc), sz_ONE), toInc);
        }

    };

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        recordPort(mBufferGraph[e]);
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        recordPort(mBufferGraph[e]);
    }

}

namespace {

struct HistogramPortListEntry {
    uint64_t Position;
    uint64_t Frequency;
    HistogramPortListEntry * Next;
};

struct HistogramPortData {
    uint32_t PortType;
    uint32_t PortNum;
    char * BindingName;
    uint64_t Size;
    void * Data; // if Size = 0, this points to a HistogramPortListEntry; otherwise its an 64-bit array of length size.

    HistogramPortListEntry * DeferredData;
};

struct HistogramKernelData {
    uint32_t Id;
    uint32_t NumOfPorts;
    char * KernelName;
    HistogramPortData * PortData;
};

void __print_pipeline_histogram_report(const void * const data, const uint64_t numOfKernels) {

    uint32_t maxKernelId = 9;
    size_t maxKernelNameLength = 11;
    size_t maxBindingNameLength = 12;
    uint32_t maxPortNum = 9;
    uint64_t maxFrequency = 9;
    uint64_t maxDeferredTransferredItemCount = 9;
    uint64_t maxTransferredItemCount = 9;

    const auto kernelData = static_cast<const HistogramKernelData *>(data);

    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto & K = kernelData[i];
        maxKernelId = std::max(maxKernelId, K.Id);
        maxKernelNameLength = std::max(maxKernelNameLength, strlen(K.KernelName));
        const auto numOfPorts = K.NumOfPorts;
        for (unsigned j = 0; j < numOfPorts; ++j) {
            const HistogramPortData & pd = K.PortData[j];
            maxPortNum = std::max(maxPortNum, pd.PortNum);
            maxBindingNameLength = std::max(maxBindingNameLength, strlen(pd.BindingName));

            auto scanThroughLinkedList = [&maxFrequency](const HistogramPortListEntry * const root) {
                uint64_t t = 0;
                for (auto e = root; e; e = e->Next) {
                    t = std::max(t, e->Position);
                    maxFrequency = std::max(maxFrequency, e->Frequency);
                    e = e->Next;
                    if (!e) break;
                }
                return t;
            };

            if (pd.Size == 0) {
                const auto t = scanThroughLinkedList(static_cast<const HistogramPortListEntry *>(pd.Data));
                maxTransferredItemCount = std::max(maxTransferredItemCount, t);
            } else {
                const auto c = pd.Size;
                maxTransferredItemCount = std::max(maxTransferredItemCount, c);
                const auto L = static_cast<const uint64_t *>(pd.Data);
                for (unsigned k = 0; k < c; ++k) {
                    maxFrequency = std::max(maxFrequency, L[k]);
                }
            }

            if (pd.DeferredData) {
                const auto t = scanThroughLinkedList(static_cast<const HistogramPortListEntry *>(pd.DeferredData));
                maxDeferredTransferredItemCount = std::max(maxDeferredTransferredItemCount, t);
            }

        }


    }

    errs() << "maxKernelId: " << maxKernelId << "\n"
              "maxKernelNameLength: " << maxKernelNameLength << "\n"
              "maxBindingNameLength: " << maxBindingNameLength << "\n"
              "maxPortNum: " << maxPortNum << "\n"
              "maxFrequency: " << maxFrequency << "\n"
              "maxDeferredTransferredItemCount: " << maxDeferredTransferredItemCount << "\n"
              "maxTransferredItemCount: " << maxTransferredItemCount << "\n";

}

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printHistogramReport
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printHistogramReport(BuilderRef b) {

    assert (mGenerateTransferredItemCountHistogram);

    // struct HistogramPortData
    FixedArray<Type *, 6> hpdFields;
    hpdFields[0] = b->getInt32Ty(); // PortType
    hpdFields[1] = b->getInt32Ty(); // PortNum
    hpdFields[2] = b->getInt8PtrTy(); // Binding Name
    hpdFields[3] = b->getInt64Ty(); // Size
    hpdFields[4] = b->getVoidPtrTy(); // Data
    hpdFields[5] = b->getVoidPtrTy(); // DeferredData
    StructType * const hpdTy = StructType::get(b->getContext(), hpdFields);

    // struct HistogramKernelData
    FixedArray<Type *, 4> hkdFields;
    hkdFields[0] = b->getInt32Ty(); // Id
    hkdFields[1] = b->getInt32Ty(); // NumOfPorts
    hkdFields[2] = b->getInt8PtrTy(); // KernelName
    hkdFields[3] = hpdTy->getPointerTo(); // PortData
    StructType * const hkdTy = StructType::get(b->getContext(), hkdFields);

    #ifndef NDEBUG
    BEGIN_SCOPED_REGION
    DataLayout dl(b->getModule());
    auto getTypeSize = [&dl](Type * const type) -> size_t {
        if (type == nullptr) {
            return 0UL;
        }
        #if LLVM_VERSION_INTEGER < LLVM_VERSION_CODE(11, 0, 0)
        return dl.getTypeAllocSize(type);
        #else
        return dl.getTypeAllocSize(type).getFixedSize();
        #endif
    };
    assert (getTypeSize(hpdTy) == sizeof(HistogramPortData));
    assert (getTypeSize(hkdTy) == sizeof(HistogramKernelData));
    END_SCOPED_REGION
    #endif

    ConstantInt * const i32_ZERO = b->getInt32(0);
    ConstantInt * const i32_ONE = b->getInt32(1);
    ConstantInt * const i32_TWO = b->getInt32(2);
    ConstantInt * const i32_THREE = b->getInt32(3);
    ConstantInt * const i32_FOUR = b->getInt32(4);
    ConstantInt * const i32_FIVE = b->getInt32(5);

    PointerType * const voidPtrTy = b->getVoidPtrTy();

    unsigned numOfKernels = 0;

    for (auto kernelId = FirstKernel; kernelId <= LastKernel; ++kernelId) {
        for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bind =  br.Binding;
            if (bind.hasAttribute(AttrId::Deferred) || !bind.getRate().isFixed()) {
                numOfKernels++;
            }
        }

        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bind =  br.Binding;
            if (bind.hasAttribute(AttrId::Deferred) || !bind.getRate().isFixed()) {
                numOfKernels++;
            }
        }
    }

    Value * const kernelData = b->CreateAlignedMalloc(hkdTy, b->getSize(numOfKernels), 0, b->getCacheAlignment());

    for (unsigned kernelId = FirstKernel, index = 0; kernelId <= LastKernel; ++kernelId) {
        const auto anyGreedy = hasAnyGreedyInput(kernelId);

        unsigned numOfPorts = 0;
        for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bind =  br.Binding;
            if (anyGreedy || bind.hasAttribute(AttrId::Deferred) || !bind.getRate().isFixed()) {
                numOfPorts++;
            }
        }

        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bind =  br.Binding;
            if (anyGreedy || bind.hasAttribute(AttrId::Deferred) || !bind.getRate().isFixed()) {
                numOfPorts++;
            }
        }

        if (numOfPorts == 0) {
            continue;
        }

        FixedArray<Value *, 2> offset;
        offset[0] = b->getInt32(index++);
        offset[1] = i32_ZERO;
        b->CreateStore(b->getInt32(kernelId), b->CreateGEP(kernelData, offset));
        offset[1] = i32_ONE;
        b->CreateStore(b->getInt32(numOfPorts), b->CreateGEP(kernelData, offset));
        offset[1] = i32_TWO;
        b->CreateStore(b->GetString(getKernel(kernelId)->getName()), b->CreateGEP(kernelData, offset));
        Value * const portData = b->CreateAlignedMalloc(hpdTy, b->getSize(numOfPorts), 0, b->getCacheAlignment());
        offset[1] = i32_THREE;
        b->CreateStore(portData, b->CreateGEP(kernelData, offset));

        unsigned portIndex = 0;

        auto writePortEntry = [&](const BufferPort & br) {
            const Binding & bind =  br.Binding;
            const ProcessingRate & pr = bind.getRate();

            if (LLVM_LIKELY(!anyGreedy && pr.isFixed() && !bind.hasAttribute(AttrId::Deferred))) {
                return;
            }

            assert (portIndex < numOfPorts);

            offset[0] = b->getInt32(portIndex++);
            offset[1] = i32_ZERO;
            b->CreateStore(b->getInt32((unsigned)br.Port.Type), b->CreateGEP(portData, offset));
            offset[1] = i32_ONE;
            b->CreateStore(b->getInt32(br.Port.Number), b->CreateGEP(portData, offset));
            offset[1] = i32_TWO;
            b->CreateStore(b->GetString(bind.getName()), b->CreateGEP(portData, offset));

            const auto prefix = makeBufferName(kernelId, br.Port);

            Value * nonDeferred = nullptr;
            if (!anyGreedy && pr.isFixed()) {
                nonDeferred = ConstantPointerNull::get(voidPtrTy);
            } else {
                nonDeferred = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
                nonDeferred = b->CreatePointerCast(nonDeferred, voidPtrTy);
            }

            b->CallPrintInt(prefix + ": nonDeferred", nonDeferred);

            uint64_t maxSize = 0;
            if (pr.isFixed() || anyGreedy || pr.isUnknown()) {
                maxSize = 0;
            } else {
                maxSize = ceiling(br.Maximum) + 1;
            }
            offset[1] = i32_THREE;
            b->CreateStore(b->getInt64(maxSize), b->CreateGEP(portData, offset));

            offset[1] = i32_FOUR;
            b->CreateStore(nonDeferred, b->CreateGEP(portData, offset));

            Value * deferred = nullptr;

            if (LLVM_UNLIKELY(bind.hasAttribute(AttrId::Deferred))) {
                deferred = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
                deferred = b->CreatePointerCast(deferred, voidPtrTy);
            } else {
                deferred = ConstantPointerNull::get(voidPtrTy);
            }

            b->CallPrintInt(prefix + ": deferred", deferred);

            offset[1] = i32_FIVE;
            b->CreateStore(deferred, b->CreateGEP(portData, offset));
        };

        for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
            writePortEntry(mBufferGraph[e]);
        }
        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            writePortEntry(mBufferGraph[e]);
        }
        assert (portIndex == numOfPorts);
    }

    // call the report function
    FixedArray<Value *, 2> args;
    args[0] = b->CreatePointerCast(kernelData, voidPtrTy);
    args[1] = b->getInt64(numOfKernels);
    Function * const reportPrinter = b->getModule()->getFunction("__print_pipeline_histogram_report");
    assert (reportPrinter);
    b->CreateCall(reportPrinter->getFunctionType(), reportPrinter, args);



    // memory cleanup
    for (unsigned kernelId = FirstKernel, index = 0; kernelId <= LastKernel; ++kernelId) {
        for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bind =  br.Binding;
            if (bind.hasAttribute(AttrId::Deferred) || !bind.getRate().isFixed()) {
                goto free_port_data;
            }
        }
        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bind =  br.Binding;
            if (bind.hasAttribute(AttrId::Deferred) || !bind.getRate().isFixed()) {
                goto free_port_data;
            }
        }
        continue;
free_port_data:
        FixedArray<Value *, 2> offset;
        offset[0] = b->getInt32(index++);
        offset[1] = i32_ZERO;
        offset[1] = i32_THREE;
        b->CreateFree(b->CreateLoad(b->CreateGEP(kernelData, offset)));
    }
    b->CreateFree(kernelData);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief linkHistogramFunctions
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::linkHistogramFunctions(BuilderRef b) {
    b->LinkFunction("__print_pipeline_histogram_report", __print_pipeline_histogram_report);
}

#if 0

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printHistogramReport
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printHistogramReport(BuilderRef b) {

    assert (mGenerateTransferredItemCountHistogram);

    size_t kernelNameLength = 11;
    size_t bindingNameLength = 12;
    unsigned maxPortNumber = 9;
    unsigned maxTransferredItemCount = 9;
    unsigned maxKernelId = 9;

    bool anyDeferred = false;

    for (auto kernelId = FirstKernel; kernelId <= LastKernel; ++kernelId) {

        bool reportKernelName = false;

        for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bind =  br.Binding;
            if (bind.hasAttribute(AttrId::Deferred)) {
                anyDeferred = true;
            } else if (bind.getRate().isFixed()) {
                continue;
            }
            reportKernelName = true;
            bindingNameLength = std::max(bindingNameLength, bind.getName().size());
            maxPortNumber = std::max(maxPortNumber, br.Port.Number);
            maxTransferredItemCount = std::max(maxTransferredItemCount, ceiling(br.Maximum));
        }

        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            if (__trackPort(br)) {
                reportKernelName = true;
                const Binding & bind =  br.Binding;
                auto len = bind.getName().size();
                if (bind.hasAttribute(AttrId::Deferred)) {
                    len += 11; // " (deferred)"
                }
                bindingNameLength = std::max(bindingNameLength, len);
                maxPortNumber = std::max(maxPortNumber, br.Port.Number);
                maxTransferredItemCount = std::max(maxTransferredItemCount, ceiling(br.Maximum));
            }
        }

        if (reportKernelName) {
            kernelNameLength = std::max(kernelNameLength, getKernel(kernelId)->getName().size());
            maxKernelId = kernelId;
        }

    }

    const auto kernelIdLength = std::ceil(std::log10(maxKernelId));

    const auto portLength = std::max<unsigned>(6, std::ceil(std::log10(maxPortNumber)));

    const auto transferredLength = std::max<unsigned>(13, std::ceil(std::log10(maxTransferredItemCount)));

    // Kernel Name, Binding Name, Port #, Transferred #, Frequency

    Function * Dprintf = b->GetDprintf();
    FunctionType * fTy = Dprintf->getFunctionType();

    Constant * const STDERR = b->getInt32(STDERR_FILENO);

    BEGIN_SCOPED_REGION

    std::string tmp;
    raw_string_ostream fmt(tmp);

    fmt << "%-" << kernelIdLength << "s "
           "%-" << kernelNameLength << "s %-" << bindingNameLength << "s "
           "%-" << (portLength + 1) << "s %-" << transferredLength << "s %s\n";

    FixedArray<Value *, 8> args;
    args[0] = STDERR;

    args[1] = b->GetString(fmt.str());
    args[2] = b->GetString("#");
    args[3] = b->GetString("Kernel Name");
    args[4] = b->GetString("Binding Name");
    args[5] = b->GetString("Port #");
    args[6] = b->GetString("Transferred #");
    args[7] = b->GetString("Frequency");

    b->CreateCall(fTy, Dprintf, args);

    END_SCOPED_REGION

    FixedArray<Value *, 9> args;
    args[0] = STDERR;

    BEGIN_SCOPED_REGION

    std::string tmp;
    raw_string_ostream fmt(tmp);

    fmt << "%-" << kernelIdLength << "d "
           "%-" << kernelNameLength << "s %-" << bindingNameLength << "s "
           "%c%-" << portLength << "d %-" << transferredLength << "d %d\n";

    args[1] = b->GetString(fmt.str());

    END_SCOPED_REGION

    IntegerType * const sizeTy = b->getSizeTy();
    ConstantInt * const sz_ZERO = b->getSize(0);
    ConstantInt * const sz_ONE = b->getSize(1);

    for (auto kernelId = FirstKernel; kernelId <= LastKernel; ++kernelId) {




        auto printLine = [&](const BufferPort & br) {

            args[2] = b->getSize(kernelId);
            args[3] = b->GetString(getKernel(kernelId)->getName());
            args[4] = b->GetString(br.Binding.get().getName());
            if (br.Port.Type == PortType::Input) {
                args[5] = b->getInt8('I');
            } else {
                args[5] = b->getInt8('O');
            }

            args[6] = b->getSize(br.Port.Number);

            BasicBlock * const loopEntry = b->GetInsertBlock();
            BasicBlock * const loopBody = b->CreateBasicBlock("");
            BasicBlock * const printEntry = b->CreateBasicBlock("");
            BasicBlock * const printExit = b->CreateBasicBlock("");
            BasicBlock * const loopExit = b->CreateBasicBlock("");

            const auto prefix = makeBufferName(kernelId, br.Port);
            Value * const history = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);

            ArrayType * ty = cast<ArrayType>(history->getType()->getPointerElementType());
            Value * const maxSize = b->getSize(ty->getArrayNumElements() - 1);

            b->CreateBr(loopBody);
            b->SetInsertPoint(loopBody);
            PHINode * const index = b->CreatePHI(sizeTy, 2);
            index->addIncoming(sz_ZERO, loopEntry);

            FixedArray<Value *, 2> offset;
            offset[0] = sz_ZERO;
            offset[1] = index;
            Value * const val = b->CreateLoad(b->CreateGEP(history, offset));
            b->CreateCondBr(b->CreateICmpNE(val, sz_ZERO), printEntry, printExit);

            b->SetInsertPoint(printEntry);

            args[7] = index;
            args[8] = val;
            b->CreateCall(fTy, Dprintf, args);

            b->CreateBr(printExit);

            b->SetInsertPoint(printExit);
            Value * const nextIndex = b->CreateAdd(index, sz_ONE);
            index->addIncoming(nextIndex, printExit);
            b->CreateCondBr(b->CreateICmpEQ(nextIndex, maxSize), loopExit, loopBody);

            b->SetInsertPoint(loopExit);
        };

        auto checkPort = [&](const BufferPort & br) {



            const Binding & bd = br.Binding;
            const ProcessingRate & pr = bd.getRate();
            if (LLVM_UNLIKELY(bd.hasAttribute(AttrId::Deferred))) {
                const auto prefix = makeBufferName(mKernelId, br.Port);
                FixedArray<Value *, 6> args;
                Value * const history = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
                args[1] = b->getSize(kernelId);
                args[2] = b->GetString(getKernel(kernelId)->getName());
                args[3] = b->GetString(bd.getName() + " (deferred)");
                if (br.Port.Type == PortType::Input) {
                    args[4] = b->getInt8('I');
                } else {
                    args[4] = b->getInt8('O');
                }
                args[5] = b->getSize(br.Port.Number);

            }
            if (pr.isFixed()) return;

            const auto prefix = makeBufferName(kernelId, br.Port);
            FixedArray<Value *, 6> args;
            args[0] = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
            args[1] = b->getSize(kernelId);
            args[2] = b->GetString(getKernel(kernelId)->getName());
            args[3] = b->GetString(bd.getName());
            if (br.Port.Type == PortType::Input) {
                args[4] = b->getInt8('I');
            } else {
                args[4] = b->getInt8('O');
            }
            args[5] = b->getSize(br.Port.Number);

        };


        for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            if (__trackPort(br)) {
                printLine(br);
            }
        }

        for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            if (__trackPort(br)) {
                printLine(br);
            }
        }

    }

}

#endif

}

#endif // HISTOGRAM_GENERATION_LOGIC_HPP
