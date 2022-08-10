#include "../pipeline_compiler.hpp"

//#ifdef ENABLE_CERN_ROOT
//#include <TH1.h>
//#endif

namespace kernel {

namespace {

struct HistogramPortListEntry {
    uint64_t ItemCount;
    uint64_t Frequency;
    HistogramPortListEntry * Next;
};

struct HistogramPortData {
    uint32_t PortType;
    uint32_t PortNum;
    const char * BindingName;
    uint64_t Size;
    void * Data; // if Size = 0, this points to a HistogramPortListEntry; otherwise its an 64-bit array of length size.
};

struct HistogramKernelData {
    uint32_t Id;
    uint32_t NumOfPorts;
    const char * KernelName;
    HistogramPortData * PortData;
};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __print_pipeline_histogram_report
 ** ------------------------------------------------------------------------------------------------------------- */
void __print_pipeline_histogram_report(const void * const data, const uint64_t numOfKernels, uint32_t reportType) {

    uint32_t maxKernelId = 0;
    size_t maxKernelNameLength = 11;
    size_t maxBindingNameLength = 12;
    uint32_t maxPortNum = 0;
    uint64_t maxItemCount = 0;
    uint64_t maxFrequency = 0;

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
            if (pd.Size == 0) {
                auto e = static_cast<const HistogramPortListEntry *>(pd.Data); assert (e);
                do {
                    maxItemCount = std::max(maxItemCount, e->ItemCount);
                    maxFrequency = std::max(maxFrequency, e->Frequency);
                    e = e->Next;
                } while (e);
            } else {
                const auto c = pd.Size;
                maxItemCount = std::max(maxItemCount, c);
                const auto L = static_cast<const uint64_t *>(pd.Data);
                for (unsigned k = 0; k < c; ++k) {
                    maxFrequency = std::max(maxFrequency, L[k]);
                }
            }

        }
    }

    auto ceil_log10 = [](const uint64_t v) {
        if (v < 10) {
            return 1U;
        }
        return (unsigned)std::ceil(std::log10(v));
    };

    const auto maxKernelIdLength = ceil_log10(maxKernelId);
    const auto maxPortNumLength =  std::max(4U, ceil_log10(maxPortNum));
    const auto cw = (reportType == HistogramReportType::TransferredItems) ? 11U : 8U;
    const auto maxItemCountLength = std::max(cw, ceil_log10(maxItemCount));
    const auto maxFrequencyLength = std::max(9U, ceil_log10(maxFrequency));

    auto & out = errs();
    if (reportType == HistogramReportType::TransferredItems) {
        out << "TRANSFERRED ITEMS HISTOGRAM:\n\n";
    } else if (reportType == HistogramReportType::DeferredItems) {
        out << "DEFERRED FROM TRANSFERRED ITEM COUNT DISTANCE HISTOGRAM:\n\n";
    }

    out << left_justify("#", maxKernelIdLength + 1); // kernel #
    out << left_justify("Kernel", maxKernelNameLength + 1);
    out << left_justify("Binding", maxBindingNameLength + 1);
    out << left_justify("Port", maxPortNumLength + 2);
    if (reportType == HistogramReportType::TransferredItems) {
        out << left_justify("Transferred", maxItemCountLength + 1);
    } else if (reportType == HistogramReportType::DeferredItems) {
        out << left_justify("Deferred", maxItemCountLength + 1);
    }
    out << left_justify("Frequency", maxFrequencyLength + 1);
    out << "\n\n";

    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto & K = kernelData[i];

        const auto id = std::to_string(K.Id);

        const StringRef kernelName = StringRef(K.KernelName, std::strlen(K.KernelName));

        const auto numOfPorts = K.NumOfPorts;
        for (unsigned j = 0; j < numOfPorts; ++j) {
            const HistogramPortData & pd = K.PortData[j];

            const StringRef bindingName = StringRef(pd.BindingName, std::strlen(pd.BindingName));

            const auto portType = (pd.PortType == (uint32_t)PortType::Input) ? "I" : "O";
            const auto portNum = std::to_string(pd.PortNum);

            auto printLine = [&](const uint64_t itemCount, const uint64_t freq) {

                const auto itemCountString = std::to_string(itemCount);
                const auto freqString = std::to_string(freq);

                out << left_justify(id, maxKernelIdLength + 1)
                    << left_justify(kernelName, maxKernelNameLength + 1)
                    << left_justify(bindingName, maxBindingNameLength + 1)
                    << portType
                    << left_justify(portNum, maxPortNumLength + 1)
                    << right_justify(itemCountString, maxItemCountLength) << ' '
                    << right_justify(freqString, maxFrequencyLength) << '\n';
            };

            const auto arrayLength = pd.Size;
            if (arrayLength == 0) {
                const HistogramPortListEntry * node = static_cast<const HistogramPortListEntry *>(pd.Data); assert (node);
                // only the root node might have a frequency of 0
                if (node->Frequency == 0) {
                    node = node->Next;
                }
                while (node) {
                    assert (node->Frequency > 0);
                    printLine(node->ItemCount, node->Frequency);
                    node = node->Next;
                }
            } else {
                const uint64_t * const array = static_cast<const uint64_t *>(pd.Data);
                for (uint64_t i = 0; i < arrayLength; ++i) {
                    const auto f = array[i];
                    if (f) {
                        printLine(i, f);
                    }
                }
            }

        }
    }

    out << '\n';
}

#ifdef ENABLE_CERN_ROOT

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief __analyze_histogram_processing_rates
 ** ------------------------------------------------------------------------------------------------------------- */
void __analyze_histogram_processing_rates(const void * const data, const uint64_t numOfKernels, uint32_t reportType) {

    uint32_t maxKernelId = 0;
    size_t maxKernelNameLength = 11;
    size_t maxBindingNameLength = 12;
    uint32_t maxPortNum = 0;
    uint64_t maxItemCount = 0;
    uint64_t maxFrequency = 0;

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
            if (pd.Size == 0) {
                auto e = static_cast<const HistogramPortListEntry *>(pd.Data); assert (e);
                do {
                    maxItemCount = std::max(maxItemCount, e->ItemCount);
                    maxFrequency = std::max(maxFrequency, e->Frequency);
                    e = e->Next;
                } while (e);
            } else {
                const auto c = pd.Size;
                maxItemCount = std::max(maxItemCount, c);
                const auto L = static_cast<const uint64_t *>(pd.Data);
                for (unsigned k = 0; k < c; ++k) {
                    maxFrequency = std::max(maxFrequency, L[k]);
                }
            }

        }
    }

    auto ceil_log10 = [](const uint64_t v) {
        if (v < 10) {
            return 1U;
        }
        return (unsigned)std::ceil(std::log10(v));
    };

    const auto maxKernelIdLength = ceil_log10(maxKernelId);
    const auto maxPortNumLength =  std::max(4U, ceil_log10(maxPortNum));
    const auto cw = (reportType == HistogramReportType::TransferredItems) ? 11U : 8U;
    const auto maxItemCountLength = std::max(cw, ceil_log10(maxItemCount));
    const auto maxFrequencyLength = std::max(9U, ceil_log10(maxFrequency));

    auto & out = errs();
    if (reportType == HistogramReportType::TransferredItems) {
        out << "TRANSFERRED ITEMS HISTOGRAM:\n\n";
    } else if (reportType == HistogramReportType::DeferredItems) {
        out << "DEFERRED FROM TRANSFERRED ITEM COUNT DISTANCE HISTOGRAM:\n\n";
    }

    out << left_justify("#", maxKernelIdLength + 1); // kernel #
    out << left_justify("Kernel", maxKernelNameLength + 1);
    out << left_justify("Binding", maxBindingNameLength + 1);
    out << left_justify("Port", maxPortNumLength + 2);
    if (reportType == HistogramReportType::TransferredItems) {
        out << left_justify("Transferred", maxItemCountLength + 1);
    } else if (reportType == HistogramReportType::DeferredItems) {
        out << left_justify("Deferred", maxItemCountLength + 1);
    }
    out << left_justify("Frequency", maxFrequencyLength + 1);
    out << "\n\n";

    for (unsigned i = 0; i < numOfKernels; ++i) {
        const auto & K = kernelData[i];

        const auto id = std::to_string(K.Id);

        const StringRef kernelName = StringRef(K.KernelName, std::strlen(K.KernelName));

        const auto numOfPorts = K.NumOfPorts;
        for (unsigned j = 0; j < numOfPorts; ++j) {
            const HistogramPortData & pd = K.PortData[j];

            const StringRef bindingName = StringRef(pd.BindingName, std::strlen(pd.BindingName));

            const auto portType = (pd.PortType == (uint32_t)PortType::Input) ? "I" : "O";
            const auto portNum = std::to_string(pd.PortNum);


            const auto arrayLength = pd.Size;
            if (arrayLength == 0) {
                const HistogramPortListEntry * node = static_cast<const HistogramPortListEntry *>(pd.Data); assert (node);
                // only the root node might have a frequency of 0
                if (node->Frequency == 0) {
                    node = node->Next;
                }
                while (node) {
                    assert (node->Frequency > 0);

                    node = node->Next;
                }
            } else {
                const uint64_t * const array = static_cast<const uint64_t *>(pd.Data);
                for (uint64_t i = 0; i < arrayLength; ++i) {
                    const auto f = array[i];
                    if (f) {

                    }
                }
            }

        }
    }

    out << '\n';
}


#endif

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief recordsAnyHistogramData
 ** ------------------------------------------------------------------------------------------------------------- */
bool PipelineCompiler::recordsAnyHistogramData() const {
    if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram || mGenerateDeferredItemCountHistogram)) {
        for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bd = br.Binding;
            if (LLVM_UNLIKELY(mGenerateDeferredItemCountHistogram && bd.hasAttribute(AttrId::Deferred))) {
                return true;
            }
            const ProcessingRate & pr = bd.getRate();
            if (mGenerateTransferredItemCountHistogram && !pr.isFixed()) {
                return true;
            }
        }
        for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
            const BufferPort & br = mBufferGraph[e];
            const Binding & bd = br.Binding;
            if (LLVM_UNLIKELY(mGenerateDeferredItemCountHistogram && bd.hasAttribute(AttrId::Deferred))) {
                return true;
            }
            const ProcessingRate & pr = bd.getRate();
            if (mGenerateTransferredItemCountHistogram && !pr.isFixed()) {
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
        if (LLVM_UNLIKELY(mGenerateDeferredItemCountHistogram && bd.hasAttribute(AttrId::Deferred))) {
            const auto prefix = makeBufferName(kernelId, br.Port);
            mTarget->addInternalScalar(listTy, prefix + STATISTICS_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX, groupId);
        }
        // fixed rate doesn't need to be tracked as the only one that wouldn't be the exact rate would be
        // the final partial one but that isn't a very interesting value to model.
        if (!mGenerateTransferredItemCountHistogram || (!anyGreedy && pr.isFixed())) {
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
 * @brief freeHistogramProperties
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::freeHistogramProperties(BuilderRef b) {

    Function * freeLinkedListFunc = nullptr;

    for (unsigned i = FirstKernel; i <= LastKernel; ++i) {
        const auto anyGreedy = hasAnyGreedyInput(i);

        auto freeLinkedList = [&](Value * root) {

            if (freeLinkedListFunc == nullptr) {


                PointerType * const voidPtrTy = b->getVoidPtrTy();
                PointerType * const listPtrTy = cast<PointerType>(root->getType());
                FunctionType * const funcTy = FunctionType::get(b->getVoidTy(), {listPtrTy}, false);

                const auto ip = b->saveIP();

                Module * const m = b->getModule();

                LLVMContext & C = m->getContext();
                freeLinkedListFunc = Function::Create(funcTy, Function::InternalLinkage, "__freeHistogramLinkedList", m);

                BasicBlock * const entry = BasicBlock::Create(C, "entry", freeLinkedListFunc);
                BasicBlock * const freeLoop = BasicBlock::Create(C, "freeLoop", freeLinkedListFunc);
                BasicBlock * const freeExit = BasicBlock::Create(C, "freeExit", freeLinkedListFunc);

                b->SetInsertPoint(entry);

                auto arg = freeLinkedListFunc->arg_begin();
                auto nextArg = [&]() {
                    assert (arg != freeLinkedListFunc->arg_end());
                    Value * const v = &*arg;
                    std::advance(arg, 1);
                    return v;
                };

                Value * const root = nextArg();
                root->setName("root");
                assert (arg == freeLinkedListFunc->arg_end());

                FixedArray<Value *, 2> offset;
                offset[0] = b->getInt32(0);
                offset[1] = b->getInt32(2);

                Value * const first = b->CreateLoad(b->CreateGEP(root, offset));
                Value * const nil = ConstantPointerNull::get(voidPtrTy);

                b->CreateCondBr(b->CreateICmpNE(first, nil), freeLoop, freeExit);

                b->SetInsertPoint(freeLoop);
                PHINode * const current = b->CreatePHI(voidPtrTy, 2);
                current->addIncoming(first, entry);
                Value * const currentList = b->CreatePointerCast(current, listPtrTy);
                Value * const next = b->CreateLoad(b->CreateGEP(currentList, offset));
                b->CreateFree(currentList);
                current->addIncoming(next, freeLoop);
                b->CreateCondBr(b->CreateICmpNE(next, nil), freeLoop, freeExit);

                b->SetInsertPoint(freeExit);
                b->CreateRetVoid();

                b->restoreIP(ip);
            }

            b->CreateCall(freeLinkedListFunc->getFunctionType(), freeLinkedListFunc, {root} );

        };

        auto freeProperties = [&](const BufferPort & br) {
            const Binding & bd = br.Binding;
            const ProcessingRate & pr = bd.getRate();

            if (LLVM_UNLIKELY(mGenerateDeferredItemCountHistogram && bd.hasAttribute(AttrId::Deferred))) {
                const auto prefix = makeBufferName(i, br.Port);
                freeLinkedList(getScalarFieldPtr(b.get(), prefix + STATISTICS_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX));
            }
            if (LLVM_UNLIKELY(mGenerateTransferredItemCountHistogram && (anyGreedy || pr.isUnknown()))) {
                const auto prefix = makeBufferName(i, br.Port);
                freeLinkedList(getScalarFieldPtr(b.get(), prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX));
            }

        };

        for (const auto e : make_iterator_range(in_edges(i, mBufferGraph))) {
            freeProperties(mBufferGraph[e]);
        }

        for (const auto e : make_iterator_range(out_edges(i, mBufferGraph))) {
            freeProperties(mBufferGraph[e]);
        }

    }


}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief updateTransferredItemsForHistogramData
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::updateTransferredItemsForHistogramData(BuilderRef b) {

    ConstantInt * const sz_ZERO = b->getSize(0);
    ConstantInt * const sz_ONE = b->getSize(1);

    const auto anyGreedy = hasAnyGreedyInput(mKernelId);

    auto recordDynamicEntry = [&](Value * const initialEntry, Value * const itemCount) {

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
            ConstantInt * const i64_ONE = b->getSize(1);

            const auto ip = b->saveIP();

            LLVMContext & C = m->getContext();
            func = Function::Create(funcTy, Function::InternalLinkage, "updateHistogramList", m);

            BasicBlock * const entry = BasicBlock::Create(C, "entry", func);
            BasicBlock * scanLoop = BasicBlock::Create(C, "scanLoop", func);
            BasicBlock * checkForUpdateOrInsert = BasicBlock::Create(C, "checkInsert", func);
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

            Value * const firstEntry = nextArg();
            firstEntry->setName("firstEntry");
            Value * const itemCount = nextArg();
            itemCount->setName("itemCount");
            assert (arg == func->arg_end());

            b->CreateUnlikelyCondBr(b->CreateICmpEQ(itemCount, sz_ZERO), updateEntry, scanLoop);

            b->SetInsertPoint(scanLoop);
            PHINode * const lastEntry = b->CreatePHI(entryPtrTy, 2, "lastEntry");
            lastEntry->addIncoming(firstEntry, entry);
            PHINode * const lastItemCount = b->CreatePHI(b->getSizeTy(), 2, "lastPosition");
            lastItemCount->addIncoming(sz_ZERO, entry);

            FixedArray<Value *, 2> offset;
            offset[0] = i32_ZERO;
            offset[1] = i32_TWO;

            Value * const currentEntryPtr = b->CreatePointerCast(b->CreateGEP(lastEntry, offset), entryPtrTy->getPointerTo());
            Value * const currentEntry = b->CreateLoad(currentEntryPtr);
            Value * const noMore = b->CreateICmpEQ(currentEntry, ConstantPointerNull::get(entryPtrTy));
            b->CreateCondBr(noMore, insertNewEntry, checkForUpdateOrInsert);

            b->SetInsertPoint(checkForUpdateOrInsert);
            offset[1] = i32_ZERO;
            Value * const currentItemCount = b->CreateLoad(b->CreateGEP(currentEntry, offset));
            if (LLVM_UNLIKELY(CheckAssertions)) {
                Value * const valid = b->CreateICmpULT(lastItemCount, currentItemCount);
                b->CreateAssert(valid, "Histogram history error: last position (%" PRIu64
                                ") >= current position (%" PRIu64 ")", lastItemCount, currentItemCount);
            }
            lastEntry->addIncoming(currentEntry, checkForUpdateOrInsert);
            lastItemCount->addIncoming(currentItemCount, checkForUpdateOrInsert);
            b->CreateCondBr(b->CreateICmpULT(currentItemCount, itemCount), scanLoop, updateOrInsertEntry);

            b->SetInsertPoint(updateOrInsertEntry);
            b->CreateCondBr(b->CreateICmpEQ(currentItemCount, itemCount), updateEntry, insertNewEntry);

            b->SetInsertPoint(insertNewEntry);
            Value * const size = ConstantExpr::getSizeOf(entryPtrTy->getPointerElementType());
            Value * const newEntry = b->CreatePointerCast(b->CreateAlignedMalloc(size, sizeof(uint64_t)), entryPtrTy);
            offset[1] = i32_ZERO;
            b->CreateStore(itemCount, b->CreateGEP(newEntry, offset));
            offset[1] = i32_ONE;
            b->CreateStore(i64_ONE, b->CreateGEP(newEntry, offset));
            offset[1] = i32_TWO;
            b->CreateStore(b->CreatePointerCast(currentEntry, voidPtrTy), b->CreateGEP(newEntry, offset));
            b->CreateStore(b->CreatePointerCast(newEntry, voidPtrTy), b->CreateGEP(lastEntry, offset));
            b->CreateRetVoid();

            b->SetInsertPoint(updateEntry);
            PHINode * const entryToUpdate = b->CreatePHI(entryPtrTy, 2);
            entryToUpdate->addIncoming(currentEntry, updateOrInsertEntry);
            entryToUpdate->addIncoming(firstEntry, entry);
            offset[1] = i32_ONE;
            Value * const ptr = b->CreateGEP(entryToUpdate, offset);
            Value * const val = b->CreateAdd(b->CreateLoad(ptr), i64_ONE);
            b->CreateStore(val, ptr);
            b->CreateRetVoid();

            b->restoreIP(ip);
        }

        FixedArray<Value *, 2> args;
        args[0] = initialEntry;
        args[1] = itemCount;

        b->CreateCall(func->getFunctionType(), func, args);

    };

    auto recordPort = [&](const BufferPort & br) {
        const Binding & bd = br.Binding;
        const ProcessingRate & pr = bd.getRate();

        auto calculateDiff = [&](Value * const A, Value * const B, StringRef Name) -> Value * {
            if (LLVM_UNLIKELY(CheckAssertions)) {
                Value * const valid = b->CreateICmpUGE(A, B);
                b->CreateAssert(valid, "Expected %s.%s (%" PRIu64 ") to exceed %s rate (%" PRIu64 ")",
                                mCurrentKernelName, b->GetString(bd.getName()), A, b->GetString(Name), B);
            }
            return b->CreateSub(A, B);
        };

        if (LLVM_UNLIKELY(mGenerateDeferredItemCountHistogram && bd.hasAttribute(AttrId::Deferred))) {
            const auto prefix = makeBufferName(mKernelId, br.Port);
            Value * const base = b->getScalarFieldPtr(prefix + STATISTICS_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
            Value * diff = nullptr;
            if (br.Port.Type == PortType::Input) {
                diff = calculateDiff(mProcessedItemCount[br.Port], mCurrentProcessedDeferredItemCountPhi[br.Port], "processed deferred");
            } else {
                diff = calculateDiff(mProducedItemCount[br.Port], mCurrentProducedDeferredItemCountPhi[br.Port], "produced deferred");
            }
            recordDynamicEntry(base, diff);
        }
        // fixed rate doesn't need to be tracked as the only one that wouldn't be the exact rate would be
        // the final partial one but that isn't a very interesting value to model.
        if (mGenerateTransferredItemCountHistogram && (anyGreedy || !pr.isFixed())) {
            const auto prefix = makeBufferName(mKernelId, br.Port);
            Value * const base = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
            Value * diff = nullptr;
            if (br.Port.Type == PortType::Input) {
                diff = calculateDiff(mProcessedItemCount[br.Port], mCurrentProcessedItemCountPhi[br.Port], "processed");
            } else {
                diff = calculateDiff(mProducedItemCount[br.Port], mCurrentProducedItemCountPhi[br.Port], "produced");
            }

            if (LLVM_UNLIKELY(anyGreedy || pr.isUnknown())) {
                recordDynamicEntry(base, diff);
            } else {
                if (LLVM_UNLIKELY(CheckAssertions)) {
                    ArrayType * ty = cast<ArrayType>(base->getType()->getPointerElementType());
                    Value * const maxSize = b->getSize(ty->getArrayNumElements() - 1);
                    Value * const valid = b->CreateICmpULE(diff, maxSize);
                    Constant * const bindingName = b->GetString(bd.getName());
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
        }
    };

    for (const auto e : make_iterator_range(in_edges(mKernelId, mBufferGraph))) {
        recordPort(mBufferGraph[e]);
    }

    for (const auto e : make_iterator_range(out_edges(mKernelId, mBufferGraph))) {
        recordPort(mBufferGraph[e]);
    }

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief printHistogramReport
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineCompiler::printHistogramReport(BuilderRef b, HistogramReportType type) const {

    // struct HistogramPortData
    FixedArray<Type *, 5> hpdFields;
    hpdFields[0] = b->getInt32Ty(); // PortType
    hpdFields[1] = b->getInt32Ty(); // PortNum
    hpdFields[2] = b->getInt8PtrTy(); // Binding Name
    hpdFields[3] = b->getInt64Ty(); // Size
    hpdFields[4] = b->getVoidPtrTy(); // Data
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

    PointerType * const voidPtrTy = b->getVoidPtrTy();

    unsigned numOfKernels = 0;

    if (type == HistogramReportType::TransferredItems) {

        for (auto kernelId = FirstKernel; kernelId <= LastKernel; ++kernelId) {

            auto addEntry = [](const BufferPort & br) {
                const Binding & bd = br.Binding;
                const ProcessingRate & pr = bd.getRate();
                return !pr.isFixed();
            };

            auto countKernelEntry = [&]() {
                for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                    if (addEntry(mBufferGraph[e])) return true;
                }

                for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                    if (addEntry(mBufferGraph[e])) return true;
                }
                return false;
            };

            numOfKernels += countKernelEntry() ? 1 : 0;
        }

    } else if (type == HistogramReportType::DeferredItems) {

        for (auto kernelId = FirstKernel; kernelId <= LastKernel; ++kernelId) {

            auto addEntry = [](const BufferPort & br) {
                const Binding & bd = br.Binding;
                return bd.hasAttribute(AttrId::Deferred);
            };

            auto countKernelEntry = [&]() {
                for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                    if (addEntry(mBufferGraph[e])) return true;
                }

                for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                    if (addEntry(mBufferGraph[e])) return true;
                }
                return false;
            };

            numOfKernels += countKernelEntry() ? 1 : 0;

        }

    }



    Value * const kernelData = b->CreateAlignedMalloc(hkdTy, b->getSize(numOfKernels), 0, b->getCacheAlignment());

    for (unsigned kernelId = FirstKernel, index = 0; kernelId <= LastKernel; ++kernelId) {

        unsigned numOfPorts = 0;

        bool anyGreedy = false;

        if (type == HistogramReportType::TransferredItems) {

            anyGreedy = hasAnyGreedyInput(kernelId);

            auto countPorts = [&](const BufferPort & br) {
                const Binding & bind =  br.Binding;
                if (anyGreedy || !bind.getRate().isFixed()) {
                    numOfPorts++;
                }
            };

            for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                countPorts(mBufferGraph[e]);
            }

            for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                countPorts(mBufferGraph[e]);
            }

        } else if (type == HistogramReportType::DeferredItems) {

            auto countPorts = [&](const BufferPort & br) {
                const Binding & bind =  br.Binding;
                if (bind.hasAttribute(AttrId::Deferred)) {
                    numOfPorts++;
                }
            };

            for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                countPorts(mBufferGraph[e]);
            }

            for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                countPorts(mBufferGraph[e]);
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

            if (type == HistogramReportType::TransferredItems) {
                if (LLVM_LIKELY(!anyGreedy && pr.isFixed())) {
                    return;
                }
            } else if (type == HistogramReportType::DeferredItems) {
                if (LLVM_LIKELY(!bind.hasAttribute(AttrId::Deferred))) {
                    return;
                }
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

            Value * data = nullptr;
            uint64_t maxSize = 0;

            if (type == HistogramReportType::TransferredItems) {
                data = b->getScalarFieldPtr(prefix + STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
                if (!anyGreedy && !pr.isUnknown()) {
                    maxSize = ceiling(br.Maximum) + 1;
                }
            } else if (type == HistogramReportType::DeferredItems) {
                data = b->getScalarFieldPtr(prefix + STATISTICS_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX);
            }

            offset[1] = i32_THREE;
            b->CreateStore(b->getInt64(maxSize), b->CreateGEP(portData, offset));

            offset[1] = i32_FOUR;
            b->CreateStore(b->CreatePointerCast(data, voidPtrTy), b->CreateGEP(portData, offset));

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
    FixedArray<Value *, 3> args;
    args[0] = b->CreatePointerCast(kernelData, voidPtrTy);
    args[1] = b->getInt64(numOfKernels);
    args[2] = b->getInt32((unsigned)type);
    Function * const reportPrinter = b->getModule()->getFunction("__print_pipeline_histogram_report");
    assert (reportPrinter);
    b->CreateCall(reportPrinter->getFunctionType(), reportPrinter, args);

    // memory cleanup
    for (unsigned kernelId = FirstKernel, index = 0; kernelId <= LastKernel; ++kernelId) {

        if (type == HistogramReportType::TransferredItems) {

            auto hasPortData = [](const BufferPort & br) {
                const Binding & bind =  br.Binding;
                return !bind.getRate().isFixed();
            };

            for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                if (hasPortData(mBufferGraph[e])) goto free_port_data;
            }

            for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                if (hasPortData(mBufferGraph[e])) goto free_port_data;
            }

        } else if (type == HistogramReportType::DeferredItems) {

            auto hasPortData = [](const BufferPort & br) {
                const Binding & bind =  br.Binding;
                return bind.hasAttribute(AttrId::Deferred);
            };

            for (const auto e : make_iterator_range(in_edges(kernelId, mBufferGraph))) {
                if (hasPortData(mBufferGraph[e])) goto free_port_data;
            }

            for (const auto e : make_iterator_range(out_edges(kernelId, mBufferGraph))) {
                if (hasPortData(mBufferGraph[e])) goto free_port_data;
            }
        }
        continue;
free_port_data:
        FixedArray<Value *, 2> offset;
        offset[0] = b->getInt32(index++);
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
