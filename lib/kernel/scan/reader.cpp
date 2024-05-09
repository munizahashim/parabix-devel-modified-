/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/scan/reader.h>

#include <kernel/core/kernel_builder.h>

using namespace llvm;

namespace kernel {

void ScanReader::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    Module * const module = b.getModule();
    Type * const sizeTy = b.getSizeTy();
    Value * const sz_ZERO = b.getSize(0);
    Value * const sz_ONE = b.getSize(1);


    Function * const fCallback = module->getFunction(mCallbackName);
    if (fCallback == nullptr) {
        llvm::report_fatal_error(llvm::StringRef(mKernelName) + ": failed to get function: " + mCallbackName);
    }
    FunctionType * const fCallbackTy = fCallback->getFunctionType();

    BasicBlock * const entryBlock = b.GetInsertBlock();
    BasicBlock * const readItem = b.CreateBasicBlock("readItem");
    BasicBlock * const exitBlock = b.CreateBasicBlock("exitBlock");
    BasicBlock * const doneBlock = mDoneCallbackName.empty() ? exitBlock :  b.CreateBasicBlock("doneBlock");
    Value * const initialStride = b.getProcessedItemCount("scan");
    Value * const isInvalidFinalItem = b.CreateAnd(b.isFinal(), b.CreateICmpEQ(b.getSize(0), b.getAccessibleItemCount("scan")));
    b.CreateCondBr(isInvalidFinalItem, doneBlock, readItem);

    b.SetInsertPoint(readItem);
    PHINode * const strideNo = b.CreatePHI(sizeTy, 2);
    strideNo->addIncoming(sz_ZERO, entryBlock);
    Value * const nextStrideNo = b.CreateAdd(strideNo, sz_ONE);
    strideNo->addIncoming(nextStrideNo, readItem);
    std::vector<Value *> callbackParams;
    Value * maxScanIndex = nullptr;
    Value * const index = b.CreateAdd(strideNo, initialStride);
    for (uint32_t i = 0; i < mNumScanStreams; ++i) {
        Value * const scanItem = b.CreateLoad(sizeTy, b.getRawInputPointer("scan", b.getInt32(i), index));
        if (maxScanIndex != nullptr) {
            maxScanIndex = b.CreateUMax(maxScanIndex, scanItem);
        } else {
            maxScanIndex = scanItem;
        }
        // FIXME: We are assuming that we have access to the entire source stream, this may not always be the case.
        Value * const scanPtr = b.getRawInputPointer("source", scanItem);
        callbackParams.push_back(scanPtr);
    }
    b.setProcessedItemCount("source", maxScanIndex);
    Value * const nextIndex = b.CreateAdd(nextStrideNo, initialStride);
    b.setProcessedItemCount("scan", nextIndex);
    for (unsigned i = 2; i < getNumOfStreamInputs(); ++i) {
        const StreamSet * const ss = getInputStreamSet(i);
        const auto & name = getInputStreamSetBinding(i).getName();
        Value * const ptr = b.getRawInputPointer(name, b.getInt32(0), index);
        Value * const item = b.CreateLoad(b.getIntNTy(ss->getFieldWidth()), ptr);
        callbackParams.push_back(item);
        b.setProcessedItemCount(name, nextIndex);
    }

    assert (fCallbackTy->getNumParams() == callbackParams.size());
    b.CreateCall(fCallbackTy, fCallback, callbackParams);
    b.CreateCondBr(b.CreateICmpNE(nextStrideNo, numOfStrides), readItem, exitBlock);

    if (doneBlock != exitBlock) {
        Function * const fDone = module->getFunction(mDoneCallbackName);
        FunctionType * fDoneTy = fDone->getFunctionType();
        if (fDone == nullptr) {
            llvm::report_fatal_error(llvm::StringRef(mKernelName) + ": failed to get function: " + mDoneCallbackName);
        }
        b.SetInsertPoint(doneBlock);
        b.CreateCall(fDoneTy, fDone, ArrayRef<Value *>({}));
        b.CreateBr(exitBlock);
    }

    b.SetInsertPoint(exitBlock);
}

static std::string ScanReader_GenerateName(StreamSet * scan, std::string const & callbackName) {
    return "ScanReader_" + std::to_string(scan->getNumElements()) + "xscan" + "_" + std::string(callbackName);
}

static std::string ScanReader_GenerateName(StreamSet * scan, std::string const & callbackName, std::initializer_list<StreamSet *> const & additionalStreams) {
    std::string name = ScanReader_GenerateName(scan, callbackName);
    for (auto const & stream : additionalStreams) {
        name += "_" + std::to_string(stream->getNumElements()) + "xi" + std::to_string(stream->getFieldWidth());
    }
    return name;
}

ScanReader::ScanReader(KernelBuilder & b, StreamSet * source, StreamSet * scanIndices, std::string const & callbackName)
: MultiBlockKernel(b, ScanReader_GenerateName(scanIndices, callbackName), {
    {"scan", scanIndices, BoundedRate(0, 1)},
    {"source", source, BoundedRate(0, 1)}
  }, {}, {}, {}, {})
, mCallbackName(callbackName)
, mNumScanStreams(scanIndices->getNumElements())
{
    assert (scanIndices->getFieldWidth() == 64);
    assert (source->getNumElements() == 1);
    addAttribute(SideEffecting());
    setStride(1);
}

ScanReader::ScanReader(KernelBuilder & b, StreamSet * source, StreamSet * scanIndices, std::string const & callbackName, std::string const & doneCallbackName)
: ScanReader(b, source, scanIndices, callbackName)
{
    mDoneCallbackName = doneCallbackName;
}

ScanReader::ScanReader(KernelBuilder & b, StreamSet * source, StreamSet * scanIndices, std::string const & callbackName, std::initializer_list<StreamSet *> additionalStreams)
: MultiBlockKernel(b, ScanReader_GenerateName(scanIndices, callbackName, additionalStreams), {
    {"scan", scanIndices, BoundedRate(0, 1)},
    {"source", source, BoundedRate(0, 1)}
  }, {}, {}, {}, {})
, mCallbackName(callbackName)
, mNumScanStreams(scanIndices->getNumElements())
{
    assert (scanIndices->getFieldWidth() == 64);
    assert (source->getNumElements() == 1);
    addAttribute(SideEffecting());
    setStride(1);
    size_t i = 0;
    assert (mInputStreamSets.size() == 2);
    for (auto const & stream : additionalStreams) {
        std::string name = "__additional_" + std::to_string(i++);
        mInputStreamSets.push_back({name, stream, BoundedRate(0, 1)});
//        mAdditionalStreamNames.push_back(name);
    }
}

ScanReader::ScanReader(KernelBuilder & b, StreamSet * source, StreamSet * scanIndices, std::string const & callbackName, std::string const & doneCallbackName, std::initializer_list<StreamSet *> additionalStreams)
: ScanReader(b, source, scanIndices, callbackName, additionalStreams)
{
    mDoneCallbackName = doneCallbackName;
}

}
