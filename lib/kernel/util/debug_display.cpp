/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <toolchain/toolchain.h>
#include <kernel/streamutils/stream_select.h>
#include <kernel/util/debug_display.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/basis/s2p_kernel.h>
#include <kernel/basis/p2s_kernel.h>
#include <pablo/pabloAST.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <pablo/pe_ones.h>
#include <pablo/bixnum/bixnum.h>
#include <re/cc/cc_compiler.h>
#include <iostream>

using namespace llvm;

namespace kernel {


/* Helper classes for ParabixIllustrator. */

class CaptureBlock : public kernel::BlockOrientedKernel {
public:
    CaptureBlock(KernelBuilder & b, Scalar * accumObj, Scalar * streamNo, StreamSet * byteStream);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
    void generateFinalBlockMethod(KernelBuilder & b, llvm::Value * const remainingByte) override;
};

class BitstreamIllustrator : public pablo::PabloKernel {
public:
    BitstreamIllustrator(KernelBuilder & b, StreamSet * bits, StreamSet * displayBasis, char zeroCh = '.', char oneCh = '1');
protected:
    void generatePabloMethod() override;

private:
    char mZeroCh;
    char mOneCh;
};

class PrintableASCII : public pablo::PabloKernel {
public:
    PrintableASCII(KernelBuilder & b, StreamSet * basisBits, StreamSet * printableBasis, char nonASCIIsubstitute = '.');
protected:
    void generatePabloMethod() override;

private:
    char mNonASCIIsubstitute;
};

class PrintableBixNum : public pablo::PabloKernel {
public:
    PrintableBixNum(KernelBuilder & b, StreamSet * bixnum, StreamSet * printable, char hexBase = 'A');
protected:
    void generatePabloMethod() override;

private:
    char mHexBase;
};

static std::string GenerateName(StringRef name, StreamSet * s) {
    return std::string("DebugDisplay::") + 
           "<i" + std::to_string(s->getFieldWidth()) + ">[" + std::to_string(s->getNumElements()) + "]" +
           "@" + name.str();
}

DebugDisplayKernel::DebugDisplayKernel(KernelBuilder & b, StringRef name, StreamSet * s)
: MultiBlockKernel(b, GenerateName(name, s), {{"s", s}}, {}, {}, {}, {InternalScalar{b.getSizeTy(), "initialStride"}})
, mName(name)
, mFW(s->getFieldWidth())
, mSCount(s->getNumElements())
{
    if (mFW != 1) {
        setStride(1);
    } else {
        setStride(b.getBitBlockWidth());
    }
    addAttribute(SideEffecting());
}

void DebugDisplayKernel::generateMultiBlockLogic(KernelBuilder & b, Value * const numOfStrides) {
    auto getRegName = [&](uint32_t i) -> std::string {
        if (mSCount != 1) {
            return std::string(mName) + "[" + std::to_string(i) + "]";
        } else {
            return std::string(mName);
        }
    };

    bool useBitblocks = mFW == 1;

    Type * const sizeTy = b.getSizeTy();
    Value * const sz_ZERO = b.getSize(0);
    Value * const sz_ONE = b.getSize(1);

    BasicBlock * const entry = b.GetInsertBlock();
    BasicBlock * const loop = b.CreateBasicBlock("loop");
    BasicBlock * const exit = b.CreateBasicBlock("exit");

    Value * initialStride = nullptr;
    if (!useBitblocks) {
        initialStride = b.getScalarField("initialStride");
    }
    
    if (!useBitblocks) {
        // Since stride width == 1, an extra final call to this kernel is made.
        // We don't want to print anything out on this final call.
        b.CreateCondBr(b.isFinal(), exit, loop);
    } else {
        b.CreateBr(loop);
    }

    b.SetInsertPoint(loop);
    PHINode * const strideNum = b.CreatePHI(sizeTy, 2);
    strideNum->addIncoming(sz_ZERO, entry);
    if (useBitblocks) {
        // strideNum is equivalent to the block number since stride width == bitblock width
        for (uint32_t i = 0; i < mSCount; ++i) {
            Value * const block = b.loadInputStreamBlock("s", b.getInt32(i), strideNum);
            b.CallPrintRegister(getRegName(i), block);
        }
    } else {
        IntegerType * const fieldTy = b.getIntNTy(mFW);
        for (uint32_t i = 0; i < mSCount; ++i) {
            Value * const ptr = b.getRawInputPointer("s", b.getInt32(i), b.CreateAdd(strideNum, initialStride));
            Value * const val = b.CreateLoad(fieldTy, ptr);
            b.CallPrintInt(getRegName(i), val);
        }
    }
    Value * const nextStrideNum = b.CreateAdd(strideNum, sz_ONE);
    strideNum->addIncoming(nextStrideNum, loop);
    if (!useBitblocks) {
        b.setScalarField("initialStride", b.CreateAdd(nextStrideNum, initialStride));
    }
    b.CreateCondBr(b.CreateICmpNE(nextStrideNum, numOfStrides), loop, exit);

    b.SetInsertPoint(exit);
}

extern "C" void appendStreamText_wrapper(intptr_t illustrator_addr, uint64_t streamNo, char * streamText, uint64_t lgth) {
    assert ("passed a null accumulator" && illustrator_addr);
    std::string text = std::string(streamText, lgth);
    reinterpret_cast<ParabixIllustrator *>(illustrator_addr)->appendStreamText(streamNo, text);
}

unsigned ParabixIllustrator::addStream(std::string streamName) {
    unsigned streamNo = mStreamNames.size();
    mStreamNames.push_back(streamName);
    mStreamData.push_back("");
    if (streamName.size() > mMaxStreamNameSize) mMaxStreamNameSize = streamName.size();
    return streamNo;
}

void ParabixIllustrator::appendStreamText(unsigned streamNo, std::string streamText) {
    mStreamData[streamNo].append(streamText);
}

void ParabixIllustrator::registerIllustrator(Scalar * illustrator) {
    if (mDisplayWidth == 0) {
        llvm::report_fatal_error("Parabix illustrator: display width cannot be 0");
    }
    mIllustrator = illustrator;
}

void ParabixIllustrator::captureByteData(ProgramBuilderRef P, std::string streamLabel, StreamSet * byteData, char nonASCIIsubstitute) {
    unsigned illustratedStreamNo = addStream(streamLabel);
    StreamSet * basis = P->CreateStreamSet(8);
    P->CreateKernelCall<S2PKernel>(byteData, basis);
    StreamSet * printableBasis = P->CreateStreamSet(8);
    P->CreateKernelCall<PrintableASCII>(basis, printableBasis, nonASCIIsubstitute);
    StreamSet * printableData = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<P2SKernel>(printableBasis, printableData);
    Scalar * streamNo = P->CreateConstant(P->getDriver().getBuilder()->getSize(illustratedStreamNo));
    Kernel * scK = P->CreateKernelCall<CaptureBlock>(mIllustrator, streamNo, printableData);
    scK->link("appendStreamText_wrapper", appendStreamText_wrapper);
}

void ParabixIllustrator::captureBitstream(ProgramBuilderRef P, std::string streamLabel, StreamSet * bitstream, char zeroCh, char oneCh) {
    unsigned illustratedStreamNo = addStream(streamLabel);
    StreamSet * printableBasis = P->CreateStreamSet(8);
    P->CreateKernelCall<BitstreamIllustrator>(bitstream, printableBasis, zeroCh, oneCh);
    StreamSet * printableData = P->CreateStreamSet(1, 8);
    P->CreateKernelCall<P2SKernel>(printableBasis, printableData);
    Scalar * streamNo = P->CreateConstant(P->getDriver().getBuilder()->getSize(illustratedStreamNo));
    Kernel * scK = P->CreateKernelCall<CaptureBlock>(mIllustrator, streamNo, printableData);
    scK->link("appendStreamText_wrapper", appendStreamText_wrapper);
}

void ParabixIllustrator::captureBixNum(ProgramBuilderRef P, std::string streamLabel, StreamSet * bixnum, char hexBase) {
    auto bixBits = bixnum->getNumElements();
    if (bixBits <= 4) {
        unsigned illustratedStreamNo = addStream(streamLabel);
        StreamSet * printableBasis = P->CreateStreamSet(8);
        P->CreateKernelCall<PrintableBixNum>(bixnum, printableBasis, hexBase);
        StreamSet * printableData = P->CreateStreamSet(1, 8);
        P->CreateKernelCall<P2SKernel>(printableBasis, printableData);
        Scalar * streamNo = P->CreateConstant(P->getDriver().getBuilder()->getSize(illustratedStreamNo));
        Kernel * scK = P->CreateKernelCall<CaptureBlock>(mIllustrator, streamNo, printableData);
        scK->link("appendStreamText_wrapper", appendStreamText_wrapper);
    } else {
        auto hexDigits = (bixBits + 3)/4;
        for (auto i = hexDigits; i >= 1; i--) {
            auto low = (i - 1) * 4;
            auto hi = bixBits;
            std::string lbl = streamLabel + "[" + std::to_string(low) + "-" + std::to_string(hi - 1) + "]";
            StreamSet * hexBasis = streamutils::Select(P, bixnum, streamutils::Range(low, hi));
            captureBixNum(P, lbl, hexBasis, hexBase);
            bixBits = low;
        }
    }
}

void ParabixIllustrator::displayAllCapturedData() {
    unsigned maxStreamSize = 0;
    for (unsigned i = 0; i < mStreamData.size(); i++) {
        if (mStreamData[i].size() > maxStreamSize) maxStreamSize = mStreamData[i].size();
    }
    unsigned pages = (maxStreamSize + mDisplayWidth - 1)/ mDisplayWidth;
    for (unsigned page = 0; page < pages; page++) {
        unsigned pagePos = mDisplayWidth * page;
        for (unsigned i = 0; i < mStreamData.size(); i++) {
            std::cerr << std::setw(mMaxStreamNameSize) << mStreamNames[i]  << " | ";
            int avail = mStreamData[i].size() - pagePos;
            if (avail > 0) {
                unsigned width = std::min(static_cast<unsigned>(avail), mDisplayWidth);
                std::cerr << mStreamData[i].substr(pagePos, width);
            }
            std::cerr << "\n";
        }
        std::cerr << "\n";
    }
}

BitstreamIllustrator::BitstreamIllustrator(KernelBuilder & b, StreamSet * bits, StreamSet * displayBasis, char zeroCh, char oneCh)
    : pablo::PabloKernel(b, "BitstreamIllustrator" + std::to_string(zeroCh) + "_" + std::to_string(oneCh),
                  {Binding{"bits", bits}},
                  {Binding{"displayBasis", displayBasis}}),
                  mZeroCh(zeroCh), mOneCh(oneCh) {}

void BitstreamIllustrator::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::PabloAST * data = getInputStreamSet("bits")[0];

    pablo::Var * displayBasis = getOutputStreamVar("displayBasis");
    for (unsigned i = 0; i < 8; i++) {
        unsigned bit = (unsigned) 1 << i;
        pablo::PabloAST * displayBit = nullptr;
        if ((bit & mOneCh & mZeroCh) == bit) {
            displayBit = pb.createOnes();
        }
        else if (((bit & mOneCh) == 0) && ((bit & mZeroCh) == 0)) {
            displayBit = pb.createZeroes();
        }
        else if (((bit & mOneCh) == bit) && ((bit & mZeroCh) == 0)) {
            displayBit = data;
        }
        else {
            displayBit = pb.createNot(data);
       }
       pb.createAssign(pb.createExtract(displayBasis, pb.getInteger(i)), displayBit);
    }
}

PrintableASCII::PrintableASCII(KernelBuilder & b, StreamSet * basisBits, StreamSet * printableBasis, char nonASCIIsubstitute)
    : pablo::PabloKernel(b, "PrintableASCII" + std::to_string(nonASCIIsubstitute),
                  {Binding{"basisBits", basisBits}},
                  {Binding{"printableBasis", printableBasis}}),
                  mNonASCIIsubstitute(nonASCIIsubstitute) {}

void PrintableASCII::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<pablo::PabloAST *> basisBits = getInputStreamSet("basisBits");
    cc::Parabix_CC_Compiler ccc(getEntryScope(), basisBits);
    PabloAST * isPrintable = ccc.compileCC(re::makeCC(0x20, 0x7E));
    PabloAST * notPrintable = pb.createNot(isPrintable);
    pablo::Var * printableVar = getOutputStreamVar("printableBasis");
    for (unsigned i = 0; i < 8; i++) {
        unsigned bit = (unsigned) 1 << i;
        PabloAST * displayBit = basisBits[i];
        if ((bit & mNonASCIIsubstitute) == bit) {
            displayBit = pb.createOr(displayBit, notPrintable);
        } else {
            displayBit = pb.createAnd(displayBit, isPrintable);
        }
       pb.createAssign(pb.createExtract(printableVar, pb.getInteger(i)), displayBit);
    }
}

PrintableBixNum::PrintableBixNum(KernelBuilder & b, StreamSet * bixnum, StreamSet * printableBasis, char hexBase)
    : pablo::PabloKernel(b, "PrintableBixNum_x" + std::to_string(bixnum->getNumElements()) + hexBase,
                  {Binding{"bixnum", bixnum}},
                  {Binding{"printableBasis", printableBasis}}), mHexBase(hexBase) {}

void PrintableBixNum::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::BixNum num = getInputStreamSet("bixnum");
    if (num.size() > 4) num = bnc.Truncate(num, 4);
    num = bnc.ZeroExtend(num, 7);
    pablo::BixNum digits = bnc.AddModular(num, 0x30);   //  ASCII for [0-9]
    pablo::BixNum hex = bnc.AddModular(num, mHexBase - 10);  // ASCII for hex digits [A-F] or [a-f]
    pablo::BixNum result = bnc.Select(bnc.UGE(num, 10), hex, digits);
    pablo::Var * printableVar = getOutputStreamVar("printableBasis");
    for (unsigned i = 0; i < result.size(); i++) {
       pb.createAssign(pb.createExtract(printableVar, pb.getInteger(i)), result[i]);
    }
    for (unsigned i = result.size(); i < 8; i++) {
       pb.createAssign(pb.createExtract(printableVar, pb.getInteger(i)), pb.createZeroes());
    }
}

CaptureBlock::CaptureBlock(KernelBuilder & b, Scalar * accumObj, Scalar * streamNo, StreamSet * byteStream)
: BlockOrientedKernel(b, "CallBack",
                      {Binding{"byteStream", byteStream}},
                      {},
                      {Binding{"accumObj", accumObj}, Binding{"streamNo", streamNo}}, //input scalars
                      {},
                      {}) {
                          addAttribute(SideEffecting());
                      }

void CaptureBlock::generateDoBlockMethod(KernelBuilder & b) {
    Value * byteStreamBasePtr = b.CreatePointerCast(b.getInputStreamBlockPtr("byteStream", b.getSize(0)), b.getInt8PtrTy());
    Value * accumObj = b.getScalarField("accumObj");
    Value * streamNo = b.getScalarField("streamNo");
    Function * callback = b.getModule()->getFunction("appendStreamText_wrapper");
    FunctionType * fty = callback->getFunctionType();
    b.CreateCall(fty, callback, {accumObj, streamNo, byteStreamBasePtr, b.getSize(codegen::BlockSize)});
}

void CaptureBlock::generateFinalBlockMethod(KernelBuilder & b, Value * const remainingBytes) {
    Value * byteStreamBasePtr = b.CreatePointerCast(b.getInputStreamBlockPtr("byteStream", b.getSize(0)), b.getInt8PtrTy());
    Value * accumObj = b.getScalarField("accumObj");
    Value * streamNo = b.getScalarField("streamNo");
    Function * callback = b.getModule()->getFunction("appendStreamText_wrapper");
    FunctionType * fty = callback->getFunctionType();
    b.CreateCall(fty, callback, {accumObj, streamNo, byteStreamBasePtr, remainingBytes});
}


}
