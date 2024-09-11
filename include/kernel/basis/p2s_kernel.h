/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <re/alphabet/alphabet.h>
#include <kernel/core/kernel.h>

namespace IDISA { class IDISA_Builder; }

namespace kernel {

class P2SKernel final : public BlockOrientedKernel {
public:
    P2SKernel(LLVMTypeSystemInterface & ts,
              StreamSet * basisBits,
              StreamSet * byteStream);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
};


class P2SMultipleStreamsKernel final : public BlockOrientedKernel {
public:
    P2SMultipleStreamsKernel(LLVMTypeSystemInterface & ts,
                             const StreamSets & inputStreams,
                             StreamSet * const outputStream);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
private:
};

class P2SKernelWithCompressedOutput final : public BlockOrientedKernel {
public:
    P2SKernelWithCompressedOutput(LLVMTypeSystemInterface & ts);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
};

class P2S16Kernel final : public BlockOrientedKernel {
public:
    P2S16Kernel(LLVMTypeSystemInterface & ts, StreamSet * u16bits, StreamSet * u16stream, cc::ByteNumbering endianness = cc::ByteNumbering::LittleEndian);
    P2S16Kernel(LLVMTypeSystemInterface & ts, StreamSets & inputSets, StreamSet * u16stream, cc::ByteNumbering endianness = cc::ByteNumbering::LittleEndian);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
    cc::ByteNumbering mByteNumbering;
};

class P2S16KernelWithCompressedOutput final : public BlockOrientedKernel {
public:
    P2S16KernelWithCompressedOutput(LLVMTypeSystemInterface & ts,
                                    StreamSet * basisBits, StreamSet * fieldCounts, StreamSet * i16Stream, cc::ByteNumbering endianness = cc::ByteNumbering::LittleEndian);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
    cc::ByteNumbering mByteNumbering;
};

class P2S16KernelWithCompressedOutputOld final : public BlockOrientedKernel {
public:
    P2S16KernelWithCompressedOutputOld(LLVMTypeSystemInterface & ts,
                                       StreamSet * basisBits, StreamSet * delCounts, StreamSet * byteStream);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
};

class P2S21Kernel final : public BlockOrientedKernel {
public:
    P2S21Kernel(LLVMTypeSystemInterface & ts, StreamSet * u21bits, StreamSet * u32stream, cc::ByteNumbering = cc::ByteNumbering::LittleEndian);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
    cc::ByteNumbering mByteNumbering;
};

}

