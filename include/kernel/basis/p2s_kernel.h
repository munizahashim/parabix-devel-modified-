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
    P2SKernel(KernelBuilder & b,
              StreamSet * basisBits,
              StreamSet * byteStream);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
};


class P2SMultipleStreamsKernel final : public BlockOrientedKernel {
public:
    P2SMultipleStreamsKernel(KernelBuilder & b,
                             const StreamSets & inputStreams,
                             StreamSet * const outputStream);
protected:
    void generateDoBlockMethod(KernelBuilder & b) override;
private:
};

class P2SKernelWithCompressedOutput final : public BlockOrientedKernel {
public:
    P2SKernelWithCompressedOutput(KernelBuilder & b);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
};

class P2S16Kernel final : public BlockOrientedKernel {
public:
    P2S16Kernel(KernelBuilder & b, StreamSet * u16bits, StreamSet * u16stream, cc::ByteNumbering endianness = cc::ByteNumbering::LittleEndian);
    P2S16Kernel(KernelBuilder & b, StreamSets & inputSets, StreamSet * u16stream, cc::ByteNumbering endianness = cc::ByteNumbering::LittleEndian);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
    cc::ByteNumbering mByteNumbering;
};

class P2S16KernelWithCompressedOutput final : public BlockOrientedKernel {
public:
    P2S16KernelWithCompressedOutput(KernelBuilder & b,
                                    StreamSet * basisBits, StreamSet * fieldCounts, StreamSet * i16Stream, cc::ByteNumbering endianness = cc::ByteNumbering::LittleEndian);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
    cc::ByteNumbering mByteNumbering;
};

class P2S16KernelWithCompressedOutputOld final : public BlockOrientedKernel {
public:
    P2S16KernelWithCompressedOutputOld(KernelBuilder & b,
                                       StreamSet * basisBits, StreamSet * delCounts, StreamSet * byteStream);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
};

class P2S21Kernel final : public BlockOrientedKernel {
public:
    P2S21Kernel(KernelBuilder & b, StreamSet * u21bits, StreamSet * u32stream, cc::ByteNumbering = cc::ByteNumbering::LittleEndian);
private:
    void generateDoBlockMethod(KernelBuilder & b) override;
    cc::ByteNumbering mByteNumbering;
};

}

