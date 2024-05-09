#pragma once

#include <kernel/core/kernel.h>
namespace kernel { class KernelBuilder; }

namespace kernel {

/* The MMapSourceKernel is a simple wrapper for an external MMap file buffer.
   The doSegment method of this kernel feeds one segment at a time to a
   pipeline. */

class MMapSourceKernel final : public SegmentOrientedKernel {
    friend class FDSourceKernel;
public:
    MMapSourceKernel(KernelBuilder & b, Scalar * const fd, StreamSet * const outputStream);
    void linkExternalMethods(KernelBuilder & b) override;
    void generateInitializeMethod(KernelBuilder & b) override {
        generateInitializeMethod(mCodeUnitWidth, mStride, b);
    }
    void generateDoSegmentMethod(KernelBuilder & b) override {
        generateDoSegmentMethod(mCodeUnitWidth, mStride, b);
    }
    void generateFinalizeMethod(KernelBuilder & b) override {
        freeBuffer(b, mCodeUnitWidth);
    }
    llvm::Value * generateExpectedOutputSizeMethod(KernelBuilder & b) override {
        return generateExpectedOutputSizeMethod(mCodeUnitWidth, b);
    }
protected:
    static void generatLinkExternalFunctions(KernelBuilder & b);
    static void generateInitializeMethod(const unsigned codeUnitWidth, const unsigned stride, KernelBuilder & b);
    static void generateDoSegmentMethod(const unsigned codeUnitWidth, const unsigned stride, KernelBuilder & b);
    static llvm::Value * generateExpectedOutputSizeMethod(const unsigned codeUnitWidth, KernelBuilder & b);
    static void freeBuffer(KernelBuilder & b, const unsigned codeUnitWidth);
protected:
    const unsigned mCodeUnitWidth;
};

class ReadSourceKernel final : public SegmentOrientedKernel {
    friend class FDSourceKernel;
public:
    ReadSourceKernel(KernelBuilder & b, Scalar * const fd, StreamSet * const outputStream);
    void linkExternalMethods(KernelBuilder & b) override;
    void generateInitializeMethod(KernelBuilder & b) override {
        generateInitializeMethod(mCodeUnitWidth, mStride, b);
    }
    void generateDoSegmentMethod(KernelBuilder & b) override {
        generateDoSegmentMethod(mCodeUnitWidth, mStride, b);
    }
    void generateFinalizeMethod(KernelBuilder & b) override {
        freeBuffer(b);
    }
    llvm::Value * generateExpectedOutputSizeMethod(KernelBuilder & b) override {
        return generateExpectedOutputSizeMethod(mCodeUnitWidth, b);
    }
protected:
    static void generatLinkExternalFunctions(KernelBuilder & b);
    static void generateInitializeMethod(const unsigned codeUnitWidth, const unsigned stride, KernelBuilder & b);
    static void generateDoSegmentMethod(const unsigned codeUnitWidth, const unsigned stride, KernelBuilder & b);
    static llvm::Value * generateExpectedOutputSizeMethod(const unsigned codeUnitWidth, KernelBuilder & b);
    static void freeBuffer(KernelBuilder & b);
    static void createInternalBuffer(KernelBuilder & b);
private:
    const unsigned mCodeUnitWidth;
};

class FDSourceKernel final : public SegmentOrientedKernel {
public:
    FDSourceKernel(KernelBuilder & b, Scalar * const useMMap, Scalar * const fd, StreamSet * const outputStream);
    void linkExternalMethods(KernelBuilder & b) override;
    void generateInitializeMethod(KernelBuilder & b) override;
    void generateDoSegmentMethod(KernelBuilder & b) override;
    void generateFinalizeMethod(KernelBuilder & b) override;
    llvm::Value * generateExpectedOutputSizeMethod(KernelBuilder &) override;
protected:
    const unsigned mCodeUnitWidth;
};

class MemorySourceKernel final : public SegmentOrientedKernel {
public:
    MemorySourceKernel(KernelBuilder & b, Scalar * fileSource, Scalar * fileItems, StreamSet * const outputStream);
protected:
    void generateInitializeMethod(KernelBuilder & b) override;
    void generateDoSegmentMethod(KernelBuilder & b) override;
    void generateFinalizeMethod(KernelBuilder & b) override;
    llvm::Value * generateExpectedOutputSizeMethod(KernelBuilder &) override;
};

}

