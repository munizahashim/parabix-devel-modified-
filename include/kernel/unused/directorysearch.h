#pragma once

#include <kernel/core/kernel.h>

namespace kernel { class KernelBuilder; }

namespace kernel {

class DirectorySearch final : public SegmentOrientedKernel {
public:
    DirectorySearch(KernelBuilder & iBuilder,
                    Scalar * const rootPath,
                    StreamSet * const directoryNameStream, StreamSet * const fileDirectoryStream, StreamSet * const fileNameStream,
                    const unsigned filesPerSegment = 1024, const bool recursive = true, const bool includeHidden = false);

    void linkExternalMethods(KernelBuilder & b) override;

    void generateInitializeMethod(KernelBuilder & b) override;

    void generateDoSegmentMethod(KernelBuilder & b) override;

    void generateFinalizeMethod(KernelBuilder & b) override;
private:

    void addToOutputStream(KernelBuilder & b, llvm::Value * const name, llvm::Value * const nameLength, llvm::StringRef field, llvm::Value * const consumed);

private:
    const bool mRecursive;
    const bool mIncludeHidden;
    llvm::Function * fOpen;
    llvm::Function * fSysCall;
};

}

