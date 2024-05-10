/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>

namespace IDISA { class IDISA_Builder; }
namespace llvm { class Function; }
namespace llvm { class Module; }

namespace kernel {

class editdScanKernel : public BlockOrientedKernel {
public:
    editdScanKernel(KernelBuilder & b, StreamSet * matchResults);

private:
    void generateDoBlockMethod(KernelBuilder & b) override;
    llvm::Function * generateScanWordRoutine(KernelBuilder & b) const;

    unsigned mNumElements;
    unsigned mScanwordBitWidth;
};

}

