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

class CCScanKernel : public BlockOrientedKernel {
public:
    CCScanKernel(KernelBuilder & b, unsigned streamNum);

private:
    void generateDoBlockMethod(KernelBuilder & iBuilder) override;
    llvm::Function * generateScanWordRoutine(KernelBuilder & iBuilder) const;

    unsigned mStreamNum;
    unsigned mScanwordBitWidth;
};

}

