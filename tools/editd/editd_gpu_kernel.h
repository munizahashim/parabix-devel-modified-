/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>

namespace llvm { class Module; }

namespace IDISA { class IDISA_Builder; }

namespace kernel {

class editdGPUKernel : public BlockOrientedKernel {
public:

    editdGPUKernel(KernelBuilder & b, unsigned dist, unsigned pattLen, unsigned groupSize);


private:
    void generateDoBlockMethod(KernelBuilder & idb) override;
    void generateFinalBlockMethod(KernelBuilder & idb, llvm::Value * remainingBytes) override;
    unsigned mEditDistance;
    unsigned mPatternLen;
    unsigned mGroupSize;

};

}
