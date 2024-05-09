/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <kernel/core/kernel.h>
#include <llvm/IR/Value.h>
#include <string>

/**
 * For every input stream set, SwizzledMultiplePDEPkernel do exactly the same thing as PDEPkernel kernel.
 * However, instead of only handing one swizzled source stream, the SwizzledMultiplePDEPkernel handle
 * the PDEP logic of multiple source streams at the same time to improve the performance in single thread
 * environment.
 */

namespace kernel {

class SwizzledMultiplePDEPkernel final : public MultiBlockKernel {
public:
    SwizzledMultiplePDEPkernel(KernelBuilder & b, const unsigned swizzleFactor = 4, const unsigned numberOfStreamSet = 1, std::string name = "SwizzledMultiplePDEP");
private:
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) final;
private:
    const unsigned mSwizzleFactor;
    const unsigned mNumberOfStreamSet;
};

}

