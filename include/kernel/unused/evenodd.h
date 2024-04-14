/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>
namespace IDISA { class IDISA_Builder; }
namespace llvm { class Value; }

namespace kernel {

class EvenOddKernel final : public BlockOrientedKernel {
public:
    EvenOddKernel(BuilderRef b);
private:
    void generateDoBlockMethod(BuilderRef iBuilder) override;
};

}
