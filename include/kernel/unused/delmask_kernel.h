/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <string>                // for string
#include <kernel/core/kernel_builder.h>

namespace kernel {

class DelMaskKernelBuilder final: public pablo::PabloKernel {
public:

    DelMaskKernelBuilder (KernelBuilder & iBuilder)
    : PabloKernel(iBuilder, "delmask_kernel", {Binding{iBuilder->getStreamSetTy(8, 1), "u8bit"}},
                       {Binding{iBuilder->getStreamSetTy(1, 1), "delMask"},
                        Binding{iBuilder->getStreamSetTy(1, 1), "neg_delMask"},
                        Binding{iBuilder->getStreamSetTy(1, 1), "errMask"}}, {}) {

    }

    void generatePabloMethod() override;

};

}
