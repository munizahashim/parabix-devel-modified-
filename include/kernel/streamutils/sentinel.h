#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel

namespace kernel {

// Extend an input stream by one position with adding a 1 bit.
class AddSentinel final : public pablo::PabloKernel {
public:
    AddSentinel(KernelBuilder & b,
               StreamSet * const input, StreamSet * const output);
    void generatePabloMethod() override;
};

}

