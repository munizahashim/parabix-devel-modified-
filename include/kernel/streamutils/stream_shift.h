/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <vector>

namespace IDISA { class IDISA_Builder; }

namespace kernel {

class ShiftForward final : public pablo::PabloKernel {
public:
    ShiftForward(KernelBuilder & b, StreamSet * inputs, StreamSet * outputs, unsigned shiftAmount = 1);
protected:
    void generatePabloMethod() override;
    unsigned mShiftAmount;
};

class ShiftBack final : public pablo::PabloKernel {
public:
    ShiftBack(KernelBuilder & b, StreamSet * inputs, StreamSet * outputs, unsigned shiftAmount = 1);
protected:
    void generatePabloMethod() override;
    unsigned mShiftAmount;
};

class IndexedAdvance final : public pablo::PabloKernel {
public:
    IndexedAdvance(KernelBuilder & b, StreamSet * inputs, StreamSet * index, StreamSet * outputs, unsigned shiftAmount = 1);
protected:
    void generatePabloMethod() override;
    unsigned mShiftAmount;
};

}

