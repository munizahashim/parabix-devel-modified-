/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <kernel/core/relationship.h>
#include <kernel/core/kernel.h>

using namespace kernel;
namespace bixnum {
    class Add final : public pablo::PabloKernel {
    public:
        Add(KernelBuilder & kb, StreamSet * a, StreamSet * b, StreamSet * sum)
        : pablo::PabloKernel(kb, "Add_" + a->shapeString() + "+" + b->shapeString() + ":" + sum->shapeString(),
                             {Binding{"a", a}, Binding{"b", b}}, {Binding{"sum", sum}}),
        mBixBits(sum->getNumElements()) {
        }
    protected:
        void generatePabloMethod() override;
    private:
        const unsigned mBixBits;
    };
}
