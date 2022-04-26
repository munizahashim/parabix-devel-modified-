/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <string>                // for string
#include <kernel/core/kernel_builder.h>

namespace kernel {

class UTF8_Decoder final: public pablo::PabloKernel {
public:
UTF8_Decoder (BuilderRef b, StreamSet * u8_basis, StreamSet * unicode_basis)
    : PabloKernel(b, "UTF8_Decoder", {Binding{"utf8_bit", u8_basis, FixedRate(1), LookAhead(3)}},
                  {Binding{"unicode_bit", unicode_basis}}) {}

    void generatePabloMethod() override;
};

}
