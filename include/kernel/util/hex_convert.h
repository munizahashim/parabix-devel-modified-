/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <kernel/core/kernel.h>

namespace kernel {
//
// Convert a byte stream of hexadecimal values into a bit stream.
// Each hexadecimal input byte generates 4 bits to the output stream.
//

class HexToBinary final : public kernel::BlockOrientedKernel {
public:
    HexToBinary(BuilderRef b, StreamSet * hexStream, StreamSet * binStream);
protected:
    void generateDoBlockMethod(BuilderRef b) override;
};

//
// Produce a hexadecimal output stream with one hexadecimal byte
// for each 4 bits of an input bit stream.
//

class BinaryToHex final : public kernel::BlockOrientedKernel {
public:
    BinaryToHex(BuilderRef b, StreamSet * binStream, StreamSet * hexStream);
protected:
    void generateDoBlockMethod(BuilderRef b) override;
    void generateFinalBlockMethod(BuilderRef b, llvm::Value * const remainingBits) override;
};

}
