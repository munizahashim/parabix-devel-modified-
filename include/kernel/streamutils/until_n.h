/*
 *  Copyright (c) 2023 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */
#pragma once

#include <kernel/core/kernel.h>

/*
   The UntilNkernel copies the initial prefix of a marker bitstream
   up to and including the position of the Nth one bit.   In the mode
   ZeroAfterN, the remainder of the output bit stream is zeroed out,
   up to the size of the input marker stream.   In the mode TerminateAtN,
   the output stream is terminated at the position of the Nth one bit.
*/

namespace kernel {

class UntilNkernel final : public MultiBlockKernel {
public:
    enum class Mode {ZeroAfterN, TerminateAtN};
    UntilNkernel(BuilderRef b, Scalar * N, StreamSet * Markers, StreamSet * FirstN,
                 UntilNkernel::Mode m = UntilNkernel::Mode::TerminateAtN);
private:
    void generateMultiBlockLogic(BuilderRef b, llvm::Value * const numOfStrides) final;
    UntilNkernel::Mode mMode;
};

}
