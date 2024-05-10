/*
 * Part of the Parabix Project, under the Open Software License 3.0.
 * SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <kernel/core/kernel.h>
#include <kernel/pipeline/pipeline_builder.h>

namespace kernel {

/**
 * Kernel driving `AssertEQ` and `AssertNE` functions.
 * 
* Should NOT be used directly. Use either `AssertEQ` or `AssertNE` instead.
 */
class StreamEquivalenceKernel : public MultiBlockKernel {
public:
    enum class Mode { EQ, NE };

    StreamEquivalenceKernel(KernelBuilder & b, Mode mode, StreamSet * x, StreamSet * y, Scalar * outPtr);
    void generateInitializeMethod(KernelBuilder & b) override;
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
    void generateFinalizeMethod(KernelBuilder & b) override;
private:
    const Mode mMode;
};

}

namespace testing {

/**
 * Compares two `StreamsSets` to see if they are equal setting the test case's 
 * state to `failing` if they are not.
 * 
 * Preconditions:
 *  - `lhs` and `rhs` must have the same field width
 *  - `lhs` and `rhs` must have the same number of elements
 */
void AssertEQ(const std::unique_ptr<kernel::ProgramBuilder> & P, kernel::StreamSet * lhs, kernel::StreamSet * rhs);

/**
 * Compares two `StreamsSets` to see if they are equal setting the test case's 
 * state to `failing` if they are.
 * 
 * Preconditions:
 *  - `lhs` and `rhs` must have the same field width
 *  - `lhs` and `rhs` must have the same number of elements
 */
void AssertNE(const std::unique_ptr<kernel::ProgramBuilder> & P, kernel::StreamSet * lhs, kernel::StreamSet * rhs);

/**
 * Prints both `lhs` and `rhs` `StreamsSets` as a mean of debug within the test scripts.
 *
 * Warning:
 * The test case will always fail when this function is called
 * as this is not meant to be used in actual tests
 *
 */
void AssertDebug(const std::unique_ptr<kernel::ProgramBuilder> & P, kernel::StreamSet * lhs, kernel::StreamSet * rhs);

}
