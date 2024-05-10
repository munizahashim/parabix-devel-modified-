/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#include <kernel/streamutils/stream_shift.h>
#include <pablo/builder.hpp>

using namespace llvm;
using namespace pablo;

namespace kernel {

ShiftForward::ShiftForward(KernelBuilder & b, StreamSet * inputs, StreamSet * outputs, unsigned shiftAmount)
: PabloKernel(b, "ShftFwd" + std::to_string(outputs->getNumElements()) + "x1_by" + std::to_string(shiftAmount),
{Binding{"inputs", inputs}}, {Binding{"outputs", outputs}}),
mShiftAmount(shiftAmount)
{   assert(outputs->getNumElements() == inputs->getNumElements());
}

void ShiftForward::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> sourceStreams = getInputStreamSet("inputs");

    for (unsigned i = 0; i < sourceStreams.size(); i++) {
        pb.createAssign(pb.createExtract(getOutput(0), i), pb.createAdvance(sourceStreams[i], mShiftAmount));
    }
}

ShiftBack::ShiftBack(KernelBuilder & b, StreamSet * inputs, StreamSet * outputs, unsigned shiftAmount)
: PabloKernel(b, "ShftBack" + std::to_string(outputs->getNumElements()) + "x1_by" + std::to_string(shiftAmount),
{Binding{"inputs", inputs, FixedRate(1), LookAhead(shiftAmount)}}, {Binding{"outputs", outputs}}),
mShiftAmount(shiftAmount)
{   assert(outputs->getNumElements() == inputs->getNumElements());
}

void ShiftBack::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> sourceStreams = getInputStreamSet("inputs");
    for (unsigned i = 0; i < sourceStreams.size(); i++) {
        pb.createAssign(pb.createExtract(getOutput(0), i), pb.createLookahead(sourceStreams[i], mShiftAmount, "shiftback_" + std::to_string(i)));
    }
}

IndexedAdvance::IndexedAdvance(KernelBuilder & b, StreamSet * inputs, StreamSet * index, StreamSet * outputs, unsigned shiftAmount)
: PabloKernel(b, "IndexedAdvance" + std::to_string(outputs->getNumElements()) + "x1_by" + std::to_string(shiftAmount),
{Binding{"inputs", inputs}, Binding{"index", index}}, {Binding{"outputs", outputs}}),
mShiftAmount(shiftAmount)
{   assert(outputs->getNumElements() == inputs->getNumElements());
}

void IndexedAdvance::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> sourceStreams = getInputStreamSet("inputs");
    PabloAST * index = getInputStreamSet("index")[0];
    for (unsigned i = 0; i < sourceStreams.size(); i++) {
        pb.createAssign(pb.createExtract(getOutput(0), i), pb.createIndexedAdvance(sourceStreams[i], index, mShiftAmount));
    }
}


}
