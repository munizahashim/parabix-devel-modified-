
#pragma once

#include <kernel/core/kernel.h>

namespace IDISA { class IDISA_Builder; }

namespace kernel {

    class FakeStreamGeneratingKernel final : public SegmentOrientedKernel {
    public:
        FakeStreamGeneratingKernel(KernelBuilder & b, StreamSet * refStream, StreamSet * outputStream);
        FakeStreamGeneratingKernel(KernelBuilder & b, StreamSet * refStream, const StreamSets & outputStreams);
    protected:
        void generateDoSegmentMethod(KernelBuilder &) final;
    };
}


