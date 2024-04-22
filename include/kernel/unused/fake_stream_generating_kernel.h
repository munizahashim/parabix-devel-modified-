
#pragma once

#include <kernel/core/kernel.h>

namespace IDISA { class IDISA_Builder; }

namespace kernel {

    class FakeStreamGeneratingKernel final : public SegmentOrientedKernel {
    public:
        FakeStreamGeneratingKernel(BuilderRef b, StreamSet * refStream, StreamSet * outputStream);
        FakeStreamGeneratingKernel(BuilderRef b, StreamSet * refStream, const StreamSets & outputStreams);
    protected:
        void generateDoSegmentMethod(BuilderRef) final;
    };
}


