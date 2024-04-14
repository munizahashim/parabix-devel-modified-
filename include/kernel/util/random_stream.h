#pragma once

#include <kernel/core/kernel.h>
namespace kernel { class KernelBuilder; }

namespace kernel {

class RandomStreamKernel final : public SegmentOrientedKernel {
public:
    RandomStreamKernel(BuilderRef iBuilder, unsigned seed, unsigned valueWidth, size_t streamLength);
    void generateInitializeMethod(BuilderRef iBuilder) override;
    void generateDoSegmentMethod(BuilderRef iBuilder) override;
protected:
    const unsigned mSeed;
    const unsigned mValueWidth;
    const size_t mStreamLength;
};

}

