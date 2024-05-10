#pragma once

#include <kernel/core/kernel.h>
namespace kernel { class KernelBuilder; }

namespace kernel {

class RandomStreamKernel final : public SegmentOrientedKernel {
public:
    RandomStreamKernel(KernelBuilder & b, unsigned seed, unsigned valueWidth, size_t streamLength);
    void generateInitializeMethod(KernelBuilder & b) override;
    void generateDoSegmentMethod(KernelBuilder & b) override;
protected:
    const unsigned mSeed;
    const unsigned mValueWidth;
    const size_t mStreamLength;
};

}

