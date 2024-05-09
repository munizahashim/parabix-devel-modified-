#pragma once

#include <kernel/core/kernel.h>

namespace kernel {

class RegionSelectionKernel final : public MultiBlockKernel {
public:

    #define STREAMSET_WITH_INDEX(NAME) \
        struct NAME { \
            StreamSet * const Stream; \
            const unsigned    Index; \
            NAME(StreamSet * stream, unsigned index) : Stream(stream), Index(index) { }; \
            NAME(std::pair<StreamSet *, unsigned> p) : Stream(std::get<0>(p)), Index(std::get<1>(p)) { }; \
        }
    STREAMSET_WITH_INDEX(Demarcators);
    STREAMSET_WITH_INDEX(Starts);
    STREAMSET_WITH_INDEX(Ends);
    STREAMSET_WITH_INDEX(Selectors);
    #undef STREAMSET_WITH_INDEX

    explicit RegionSelectionKernel(KernelBuilder & b, Starts starts, Ends ends, StreamSet * const regionSpans);

    explicit RegionSelectionKernel(KernelBuilder & b, Demarcators, Selectors selectors, StreamSet * const regionSpans);

    explicit RegionSelectionKernel(KernelBuilder & b, Starts starts, Ends ends, Selectors selectors, StreamSet * const regionSpans);

    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) final;

protected:

    llvm::Value * getRegionStarts(KernelBuilder & b, llvm::Value * const offset) const;
    llvm::Value * getNumOfRegionStarts(KernelBuilder & b) const;
    llvm::Value * getRegionEnds(KernelBuilder & b, llvm::Value * const offset) const;
    llvm::Value * getNumOfRegionEnds(KernelBuilder & b) const;
    llvm::Value * getSelectorStream(KernelBuilder & b, llvm::Value * const offset) const;
    LLVM_READNONE bool hasIndependentStartEndStreams() const;
    LLVM_READNONE bool hasSelectorStream() const;


private:

    const unsigned mStartStreamIndex;
    const unsigned mEndStreamIndex;
    const unsigned mSelectorStreamIndex;
    const bool mSelectorsAreAlignedWithRegionEnds;
    const bool mAlwaysExtendSelectedRegionsToRegionEnds;

};

}

