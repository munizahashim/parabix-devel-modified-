#ifndef STRING_INSERT_H
#define STRING_INSERT_H

#include <pablo/pablo_kernel.h>

namespace kernel {
    
//
//  Given a set of insertion amounts (numbers of zeroes to be inserted
//  at indexed positions, and a stream set (possibly multiplexed) identifying
//  position indices at which insertion is to occur, a bixNum stream set is
//  calculated such that the bixNum at position p is n if the zeroes to be
//  inserted at the position is n, or 0 if no insertion is to occur.
//  If the number of streams in the insertMarks stream set is less than
//  the size of the insertion amount vector, then it is interpreted as a
//  multiplexed set, i.e., a bixnum whose index selects the insertion amount
//  to apply at a particular position.
//
//  The result may then be used for calculation of a SpreadMask by InsertionSpreadMask.
//

class StringInsertBixNum final : public pablo::PabloKernel {
public:
    StringInsertBixNum(BuilderRef b, const std::vector<unsigned> &insertAmts,
                       StreamSet * insertMarks, StreamSet * insertBixNum);
    void generatePabloMethod() override;
    bool hasSignature() const override { return true; }
    llvm::StringRef getSignature() const override {
        return mSignature;
    }
private:
    const std::vector<unsigned>  mInsertAmounts;
    const bool                      mMultiplexing;
    const unsigned                  mBixNumBits;
    const std::string               mSignature;
};

class StringReplaceKernel final : public pablo::PabloKernel {
public:
    StringReplaceKernel(BuilderRef b, const std::vector<std::string> & insertStrs,
                        StreamSet * basis, StreamSet * spreadMask,
                        StreamSet * insertMarks, StreamSet * runIndex,
                        StreamSet * output);
    void generatePabloMethod() override;
    bool hasSignature() const override { return true; }
    llvm::StringRef getSignature() const override {
        return mSignature;
    }
private:
    const std::vector<std::string>  mInsertStrings;
    const bool                      mMultiplexing;
    const std::string               mSignature;
};
}

#endif // STRING_INSERT_H
