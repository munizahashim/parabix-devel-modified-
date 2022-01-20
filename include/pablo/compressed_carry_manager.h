/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#ifndef COMPRESSED_CARRY_MANAGER_H
#define COMPRESSED_CARRY_MANAGER_H

#include "carry_manager.h"

namespace pablo {

class CompressedCarryManager final : public CarryManager {

public:

    CompressedCarryManager() noexcept;

    void initializeCodeGen(BuilderRef b) override;

    /* Entering and leaving ifs. */

    void enterIfScope(BuilderRef b, const PabloBlock * const scope) override;

protected:

    llvm::StructType * analyse(BuilderRef b, const PabloBlock * const scope, const unsigned ifDepth = 0, const unsigned whileDepth = 0, const bool isNestedWithinNonCarryCollapsingLoop = false) override;

    /* Methods for processing individual carry-generating operations. */

    llvm::Value * shortIndexedAdvanceCarryInCarryOut(BuilderRef b, const unsigned shiftAmount, llvm::Value * const strm, llvm::Value * const index_strm) override;

    llvm::Value * readCarryInSummary(BuilderRef b) const override;

    /* Summary handling routines */

    void combineCarryOutSummary(BuilderRef b, const unsigned offset) override;

    llvm::Value * convertFrameToImplicitSummary(BuilderRef b) const;
    llvm::Value * convertFrameToImplicitSummaryPtr(BuilderRef b) const;
    llvm::Type * getSummaryTypeFromCurrentFrame(BuilderRef b) const override;

private:

    llvm::Type * mBaseSummaryType = nullptr;


};

}

#endif
