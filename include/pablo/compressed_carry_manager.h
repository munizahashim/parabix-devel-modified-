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

protected:

    llvm::StructType * analyse(BuilderRef b, const PabloBlock * const scope) override;

    llvm::Value * shortIndexedAdvanceCarryInCarryOut(BuilderRef b, const unsigned shiftAmount, llvm::Value * const strm, llvm::Value * const index_strm) override;

    llvm::Value * readCarryInSummary(BuilderRef b) const override;

    void writeCurrentCarryOutSummary(BuilderRef b) override;

    void combineCarryOutSummary(BuilderRef b, const unsigned offset) override;

    llvm::Type * getSummaryTypeFromCurrentFrame(BuilderRef b) const override;

};

}

#endif
