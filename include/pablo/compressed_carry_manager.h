/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

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

