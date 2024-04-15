#pragma once

#include <idisa/idisa_builder.h>

namespace IDISA {

const unsigned ARM_width = 128;

class IDISA_ARM_Builder : public virtual IDISA_Builder {
public:
    static const unsigned NativeBitBlockWidth = ARM_width;
    IDISA_ARM_Builder(llvm::LLVMContext & C, unsigned bitBlockWidth, unsigned laneWidth)
    : IDISA_Builder(C, ARM_width, bitBlockWidth, laneWidth) {

    }
    virtual std::string getBuilderUniqueName() override;
    llvm::Value * hsimd_signmask(unsigned fw, llvm::Value * a) override;
    llvm::Value * esimd_mergeh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * esimd_mergel(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * mvmd_compress(unsigned fw, llvm::Value * a, llvm::Value * select_mask) override;
    llvm::Value * hsimd_packh(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packl(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    llvm::Value * hsimd_packus(unsigned fw, llvm::Value * a, llvm::Value * b) override;
    std::pair<llvm::Value *, llvm::Value *> bitblock_advance(llvm::Value * a, llvm::Value * shiftin, unsigned shift) override;
    llvm::Value * mvmd_shuffle(unsigned fw, llvm::Value * data_table, llvm::Value * index_vector) override;

    ~IDISA_ARM_Builder() {}
};

}

