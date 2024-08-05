/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <re/alphabet/alphabet.h>

namespace kernel { class KernelBuilder; }
namespace IDISA { class IDISA_Builder; }
namespace re { class RE; class CC; }
namespace kernel {

class CharClassesKernel final : public pablo::PabloKernel {
public:
    CharClassesKernel(KernelBuilder & b, std::vector<re::CC *> ccs, StreamSet * BasisBits, StreamSet * CharClasses);
    bool hasSignature() const override { return true; }
    llvm::StringRef getSignature() const override;
protected:
    CharClassesKernel(KernelBuilder & b, std::string signature, std::vector<re::CC *> && ccs, StreamSet * BasisBits, StreamSet * CharClasses);
    void generatePabloMethod() override;
protected:
    const std::vector<re::CC *> mCCs;
    const std::string mSignature;
};


class ByteClassesKernel final : public pablo::PabloKernel {
public:
    ByteClassesKernel(KernelBuilder & b, std::vector<re::CC *> ccs, StreamSet * inputStream, StreamSet * CharClasses);
    bool hasSignature() const override { return true; }
    llvm::StringRef getSignature() const override;
protected:
    ByteClassesKernel(KernelBuilder & b, std::string signature, std::vector<re::CC *> && ccs, StreamSet * inputStream, StreamSet * CharClasses);
    void generatePabloMethod() override;
protected:
    const std::vector<re::CC *> mCCs;
    const std::string mSignature;
};


}
