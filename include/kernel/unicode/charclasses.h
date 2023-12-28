/*
 *  Copyright (c) 2017 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */
#ifndef CHARCLASSES_H
#define CHARCLASSES_H

#include <pablo/pablo_kernel.h>  // for PabloKernel
#include <re/alphabet/alphabet.h>

namespace kernel { class KernelBuilder; }
namespace IDISA { class IDISA_Builder; }
namespace re { class RE; class CC; }
namespace kernel {

class CharClassesKernel final : public pablo::PabloKernel {
public:
    CharClassesKernel(BuilderRef b, std::vector<re::CC *> ccs, StreamSet * BasisBits, StreamSet * CharClasses);
    bool hasSignature() const override { return true; }
    llvm::StringRef getSignature() const override;
protected:
    CharClassesKernel(BuilderRef b, std::string signature, std::vector<re::CC *> && ccs, StreamSet * BasisBits, StreamSet * CharClasses);
    void generatePabloMethod() override;
protected:
    const std::vector<re::CC *> mCCs;
    const std::string mSignature;
};


class ByteClassesKernel final : public pablo::PabloKernel {
public:
    ByteClassesKernel(BuilderRef b, std::vector<re::CC *> ccs, StreamSet * inputStream, StreamSet * CharClasses);
    bool hasSignature() const override { return true; }
    llvm::StringRef getSignature() const override;
protected:
    ByteClassesKernel(BuilderRef b, std::string signature, std::vector<re::CC *> && ccs, StreamSet * inputStream, StreamSet * CharClasses);
    void generatePabloMethod() override;
protected:
    const std::vector<re::CC *> mCCs;
    const std::string mSignature;
};


}
#endif
