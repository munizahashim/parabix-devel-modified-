/*
 *  Copyright (c) 2018 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */
#ifndef UCD_PROPERTY_KERNEL_H
#define UCD_PROPERTY_KERNEL_H

#include <pablo/pablo_kernel.h>  // for PabloKernel

namespace re { class Name; }

namespace kernel {

class UnicodePropertyKernelBuilder : public pablo::PabloKernel {
public:
    UnicodePropertyKernelBuilder(BuilderRef kb, re::Name * property_value_name, StreamSet * BasisBits, StreamSet * property);
protected:
    llvm::StringRef getSignature() const override;
    bool hasSignature() const override { return true; }
    void generatePabloMethod() override;
private:
    UnicodePropertyKernelBuilder(BuilderRef kb, re::Name * property_value_name, StreamSet * BasisBits, StreamSet * property, std::string && propValueName);
private:
    std::string mPropNameValue;
    re::Name * mName;
};

}
#endif
