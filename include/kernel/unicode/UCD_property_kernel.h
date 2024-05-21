/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */
#pragma once

#include <pablo/pablo_kernel.h>  // for PabloKernel

namespace re { class Name; }
namespace UCD { class EnumeratedPropertyObject; }

namespace kernel {

class UnicodePropertyKernelBuilder : public pablo::PabloKernel {
public:
    UnicodePropertyKernelBuilder(KernelBuilder & b, re::Name * property_value_name, StreamSet * BasisBits, StreamSet * property);
protected:
    llvm::StringRef getSignature() const override;
    bool hasSignature() const override { return true; }
    void generatePabloMethod() override;
private:
    UnicodePropertyKernelBuilder(KernelBuilder & b, re::Name * property_value_name, StreamSet * BasisBits, StreamSet * property, std::string && propValueName);
private:
    std::string mPropNameValue;
    re::Name * mName;
};

class UnicodePropertyBasis : public pablo::PabloKernel {
public:
    UnicodePropertyBasis(KernelBuilder & b, UCD::EnumeratedPropertyObject * enumObj, StreamSet * BasisBits, StreamSet * PropertyBasis);
protected:
    void generatePabloMethod() override;
private:
    UCD::EnumeratedPropertyObject * mEnumObj;
};

}
