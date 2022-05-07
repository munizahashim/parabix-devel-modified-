/*
 *  Copyright (c) 2018 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include <kernel/unicode/UCD_property_kernel.h>

#include <kernel/core/kernel.h>
#include <re/adt/re_name.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <unicode/utf/utf_compiler.h>
#include <kernel/core/kernel_builder.h>
#include <pablo/builder.hpp>
#include <pablo/pe_zeroes.h>
#include <llvm/Support/ErrorHandling.h>

using namespace kernel;
using namespace pablo;
using namespace cc;

UnicodePropertyKernelBuilder::UnicodePropertyKernelBuilder(BuilderRef iBuilder, re::Name * property_value_name, StreamSet * Source, StreamSet * property)
: PabloKernel(iBuilder,
"UCD:" + std::to_string(Source->getNumElements()) + "x" + std::to_string(Source->getFieldWidth()) + getStringHash(property_value_name->getFullName()),
{Binding{"source", Source}},
{Binding{"property_stream", property}}),
  mName(property_value_name) {

}

void UnicodePropertyKernelBuilder::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    UTF::UTF_Compiler unicodeCompiler(getInput(0), pb);
    pablo::Var * propertyVar = pb.createVar(mName->getFullName(), pb.createZeroes());
    re::RE * property_defn = mName->getDefinition();
    if (re::CC * propertyCC = llvm::dyn_cast<re::CC>(property_defn)) {
        unicodeCompiler.addTarget(propertyVar, propertyCC);
    } else if (re::PropertyExpression * pe = llvm::dyn_cast<re::PropertyExpression>(property_defn)) {
        if (pe->getKind() == re::PropertyExpression::Kind::Codepoint) {
            re::CC * propertyCC = llvm::cast<re::CC>(pe->getResolvedRE());
            unicodeCompiler.addTarget(propertyVar, propertyCC);
        }
    }
    unicodeCompiler.compile();
    Var * const property_stream = getOutputStreamVar("property_stream");
    pb.createAssign(pb.createExtract(property_stream, pb.getInteger(0)), propertyVar);
}

