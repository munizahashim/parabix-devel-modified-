/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include "ztf-phrase-logic.h"
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pe_zeroes.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <pablo/pe_debugprint.h>
#include <re/ucd/ucd_compiler.hpp>
#include <re/unicode/resolve_properties.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>
#include <llvm/Support/raw_ostream.h>
#include <sstream>

using namespace pablo;
using namespace kernel;
using namespace llvm;

InverseStream::InverseStream(BuilderRef kb,
                StreamSet * hashMarks,
                StreamSet * prevMarks,
                unsigned groupNum,
                StreamSet * selected)
: PabloKernel(kb, "InverseStream" + std::to_string(groupNum),
            {Binding{"hashMarks", hashMarks},
             Binding{"prevMarks", prevMarks}},
            {Binding{"selected", selected}}), mGroupNum(groupNum) { }

void InverseStream::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    PabloAST * hashMarks = getInputStreamSet("hashMarks")[0];
    PabloAST * prevMarks = getInputStreamSet("prevMarks")[0];
    if (mGroupNum == 1) {
        prevMarks = pb.createNot(prevMarks);
    }
    Var * groupStreamVar = getOutputStreamVar("selected");
    PabloAST * result = pb.createNot(hashMarks);
    result = pb.createOr(result, prevMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("selected"), pb.getInteger(0)), result);

}