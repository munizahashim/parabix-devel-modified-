#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <toolchain/pablo_toolchain.h>
#include <pablo/bixnum/bixnum.h>

//
//  FieldNumberingKernel(N) 
//  two input streams: record marks, field marks, N fields per record
//  output: at each field mark, a bixnum value equal to the sequential field number (counting from 0)
//          - note: record mark must also be a field mark...
// 

class FieldNumberingKernel : public pablo::PabloKernel {
public:
    FieldNumberingKernel(BuilderRef kb, StreamSet * Marks, StreamSet * FieldBixNum, unsigned fieldCount);
protected:
    void generatePabloMethod() override;
    unsigned mFieldCount;
};

FieldNumberingKernel::FieldNumberingKernel(BuilderRef kb, StreamSet * Marks, StreamSet * FieldBixNum, unsigned fieldCount)
: PabloKernel(kb, "FieldNumbering" + std::to_string(fieldcount), {Binding{"Marks", Marks}}, {Binding{"FieldBixNum", FieldBixNum}}), mFieldCount(fieldCount) { }

void FieldNumberingKernel::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::PabloAST * recordMarks = getInputStreamSet("Marks")[0];
    pablo::PabloAST * fieldMmarks = getInputStreamSet("Marks")[1];

    unsigned N = ceil_log2(mFieldCount);
    pablo::BixNum(n) fieldNumbering;
    // Initially only the recordMarks positions are correctly numbered.
    PabloAST * numbered = recordMarks;
    // Work through the numbering bits from the most significant down.
    for (unsigned k = n - 1; k >= 0; k--) {
        unsigned K = 1 << k;
        // Determine which numbered positions will still be within range when
        // advancing through the fieldMarks index stream.
        toAdvance = bnc.ULT(fieldNumbering, mFieldCount - K);
        fieldNumbering[k] = pb.createIndexedAdvance(pb.createAnd(numbered, toAdvance), fieldMarks, K);
        // Now the positions just identified are correctly numbered.
        numbered = pb.createOr(numbered, fieldNumbering[k]);
    }
    pablo::Var * fieldBixNum = getOutputStreamVar("FieldBixNum");
    for (unsigned i = 0; i < n; i++) {
        pb.createAssign(pb.createExtract(fieldBixNum, i), numbered[i]);
    }
}
