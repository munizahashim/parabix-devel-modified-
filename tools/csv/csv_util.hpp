#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <toolchain/pablo_toolchain.h>
#include <pablo/bixnum/bixnum.h>
#include <boost/intrusive/detail/math.hpp>

using boost::intrusive::detail::ceil_log2;
//
//  FieldNumberingKernel(N) 
//  two input streams: record marks, field marks, N fields per record
//  output: at the start position after each mark, a bixnum value equal to the
//          sequential field number (counting from 0 at each record start).
//

class FieldNumberingKernel : public pablo::PabloKernel {
public:
    FieldNumberingKernel(BuilderRef kb, kernel::StreamSet * Marks, kernel::StreamSet * FieldBixNum, unsigned fieldCount);
protected:
    void generatePabloMethod() override;
    unsigned mFieldCount;
};

FieldNumberingKernel::FieldNumberingKernel(BuilderRef kb, kernel::StreamSet * Marks, kernel::StreamSet * FieldBixNum, unsigned fieldCount)
   : PabloKernel(kb, "FieldNumbering" + std::to_string(fieldCount),
                   {kernel::Binding{"Marks", Marks}}, {kernel::Binding{"FieldBixNum", FieldBixNum}}),
   mFieldCount(fieldCount) { }

void FieldNumberingKernel::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    pablo::BixNumCompiler bnc(pb);
    pablo::PabloAST * recordMarks = getInputStreamSet("Marks")[0];
    pablo::PabloAST * fieldMarks = getInputStreamSet("Marks")[1];
    pablo::PabloAST * recordStarts = pb.createNot(pb.createAdvance(pb.createNot(recordMarks), 1));
    pablo::PabloAST * fieldStarts = pb.createOr(recordStarts, pb.createAdvance(fieldMarks, 1));

    unsigned n = ceil_log2(mFieldCount);
    pablo::BixNum fieldNumbering(n, pb.createZeroes());
    // Initially only the recordStarts positions are correctly numbered.
    pablo::PabloAST * numbered = recordStarts;
    // Work through the numbering bits from the most significant down.
    for (int k = n - 1; k >= 0; k--) {
        unsigned K = 1U << k;
        // Determine which numbered positions will still be within range when
        // advancing through the fieldStarts index stream.
        pablo::PabloAST * toAdvance = bnc.ULT(fieldNumbering, mFieldCount - K);
        fieldNumbering[k] = pb.createIndexedAdvance(pb.createAnd(numbered, toAdvance), fieldStarts, K);
        // Now the positions just identified are correctly numbered.
        numbered = pb.createOr(numbered, fieldNumbering[k]);
    }
    pablo::Var * fieldBixNum = getOutputStreamVar("FieldBixNum");
    for (unsigned i = 0; i < n; i++) {
        pb.createAssign(pb.createExtract(fieldBixNum, i), fieldNumbering[i]);
    }
}


