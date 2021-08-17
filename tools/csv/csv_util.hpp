#include <fstream>
#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <toolchain/pablo_toolchain.h>
#include <pablo/bixnum/bixnum.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/intrusive/detail/math.hpp>

using namespace pablo;
using namespace kernel;

using boost::intrusive::detail::ceil_log2;

std::vector<std::string> parse_CSV_headers(std::string headerString) {
    std::vector<std::string> headers;
    boost::algorithm::split(headers, headerString, [] (char c) {return (c == ',');});
    for (unsigned i = 0; i < headers.size(); i++) {
        boost::algorithm::trim(headers[i]);
    }
    return headers;
}

std::vector<std::string> get_CSV_headers(std::string filename) {
    std::vector<std::string> headers;
    std::ifstream headerFile(filename.c_str());
    std::string line1;
    if (headerFile.is_open()) {
        std::getline(headerFile, line1);
        headerFile.close();
        headers = parse_CSV_headers(line1);
    }
    return headers;
}

std::vector<std::string> createJSONtemplateStrings(std::vector<std::string> headers) {
    std::vector<std::string> tmp;
    if (headers.size() == 0) return tmp;
    tmp.push_back("{\"" + headers[0] + "\":\"");
    for (unsigned i = 1; i < headers.size(); i++) {
        tmp.push_back("\",\"" + headers[i] + "\":\"");
    }
    tmp.push_back("\"},\n");
    return tmp;
}

//
//  FieldNumberingKernel(N) 
//  two input streams: record marks, field marks, N fields per record
//  output: at the start position after each mark, a bixnum value equal to the
//          sequential field number (counting from 0 at each record start).
//

class FieldNumberingKernel : public PabloKernel {
public:
    FieldNumberingKernel(BuilderRef kb, StreamSet * Marks, StreamSet * FieldBixNum, unsigned fieldCount);
protected:
    void generatePabloMethod() override;
    unsigned mFieldCount;
};

FieldNumberingKernel::FieldNumberingKernel(BuilderRef kb, StreamSet * Marks, StreamSet * FieldBixNum, unsigned fieldCount)
   : PabloKernel(kb, "FieldNumbering" + std::to_string(fieldCount),
                   {Binding{"Marks", Marks}}, {Binding{"FieldBixNum", FieldBixNum}}),
   mFieldCount(fieldCount) { }

void FieldNumberingKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * recordMarks = getInputStreamSet("Marks")[0];
    PabloAST * fieldMarks = getInputStreamSet("Marks")[1];
    PabloAST * recordStarts = pb.createNot(pb.createAdvance(pb.createNot(recordMarks), 1));
    PabloAST * fieldStarts = pb.createOr(recordStarts, pb.createAdvance(fieldMarks, 1));

    unsigned n = ceil_log2(mFieldCount);
    BixNum fieldNumbering(n, pb.createZeroes());
    // Initially only the recordStarts positions are correctly numbered.
    PabloAST * numbered = recordStarts;
    // Work through the numbering bits from the most significant down.
    for (int k = n - 1; k >= 0; k--) {
        unsigned K = 1U << k;
        // Determine which numbered positions will still be within range when
        // advancing through the fieldStarts index stream.
        PabloAST * toAdvance = bnc.ULT(fieldNumbering, mFieldCount - K);
        fieldNumbering[k] = pb.createIndexedAdvance(pb.createAnd(numbered, toAdvance), fieldStarts, K);
        // Now the positions just identified are correctly numbered.
        numbered = pb.createOr(numbered, fieldNumbering[k]);
    }
    Var * fieldBixNum = getOutputStreamVar("FieldBixNum");
    for (unsigned i = 0; i < n; i++) {
        pb.createAssign(pb.createExtract(fieldBixNum, i), fieldNumbering[i]);
    }
}

class CSV_Char_Replacement : public PabloKernel {
public:
    CSV_Char_Replacement(BuilderRef kb, StreamSet * csvMarks, StreamSet * basis, StreamSet * translatedBasis)
        : PabloKernel(kb, "CSV_Char_Replacement",
                      {Binding{"csvMarks", csvMarks}, Binding{"basis", basis}},
                      {Binding{"translatedBasis", translatedBasis}}) {}
protected:
    void generatePabloMethod() override;
};

void CSV_Char_Replacement::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvMarks");
    enum {recordDelim, fieldDelim, quoteEscape};
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    //
    // Zero out the field and record delimiters.
    // Translate "" to \"    ASCII value of " = 0x22, ASCII value of \ = 0x5C
    //    this translation flips bits 1 through 6
    PabloAST * toZero = pb.createNot(pb.createOr(csvMarks[recordDelim], csvMarks[fieldDelim]));
    PabloAST * toXlate = csvMarks[quoteEscape];
    std::vector<PabloAST *> translated_basis(8, nullptr);
    translated_basis[0] = pb.createAnd(basis[0], toZero);
    translated_basis[1] = pb.createAnd(pb.createXor(basis[1], toXlate), toZero);  // flip
    translated_basis[2] = pb.createAnd(pb.createXor(basis[2], toXlate), toZero);  // flip
    translated_basis[3] = pb.createAnd(pb.createXor(basis[3], toXlate), toZero);  // flip
    translated_basis[4] = pb.createAnd(pb.createXor(basis[4], toXlate), toZero);  // flip
    translated_basis[5] = pb.createAnd(pb.createXor(basis[5], toXlate), toZero);  // flip
    translated_basis[6] = pb.createAnd(pb.createXor(basis[6], toXlate), toZero);  // flip
    translated_basis[7] = pb.createAnd(basis[7], toZero);

    Var * translatedVar = getOutputStreamVar("translatedBasis");
    for (unsigned i = 0; i < 8; i++) {
        pb.createAssign(pb.createExtract(translatedVar, pb.getInteger(i)), translated_basis[i]);
    }
}

