#include <fstream>
#include <pablo/builder.hpp>
#include <pablo/pablo_kernel.h>
#include <pablo/pe_ones.h>
#include <pablo/pe_zeroes.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pe_ones.h>
#include <pablo/pablo_toolchain.h>
#include <pablo/bixnum/bixnum.h>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/intrusive/detail/math.hpp>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>

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
    } else {
        llvm::report_fatal_error("Cannot open " + filename);
    }
    return headers;
}

std::vector<std::string> createJSONtemplateStrings(std::vector<std::string> headers) {
    std::vector<std::string> tmp;
    if (headers.size() == 0) return tmp;
    tmp.push_back("\"},\n{\"" + headers[0] + "\":");
    for (unsigned i = 1; i < headers.size(); i++) {
        tmp.push_back("\",\"" + headers[i] + "\":");
    }
    tmp.push_back("\"},\n");
    return tmp;
}


char charLF = 0xA;
char charCR = 0xD;
char charDQ = 0x22;
char charComma = 0x2C;

class CSVlexer : public PabloKernel {
public:
    CSVlexer(BuilderRef kb, StreamSet * Source, StreamSet * CSVlexical)
        : PabloKernel(kb, "CSVlexer",
                      {Binding{"Source", Source}},
                      {Binding{"CSVlexical", CSVlexical, FixedRate(), Add1()}}) {}
protected:
    void generatePabloMethod() override;
};

enum {markLF = 0, markCR = 1, markDQ = 2, markComma = 3, markEOF = 4};

void CSVlexer::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::unique_ptr<cc::CC_Compiler> ccc;
    ccc = std::make_unique<cc::Parabix_CC_Compiler_Builder>(getEntryScope(), getInputStreamSet("Source"));
    PabloAST * LF = ccc->compileCC(re::makeCC(charLF, &cc::Byte));
    PabloAST * CR = ccc->compileCC(re::makeCC(charCR, &cc::Byte));
    PabloAST * DQ = ccc->compileCC(re::makeCC(charDQ, &cc::Byte));
    PabloAST * Comma = ccc->compileCC(re::makeCC(charComma, &cc::Byte));
    PabloAST * EOFbit = pb.createAtEOF(pb.createAdvance(pb.createOnes(), 1));
    Var * lexOut = getOutputStreamVar("CSVlexical");
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(markLF)), LF);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(markCR)), CR);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(markDQ)), DQ);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(markComma)), Comma);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(markEOF)), EOFbit);
}

class CSVparser : public PabloKernel {
public:
    CSVparser(BuilderRef kb, StreamSet * csvMarks, StreamSet * recordSeparators, StreamSet * fieldSeparators, StreamSet * quoteEscape)
        : PabloKernel(kb, "CSVparser",
                      {Binding{"csvMarks", csvMarks, FixedRate(), LookAhead(1)}},
                      {Binding{"recordSeparators", recordSeparators},
                       Binding{"fieldSeparators", fieldSeparators},
                       Binding{"quoteEscape", quoteEscape}}) {}
protected:
    void generatePabloMethod() override;
};

void CSVparser::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvMarks");
    PabloAST * dquote = csvMarks[markDQ];
    PabloAST * dquote_odd = pb.createEveryNth(dquote, pb.getInteger(2));
    PabloAST * dquote_even = pb.createXor(dquote, dquote_odd);
    PabloAST * quote_escape = pb.createAnd(dquote_even, pb.createLookahead(dquote, 1));
    PabloAST * escaped_quote = pb.createAdvance(quote_escape, 1);
    PabloAST * start_dquote = pb.createXor(dquote_odd, escaped_quote);
    PabloAST * end_dquote = pb.createXor(dquote_even, quote_escape);
    PabloAST * quoted_data = pb.createIntrinsicCall(pablo::Intrinsic::InclusiveSpan, {start_dquote, end_dquote});
    PabloAST * unquoted = pb.createNot(quoted_data);
    PabloAST * recordMarks = pb.createAnd(csvMarks[markLF], unquoted);
    PabloAST * fieldMarks = pb.createOr(pb.createAnd(csvMarks[markComma], unquoted), recordMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("recordSeparators"), pb.getInteger(0)), recordMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("fieldSeparators"), pb.getInteger(0)), fieldMarks);
    pb.createAssign(pb.createExtract(getOutputStreamVar("quoteEscape"), pb.getInteger(0)), quote_escape);
}

class CSVdataFieldMask : public PabloKernel {
public:
    CSVdataFieldMask(BuilderRef kb, StreamSet * csvMarks, StreamSet * recordSeparators, StreamSet * quoteEscape, StreamSet * toKeep, bool deleteHeader = true)
        : PabloKernel(kb, "CSVdataFieldMask" + std::to_string(deleteHeader),
                      {Binding{"csvMarks", csvMarks, FixedRate(), LookAhead(1)},
                       Binding{"recordSeparators", recordSeparators},
                       Binding{"quoteEscape", quoteEscape}},
                      {Binding{"toKeep", toKeep}})
    , mDeleteHeader(deleteHeader) {}
protected:
    void generatePabloMethod() override;
    bool mDeleteHeader;
};

void CSVdataFieldMask::generatePabloMethod() {
    pablo::PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> csvMarks = getInputStreamSet("csvMarks");
    PabloAST * recordMarks = pb.createExtract(getInputStreamVar("recordSeparators"), pb.getInteger(0));
    PabloAST * quoteEscape = pb.createExtract(getInputStreamVar("quoteEscape"), pb.getInteger(0));
    PabloAST * CRbeforeLF = pb.createAnd(csvMarks[markCR], pb.createLookahead(csvMarks[markLF], 1));
    PabloAST * escaped_quote = pb.createAdvance(quoteEscape, 1);
    PabloAST * fieldQuotes = pb.createAnd(csvMarks[markDQ], pb.createNot(pb.createOr(quoteEscape, escaped_quote)));
    PabloAST * toDelete = pb.createOr(CRbeforeLF, fieldQuotes);
    if (mDeleteHeader) {
        PabloAST * afterHeader = pb.createMatchStar(pb.createAdvance(recordMarks, 1), pb.createOnes());
        toDelete = pb.createOr(toDelete, pb.createNot(afterHeader));
    }
    // Delete the final LF position, so that we won't generate a template string at EOF.
    toDelete = pb.createOr(toDelete, pb.createAnd(recordMarks, pb.createLookahead(csvMarks[markEOF], 1)));
    // Also delete the final EOFbit position generated by the Add1 attribute of the CSV lexer to avoid a null.
    toDelete = pb.createOr(toDelete, csvMarks[markEOF]);
    PabloAST * toKeep = pb.createInFile(pb.createNot(toDelete));
    pb.createAssign(pb.createExtract(getOutputStreamVar("toKeep"), pb.getInteger(0)), toKeep);
}

//
//  FieldNumberingKernel(N) 
//  two input streams: record marks, field marks, N fields per record
//  output: at the start position after each mark, a bixnum value equal to the
//          sequential field number (counting from 1 at each record start).
//

class FieldNumberingKernel : public PabloKernel {
public:
    FieldNumberingKernel(BuilderRef kb, StreamSet * SeparatorNum, StreamSet * RecordMarks, StreamSet * FieldBixNum);
protected:
    void generatePabloMethod() override;
    unsigned mNumberingBits;
};

FieldNumberingKernel::FieldNumberingKernel(BuilderRef kb, StreamSet * SeparatorNum, StreamSet * RecordMarks, StreamSet * FieldBixNum)
   : PabloKernel(kb, "FieldNumbering" + std::to_string(SeparatorNum->getNumElements()),
                 {Binding{"RecordMarks", RecordMarks}, Binding{"SeparatorNum", SeparatorNum}}, {Binding{"FieldBixNum", FieldBixNum}}),
   mNumberingBits(SeparatorNum->getNumElements()) { }

void FieldNumberingKernel::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);
    PabloAST * recordMarks = getInputStreamSet("RecordMarks")[0];   //  1 at record positions, 0 elsewhere
    BixNum separatorNum = getInputStreamSet("SeparatorNum"); //  consecutively numbered from 0
    BixNum increment(2);
    increment[0] = recordMarks;   //  Add 1 at record positions
    increment[1] = pb.createNot(recordMarks);  // Add 2 at field mark positions.
    BixNum fieldNumbering = bnc.AddFull(separatorNum, increment);
    Var * fieldBixNum = getOutputStreamVar("FieldBixNum");
    for (unsigned i = 0; i < mNumberingBits; i++) {
        pb.createAssign(pb.createExtract(fieldBixNum, i), fieldNumbering[i]);
    }
}

class CSV_Char_Replacement : public PabloKernel {
public:
    CSV_Char_Replacement(BuilderRef kb, StreamSet * separatorsLF, StreamSet * separatorsComma, StreamSet * quoteEscape, StreamSet * basis,
                         StreamSet * translatedBasis)
        : PabloKernel(kb, "CSV_Char_Replacement",
                      {Binding{"separatorsLF", separatorsLF}, Binding{"separatorsComma", separatorsComma}, Binding{"quoteEscape", quoteEscape}, Binding{"basis", basis}},
                      {Binding{"translatedBasis", translatedBasis}}) {}
protected:
    void generatePabloMethod() override;
};

void CSV_Char_Replacement::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * separatorsLF = getInputStreamSet("separatorsLF")[0];
    PabloAST * separatorsComma = getInputStreamSet("separatorsComma")[0];
    separatorsComma = pb.createAnd(separatorsComma, pb.createNot(separatorsLF));
    PabloAST * quoteEscape = getInputStreamSet("quoteEscape")[0];
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    //
    // Translate "" to \"  ASCII value of " = 0x22, ASCII value of \ = 0x5C
    // Translate , to "    ASCII value of , = 0x2C, ASCII value of " = 0x22
    // Translate LF to "   ASCII value of LF = 0x0A, ASCII value of " = 0x22
    std::vector<PabloAST *> translated_basis(8, nullptr);
    translated_basis[0] = basis[0];
    translated_basis[1] = pb.createXor(basis[1], pb.createOr(quoteEscape, separatorsComma));
    translated_basis[2] = pb.createXor(basis[2], pb.createOr(quoteEscape, separatorsComma));
    translated_basis[3] = pb.createXor(basis[3], pb.createOr3(quoteEscape, separatorsLF, separatorsComma));
    translated_basis[4] = pb.createXor(basis[4], quoteEscape);  // flip only for quoteEscape
    translated_basis[5] = pb.createXor(basis[5], pb.createOr(quoteEscape, separatorsLF));  // flip
    translated_basis[6] = pb.createXor(basis[6], quoteEscape);  // flip only for quoteEscape
    translated_basis[7] = basis[7];

    Var * translatedVar = getOutputStreamVar("translatedBasis");
    for (unsigned i = 0; i < 8; i++) {
        pb.createAssign(pb.createExtract(translatedVar, pb.getInteger(i)), translated_basis[i]);
    }
}

class Extend1Zeroes : public PabloKernel {
public:
    Extend1Zeroes(BuilderRef kb, StreamSet * mask, StreamSet * extended)
    : PabloKernel(kb, "Extend1Zeroes",
                  {Binding{"mask", mask}},
                  {Binding{"extended", extended}}) {}
protected:
    void generatePabloMethod() override;
};

void Extend1Zeroes::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * mask = getInputStreamSet("mask")[0];
    PabloAST * inverted = pb.createNot(mask);
    PabloAST * extended = pb.createNot(pb.createOr(inverted, pb.createAdvance(inverted, 1)));
    Var * outputVar = getOutputStreamVar("extended");
    pb.createAssign(pb.createExtract(outputVar, pb.getInteger(0)), extended);
}
