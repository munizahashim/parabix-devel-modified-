/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */
#ifndef JSON_KERNEL_H
#define JSON_KERNEL_H

#include <pablo/pablo_kernel.h>
#include <kernel/core/kernel_builder.h>

namespace kernel {

// This enum MUST reflect the type Lex on json.pablo file
enum Lex {
    lCurly = 0,
    rCurly,
    lBracket,
    rBracket,
    colon,
    comma,
    dQuote,
    hyphen,
    digit,
    backslash,
    n, // # first letter of null
    f, // # first letter of false
    t, // # first letter of true
    ws
};

enum KwMarker {
    kwNullEnd = 0,
    kwTrueEnd,
    kwFalseEnd,
};

/*
    Given the JSON lex for characters backslash and double quotes,
    this kernel returns the marker of a JSON string, based on paper
    Parsing Gigabytes of JSON per Second (Daniel Lemire and Geoff Langdale)

             json: { "key1\"": value1, "key2"  : null }
    input example: ..1.....11..........1....1..........
    output marker: ..1......1..........1....1..........
      output span: ...111111............1111...........
*/
class JSONStringMarker : public pablo::PabloKernel {
public:
    JSONStringMarker(const std::unique_ptr<KernelBuilder> & b,
                     StreamSet * const lexIn,
                     StreamSet * strMarker, StreamSet * strSpan)
    : pablo::PabloKernel(b,
                         "jsonStrMarker",
                         {Binding{"lexIn", lexIn}},
                         {Binding{"marker", strMarker}, Binding{"span", strSpan}}) {}
    bool isCachable() const override { return true; }
    bool hasSignature() const override { return false; }
protected:
    void generatePabloMethod() override;
};

/*
   Marks keywords letters such as l', 'a', 's', 'r', 'u', 'e',
   joining it at the end with 'n', 't' and 'f'

            json: { "keynull": false, "keyt": true }
   input example: ......1......1..........1...1.....
          output:..................1.............1..

    Note: we do not return the beginning of the marker here because lookahead
    only works on input streams, so this will be done in a further step.
*/
class JSONKeywordEndMarker : public pablo::PabloKernel {
public:
    JSONKeywordEndMarker(const std::unique_ptr<KernelBuilder> & b,
                      StreamSet * const basis,
                      StreamSet * const lexIn, StreamSet * const strSpan,
                      StreamSet * kwMarker)
    : pablo::PabloKernel(b,
                         "jsonKeywordMarker",
                         {
                            Binding{"basis", basis},
                            Binding{"lexIn", lexIn},
                            Binding{"strSpan", strSpan}
                         },
                         {
                            Binding{"kwEndMarker", kwMarker},
                         }) {}
    bool isCachable() const override { return true; }
    bool hasSignature() const override { return false; }
protected:
    void generatePabloMethod() override;
};

/*
   Finds symbols used in numbers such as 'e', 'E', '.'
   and join them at the end if they match the expression:
   \-?(0|[1-9][0-9]*)(.[0-9]+)?([Ee][+-]?[0-9]+)?
*/
class JSONNumberSpan : public pablo::PabloKernel {
public:
    JSONNumberSpan(const std::unique_ptr<KernelBuilder> & b,
                   StreamSet * const basis,
                   StreamSet * const lexIn,
                   StreamSet * const strSpan,
                   StreamSet * nbrLex, StreamSet * nbrSpan, StreamSet * nbrErr)
    : pablo::PabloKernel(b,
                         "jsonNumberMarker",
                         {
                            Binding{"basis", basis, FixedRate(1), LookAhead(1)},
                            Binding{"lexIn", lexIn},
                            Binding{"strSpan", strSpan}
                         },
                         {Binding{"nbrLex", nbrLex}, Binding{"nbrSpan", nbrSpan}, Binding{"nbrErr", nbrErr}}) {}
    bool isCachable() const override { return true; }
    bool hasSignature() const override { return false; }
protected:
    void generatePabloMethod() override;
};

/*
   Marks keywords and find if there are other extra chars
   where they should not be

            json: { "keynull": false, "key{": truenull2 }
 firstLexsClning: 1..........1......1.....1.1...........1
      stringSpan: ..111111111.........111111.............
      numberSpan: ....................................1..
    kwEndMarkers: .................1.............1...1...
 output kwMarker: .............1..............1..........
 output firstLex: 1..........1......1.......1...........1
 output extraErr: ................................1......

*/
class JSONFindKwAndExtraneousChars : public pablo::PabloKernel {
    public:
    JSONFindKwAndExtraneousChars(
                        const std::unique_ptr<KernelBuilder> & b,
                        StreamSet * const lexIn,
                        StreamSet * const stringSpan,
                        StreamSet * const numberSpan,
                        StreamSet * const kwEndMarkers,
                        StreamSet * const kwMarker,
                        StreamSet * const firstLexs,
                        StreamSet * const combinedBrackets,
                        StreamSet * const extraErr
    )
    : pablo::PabloKernel(b,
                         "jsonFindKwAndExtraneousChars",
                         {
                            Binding{"lexIn", lexIn},
                            Binding{"strSpan", stringSpan},
                            Binding{"numSpan", numberSpan},
                            Binding{"kwEndMarkers", kwEndMarkers, FixedRate(1), LookAhead(4)},
                         },
                         {
                            Binding{"kwMarker", kwMarker},
                            Binding{"firstLexs", firstLexs},
                            Binding{"combinedBrackets", combinedBrackets},
                            Binding{"extraErr", extraErr}
                         }) {}
    bool isCachable() const override { return true; }
    bool hasSignature() const override { return false; }
protected:
    void generatePabloMethod() override;
};

class JSONValidateAndDeleteInnerBrackets : public pablo::PabloKernel {
    public:
    JSONValidateAndDeleteInnerBrackets(
                        const std::unique_ptr<KernelBuilder> & b,
                        StreamSet * const lexIn,
                        StreamSet * const compressed,
                        StreamSet * const stringMarker,
                        StreamSet * const keywordMarker,
                        StreamSet * const numberMarker,
                        StreamSet * const toPostProcess,
                        StreamSet * const syntaxErr
    )
    : pablo::PabloKernel(b,
                         "jsonValidateAndDeleteInnerBrackets",
                         {
                            Binding{"lexIn", lexIn},
                            Binding{"compressed", compressed, FixedRate(1), LookAhead(1)},
                            Binding{"strMarker", stringMarker},
                            Binding{"kwMarker", keywordMarker},
                            Binding{"nbrMarker", numberMarker},
                         },
                         {
                            Binding{"toPostProcess", toPostProcess},
                            Binding{"syntaxErr", syntaxErr}
                         }) {}
    bool isCachable() const override { return true; }
    bool hasSignature() const override { return false; }
protected:
    void generatePabloMethod() override;
};

}

#endif
