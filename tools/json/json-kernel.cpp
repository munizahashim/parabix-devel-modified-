/*
 *  Copyright (c) 2019 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 *  icgrep is a trademark of International Characters.
 */

#include "json-kernel.h"
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <pablo/bixnum/bixnum.h>
#include <pablo/pe_zeroes.h>
#include <pablo/builder.hpp>
#include <pablo/pe_ones.h>
#include <re/ucd/ucd_compiler.hpp>
#include <re/unicode/re_name_resolve.h>
#include <re/cc/cc_compiler.h>
#include <re/cc/cc_compiler_target.h>

using namespace pablo;
using namespace kernel;

static PabloAST * sanitizeLexInput(PabloBuilder & pb, PabloAST * span, PabloAST * out) {
    return pb.createXorAnd(out, out, span);
}

void JSONStringMarker::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * dQuotes = getInputStreamSet("lexIn")[Lex::dQuote];
    PabloAST * backslash = getInputStreamSet("lexIn")[Lex::backslash];
    Var * const strMarker = getOutputStreamVar("marker");
    Var * const strSpan = getOutputStreamVar("span");

    // keeping the names as the ones in paper PGJS (Lemire)
    PabloAST * B = backslash;
    PabloAST * E = pb.createRepeat(1, pb.getInteger(0xAAAAAAAAAAAAAAAA, 64)); // constant
    PabloAST * O = pb.createRepeat(1, pb.getInteger(0x5555555555555555, 64)); // constant

    // identify 'starts' - backslashes not preceded by backslashes
    // paper does S = B & ~(B << 1), but we can't Advance(-1)
    PabloAST * notB = pb.createNot(B);
    PabloAST * S = pb.createAnd(B, pb.createAdvance(notB, 1));
    
    // paper does S & E, but we advanced notB by 1, so it became S & O
    PabloAST * ES = pb.createAnd(S, O);
    // we don't have add, so we will have to use ScanThru on ES
    // eg.:          ES = ..............1............
    //                B = ..............1111.........
    // ScanThru(ES, EB) = ..................1........
    // This way we don't need to filter after the sum :)
    PabloAST * EC = pb.createScanThru(ES, B);
    // Checking with odd instead of even because we're one step ahead
    PabloAST * OD1 = pb.createAnd(EC, pb.createNot(O));
    // inverted even/odd
    PabloAST * OS = pb.createAnd(S, E);
    // no add/clean again
    PabloAST * OC = pb.createScanThru(OS, B);
     // Checking with odd instead of even because we're one step ahead
    PabloAST * OD2 = pb.createAnd(OC, O);
    PabloAST * OD = pb.createOr(OD1, OD2);

    // There is a bug on the quotes
    PabloAST * Q = dQuotes;
    PabloAST * QEq = pb.createAnd(Q, pb.createNot(OD));

    // Find the string spans
    PabloAST * beginDQuotes = pb.createEveryNth(QEq, pb.getInteger(2));
    PabloAST * endDQuotes = pb.createXor(QEq, beginDQuotes);
    PabloAST * inSpan = pb.createIntrinsicCall(Intrinsic::InclusiveSpan, {beginDQuotes, endDQuotes});

    pb.createAssign(pb.createExtract(strMarker, pb.getInteger(0)), QEq);
    pb.createAssign(pb.createExtract(strSpan, pb.getInteger(0)), inSpan);
}

void JSONKeywordEndMarker::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    cc::Parabix_CC_Compiler_Builder ccc(getEntryScope(), basis);
    PabloAST * strSpan = getInputStreamSet("strSpan")[0];
    Var * const kwEndMarker = getOutputStreamVar("kwEndMarker");

    PabloAST * notStrSpan = pb.createNot(strSpan);
    PabloAST * N = pb.createAnd(notStrSpan, getInputStreamSet("lexIn")[Lex::n]);
    PabloAST * T = pb.createAnd(notStrSpan, getInputStreamSet("lexIn")[Lex::t]);
    PabloAST * F = pb.createAnd(notStrSpan, getInputStreamSet("lexIn")[Lex::f]);

    // null
    PabloAST * U = ccc.compileCC(re::makeByte(0x75));
    PabloAST * L = ccc.compileCC(re::makeByte(0x6C));
    // true
    PabloAST * R = ccc.compileCC(re::makeByte(0x72));
    PabloAST * E = ccc.compileCC(re::makeByte(0x65));
    // false
    PabloAST * A = ccc.compileCC(re::makeByte(0x61));
    PabloAST * S = ccc.compileCC(re::makeByte(0x73));

    PabloAST * advNU = pb.createAnd(U, pb.createAdvance(N, 1));
    PabloAST * advNUL = pb.createAnd(L, pb.createAdvance(advNU, 1));
    Var * seqNULL = pb.createVar("null", pb.createZeroes());
    auto itNUL = pb.createScope();
    pb.createIf(advNUL, itNUL);
    {
        PabloAST * advNULL = itNUL.createAnd(L, itNUL.createAdvance(advNUL, 1));
        itNUL.createAssign(seqNULL, advNULL);
    }

    PabloAST * advTR = pb.createAnd(R, pb.createAdvance(T, 1));
    PabloAST * advTRU = pb.createAnd(U, pb.createAdvance(advTR, 1));
    Var * seqTRUE = pb.createVar("true", pb.createZeroes());
    auto itTRU = pb.createScope();
    pb.createIf(advTRU, itTRU);
    {
        PabloAST * advTRUE = pb.createAnd(E, pb.createAdvance(advTRU, 1));
        itTRU.createAssign(seqTRUE, advTRUE);
    }

    PabloAST * advFA = pb.createAnd(A, pb.createAdvance(F, 1));
    PabloAST * advFAL = pb.createAnd(L, pb.createAdvance(advFA, 1));
    PabloAST * advFALS = pb.createAnd(S, pb.createAdvance(advFAL, 1));
    Var * seqFALSE = pb.createVar("false", pb.createZeroes());
    auto itFALS = pb.createScope();
    pb.createIf(advFALS, itFALS);
    {
        PabloAST * advFALSE = pb.createAnd(E, pb.createAdvance(advFALS, 1));
        itFALS.createAssign(seqFALSE, advFALSE);
    }

    pb.createAssign(pb.createExtract(kwEndMarker, pb.getInteger(KwMarker::kwNullEnd)), seqNULL);
    pb.createAssign(pb.createExtract(kwEndMarker, pb.getInteger(KwMarker::kwTrueEnd)), seqTRUE);
    pb.createAssign(pb.createExtract(kwEndMarker, pb.getInteger(KwMarker::kwFalseEnd)), seqFALSE);
}

void JSONNumberSpan::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> basis = getInputStreamSet("basis");
    cc::Parabix_CC_Compiler_Builder ccc(getEntryScope(), basis);
    PabloAST * hyphenIn = getInputStreamSet("lexIn")[Lex::hyphen];
    PabloAST * digitIn = getInputStreamSet("lexIn")[Lex::digit];

    PabloAST * strSpan = getInputStreamSet("strSpan")[0];
    Var * const nbrLex = getOutputStreamVar("nbrLex");
    Var * const nbrSpan = getOutputStreamVar("nbrSpan");
    Var * const nbrErr = getOutputStreamVar("nbrErr");

    PabloAST * alleE = pb.createOr(ccc.compileCC(re::makeByte(0x45)), ccc.compileCC(re::makeByte(0x65)));
    PabloAST * allDot = ccc.compileCC(re::makeByte(0x2E));
    PabloAST * allPlusMinus = pb.createOr(hyphenIn, ccc.compileCC(re::makeByte(0x2B)));

    PabloAST * notStrSpan = pb.createNot(strSpan);
    PabloAST * hyphen = pb.createAnd(notStrSpan, hyphenIn);
    PabloAST * digit = pb.createAnd(notStrSpan, digitIn);
    PabloAST * alleEAfterDigit = pb.createAnd(pb.createAdvance(digit, 1), alleE);
    PabloAST * eE = pb.createAnd(notStrSpan, alleEAfterDigit);
    PabloAST * allDotAfterDigit = pb.createAnd(pb.createAdvance(digit, 1), allDot);
    PabloAST * dot = pb.createAnd(notStrSpan, allDotAfterDigit);
    PabloAST * allPlusMinusAftereE = pb.createAnd(pb.createAdvance(eE, 1), allPlusMinus);
    PabloAST * plusMinus = pb.createAnd(notStrSpan, pb.createOr(hyphen, allPlusMinusAftereE));

    PabloAST * nondigit = pb.createNot(digit);
    PabloAST * nonDigitNorEe = pb.createAnd(nondigit, pb.createNot(eE));
    PabloAST * begin = pb.createNot(pb.createAdvance(pb.createOnes(), 1));
    PabloAST * beginIsHyphen = pb.createAnd(hyphen, begin);
    PabloAST * otherIsHyphen = pb.createAnd(hyphen, pb.createAdvance(nonDigitNorEe, 1));
    PabloAST * advHyphen = pb.createOr(beginIsHyphen, otherIsHyphen);

    PabloAST * nonDigitEePlusMinus = pb.createAnd(nonDigitNorEe, pb.createNot(plusMinus));
    PabloAST * nonDigitEePlusMinusDot = pb.createAnd(nonDigitEePlusMinus, pb.createNot(dot));
    PabloAST * advDigit = pb.createAnd(digit, pb.createAdvance(nonDigitEePlusMinusDot, 1));
    PabloAST * beginNbr = pb.createOr(advDigit, advHyphen);
    pb.createAssign(pb.createExtract(nbrLex, pb.getInteger(0)), beginNbr);

    PabloAST * errDot = pb.createAnd(pb.createAdvance(dot, 1), nondigit);
    PabloAST * errPlusMinus = pb.createAnd(pb.createAdvance(plusMinus, 1), nondigit);
    PabloAST * eENotPlusMinus = pb.createAnd(pb.createAdvance(eE, 1), pb.createNot(plusMinus));
    PabloAST * erreENotPlusMinus = pb.createAnd(eENotPlusMinus, nondigit);
    PabloAST * err = pb.createOr3(errDot, errPlusMinus, erreENotPlusMinus);
    pb.createAssign(pb.createExtract(nbrErr, pb.getInteger(0)), err);

    PabloAST * endNbr = pb.createAnd(pb.createAdvance(digit, 1), nonDigitEePlusMinusDot);
    PabloAST * finalNbr = pb.createIntrinsicCall(Intrinsic::SpanUpTo, {beginNbr, endNbr});
    pb.createAssign(pb.createExtract(nbrSpan, pb.getInteger(0)), finalNbr);
}

void JSONFindKwAndExtraneousChars::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());

    PabloAST * strSpan = getInputStreamSet("strSpan")[0];
    PabloAST * numSpan = getInputStreamSet("numSpan")[0];

    std::vector<PabloAST *> kwEndMarkers = getInputStreamSet("kwEndMarkers");
    PabloAST * ws = getInputStreamSet("lexIn")[Lex::ws];

    Var * const nbrErr = getOutputStreamVar("extraErr");
    Var * const combinedOut = getOutputStreamVar("combinedLexs");

    PabloAST * sanitizelCurly = sanitizeLexInput(pb, strSpan, getInputStreamSet("lexIn")[Lex::lCurly]);
    PabloAST * sanitizerCurly = sanitizeLexInput(pb, strSpan, getInputStreamSet("lexIn")[Lex::rCurly]);
    PabloAST * sanitizelBracket = sanitizeLexInput(pb, strSpan, getInputStreamSet("lexIn")[Lex::lBracket]);
    PabloAST * sanitizerBracket = sanitizeLexInput(pb, strSpan, getInputStreamSet("lexIn")[Lex::rBracket]);
    PabloAST * sanitizeColon = sanitizeLexInput(pb, strSpan, getInputStreamSet("lexIn")[Lex::colon]);
    PabloAST * sanitizeComma = sanitizeLexInput(pb, strSpan, getInputStreamSet("lexIn")[Lex::comma]);
    PabloAST * sanitizeHyphen = sanitizeLexInput(pb, strSpan, getInputStreamSet("lexIn")[Lex::hyphen]);

    PabloAST * first3Lex = pb.createOr3(sanitizelCurly, sanitizerCurly, sanitizelBracket);
    PabloAST * last3Lex = pb.createOr3(sanitizerBracket, sanitizeColon, sanitizeComma);
    PabloAST * validLexs = pb.createOr3(first3Lex, last3Lex, sanitizeHyphen);

    PabloAST * combinedSpans = pb.createOr3(strSpan, numSpan, validLexs);

    PabloAST * nBegin = pb.createLookahead(kwEndMarkers[KwMarker::kwNullEnd], 3);
    PabloAST * tBegin = pb.createLookahead(kwEndMarkers[KwMarker::kwTrueEnd], 3);
    PabloAST * fBegin = pb.createLookahead(kwEndMarkers[KwMarker::kwFalseEnd], 4);

    PabloAST * nSpan = pb.createIntrinsicCall(
        Intrinsic::InclusiveSpan,
        { nBegin, kwEndMarkers[KwMarker::kwNullEnd] }
    );

    PabloAST * tSpan = pb.createIntrinsicCall(
        Intrinsic::InclusiveSpan,
        { tBegin, kwEndMarkers[KwMarker::kwTrueEnd] }
    );

    PabloAST * fSpan = pb.createIntrinsicCall(
        Intrinsic::InclusiveSpan,
        { fBegin, kwEndMarkers[KwMarker::kwFalseEnd] }
    );

    PabloAST * keywordSpans = pb.createOr3(nSpan, tSpan, fSpan);

    PabloAST * extraneousChars = pb.createNot(pb.createOr(keywordSpans, combinedSpans));
    PabloAST * sanitizedErr = sanitizeLexInput(pb, ws, extraneousChars);

    // ------------------- Validate values and terminals

    PabloAST * beforeKwMarker = pb.createOr3(
        kwEndMarkers[KwMarker::kwNullEnd],
        kwEndMarkers[KwMarker::kwTrueEnd],
        kwEndMarkers[KwMarker::kwFalseEnd]
    );
    PabloAST * kwMarker = pb.createAdvance(beforeKwMarker, 1);

    PabloAST * realNumSpan = pb.createOr(sanitizeHyphen, numSpan);
    PabloAST * notNumSpan = pb.createNot(realNumSpan);
    PabloAST * prepNumMarker = pb.createAnd(pb.createAdvance(realNumSpan, 1), notNumSpan);
    PabloAST * numMarker = sanitizeLexInput(pb, beforeKwMarker, prepNumMarker);

    PabloAST * notStrSpan = pb.createNot(strSpan);
    PabloAST * strMarker = pb.createAnd(pb.createAdvance(strSpan, 1), notStrSpan);

    PabloAST * allValues = pb.createOr3(kwMarker, numMarker, strMarker);
    PabloAST * lBrak = pb.createOr(sanitizelCurly, sanitizelBracket);
    PabloAST * rBrak = pb.createOr(sanitizerCurly, sanitizerBracket);
    PabloAST * specialSymbols = pb.createOr(sanitizeColon, sanitizeComma);
    PabloAST * allSymbols = pb.createOr3(lBrak, rBrak, specialSymbols);

    pb.createAssign(pb.createExtract(nbrErr, pb.getInteger(0)), sanitizedErr);
    pb.createAssign(pb.createExtract(combinedOut, pb.getInteger(Combined::symbols)), allSymbols);
    pb.createAssign(pb.createExtract(combinedOut, pb.getInteger(Combined::lBrak)), lBrak);
    pb.createAssign(pb.createExtract(combinedOut, pb.getInteger(Combined::rBrak)), rBrak);
    pb.createAssign(pb.createExtract(combinedOut, pb.getInteger(Combined::values)), allValues);
}

void JSONParser::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    BixNumCompiler bnc(pb);

    bool genSingleBlock = mOnlyDepth > -1;
    BixNum ND = getInputStreamSet("ND");

    PabloAST * symbols = getInputStreamSet("combinedLexs")[Combined::symbols];
    PabloAST * validRBrak = getInputStreamSet("combinedLexs")[Combined::rBrak];
    PabloAST * allValues = getInputStreamSet("combinedLexs")[Combined::values];
    PabloAST * valueToken = pb.createLookahead(allValues, 1);
    PabloAST * anyToken = pb.createOr(symbols, valueToken);

    PabloAST * lCurly = pb.createAnd(symbols, getInputStreamSet("lexIn")[Lex::lCurly]);
    PabloAST * rCurly = pb.createAnd(symbols, getInputStreamSet("lexIn")[Lex::rCurly]);
    PabloAST * lBracket = pb.createAnd(symbols, getInputStreamSet("lexIn")[Lex::lBracket]);
    PabloAST * rBracket = pb.createAnd(symbols, getInputStreamSet("lexIn")[Lex::rBracket]);
    PabloAST * comma = pb.createAnd(symbols, getInputStreamSet("lexIn")[Lex::comma]);
    PabloAST * colon = pb.createAnd(symbols, getInputStreamSet("lexIn")[Lex::colon]);
    PabloAST * ws = getInputStreamSet("lexIn")[Lex::ws];
    PabloAST * str = pb.createAnd(valueToken, getInputStreamSet("lexIn")[Lex::dQuote]);
    PabloAST * valueTokenMinusStr = pb.createXor(valueToken, str);

    Var * const syntaxErr = getOutputStreamVar("syntaxErr");

    // parsing non-nesting values
    PabloAST * otherND = bnc.UGT(ND, 0);
    PabloAST * zeroND = bnc.EQ(ND, 0);
    PabloAST * EOFbit = pb.createAtEOF(pb.createAdvance(pb.createOnes(), 1));
    PabloAST * begin = pb.createNot(pb.createAdvance(pb.createOnes(), 1));
    PabloAST * valueAtZero = pb.createAnd(valueToken, zeroND);
    PabloAST * stopAtEOF = pb.createXor(pb.createOnes(), EOFbit);

    // If we have simple value at depth 0, we cannot have any other token
    PabloAST * firstValue = pb.createScanTo(begin, valueAtZero);
    PabloAST * nonNestedValue = pb.createScanTo(pb.createAdvance(firstValue, 1), anyToken);
    PabloAST * errValue = pb.createScanThru(pb.createAdvance(nonNestedValue, 1), stopAtEOF);
    
    // If we have any symbol, we cannot have any value at depth 0
    PabloAST * firstSymbol = pb.createScanTo(begin, symbols);
    PabloAST * valueAtZeroAfterSymbol = pb.createScanTo(pb.createAdvance(firstSymbol, 1), valueAtZero);
    PabloAST * errSymbol = pb.createScanThru(pb.createAdvance(valueAtZeroAfterSymbol, 1), stopAtEOF);
    
    // EOFbit is always at depth 0, otherwise we have unmatched parens
    PabloAST * errEOF = pb.createAnd(EOFbit, otherND);
    PabloAST * errSimpleValue = pb.createOr3(errValue, errSymbol, errEOF);

    // parsing arr
    Var * const errArray = pb.createVar("errArray", pb.createZeroes());
    for (int i = mMaxDepth; i >= 0; --i) {
        PabloAST * atDepth = bnc.EQ(ND, genSingleBlock ? mOnlyDepth : i);
        PabloAST * nested = bnc.UGT(ND, genSingleBlock ? mOnlyDepth : i);
        PabloAST * arrayStart = pb.createAnd(atDepth, lBracket);
        PabloAST * atDepthSpan = pb.createAnd(atDepth, pb.createNot(validRBrak));
        PabloAST * arrayEnd = pb.createScanThru(arrayStart, pb.createOr(nested, atDepthSpan));
        // it must not finish in rCurly
        PabloAST * errorAtEnd = pb.createAnd(arrayEnd, rCurly);

        PabloAST * arraySpan = pb.createIntrinsicCall(
            Intrinsic::ExclusiveSpan,
            { arrayStart, arrayEnd }
        );

        // Now validate that every value or nested item is followed
        // either by a comma or a the end rBracket.
        PabloAST * nestedSpan = pb.createAnd(nested, arraySpan);
        PabloAST * afterNested = pb.createAnd(pb.createAdvance(nestedSpan, 1), atDepth);
        PabloAST * valueAtDepth = pb.createAnd(atDepth, valueToken);
        PabloAST * afterToken = pb.createAdvance(pb.createAnd(valueAtDepth, arraySpan), 1);
        PabloAST * tokenNext = pb.createScanThru(pb.createOr(afterNested, afterToken), ws);
        PabloAST * notCommaRBracket = pb.createNot(pb.createOr(comma, rBracket));
        PabloAST * errAfterValue = pb.createAnd(tokenNext, notCommaRBracket);

        // Every comma must be followed by a value
        PabloAST * commaAtDepth = pb.createAnd3(comma, atDepth, arraySpan);
        PabloAST * nestedOrVTk = pb.createOr(nested, valueAtDepth);
        PabloAST * scanAnyTkAfterComma = pb.createScanTo(pb.createAdvance(commaAtDepth, 1), anyToken);
        PabloAST * errAfterComma = pb.createAnd(scanAnyTkAfterComma, pb.createNot(nestedOrVTk));

        // After the lBracket we must have either a value or an rBracket.
        PabloAST * nestedOrVTkRBracket = pb.createOr(nestedOrVTk, rBracket);
        PabloAST * scanAnyTkAfterArrStart = pb.createScanTo(pb.createAdvance(arrayStart, 1), anyToken);
        PabloAST * errAfterLBracket = pb.createAnd(scanAnyTkAfterArrStart, pb.createNot(nestedOrVTkRBracket));

        PabloAST * errBracket = pb.createOr(errorAtEnd, errAfterLBracket);
        PabloAST * errElement = pb.createOr(errAfterComma, errAfterValue);
        pb.createAssign(errArray, pb.createOr3(errArray, errBracket, errElement));

        if (genSingleBlock) { break; }
    }

    // parsing objects
    Var * const errObj = pb.createVar("errObj", pb.createZeroes());
    for (int i = mMaxDepth; i >= 0; --i) {
        PabloAST * atDepth = bnc.EQ(ND, genSingleBlock ? mOnlyDepth : i);
        PabloAST * nested = bnc.UGT(ND, genSingleBlock ? mOnlyDepth : i);
        PabloAST * objStart = pb.createAnd(atDepth, lCurly);
        PabloAST * atDepthSpan = pb.createAnd(atDepth, pb.createNot(validRBrak));
        PabloAST * objEnd = pb.createScanThru(objStart, pb.createOr(nested, atDepthSpan));
        // it must not finish in rBracket
        PabloAST * errorAtEnd = pb.createAnd(objEnd, rBracket);

        PabloAST * objSpan = pb.createIntrinsicCall(
            Intrinsic::ExclusiveSpan,
            { objStart, objEnd }
        );

        // Now validate that every value or nested item is followed
        // either by a comma or a the end rBracket.
        PabloAST * nestedSpan = pb.createAnd(nested, objSpan);
        PabloAST * afterNested = pb.createAnd(pb.createAdvance(nestedSpan, 1), atDepth);

        // process all values that are not str
        PabloAST * valueMinusStrAtDepth = pb.createAnd(atDepth, valueTokenMinusStr);
        PabloAST * afterTokenMinusStr = pb.createAdvance(pb.createAnd(valueMinusStrAtDepth, objSpan), 1);
        PabloAST * tokenNextMinusStr = pb.createScanThru(pb.createOr(afterNested, afterTokenMinusStr), ws);
        PabloAST * commaRCurly = pb.createOr(comma, rCurly);
        PabloAST * errAfterValueMinusStr = pb.createAnd(tokenNextMinusStr, pb.createNot(commaRCurly));

        // process str as both key and value
        PabloAST * strAtDepth = pb.createAnd3(str, atDepth, objSpan);
        PabloAST * afterTokenStr = pb.createAdvance(pb.createAnd(strAtDepth, objSpan), 1);
        PabloAST * tokenNextStr = pb.createScanThru(pb.createOr(afterNested, afterTokenStr), ws);
        PabloAST * commaColonRCurly = pb.createOr(commaRCurly, colon);
        PabloAST * errAfterValueStr = pb.createAnd(tokenNextStr, pb.createNot(commaColonRCurly));

        PabloAST * errAfterValue = pb.createOr(errAfterValueStr, errAfterValueMinusStr);

        // Every colon must be followed by a value
        PabloAST * colonAtDepth = pb.createAnd3(colon, atDepth, objSpan);
        PabloAST * nestedOrVTk = pb.createOr(nested, valueToken);
        PabloAST * scanAnyTkAfterColon = pb.createScanTo(pb.createAdvance(colonAtDepth, 1), anyToken);
        PabloAST * errAfterColon = pb.createAnd(scanAnyTkAfterColon, pb.createNot(nestedOrVTk));

        // Every comma must be followed by a key string
        PabloAST * commaAtDepth = pb.createAnd3(comma, atDepth, objSpan);
        PabloAST * scanAnyTkAfterComma = pb.createScanTo(pb.createAdvance(commaAtDepth, 1), anyToken);
        PabloAST * errAfterComma = pb.createAnd(scanAnyTkAfterComma, pb.createNot(strAtDepth));

        // After the lCurly we must have either a value or an rCurly.
        PabloAST * nestedOrVTkRCurly = pb.createOr(nestedOrVTk, rCurly);
        PabloAST * scanAnyTkAfterObjStart = pb.createScanTo(pb.createAdvance(objStart, 1), anyToken);
        PabloAST * errAfterLCurly = pb.createAnd(scanAnyTkAfterObjStart, pb.createNot(nestedOrVTkRCurly));

        PabloAST * errCurly = pb.createOr(errorAtEnd, errAfterLCurly);
        PabloAST * errElement = pb.createOr3(errAfterColon, errAfterComma, errAfterValue);
        pb.createAssign(errObj, pb.createOr3(errObj, errCurly, errElement));

        if (genSingleBlock) { break; }
    }

    PabloAST * allErrs = pb.createOr3(errSimpleValue, errArray, errObj);

    pb.createAssign(pb.createExtract(syntaxErr, pb.getInteger(0)), allErrs);
}
