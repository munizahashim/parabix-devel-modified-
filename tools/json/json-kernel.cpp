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

static PabloAST * sanitizeLexInput(PabloBuilder & pb, PabloAST * strSpan, PabloAST * lexInMarker) {
    PabloAST * conflict = pb.createAnd(strSpan, lexInMarker);
    return pb.createXor(lexInMarker, conflict);
}

void JSONStringMarker::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * dQuotes = getInputStreamSet("dQuotes")[0];
    PabloAST * backslash = getInputStreamSet("backslash")[0];
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
    PabloAST * N = pb.createAnd(notStrSpan, getInputStreamSet("n")[0]);
    PabloAST * T = pb.createAnd(notStrSpan, getInputStreamSet("t")[0]);
    PabloAST * F = pb.createAnd(notStrSpan, getInputStreamSet("f")[0]);

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
    PabloAST * hyphenIn = getInputStreamSet("hyphen")[0];
    PabloAST * digitIn = getInputStreamSet("digit")[0];
    PabloAST * rCurlyIn = getInputStreamSet("rCurly")[0];
    PabloAST * rBracketIn = getInputStreamSet("rBracket")[0];
    PabloAST * commaIn = getInputStreamSet("comma")[0];

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
    PabloAST * eE = pb.createAnd(notStrSpan, alleE);
    PabloAST * dot = pb.createAnd(notStrSpan, allDot);
    PabloAST * plusMinus = pb.createAnd(notStrSpan, allPlusMinus);

    PabloAST * nondigit = pb.createNot(digit);
    PabloAST * nonDigitNorEe = pb.createAnd(nondigit, pb.createNot(eE));
    PabloAST * advHyphen = pb.createAnd(hyphen, pb.createAdvance(nonDigitNorEe, 1));

    PabloAST * nonDigitEePlusMinus = pb.createAnd(nonDigitNorEe, pb.createNot(plusMinus));
    PabloAST * nonDigitEePlusMinusDot = pb.createAnd(nonDigitEePlusMinus, pb.createNot(dot));
    PabloAST * advDigit = pb.createAnd(digit, pb.createAdvance(nonDigitEePlusMinusDot, 1));
    PabloAST * beginNbr = pb.createOr(advDigit, advHyphen);
    pb.createAssign(pb.createExtract(nbrLex, pb.getInteger(0)), beginNbr);

    PabloAST * errDot = pb.createAnd(pb.createAdvance(dot, 1), nondigit);
    PabloAST * errPlusMinus = pb.createAnd(pb.createAdvance(plusMinus, 1), nondigit);
    PabloAST * eENotPlusMinus = pb.createAnd(pb.createAdvance(eE, 1), pb.createNot(plusMinus));
    PabloAST * erreENotPlusMinus = pb.createAnd(eENotPlusMinus, nondigit);
    PabloAST * potentialErr = pb.createOr3(errDot, errPlusMinus, erreENotPlusMinus);

    PabloAST * nextValidToken = pb.createOr3(rCurlyIn, rBracketIn, commaIn);
    PabloAST * err = sanitizeLexInput(pb, nextValidToken, potentialErr);
    pb.createAssign(pb.createExtract(nbrErr, pb.getInteger(0)), err);

    PabloAST * fstPartNbr = pb.createIntrinsicCall(Intrinsic::InclusiveSpan, {beginNbr, digit});
    PabloAST * sndPartNbr = pb.createIntrinsicCall(Intrinsic::InclusiveSpan, {eE, digit});
    PabloAST * trdPartNbr = pb.createIntrinsicCall(Intrinsic::InclusiveSpan, {dot, digit});
    PabloAST * finalNbr = pb.createOr3(fstPartNbr, sndPartNbr, trdPartNbr);
    pb.createAssign(pb.createExtract(nbrSpan, pb.getInteger(0)), finalNbr);
}

void JSONFindKwAndExtraneousChars::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * combinedSpans = getInputStreamSet("combinedSpans")[0];
    std::vector<PabloAST *> kwEndMarkers = getInputStreamSet("kwEndMarkers");

    Var * const nbrErr = getOutputStreamVar("extraErr");
    Var * const keywordMarker = getOutputStreamVar("kwMarker");

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

    PabloAST * finalKeywordMarker = pb.createOr3(nBegin, tBegin, fBegin);
    PabloAST * keywordSpans = pb.createOr3(nSpan, tSpan, fSpan);

    PabloAST * extraneousChars = pb.createNot(pb.createOr(keywordSpans, combinedSpans));

    pb.createAssign(pb.createExtract(keywordMarker, pb.getInteger(0)), finalKeywordMarker);
    pb.createAssign(pb.createExtract(nbrErr, pb.getInteger(0)), extraneousChars);
}

void JSONLexSanitizer::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    std::vector<PabloAST *> lexIn = getInputStreamSet("lexIn");
    PabloAST * strSpan = getInputStreamSet("strSpan")[0];
    PabloAST * validDQuotes = getInputStreamSet("validDQuotes")[0];
    Var * const lexOut = getOutputStreamVar("lexOut");

    PabloAST * sanitizelCurly = sanitizeLexInput(pb, strSpan, lexIn[Lex::lCurly]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::lCurly)), sanitizelCurly);

    PabloAST * sanitizerCurly = sanitizeLexInput(pb, strSpan, lexIn[Lex::rCurly]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::rCurly)), sanitizerCurly);

    PabloAST * sanitizelBracket = sanitizeLexInput(pb, strSpan, lexIn[Lex::lBracket]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::lBracket)), sanitizelBracket);

    PabloAST * sanitizerBracket = sanitizeLexInput(pb, strSpan, lexIn[Lex::rBracket]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::rBracket)), sanitizerBracket);

    PabloAST * sanitizeColon = sanitizeLexInput(pb, strSpan, lexIn[Lex::colon]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::colon)), sanitizeColon);

    PabloAST * sanitizeComma = sanitizeLexInput(pb, strSpan, lexIn[Lex::comma]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::comma)), sanitizeComma);

    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::dQuote)), validDQuotes);

    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::hyphen)), lexIn[Lex::hyphen]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::digit)), lexIn[Lex::digit]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::backslash)), lexIn[Lex::backslash]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::n)), lexIn[Lex::n]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::t)), lexIn[Lex::t]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::f)), lexIn[Lex::f]);
    pb.createAssign(pb.createExtract(lexOut, pb.getInteger(Lex::ws)), lexIn[Lex::ws]);
}

void JSONErrsSanitizer::generatePabloMethod() {
    PabloBuilder pb(getEntryScope());
    PabloAST * ws = getInputStreamSet("ws")[0];
    PabloAST * errsIn = getInputStreamSet("errsIn")[0];
    Var * const errsOut = getOutputStreamVar("errsOut");

    PabloAST * sanitizedErr = sanitizeLexInput(pb, ws, errsIn);

    pb.createAssign(pb.createExtract(errsOut, pb.getInteger(0)), sanitizedErr);
}
