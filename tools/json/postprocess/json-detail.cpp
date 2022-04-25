#include "json-detail.h"

#include <cassert>
#include <cstdlib>
#include <string>
#include <sstream>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/ADT/SmallVector.h>

/*
 * <Json> ::= <Value>
 * 
 * <Object> ::= '{' '}'
 *            | '{' <Members> '}'
 * 
 * <Members> ::= <Pair>
 *             | <Pair> ',' <Members>
 * 
 * <Pair> ::= String ':' <Value>
 * 
 * <Array> ::= '[' ']'
 *           | '[' <Elements> ']'
 * 
 * <Elements> ::= <Value>
 *              | <Value> ',' <Elements>
 * 
 * <Value> ::= String
 *           | Number
 *           | <Object>
 *           | <Array>
 *           | true
 *           | false
 *           | null
 */

enum JSONState {
    JInit = 0,
    JKStrBegin,
    JKStrEnd,
    JVStrBegin,
    JVStrEnd,
    JObjInit,
    JObjColon,
    JArrInit,
    JValue,
    JNextComma,
    JDone
};

static llvm::SmallVector<const u_int8_t *, 32> stack{};
static JSONState currentState = JInit;

static void postproc_reportError(const std::string str) {
    fprintf(stderr, "%s\n", str.c_str());
    exit(-1);
}

static ptrdiff_t postproc_getColumn(const uint8_t * ptr, const uint8_t * lineBegin) {
    if (lineBegin == nullptr) {
        return 0;
    }
    ptrdiff_t column = ptr - lineBegin;
    assert (column >= 0);
    return column;
}

static std::string postproc_getLineAndColumnInfo(const std::string str, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum) {
    ptrdiff_t column = postproc_getColumn(ptr, lineBegin);
    std::stringstream ss;
    ss << str << " at line " << lineNum << " column " << column << " starting in:\n\n" << ptr;
    return ss.str();
}

static bool isControl(const uint8_t * ptr) {
    if (*ptr == '}' || *ptr == ']' || *ptr == ',' || *ptr == ':') {
        return true;
    } else {
        return false;
    }
}

static void postproc_popAndFindNewState() {
    assert(!stack.empty());
    stack.pop_back();
    if (stack.empty()) {
        currentState = JDone;
        return;
    }
    const uint8_t last = *stack.back();
    if (last == '{' || last == '[') {
        currentState = JNextComma;
    } else {
        postproc_reportError("The stack has an unknown char");
    }
}

static void postproc_parseArrOrObj(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '{') {
        currentState = JObjInit;
    } else if (*ptr == '[') {
        currentState = JArrInit;
    } else {
        postproc_reportError(postproc_getLineAndColumnInfo("Error parsing object or array", ptr, lineBegin, lineNum));
    }
    stack.push_back(ptr);
}

static void postproc_parseStrOrPop(bool popAllowed, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '"') {
        currentState = JKStrBegin;
    } else if (*ptr == '}' && popAllowed) {
        postproc_popAndFindNewState();
    } else {
        postproc_reportError(postproc_getLineAndColumnInfo("Error parsing object", ptr, lineBegin, lineNum));
    }
}

static bool postproc_parseValue(bool strict, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '"') {
        currentState = JVStrBegin;
        return true;
    } else if (*ptr == '{' || *ptr == '[') {
        postproc_parseArrOrObj(ptr, lineBegin, lineNum, position);
        return true;
    } else if (!isControl(ptr)) {
        currentState = JValue;
        return true;
    } else if (strict) {
        postproc_reportError(postproc_getLineAndColumnInfo("Error parsing value", ptr, lineBegin, lineNum));
    }
    return false;
}

static void postproc_parseValueOrPop(bool popAllowed, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (!postproc_parseValue(false, ptr, lineBegin, lineNum, position)) {
        if (*ptr == ']' && popAllowed) {
            postproc_popAndFindNewState();
        } else {
            postproc_reportError(postproc_getLineAndColumnInfo("Error parsing array", ptr, lineBegin, lineNum));
        }
    }
}

static void postproc_parseStr(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '"' && currentState == JKStrBegin) {
        currentState = JKStrEnd;
    } else if (*ptr == '"' && currentState == JVStrBegin) {
        currentState = JVStrEnd;
    } else {
        postproc_reportError(postproc_getLineAndColumnInfo("Error parsing string", ptr, lineBegin, lineNum));
    }
}

static void postproc_parseColon(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == ':') {
        currentState = JObjColon;
    } else {
        postproc_reportError(postproc_getLineAndColumnInfo("Error parsing object", ptr, lineBegin, lineNum));
    }
}

static void postproc_parseCommaOrPop(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (stack.empty()) {
        postproc_reportError(postproc_getLineAndColumnInfo("Stack is empty", ptr, lineBegin, lineNum));
        return;
    }
    const uint8_t last = *stack.back();
    if (*ptr == ',') {
        if (last == '{') {
            currentState = JObjInit;
        } else if (last == '[') {
            currentState = JArrInit;
        } else {
            postproc_reportError(postproc_getLineAndColumnInfo("Wrong char in stack", ptr, lineBegin, lineNum));
        }
    } else if (*ptr == '}' && last == '{') {
        postproc_popAndFindNewState();
    } else if (*ptr == ']' && last == '[') {
        postproc_popAndFindNewState();
    } else {
        postproc_reportError(postproc_getLineAndColumnInfo("Error parsing object", ptr, lineBegin, lineNum));
    }
}

void postproc_doneCallback() {
    if (stack.empty()) {
        currentState = JDone;
        return;
    }
    postproc_reportError("Found EOF but the JSON is missing elements");
}

void postproc_validateObjectsAndArrays(const uint8_t * ptr, const uint8_t * lineBegin, const uint8_t * /*lineEnd*/, uint64_t lineNum, uint64_t position) {
    if (currentState == JInit) {
        postproc_parseValue(true, ptr, lineBegin, lineNum, position);
    } else if (currentState == JObjInit) {
        postproc_parseStrOrPop(true, ptr, lineBegin, lineNum, position);
    } else if (currentState == JArrInit) {
        postproc_parseValueOrPop(true, ptr, lineBegin, lineNum, position);
    } else if (currentState == JKStrBegin || currentState == JVStrBegin) {
        postproc_parseStr(ptr, lineBegin, lineNum, position);
    } else if (currentState == JKStrEnd) {
        postproc_parseColon(ptr, lineBegin, lineNum, position);
    } else if (currentState == JObjColon) {
        postproc_parseValue(true, ptr, lineBegin, lineNum, position);
    } else if (currentState == JVStrEnd || currentState == JValue || currentState == JNextComma) {
        postproc_parseCommaOrPop(ptr, lineBegin, lineNum, position);
    } else if (currentState == JDone) {
        postproc_reportError(postproc_getLineAndColumnInfo("JSON has been already processed", ptr, lineBegin, lineNum));
    }
}
