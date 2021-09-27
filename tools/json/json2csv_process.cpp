#include "json2csv_process.h"

#include <cassert>
#include <cstdlib>
#include <string>
#include <sstream>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/ADT/SmallVector.h>

/*
 * <Json2CSV> ::= <Array>
 *
 * <Array> ::= '[' ']'
 *           | '[' <Elements> ']'
 * 
 * <Elements> ::= <Object>
 *              | <Object> ',' <Elements>
 *
 * <Object> ::= '{' '}'
 *            | '{' <Members> '}'
 * 
 * <Members> ::= <Pair>
 *             | <Pair> ',' <Members>
 * 
 * <Pair> ::= String ':' String
 * 
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
    JNextComma,
    JDone
};

static llvm::SmallVector<const u_int8_t *, 2> stack{};
static JSONState currentState = JInit;

static ptrdiff_t json2csv_getColumn(const uint8_t * ptr, const uint8_t * lineBegin) {
    ptrdiff_t column = ptr - lineBegin;
    assert (column >= 0);
    return column;
}

static std::string json2csv_getLineAndColumnInfo(const std::string str, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum) {
    ptrdiff_t column = json2csv_getColumn(ptr, lineBegin);
    std::stringstream ss;
    ss << str << " at line " << lineNum << " column " << column << " starting in:\n\n" << ptr;
    return ss.str();
}

static void json2csv_popAndFindNewState() {
    assert(!stack.empty());
    stack.pop_back();
    if (stack.empty()) {
        currentState = JDone;
        return;
    }
    const uint8_t last = *stack.back();
    if (last == '{') {
        currentState = JNextComma;
    } else {
        llvm_unreachable("The stack has an unknown char");
    }
}

static void json2csv_parseArr(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '[') {
        currentState = JArrInit;
    } else {
        llvm::report_fatal_error(json2csv_getLineAndColumnInfo("Error parsing array", ptr, lineBegin, lineNum));
    }
    stack.push_back(ptr);
}

static bool json2csv_parseObj(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '{') {
        currentState = JObjInit;
    } else {
        llvm::report_fatal_error(json2csv_getLineAndColumnInfo("Error parsing object", ptr, lineBegin, lineNum));
        return false;
    }
    stack.push_back(ptr);
    return true;
}

static void json2csv_parseStrOrPop(bool popAllowed, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '"') {
        currentState = JKStrBegin;
    } else if (*ptr == '}' && popAllowed) {
        json2csv_popAndFindNewState();
    } else {
        llvm::report_fatal_error(json2csv_getLineAndColumnInfo("Error parsing object", ptr, lineBegin, lineNum));
    }
}

static bool json2csv_parseStrValue(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '"') {
        currentState = JVStrBegin;
        return true;
    } else {
        llvm::report_fatal_error(json2csv_getLineAndColumnInfo("Error parsing string value", ptr, lineBegin, lineNum));
    }
    return false;
}

static void json2csv_parseObjOrPop(bool popAllowed, const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (!json2csv_parseObj(ptr, lineBegin, lineNum, position)) {
        if (*ptr == ']' && popAllowed) {
            currentState = JDone;
        } else {
            llvm::report_fatal_error(json2csv_getLineAndColumnInfo("Error parsing array", ptr, lineBegin, lineNum));
        }
    }
}

static void json2csv_parseStr(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == '"' && currentState == JKStrBegin) {
        currentState = JKStrEnd;
    } else if (*ptr == '"' && currentState == JVStrBegin) {
        currentState = JVStrEnd;
    } else {
        llvm::report_fatal_error(json2csv_getLineAndColumnInfo("Error parsing string", ptr, lineBegin, lineNum));
    }
}

static void json2csv_parseColon(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (*ptr == ':') {
        currentState = JObjColon;
    } else {
        llvm::report_fatal_error(json2csv_getLineAndColumnInfo("Error parsing object", ptr, lineBegin, lineNum));
    }
}

static void json2csv_parseCommaOrPop(const uint8_t * ptr, const uint8_t * lineBegin, uint64_t lineNum, uint64_t position) {
    if (stack.empty()) {
        llvm::report_fatal_error(json2csv_getLineAndColumnInfo("Stack is empty", ptr, lineBegin, lineNum));
        return;
    }
    const uint8_t last = *stack.back();
    if (*ptr == ',') {
        if (last == '{') {
            currentState = JObjInit;
        } else {
            llvm::report_fatal_error(json2csv_getLineAndColumnInfo("Wrong char in stack", ptr, lineBegin, lineNum));
        }
    } else if (*ptr == '}' && last == '{') {
        json2csv_popAndFindNewState();
    } else if (*ptr == ']' && last == '[') {
        json2csv_popAndFindNewState();
    } else {
        llvm::report_fatal_error(json2csv_getLineAndColumnInfo("Error parsing object", ptr, lineBegin, lineNum));
    }
}

void json2csv_doneCallback() {
    if (stack.empty()) {
        currentState = JDone;
        return;
    }
    llvm::report_fatal_error("Found EOF but the JSON is missing elements");
}

void json2csv_validateObjectsAndArrays(const uint8_t * ptr, const uint8_t * lineBegin, const uint8_t * /*lineEnd*/, uint64_t lineNum, uint64_t position) {
    if (currentState == JInit) {
        json2csv_parseArr(ptr, lineBegin, lineNum, position);
    } else if (currentState == JObjInit) {
        json2csv_parseStrOrPop(true, ptr, lineBegin, lineNum, position);
    } else if (currentState == JArrInit) {
        json2csv_parseObjOrPop(true, ptr, lineBegin, lineNum, position);
    } else if (currentState == JKStrBegin || currentState == JVStrBegin) {
        json2csv_parseStr(ptr, lineBegin, lineNum, position);
    } else if (currentState == JKStrEnd) {
        json2csv_parseColon(ptr, lineBegin, lineNum, position);
    } else if (currentState == JObjColon) {
        json2csv_parseStrValue(ptr, lineBegin, lineNum, position);
    } else if (currentState == JVStrEnd || currentState == JNextComma) {
        json2csv_parseCommaOrPop(ptr, lineBegin, lineNum, position);
    } else if (currentState == JDone) {
        llvm::report_fatal_error(json2csv_getLineAndColumnInfo("JSON has been already processed", ptr, lineBegin, lineNum));
    }
}
