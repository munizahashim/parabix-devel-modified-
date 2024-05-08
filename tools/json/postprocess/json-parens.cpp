#include "json-detail.h"
#include "json-parens.h"

#include <cassert>
#include <cstdlib>
#include <string>
#include <sstream>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/ADT/SmallVector.h>
#include <toolchain/toolchain.h>

/*
 * <Json> ::= <Value>
 * 
 * <Object> ::= '{' '}'
 *            | '{' <Value> '}'
 * 
 * <Array> ::= '[' ']'
 *           | '[' <Value> ']'
 * 
 * <Value> ::= <Object>
 *           | <Array>
 *
 */

enum JSONState {
    JValue = 0,
    JObjInit,
    JArrInit,
    JFindNewState,
    JDone
};

static llvm::SmallVector<const u_int8_t *, 32> stack{};
static JSONState currentState = JValue;

static void parseValue(const uint8_t *ptr);
static void parseObj(const uint8_t *ptr);
static void parseArr(const uint8_t *ptr);
static void parseNewState(const uint8_t *ptr);
static void parseDone(const uint8_t *ptr);

static void popAndFindNewState();

typedef void (*HandlerFnTy)(const uint8_t *);
static HandlerFnTy jsonJumpTable[JDone + 1] = {
    parseValue,
    parseObj,
    parseArr,
    parseNewState,
    parseDone
};

static void parseValue(const uint8_t *ptr) {
    if (*ptr == '{') {
        stack.push_back(ptr);
        currentState = JObjInit;
    } else if (*ptr == '[') {
        stack.push_back(ptr);
        currentState = JArrInit;
    } else {
        postproc_parensError(1);
    }
}

static void parseObj(const uint8_t *ptr) {
    if (*ptr == '}') {
        popAndFindNewState();
    } else {
        parseValue(ptr);
    }
}

static void parseArr(const uint8_t *ptr) {
    if (*ptr == ']') {
        popAndFindNewState();
    } else {
        parseValue(ptr);
    }
}

static void popAndFindNewState() {
    stack.pop_back();
    if (stack.empty()) {
        currentState = JDone;
        return;
    }

    const uint8_t last = *stack.back();
    if (last == '{') {
        currentState = JFindNewState;
    } else if (last == '[') {
        currentState = JFindNewState;
    } else {
        postproc_parensError(1);
    }
}

static void parseNewState(const uint8_t *ptr) {
    assert(!stack.empty());
    if (*ptr == ']') {
        popAndFindNewState();
    } else if (*ptr == '}') {
        popAndFindNewState();
    } else {
        parseValue(ptr);
    }
}

static void parseDone(const uint8_t *ptr) {
    postproc_doneCallback();
}

void postproc_parensValidate(const uint8_t * ptr) {
    jsonJumpTable[currentState](ptr);
}

void postproc_parensError(const uint64_t errsCount) {
    if (errsCount > 0) {
        fprintf(stderr, "The JSON document has an improper structure: parentheses don't match.\n");
        if (!codegen::EnableIllustrator) exit(-1);
    }
}
