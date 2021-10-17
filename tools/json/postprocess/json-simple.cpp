#include "json-detail.h"
#include "json-simple.h"

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
 *             | <Pair> <ObjComma> <Members>
 * 
 * <Pair> ::= StringKey ':' <Value>
 * 
 * <Array> ::= '[' ']'
 *           | '[' <Elements> ']'
 * 
 * <Elements> ::= <Value>
 *              | <Value> <ArrComma> <Elements>
 * 
 * <Value> ::= StringValue
 *           | Number
 *           | <Object>
 *           | <Array>
 *           | true
 *           | false
 *           | null
 *
 * <StringKey> ::= '"' '"'
 * <StringValue> ::= '"' '"'
 *
 * <ObjComma> ::= ','
 * <ArrComma> ::= ','
 */

enum JSONState {
    JValue = 0,
    JKStrBegin,
    JKStrEnd,
    JVStrBegin,
    JVStrEnd,
    JObjInit,
    JObjColon,
    JObjComma,
    JArrInit,
    JArrComma,
    JFindNewState,
    JDone
};

static llvm::SmallVector<const u_int8_t *, 32> stack{};
static JSONState currentState = JValue;

static void parseValue(const uint8_t *ptr);
static void parseKStrBegin(const uint8_t *ptr);
static void parseKStrEnd(const uint8_t *ptr);
static void parseVStrBegin(const uint8_t *ptr);
static void parseVStrEnd(const uint8_t *ptr);
static void parseObj(const uint8_t *ptr);
static void parseObjColon(const uint8_t *ptr);
static void parseObjAfterComma(const uint8_t *ptr);
static void parseArr(const uint8_t *ptr);
static void parseArrAfterComma(const uint8_t *ptr);
static void parseNewState(const uint8_t *ptr);
static void parseDone(const uint8_t *ptr);

typedef void (*HandlerFnTy)(const uint8_t *);
static HandlerFnTy jsonJumpTable[JDone + 1] = {
    parseValue,
    parseKStrBegin,
    parseKStrEnd,
    parseVStrBegin,
    parseVStrEnd,
    parseObj,
    parseObjColon,
    parseObjAfterComma,
    parseArr,
    parseArrAfterComma,
    parseNewState,
    parseDone
};

static void parseValue(const uint8_t *ptr) {
    if (*ptr == '"') {
        currentState = JVStrBegin;
    } else if (*ptr == '{') {
        currentState = JObjInit;
    } else if (*ptr == '[') {
        currentState = JArrInit;
    } else if (*ptr >= '0' && *ptr <= '9') {
        currentState = JFindNewState;
    } else if (*ptr == 'n') {
        currentState = JFindNewState;
    } else if (*ptr == 't') {
        currentState = JFindNewState;
    } else if (*ptr == 'f') {
        currentState = JFindNewState;
    } else {
        postproc_simpleError(nullptr);
    }
}

static void parseObj(const uint8_t *ptr) {
    if (*ptr == '"') {
        currentState = JKStrBegin;
    } else if (*ptr == '}') {
        currentState = JFindNewState;
    } else {
        postproc_simpleError(nullptr);
    }
}

static void parseArr(const uint8_t *ptr) {
    if (*ptr == ']') {
        currentState = JFindNewState;
    } else {
        parseValue(ptr);
    }
}

static void parseKStrBegin(const uint8_t *ptr) {
    if (*ptr == '"') {
        currentState = JKStrEnd;
    } else {
        postproc_simpleError(nullptr);
    }
}

static void parseKStrEnd(const uint8_t *ptr) {
    if (*ptr == ':') {
        currentState = JObjColon;
    } else {
        postproc_simpleError(nullptr);
    }
}

static void parseVStrBegin(const uint8_t *ptr) {
    if (*ptr == '"') {
        currentState = JVStrEnd;
    } else {
        postproc_simpleError(nullptr);
    }
}

static void parseVStrEnd(const uint8_t *ptr) {
    parseNewState(ptr);
}

static void parseObjColon(const uint8_t *ptr) {
    parseValue(ptr);
}

static void parseObjAfterComma(const uint8_t *ptr) {
    if (*ptr == '{') {
        currentState = JObjInit;
    } else {
        postproc_simpleError(nullptr);
    }
}

static void parseArrAfterComma(const uint8_t *ptr) {
    if (*ptr == '[') {
        currentState = JArrInit;
    } else {
        postproc_simpleError(nullptr);
    }
}

static void popAndFindNewState(const uint8_t *last) {
    stack.pop_back();
    if (stack.empty()) {
        currentState = JDone;
        return;
    }

    if (*last == '{') {
        currentState = JObjInit;
    } else if (*last == '[') {
        currentState = JArrInit;
    } else {
        postproc_simpleError(nullptr);
    }
}

static void parseNewState(const uint8_t *ptr) {
    assert(!stack.empty());
    const uint8_t last = *stack.back();
    if (*ptr == ',') {
        if (last == '{') {
            currentState = JObjComma;
        } else if (last == '[') {
            currentState = JArrComma;
        } else {
            postproc_simpleError(nullptr);
        }
    } else {
        popAndFindNewState(&last);
    }
}

static void parseDone(const uint8_t *ptr) {
    postproc_doneCallback();
}

void postproc_simpleValidateJSON(const uint8_t * ptr) {
    jsonJumpTable[currentState](ptr);
}

void postproc_simpleError(const uint8_t * /*ptr*/) {
    fprintf(stderr, "The JSON document has an improper structure: missing or superfluous commas, braces, missing keys, etc.\n");
    exit(-1);
}