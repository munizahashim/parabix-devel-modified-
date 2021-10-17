#include "json-simple.h"

#include <cassert>
#include <cstdlib>
#include <string>
#include <sstream>
#include <llvm/Support/Compiler.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/ADT/SmallVector.h>

/*
 * <Json> ::= <Object>
 *          | <Array>
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

void postproc_simpleValidateObjectsAndArrays(const uint8_t * ptr) {
    
}

void postproc_simpleError(const uint8_t * /*ptr*/) {
    fprintf(stderr, "The JSON document has an improper structure: missing or superfluous commas, braces, missing keys, etc.\n");
}