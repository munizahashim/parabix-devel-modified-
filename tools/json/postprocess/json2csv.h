/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <cinttypes>
#include <cstdlib>

extern "C" {

/*
 * JSON input:
 *     [{ "key1" : "Luiz"}, {"wow": "value"}, {"key1": "ooh", "wow": "foo"}]
 *
 *  map = {
 *     "key1": {0: "Luiz"}, {2: "ooh"},
 *     "wow": {1: "value"}, {2: "foo"}
 *  }
 *
 *  CSV output:
 *
 *  key1, wow
 *  Luiz,
 *  ,value
 *  ooh,foo
 */
void json2csv_validateObjectsAndArrays(const uint8_t * ptr, const uint8_t * lineBegin, const uint8_t * /*lineEnd*/, uint64_t lineNum, uint64_t position);
void json2csv_simpleValidateObjectsAndArrays(const uint8_t * ptr);
void json2csv_doneCallback();

} // extern "C"
