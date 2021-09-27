/*
 *  Copyright (c) 2020 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#pragma once

#include <cinttypes>
#include <cstdlib>

extern "C" {

void json2csv_validateObjectsAndArrays(const uint8_t * ptr, const uint8_t * lineBegin, const uint8_t * /*lineEnd*/, uint64_t lineNum, uint64_t position);

void json2csv_doneCallback();

} // extern "C"
