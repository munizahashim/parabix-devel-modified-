/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <cinttypes>
#include <cstdlib>

extern "C" {

void postproc_validateObjectsAndArrays(const uint8_t * ptr, const uint8_t * lineBegin, const uint8_t * /*lineEnd*/, uint64_t lineNum, uint64_t position);
void postproc_doneCallback();

} // extern "C"
