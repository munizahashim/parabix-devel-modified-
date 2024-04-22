/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <cinttypes>
#include <cstdlib>

extern "C" {

void postproc_parensValidate(const uint8_t * ptr);
void postproc_parensError(const uint64_t errsCount);

} // extern "C"