/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <cinttypes>
#include <cstdlib>

extern "C" {

void postproc_simpleValidateJSON(const uint8_t * ptr);
void postproc_simpleError(const uint8_t * /*ptr*/);

}