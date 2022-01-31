/*
 *  Copyright (c) 2020 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#pragma once

#include <cinttypes>
#include <cstdlib>

extern "C" {

void postproc_parensValidate(const uint8_t * ptr);
void postproc_parensError(const uint8_t * /*ptr*/);

} // extern "C"