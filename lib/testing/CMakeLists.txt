#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: testing

parabix_add_module(
NAME
    testing
SRC
    assert.cpp
    runtime.cpp
DEPS
    pablo
    kernel.core
    kernel.io
    kernel.pipeline
    kernel.util
)
