#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: kernel.io

parabix_add_module(
NAME
    kernel.io
SRC
    source_kernel.cpp
    stdout_kernel.cpp
DEPS
    kernel.core
)
