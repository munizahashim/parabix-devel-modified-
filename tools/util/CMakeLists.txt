#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: util

parabix_add_executable(
NAME
    hexlines
SRC
    hexlines.cpp
DEPS
    pablo
    kernel.basis
    kernel.io
    kernel.pipeline
    kernel.streamutils
    kernel.util
    toolchain
)


parabix_add_executable(
NAME
    hashdemo
SRC
    hashdemo.cpp
DEPS
    grep
    pablo
    kernel.basis
    kernel.io
    kernel.pipeline
    kernel.streamutils
    kernel.util
    toolchain
)


