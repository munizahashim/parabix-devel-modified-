#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: csv

parabix_add_executable(
NAME
    ustats
SRC
    ustats
DEPS
    kernel.basis
    kernel.core
    kernel.io
    kernel.pipeline
    kernel.util
    kernel.unicode

    re.adt
    re.cc
    re.unicode

    toolchain
)

