#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: csv

parabix_add_executable(
NAME
    csv2json
SRC
    csv2json.cpp
DEPS
    pablo
    kernel.basis
    kernel.bitwise
    kernel.io
    kernel.pipeline
    kernel.streamutils
    kernel.util
    toolchain
)

parabix_add_executable(
NAME
    csv_column
SRC
    csv_column.cpp
DEPS
    pablo
    kernel.basis
    kernel.io
    kernel.pipeline
    kernel.streamutils
    kernel.util
    toolchain
)

