# This software is licensed to the public under the Open Software License 3.0.

# module: csv

parabix_add_executable(
NAME
    csv2json
SRC
    csv2json
DEPS
    pablo
    kernel.basis
    kernel.io
    kernel.pipeline
    kernel.streamutils
    kernel.util
    toolchain
)

