#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: json

parabix_add_pablo_src(json.pablosrc json.pablo)

parabix_add_executable(
NAME
    json
SRC
    json.cpp
    postprocess/json-simple.cpp
    postprocess/json-detail.cpp
    postprocess/json-parens.cpp
    postprocess/json2csv.cpp
    json-kernel.cpp
DEPS
    grep
    pablo.parse
    kernel.basis
    kernel.io
    kernel.pipeline
    kernel.scan
    kernel.streamutils
    kernel.util
    toolchain
)

add_dependencies(json json.pablosrc)
