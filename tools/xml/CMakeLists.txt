#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: xml

parabix_add_pablo_src(xml.pablosrc xml.pablo)

parabix_add_executable(
NAME
    xml
SRC
    post_process.cpp
    test_suite_error.cpp
    xml.cpp
DEPS
    pablo.parse
    kernel.basis
    kernel.io
    kernel.pipeline
    kernel.scan
    kernel.streamutils
    kernel.util
    toolchain
)

add_dependencies(xml xml.pablosrc)
