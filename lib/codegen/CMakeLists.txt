#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: codegen

parabix_add_module(
NAME
    codegen
SRC
    CBuilder.cpp
DEPS
    toolchain
    ${Boost_LIBRARIES}
)

if(ARTIFACT_MODE STREQUAL "Single")
    add_dependencies(${PARABIX_MONO_TARGET} recompile-object-cache)
else()
    add_dependencies(codegen recompile-object-cache)
endif()
