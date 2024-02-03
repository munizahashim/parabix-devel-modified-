# Copyright (c) 2022 International Characters.
# This software is licensed to the public under the Open Software License 3.0.

# module: kernel.unicode

parabix_add_module(
NAME
    kernel.unicode
SRC
    charclasses.cpp
    boundary_kernels.cpp
    UCD_property_kernel.cpp
    utf8gen.cpp
    utf8_decoder.cpp
    utf16gen.cpp
DEPS
    kernel.pipeline
    kernel.streamutils
    re.unicode
    pablo
    pablo.bixnum
)
