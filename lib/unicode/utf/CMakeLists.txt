# Copyright (c) 2019 International Characters.
# This software is licensed to the public under the Open Software License 3.0.

# module: unicode.utf

parabix_add_module(
NAME
    unicode.utf
SRC
    utf_encoder.cpp
    utf_compiler.cpp
DEPS
    re.cc
    unicode.core
)
