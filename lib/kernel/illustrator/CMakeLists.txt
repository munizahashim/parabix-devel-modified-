# Copyright (c) 2019 International Characters.
# This software is licensed to the public under the Open Software License 3.0.

# module: kernel.illustrator

parabix_add_module(
NAME
    kernel.illustrator
SRC
    illustrator.cpp
DEPS
    toolchain
    ${Boost_LIBRARIES}
)
