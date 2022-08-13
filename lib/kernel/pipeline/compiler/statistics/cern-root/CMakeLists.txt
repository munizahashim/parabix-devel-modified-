# Copyright (c) 2019 International Characters.
# This software is licensed to the public under the Open Software License 3.0.

# module: kernel.pipeline

parabix_add_module(
USES_CERNROOT
REQUIRES_RTTI
NAME
    kernel.pipeline.cernroot
SRC
    root_histogram_analysis.cpp
)
