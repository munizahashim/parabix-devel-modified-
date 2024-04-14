#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: re.compile

parabix_add_module(
NAME
    re.compile
SRC
    re_compiler.cpp
DEPS
    re.analysis
    re.transforms
    re.printer
)
