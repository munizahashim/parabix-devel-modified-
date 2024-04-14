#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: re.printer

parabix_add_module(
NAME
    re.printer
SRC
    re_printer.cpp
DEPS
    re.adt
    unicode.data
)
