#  Part of the Parabix Project, under the Open Software License 3.0.
#  SPDX-License-Identifier: OSL-3.0

# module: icgrep

parabix_add_executable(
NAME
    icgrep
SRC
    grep_interface.cpp
    icgrep.cpp
DEPS
    fileselect
    grep
)
