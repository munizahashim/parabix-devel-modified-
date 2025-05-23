# Copyright (c) 2019 International Characters.
# This software is licensed to the public under the Open Software License 3.0.

# Place test executables in build/tests/bin
set(TEST_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${TEST_BIN_DIR})

# Add a new kernel-test target.
#
# Creates a new target with a given `NAME` from a source file with the same name
# (e.g., target `test0` from `test0.cpp`). Additional dependencies may be added
# after `NAME` and they will be linked with the resultant executable.
#
# Also adds a new test to `make check`.
function(parabix_add_kernel_test NAME)
    parabix_add_executable(NAME ${NAME} SRC ${NAME}.cpp DEPS testing ${ARGN})
    set(KERNEL_TEST_LIST ${KERNEL_TEST_LIST} ${TEST_BIN_DIR}/${NAME} PARENT_SCOPE)
    set(KERNEL_TEST_TARGET_LIST ${KERNEL_TEST_TARGET_LIST} ${NAME} PARENT_SCOPE)
    add_test(NAME "${NAME}"
        WORKING_DIRECTORY ${TEST_BIN_DIR}
        COMMAND ./${NAME})
endfunction(parabix_add_kernel_test)


###   Test Targets   ###
parabix_add_kernel_test(meta_tests kernel.streamutils)
parabix_add_kernel_test(test_scan kernel.basis kernel.scan kernel.unicode)
parabix_add_kernel_test(test_bit_movement kernel.streamutils)
parabix_add_kernel_test(index_test kernel.streamutils)
parabix_add_kernel_test(nesting_tests kernel.util)
parabix_add_kernel_test(everynth_tests pablo)
parabix_add_kernel_test(test_repeatingstreamset kernel.io)
parabix_add_kernel_test(test_bytefilterandspread kernel.io kernel.streamutils)
parabix_add_kernel_test(test_emptyprogram kernel.io)
parabix_add_kernel_test(test_truncatedstreamset kernel.io)

# `make kernel-tests` to run all kernel tests
add_custom_target(kernel-tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND python3 run_kernel_tests.py ${KERNEL_TEST_LIST})

add_dependencies(kernel-tests ${KERNEL_TEST_TARGET_LIST})

# Build target for all kernel tests: does not run any tests
add_custom_target(kernel-test-suite DEPENDS ${KERNEL_TEST_TARGET_LIST})
