# Try to find Libbacktrace headers and libraries.
#
#    https://github.com/ianlancetaylor/libbacktrace
#
# Usage of this module as follows:
#
#     find_package(Libbacktrace)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  LIBBACKTRACE_PREFIX         Set this variable to the root installation of
#                      libpapi if the module has problems finding the
#                      proper installation path.
#
# Variables defined by this module:
#
#  LIBBACKTRACE_FOUND              System has PAPI libraries and headers
#  LIBBACKTRACE_LIBRARIES          The PAPI library
#  LIBBACKTRACE_INCLUDE_DIRS       The location of PAPI headers

find_path(LIBBACKTRACE_PREFIX
    NAMES include/backtrace-supported.h
)

find_library(LIBBACKTRACE_LIBRARIES
    # Pick the static library first for easier run-time linking.
    NAMES libbacktrace.a libbacktrace.so
    HINTS ${LIBBACKTRACE_PREFIX}/lib ${HILTIDEPS}/lib
)

find_path(LIBBACKTRACE_INCLUDE_DIRS
    NAMES backtrace.h backtrace-supported.h
    HINTS ${LIBBACKTRACE_PREFIX}/include ${HILTIDEPS}/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LIBBACKTRACE DEFAULT_MSG
    LIBBACKTRACE_LIBRARIES
    LIBBACKTRACE_INCLUDE_DIRS
)

mark_as_advanced(
    LIBBACKTRACE_PREFIX
    LIBBACKTRACE_LIBRARIES
    LIBBACKTRACE_INCLUDE_DIRS
)

function(check_libbacktrace)
  file(WRITE ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testlibbacktrace.c
       "#include <backtrace-supported.h>
        #if BACKTRACE_SUPPORTED != 1
        #error not supported
        #endif
        #include <backtrace.h>
        #include <stdio.h>
        int main() {
            return 0;
        }")
endfunction(check_libbacktrace)

set(LIBBACKTRACE_FOUND OFF)

check_libbacktrace()

if(LIBBACKTRACE_INCLUDE_DIRS AND LIBBACKTRACE_LIBRARIES)
  try_run(
    LIBBACKTRACE_RETURNCODE
    LIBBACKTRACE_COMPILED
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/testlibbacktrace.c
    COMPILE_DEFINITIONS -I"${LIBBACKTRACE_INCLUDE_DIRS}"
    LINK_LIBRARIES -L${LIBBACKTRACE_LIBRARIES}
    RUN_OUTPUT_VARIABLE SRC_OUTPUT
  )
  set(LIBBACKTRACE_FOUND ${LIBBACKTRACE_COMPILED})
endif()


