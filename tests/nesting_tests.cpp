/*
 * Copyright (c) 2022 International Characters.
 * This software is licensed to the public under the Open Software License 3.0.
 */

#include <testing/testing.h>
#include <kernel/util/nesting.h>

using namespace kernel;
using namespace testing;


auto brackets1 = BinaryStreamSet({"...1..1....1...1.1..............",
                                  "........1.........1..1..1...1..."});
auto depth_e1  = BinaryStreamSet({"...111...11....11..111...1111...",
                                  "......111..111111..111111.......",
                                  ".................11............."});
auto errs_e1      = BinaryStream({"................................."});

TEST_CASE(nesting1, brackets1, depth_e1, errs_e1) {
    auto DepthResult = T->CreateStreamSet(3);
    auto ErrResult = T->CreateStreamSet(1);
    P->CreateKernelCall<NestingDepth>(Input<0>(T), DepthResult, ErrResult, 5);
    AssertEQ(T, DepthResult, Input<1>(T));
    AssertEQ(T, ErrResult, Input<2>(T));
}

auto brackets2 = BinaryStreamSet({"...1111.........................",
                                  "..................1..1..1......."});
auto depth_e2  = BinaryStreamSet({"...1.1.............111...1111111",
                                  "....11.............111111.......",
                                  "......1111111111111............."});
auto errs_e2      = BinaryStream({"................................1"});

TEST_CASE(unclosed, brackets2, depth_e2, errs_e2) {
    auto DepthResult = T->CreateStreamSet(3);
    auto ErrResult = T->CreateStreamSet(1);
    P->CreateKernelCall<NestingDepth>(Input<0>(T), DepthResult, ErrResult, 5);
    AssertEQ(T, DepthResult, Input<1>(T));
    AssertEQ(T, ErrResult, Input<2>(T));
}

auto brackets3 = BinaryStreamSet({"...1111.........................",
                                  "..................1..1..1..1..1."});
auto depth_e3  = BinaryStreamSet({"...1.1.............111...111....",
                                  "....11.............111111.......",
                                  "......1111111111111............."});
auto errs_e3      = BinaryStream({"..............................1.."});

TEST_CASE(unmatchedR, brackets3, depth_e3, errs_e3) {
    auto DepthResult = T->CreateStreamSet(3);
    auto ErrResult = T->CreateStreamSet(1);
    P->CreateKernelCall<NestingDepth>(Input<0>(T), DepthResult, ErrResult, 5);
    AssertEQ(T, DepthResult, Input<1>(T));
    AssertEQ(T, ErrResult, Input<2>(T));
}

auto long_brak =  BinaryStreamSet({"..1.111...... .{333} ...................",
                                   "............. .{333} ........1..1..1..1."});
auto long_depth = BinaryStreamSet({"..11.1....... .{333} .........111...111.",
                                   "....11....... .{333} .........111111....",
                                   "......1111111 1{333} 111111111.........."});
auto long_errs     = BinaryStream({"............. .{333} ...................."});

TEST_CASE(multiblock, long_brak, long_depth, long_errs) {
    auto DepthResult = T->CreateStreamSet(3);
    auto ErrResult = T->CreateStreamSet(1);
    P->CreateKernelCall<NestingDepth>(Input<0>(T), DepthResult, ErrResult, 5);
    AssertEQ(T, DepthResult, Input<1>(T));
    AssertEQ(T, ErrResult, Input<2>(T));
}



RUN_TESTS(
          CASE(nesting1),
          CASE(unclosed),
          CASE(unmatchedR),
          CASE(multiblock)
)
