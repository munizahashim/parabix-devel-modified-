/*
 * Copyright (c) 2019 International Characters.
 * This software is licensed to the public under the Open Software License 3.0.
 */

#include <testing/testing.h>
#include <kernel/streamutils/deletion.h>
#include <kernel/streamutils/pdep_kernel.h>
#include <kernel/streamutils/stream_shift.h>
#include <kernel/pipeline/program_builder.h>

using namespace kernel;
using namespace testing;

auto insert_mask_i = BinaryStream({"11100001"});
auto insert_before_e = BinaryStream({"010101111101"});

TEST_CASE(insert_before1, insert_mask_i, insert_before_e) {
    auto Result = UnitInsertionSpreadMask(T, Input<0>(T), InsertPosition::Before);
    AssertEQ(P, Result, Input<1>(T));
}

auto insert_after_e = BinaryStream({"101010111110"});

TEST_CASE(insert_after1, insert_mask_i, insert_after_e) {
    auto Result = UnitInsertionSpreadMask(T, Input<0>(T), InsertPosition::After);
    AssertEQ(P, Result, Input<1>(T));
}

auto insert_counts = BinaryStreamSet({"001000101", "001000000", "100000000"});
auto insert_mult_before_e = BinaryStream("000011000111101101");
auto insert_mult_after_e = BinaryStream("100001100011110110");

TEST_CASE(insert_mult_after, insert_counts, insert_mult_after_e) {
    auto Result = InsertionSpreadMask(T, Input<0>(T), InsertPosition::After);
    AssertEQ(P, Result, Input<1>(T));
}

TEST_CASE(insert_mult_before, insert_counts, insert_mult_before_e) {
    auto Result = InsertionSpreadMask(T, Input<0>(T), InsertPosition::Before);
    AssertEQ(P, Result, Input<1>(T));
}

auto to_filter = BinaryStreamSet({"01010101011", "11110000111", "11010011000"});
auto filter_mask = BinaryStream({"11110000111"});
auto filtered = BinaryStreamSet({"0101011", "1111111", "1101000"});

TEST_CASE(filter1, filter_mask, to_filter, filtered) {
    auto Result = P.CreateStreamSet(3);
    FilterByMask(T, Input<0>(T), Input<1>(T), Result);
    AssertEQ(P, Result, Input<2>(T));
}

auto spread = BinaryStreamSet({"01010000011", "11110000111", "11010000000"});

TEST_CASE(spread1, filter_mask, filtered, spread) {
    auto Result = P.CreateStreamSet(3);
    SpreadByMask(T, Input<0>(T), Input<1>(T), Result);
    AssertEQ(P, Result, Input<2>(T));
}

auto marker =    BinaryStreamSet({"1011101001010000"});
auto indexStrm = BinaryStreamSet({"1011101101011000"});
auto advmarker = BinaryStreamSet({"0011101100011000"});

TEST_CASE(indexedadvance1, indexStrm, marker, advmarker) {
    auto Result = P.CreateStreamSet(1);
    P.CreateKernelCall<IndexedAdvance>(Input<0>(T), Input<1>(T), Result, 1);
    AssertEQ(P, Result, Input<2>(T));
}

auto bakmarker = BinaryStreamSet({"1011100101000000"});

TEST_CASE(indexedshiftback1, indexStrm, marker, bakmarker) {
    auto Result = P.CreateStreamSet(1);
    P.CreateKernelCall<IndexedShiftBack>(Input<0>(T), Input<1>(T), Result);
    AssertEQ(P, Result, Input<2>(T));
}

auto longmarker =    BinaryStreamSet({"..1.{250}0.{270}1.1"});
auto longindexStrm = BinaryStreamSet({"..1.{250}1.{270}1.1"});
auto longbakmarker = BinaryStreamSet({"..0.{250}1.{270}1.0"});

TEST_CASE(longindexedshiftback, longindexStrm, longmarker, longbakmarker) {
    auto Result = P.CreateStreamSet(1);
    P.CreateKernelCall<IndexedShiftBack>(Input<0>(T), Input<1>(T), Result);
    AssertEQ(P, Result, Input<2>(T));
}


RUN_TESTS(
          CASE(insert_before1),
          CASE(insert_after1),
          CASE(insert_mult_after),
          CASE(insert_mult_before),
          CASE(filter1),
          CASE(spread1),
          CASE(indexedadvance1),
          CASE(indexedshiftback1),
          CASE(longindexedshiftback)
)
