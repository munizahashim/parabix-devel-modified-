/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include <kernel/illustrator/illustrator.h>
#include <kernel/illustrator/illustrator_binding.h>
#include <llvm/Support/raw_os_ostream.h>
#include <kernel/core/kernel_builder.h>
#include <util/slab_allocator.h>
#include <boost/intrusive/detail/math.hpp>
#include <boost/container/flat_map.hpp>
#include <mutex>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <Windows.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <sys/ioctl.h>
#include <stdio.h>
#endif

using namespace boost::container;
using boost::intrusive::detail::floor_log2;
using boost::intrusive::detail::is_pow2;
using namespace llvm;

#define ELEMENTS_PER_ALLOCATION 1024

inline static size_t udiv(const size_t x, const size_t y) {
    assert (is_pow2(y));
    const unsigned z = x >> floor_log2(y);
    assert (z == (x / y));
    return z;
}

inline static size_t ceil_udiv(const size_t x, const size_t y) {
    return udiv(((x - 1) | (y - 1)) + 1, y);
}

inline size_t get_terminal_width(const size_t fileDefault) {
#if defined(_WIN32)
    #warning UNTESTED CODE
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD type = GetFileType(h);
    if (type == FILE_TYPE_DISK) {
        return fileDefault;
    }
    GetConsoleScreenBufferInfo(h, &csbi);
    return = csbi.srWindow.Right-csbi.srWindow.Left+1;
#elif defined(__linux__) || defined(__APPLE__)
    if (isatty(STDERR_FILENO)) {
        struct winsize w;
        ioctl(STDERR_FILENO, TIOCGWINSZ, &w);
        return w.ws_col;
    } else {
        return fileDefault;
    }
#endif // Windows/Linux
}

namespace kernel {

using MemoryOrdering = KernelBuilder::MemoryOrdering;

using StreamDataKey = std::tuple<const char *, const char *, const void *>;

struct StreamDataElement {
    size_t StrideNum;
    size_t From;
    size_t To;
    uint8_t * Data;
};

struct StreamDataChunk {
    StreamDataElement Data[ELEMENTS_PER_ALLOCATION];
    StreamDataChunk * Next = nullptr;
};

struct StreamDataGroup {
    size_t Rows;
    size_t Cols;
    size_t ItemWidth;
    MemoryOrdering Ordering;
    IllustratorTypeId IllustratorType;
    char Replacement0;
    char Replacement1;

    size_t CurrentIndex;

    StreamDataChunk Root;
    StreamDataChunk * Current;

    SlabAllocator<uint8_t, 1024 * 1024> InternalAllocator;

    StreamDataGroup(size_t rows, size_t cols, size_t iw, uint8_t ordering, uint8_t illustratorType, char rep0, char rep1)
    : Rows(rows), Cols(cols), ItemWidth(iw), Ordering((MemoryOrdering)ordering)
    , IllustratorType((IllustratorTypeId)illustratorType), Replacement0(rep0), Replacement1(rep1)
    , CurrentIndex(0), Current(&Root) {
        assert (Ordering == MemoryOrdering::ColumnMajor || Ordering == MemoryOrdering::RowMajor);
        assert (IllustratorType == IllustratorTypeId::Bitstream || IllustratorType == IllustratorTypeId::BixNum || IllustratorType == IllustratorTypeId::ByteData);
        assert (is_pow2(ItemWidth));
    }
};

using StreamDataEntry = std::tuple<const char *, const char *, const void *, const StreamDataGroup *>;

class StreamDataIllustrator {
public:

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief registerStreamDataCapture
 ** ------------------------------------------------------------------------------------------------------------- */
inline void registerStreamDataCapture(const char * kernelName, const char * streamName, const void * stateObject,
                                      const size_t rows, const size_t cols, const size_t itemWidth, const uint8_t memoryOrdering,
                                      const uint8_t illustratorType, const char replacement0, const char replacement1) {

 //   std::lock_guard<std::mutex> L(AllocatorLock);

    StreamDataGroup * newGroup = GroupAllocator.allocate(1);
    newGroup = new (newGroup) StreamDataGroup{rows, cols, itemWidth, memoryOrdering, illustratorType, replacement0, replacement1};
    RegisteredCaptures.emplace(std::make_tuple(kernelName, streamName, stateObject), newGroup);
    InstallOrderCaptures.emplace_back(kernelName, streamName, stateObject, newGroup);
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief doStreamDataCapture
 ** ------------------------------------------------------------------------------------------------------------- */
inline void doStreamDataCapture(const char * kernelName, const char * streamName, const void * stateObject,
                                const size_t strideNum, const uint8_t * streamData, const size_t from, const size_t to,
                                const size_t blockWidth) {

    auto capture = RegisteredCaptures.find(std::make_tuple(kernelName, streamName, stateObject));
    assert (capture != RegisteredCaptures.end());
    StreamDataGroup & group = *capture->second;
    assert (to >= from);

    StreamDataChunk * C = group.Current;
    auto & A = group.InternalAllocator;
    if (LLVM_UNLIKELY(group.CurrentIndex == ELEMENTS_PER_ALLOCATION)) {
        StreamDataChunk * N = new (A.aligned_allocate(sizeof(StreamDataChunk), sizeof(size_t))) StreamDataChunk{};
        assert (N);
        C->Next = N;
        group.Current = N;
        group.CurrentIndex = 0;
        C = N;
    }
    assert (group.CurrentIndex < ELEMENTS_PER_ALLOCATION);
    StreamDataElement & E = C->Data[group.CurrentIndex++];
    E.StrideNum = strideNum;
    E.From = from;
    E.To = to;

    assert (from <= to);

    if (LLVM_UNLIKELY(from == to)) {
        E.Data = nullptr;
    } else if (LLVM_UNLIKELY(group.ItemWidth >= CHAR_BIT && group.Rows == 1 && group.Cols == 1)) {
        // may be unsafe; do not memcpy any data that isn't explicitly given
        const auto offset = udiv(from * group.ItemWidth, CHAR_BIT);
        const uint8_t * start = streamData + offset;
        const size_t length  = (to - from) * group.ItemWidth / CHAR_BIT;
        assert (length > 0);
        assert (group.ItemWidth % CHAR_BIT == 0);
        E.Data = A.aligned_allocate(length, group.ItemWidth / CHAR_BIT);
        std::memcpy(E.Data, start, length);
    } else {
        // each "block" of streamData will contain blockWidth items, regardless of the item width.
        const size_t blockSize = (group.Rows * group.Cols * group.ItemWidth * blockWidth) / CHAR_BIT;
        const uint8_t * start = streamData + (udiv(from, blockWidth) * blockSize);
        const auto end = (from & (blockWidth - 1)) + (to - from);
        const auto length = ceil_udiv(end, blockWidth) * blockSize;
        assert (length > 0);
        E.Data = A.aligned_allocate(length, blockWidth / CHAR_BIT);
        std::memcpy(E.Data, start, length);
    }

};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief displayCapturedData
 ** ------------------------------------------------------------------------------------------------------------- */
inline void displayCapturedData(const size_t blockWidth) const {

    struct KernelNameNode {
        StringRef Label;
        size_t NumOfCopies;
        size_t CurrentCopyNum;

        KernelNameNode(StringRef kernelName)
        : Label(kernelName)
        , NumOfCopies(0)
        , CurrentCopyNum(0) {

        }
    };

    struct StreamNameNode {
        StringRef Label;
        std::vector<KernelNameNode> Children;

        StreamNameNode(StringRef streamName, StringRef kernelName)
        : Label(streamName) {
            Children.emplace_back(kernelName);
        }
    };

    const auto n = InstallOrderCaptures.size();

    assert (n == RegisteredCaptures.size());

    std::vector<StreamNameNode> roots;

    assert (is_pow2(blockWidth));

    // initialize some position information and determine whether the streamNames are unique
    // so that we can make it easier for the user to read the output.

    for (size_t i = 0; i < n; ++i) {
        const char * kernelName; const char * streamName; const void * stateObject;
        const StreamDataGroup * group;
        std::tie(kernelName, streamName, stateObject, group) = InstallOrderCaptures[i];

        // determine if the streamName is unique or if we need
        const auto m = roots.size();

        StringRef curStreamName{streamName};
        StringRef curKernelName{kernelName};

        for (size_t j = 0; j < m; ++j) {
            auto & A = roots[j];
            if (LLVM_UNLIKELY(curStreamName.equals(A.Label))) {
                const auto l = A.Children.size();
                for (size_t k = 0; k < l; ++k) {
                    auto & B = A.Children[k];
                    if (LLVM_UNLIKELY(curKernelName.equals(B.Label))) {
                        B.NumOfCopies++;
                        goto updated_trie;
                    }
                }
                // add new kernel name entry
                A.Children.emplace_back(curKernelName);
            }
        }
        roots.emplace_back(curStreamName, curKernelName);
updated_trie:
        continue;
    }

    // Initialize some information and construct the displayed names

    struct SubField {
        std::string Label;
    };

    // codegen::IllustratorDisplay

    struct CurrentRecord {
        std::string Name;
        const StreamDataGroup * Group;
        const StreamDataChunk * Current;
        size_t Index;
        std::vector<std::string> SubField;

        CurrentRecord()
        : Current(nullptr)
        , Index(0) {

        }
    };


    std::vector<CurrentRecord> record(n);

    const auto m = roots.size();

    size_t longestNameLength = 0;
    size_t maxNumOfRows = 0;

    for (size_t i = 0; i < n; ++i) {
        const char * kernelName; const char * streamName; const void * stateObject;
        const StreamDataGroup * group;
        std::tie(kernelName, streamName, stateObject, group) = InstallOrderCaptures[i];

        auto & R = record[i];
        R.Group = group;
        R.Current = &group->Root;

        if (group->IllustratorType == IllustratorTypeId::BixNum) {
            const auto r = group->Rows;
            const auto bixNumRows = ceil_udiv(r, 4);
            R.SubField.resize(bixNumRows);
            maxNumOfRows = std::max(maxNumOfRows, bixNumRows);

            if (bixNumRows == 1) {
                R.SubField[0] = "";
            } else {
                for (unsigned j = 0; j < bixNumRows; ++j) {
                    const auto a = (j * 4);
                    const auto b = std::min<size_t>(a + 3, r);
                    R.SubField[j] = "[" + std::to_string(a) + "-" + std::to_string(b) + "]";
                }
            }
        } else {
            const auto r = group->Rows;
            R.SubField.resize(r);
            maxNumOfRows = std::max(maxNumOfRows, r);
            if (r == 1) {
                R.SubField[0] = "";
            } else {
                for (unsigned j = 0; j < r; ++j) {
                    R.SubField[j] = "[" + std::to_string(j) + "]";
                }
            }
        }

        StringRef curStreamName{streamName};
        StringRef curKernelName{kernelName};

        for (size_t j = 0; j < m; ++j) {
            auto & A = roots[j];
            if (LLVM_UNLIKELY(curStreamName.equals(A.Label))) {
                const auto l = A.Children.size();
                for (size_t k = 0; k < l; ++k) {
                    auto & B = A.Children[k];
                    if (LLVM_UNLIKELY(curKernelName.equals(B.Label))) {
                        const auto c = B.NumOfCopies;
                        if (l == 1 && c == 0) {
                            R.Name = curStreamName.str();
                        } else {
                            std::string name;
                            if (LLVM_LIKELY(l != 1)) {
                                name = curKernelName.str() + ".";
                            }
                            name += curStreamName.str();
                            if (c > 0) {
                                name += std::to_string(++B.CurrentCopyNum);
                            }
                            R.Name = name;
                        }
                        size_t maxFieldLength = 0;
                        for (unsigned j = 0; j < R.SubField.size(); ++j) {
                            auto & Sj = R.SubField[j];
                            maxFieldLength = std::max(maxFieldLength, Sj.size());
                        }
                        auto length = R.Name.length() + maxFieldLength;
                        longestNameLength = std::max(longestNameLength, length);
                        goto next_entry;
                    }
                }
            }
        }
        llvm_unreachable("Failed to locate stream group in trie?");
next_entry:
        continue;
    }


    // display item aligned data

    auto & out = errs();

    std::vector<std::vector<char>> FormattedOutput(maxNumOfRows);

    size_t charsPerRow = codegen::IllustratorDisplay;
    if (charsPerRow == 0) {
        const auto max = get_terminal_width(blockWidth);
        if (LLVM_LIKELY(longestNameLength + 3 < max)) {
            charsPerRow = max - (longestNameLength + 3);
        } else {
            // TODO: what is a good default here for when we cannot even fit the names in the console?
            charsPerRow = 100;
        }
    }

    for (size_t i = 0; i < maxNumOfRows; ++i) {
        FormattedOutput[i].resize(charsPerRow);
    }

    size_t startPosition = 0;

    for (;;) {

        bool noRowCompletelyFilled = true;
        for (size_t groupNum = 0; groupNum < n; ++groupNum) {

            auto & R = record[groupNum];
            const auto & G = *R.Group;
            const auto & F = R.SubField;

            size_t position = startPosition;

            assert (G.Ordering == MemoryOrdering::RowMajor);

            size_t scale = G.ItemWidth;
            if (G.IllustratorType == IllustratorTypeId::ByteData) {
                scale /= CHAR_BIT;
            }

            const auto rowSize = G.Cols * (blockWidth * G.ItemWidth) / CHAR_BIT;

            const auto chunkSize = G.Rows * rowSize;

            size_t end = 0;

get_more_data:

            // fill in the data for the i-th illustrated streamset
            const StreamDataElement * E = nullptr;
            if (LLVM_LIKELY(R.Current)) {
                size_t limit = (R.Current->Next == nullptr) ? G.CurrentIndex : ELEMENTS_PER_ALLOCATION;
                for (;;) {
                    assert (R.Current);
                    assert (R.Index < limit);
                    const auto & t = R.Current->Data[R.Index];
                    if ((t.To * scale) > position) {
                        E = &t;
                        break;
                    }
                    ++R.Index;
                    if (LLVM_UNLIKELY(R.Index == limit)) {
                        R.Current = R.Current->Next;
                        R.Index = 0;
                        if (LLVM_UNLIKELY(R.Current == nullptr)) {
                            break;
                        }
                        limit = (R.Current->Next == nullptr) ? G.CurrentIndex : ELEMENTS_PER_ALLOCATION;
                    }
                }
            }

            const auto m = F.size();

            if (LLVM_LIKELY(E)) {

                // each chunk is aligned in blockWidth x itemWidth bits

                assert (E->From <= position);

                const auto pos = position - E->From;

                const uint8_t * blockData = E->Data + ((pos / blockWidth) * chunkSize);
                const size_t readStart = (pos & (blockWidth - 1));
                const size_t writeStart = (position % charsPerRow);
                const size_t blockDataLimit = std::min(blockWidth - readStart, charsPerRow - writeStart);
                const auto to = (E->To * scale);
                assert (position < to);
                const size_t length = std::min(blockDataLimit, to - position);
                assert (length > 0);

                assert ((readStart + length) <= blockWidth);
                assert ((writeStart + length) <= charsPerRow);

                if (G.IllustratorType == IllustratorTypeId::Bitstream) {

                    const char zeroCh = G.Replacement0;
                    const char oneCh = G.Replacement1;
                    for (size_t j = 0; j < m; ++j) {

                        assert (j < FormattedOutput.size());
                        auto & toFill = FormattedOutput[j];
                        assert (toFill.size() == charsPerRow);

                        const uint8_t * rowData = blockData + (j * rowSize);
                        for (size_t k = 0; k < length; ++k) {
                            const auto in = (readStart + k);
                            assert (in < blockWidth);
                            const uint8_t v = rowData[in / CHAR_BIT] & (1UL << (in & (CHAR_BIT - 1)));
                            const auto ch = (v == 0) ? zeroCh : oneCh;
                            const auto out = (writeStart + k);
                            assert (out < charsPerRow);
                            toFill[out] = ch;
                        }
                    }

                } else if (G.IllustratorType == IllustratorTypeId::BixNum) {

                    const char hexBase = G.Replacement0 - 10;

                    for (size_t j = 0; j < m; ++j) {

                        const auto s = (j * 4);
                        assert (s < G.Rows);
                        const auto t = std::min(G.Rows - s, 3UL);

                        assert (j < FormattedOutput.size());
                        auto & toFill = FormattedOutput[j];
                        assert (toFill.size() == charsPerRow);

                        for (size_t k = 0; k < length; ++k) {
                            const auto out = (writeStart + k);
                            assert (out < charsPerRow);
                            toFill[out] = 0;
                        }

                        for (size_t r = 0; r < t; ++r) {
                            const uint8_t * rowData = blockData + ((s + r) * rowSize);
                            for (size_t k = 0; k < length; ++k) {
                                const auto in = (readStart + k);
                                assert (in < blockWidth);
                                const uint8_t v = (rowData[in / CHAR_BIT] & (1UL << (in & (CHAR_BIT - 1)))) != 0;
                                assert (v == 0 || v == 1);
                                const auto out = (writeStart + k);
                                assert (out < charsPerRow);
                                toFill[out] |= (uint8_t)(v << r);
                            }
                        }

                        for (size_t k = 0; k < length; ++k) {
                            const auto out = (writeStart + k);
                            assert (out < charsPerRow);
                            auto & c = toFill[out];
                            if (c < 10) {
                                c += '0';
                            } else {
                                c += hexBase;
                            }
                        }

                    }

                } else if (G.IllustratorType == IllustratorTypeId::ByteData) {

                    const char nonAsciiRep = G.Replacement0;
                    for (size_t j = 0; j < m; ++j) {

                        assert (j < FormattedOutput.size());
                        auto & toFill = FormattedOutput[j];
                        assert (toFill.size() == charsPerRow);

                        const uint8_t * rowData = blockData + (j * rowSize);
                        for (size_t k = 0; k < length; ++k) {
                            const auto in = (readStart + k);
                            assert (in < blockWidth);
                            auto ch = rowData[in];
                            const auto out = (writeStart + k);
                            if (LLVM_UNLIKELY((ch < 32) || (ch > 126))) {
                                switch (ch) {
                                    case '\t': case '\n': case '\r':
                                        ch = ' ';
                                        break;
                                    default:
                                        ch = nonAsciiRep;
                                }
                            }
                            assert (out < charsPerRow);
                            toFill[out] = ch;
                        }
                    }
                }

                position += length;
                end = (writeStart + length);
                assert (end == (position - startPosition));
                if (end < charsPerRow) {
                    goto get_more_data;
                }
                noRowCompletelyFilled = false;
            }

            assert (end <= charsPerRow);
            for (size_t j = 0; j < m; ++j) {
                const auto & Fj = F[j];
                out.indent(longestNameLength - R.Name.size() - Fj.size());
                out << R.Name << Fj << " | ";
                auto & output = FormattedOutput[j];
                for (size_t i = 0; i < end; ++i) {
                    out << output[i];
                }
                out << '\n';
            }
        }

        if (noRowCompletelyFilled) {
            break;
        }

        out << '\n';

        startPosition += charsPerRow;
    }
}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief deconstructor
 ** ------------------------------------------------------------------------------------------------------------- */
~StreamDataIllustrator() {
    // slab allocated but group deconstructor needs to be called too
    for (auto & rc : RegisteredCaptures) {
        rc.second->~StreamDataGroup();
    }
}

private:

SlabAllocator<StreamDataGroup, sizeof(StreamDataGroup) * 64> GroupAllocator;
flat_map<StreamDataKey, StreamDataGroup *> RegisteredCaptures;
std::vector<StreamDataEntry> InstallOrderCaptures;

};

extern "C"
StreamDataIllustrator * createStreamDataIllustrator() {
    return new StreamDataIllustrator();
}

// Each kernel can verify that the display Name of every illustrated value is locally unique but since multiple instances
// of a kernel can be instantiated, we also need the address of the state object to identify each value. Additionally, the
// presence of family kernels means we cannot guarantee that all kernels will be compiled at the same time so we cannot
// number the illustrated values at compile time.
extern "C"
void illustratorRegisterCapturedData(StreamDataIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject,
                                     const size_t rows, const size_t cols, const size_t itemWidth, const uint8_t memoryOrdering,
                                     const uint8_t illustratorTypeId, const char replacement0, const char replacement1) {
    illustrator->registerStreamDataCapture(kernelName, streamName, stateObject, rows, cols, itemWidth, memoryOrdering, illustratorTypeId, replacement0, replacement1);
}

extern "C"
void illustratorCaptureStreamData(StreamDataIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject,
                                  const size_t strideNum, const uint8_t * streamData, const size_t from, const size_t to, const size_t blockWidth) {
    illustrator->doStreamDataCapture(kernelName, streamName, stateObject, strideNum, streamData, from, to, blockWidth);
}

extern "C"
void illustratorDisplayCapturedData(const StreamDataIllustrator * illustrator, const size_t blockWidth) {
    illustrator->displayCapturedData(blockWidth);
}

extern "C"
void destroyStreamDataIllustrator(StreamDataIllustrator * illustrator) {
    delete illustrator;
}


}
