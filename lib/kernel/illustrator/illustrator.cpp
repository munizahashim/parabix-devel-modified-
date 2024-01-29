/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include <kernel/illustrator/illustrator.h>
#include <kernel/illustrator/illustrator_binding.h>
#include <boost/container/flat_map.hpp>
#include <llvm/Support/raw_os_ostream.h>
#include <kernel/core/kernel_builder.h>
#include <util/slab_allocator.h>
#include <mutex>

using namespace boost::container;

namespace kernel {

using MemoryOrdering = KernelBuilder::MemoryOrdering;

class StreamDataIllustrator {
public:

inline void registerStreamDataCapture(const char * kernelName, const char * streamName, const void * stateObject,
                                      const size_t dim0, const size_t dim1, const size_t itemWidth, const uint8_t memoryOrdering,
                                      const uint8_t illustratorType, const char replacement0, const char replacement1) {

    llvm::errs() << " -- registering stream capture " << kernelName << "." << streamName << "  (";
    llvm::errs().write_hex((uintptr_t)stateObject) << ")\n";

    mRegisteredCaptures.emplace(std::make_tuple(kernelName, streamName, stateObject),
                                StreamDataGroup{dim0, dim1, itemWidth, memoryOrdering, illustratorType, replacement0, replacement1});
}

inline void doStreamDataCapture(const char * kernelName, const char * streamName, const void * stateObject,
                                const size_t strideNum, const void * streamData, const size_t from, const size_t to) {

};


using StreamDataKey = std::tuple<const char *, const char *, const void *>;

struct StreamDataGroup {
    size_t Dim0;
    size_t Dim1;
    size_t ItemWidth;
    MemoryOrdering Ordering;
    IllustratorTypeId IllustratorType;
    char Replacement0;
    char Replacement1;

    StreamDataGroup(size_t dim0, size_t dim1, size_t iw, uint8_t ordering, uint8_t illustratorType, char rep0, char rep1)
    : Dim0(dim0), Dim1(dim1), ItemWidth(iw), Ordering((MemoryOrdering)ordering)
    , IllustratorType((IllustratorTypeId)illustratorType), Replacement0(rep0), Replacement1(rep1) {
        assert (Ordering == MemoryOrdering::ColumnMajor || Ordering == MemoryOrdering::RowMajor);
        assert (IllustratorType == IllustratorTypeId::Bitstream || IllustratorType == IllustratorTypeId::BixNum || IllustratorType == IllustratorTypeId::ByteData);
    }

};

flat_map<StreamDataKey, StreamDataGroup> mRegisteredCaptures;
SlabAllocator<> mInternalAllocator;
std::mutex mAllocatorLock;

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
                                     const size_t dim0, const size_t dim1, const size_t itemWidth, const uint8_t memoryOrdering,
                                     const uint8_t illustratorTypeId, const char replacement0, const char replacement1) {
    illustrator->registerStreamDataCapture(kernelName, streamName, stateObject, dim0, dim1, itemWidth, memoryOrdering, illustratorTypeId, replacement0, replacement1);
}

extern "C"
void illustratorCaptureStreamData(StreamDataIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject,
                                  const size_t strideNum, const void * streamData, const size_t from, const size_t to) {

}

extern "C"
void illustratorDisplayCapturedData(const StreamDataIllustrator * illustrator) {

}

extern "C"
void destroyStreamDataIllustrator(StreamDataIllustrator * illustrator) {
    delete illustrator;
}


}
