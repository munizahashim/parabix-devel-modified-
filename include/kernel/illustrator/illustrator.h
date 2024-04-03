#ifndef ILLUSTRATOR_H
#define ILLUSTRATOR_H

#include <stdlib.h>
#include <stdint.h>

namespace kernel {

class StreamDataIllustrator;

extern "C"
StreamDataIllustrator * createStreamDataIllustrator();

// Each kernel can verify that the display Name of every illustrated value is locally unique but since multiple instances
// of a kernel can be instantiated, we also need the address of the state object to identify each value. Additionally, the
// presence of family kernels means we cannot guarantee that all kernels will be compiled at the same time so we cannot
// number the illustrated values at compile time.
extern "C"
void illustratorRegisterCapturedData(StreamDataIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject,
                                     const size_t rows, const size_t cols, const size_t itemWidth, const uint8_t memoryOrdering,
                                     const uint8_t illustratorTypeId, const char replacement0, const char replacement1,
                                     const size_t * loopIdArray);


extern "C"
void illustratorCaptureStreamData(StreamDataIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject,
                                 const size_t strideNum, const uint8_t * streamData, const size_t from, const size_t to, const size_t blockWidth);

extern "C"
void illustratorEnterKernel(StreamDataIllustrator * illustrator, const void * stateObject);

extern "C"
void illustratorEnterLoop(StreamDataIllustrator * illustrator, const void * stateObject, const size_t loopId);

extern "C"
void illustratorIterateLoop(StreamDataIllustrator * illustrator, const void * stateObject);

extern "C"
void illustratorExitLoop(StreamDataIllustrator * illustrator, const void * stateObject);

extern "C"
void illustratorExitKernel(StreamDataIllustrator * illustrator, const void * stateObject);

extern "C"
void illustratorDisplayCapturedData(const StreamDataIllustrator * illustrator, const size_t blockWidth);

extern "C"
void destroyStreamDataIllustrator(StreamDataIllustrator * illustrator);

}

#endif // ILLUSTRATOR_H
