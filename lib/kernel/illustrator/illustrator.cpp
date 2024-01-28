/*
 *  Copyright (c) 2022 International Characters.
 *  This software is licensed to the public under the Open Software License 3.0.
 */

#include <kernel/illustrator/illustrator.h>

namespace kernel {

class ParabixIllustrator;

extern "C"
ParabixIllustrator * createParabixIllustrator() {
    return nullptr;
}

// Each kernel can verify that the display Name of every illustrated value is locally unique but since multiple instances
// of a kernel can be instantiated, we also need the address of the state object to identify each value. Additionally, the
// presence of family kernels means we cannot guarantee that all kernels will be compiled at the same time so we cannot
// number the illustrated values at compile time.
extern "C"
void illustratorRegisterCapturedData(ParabixIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject) {

}

extern "C"
void illustratorCaptureBitstream(ParabixIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject,
                                 const size_t strideNum, const void * bitstream, const size_t from, const size_t to, const char zeroCh, const char oneCh) {

}

extern "C"
void illustratorCaptureBixNum(ParabixIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject,
                              const size_t strideNum, const void * bitstream, const size_t from, const size_t to, const char hexBase) {

}
extern "C"
void illustratorCaptureByteData(ParabixIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject,
                                const size_t strideNum, const void * bitstream, const size_t from, const size_t to, const char nonASCIIsubstitute) {

}

extern "C"
void illustratorDisplayCapturedData(const ParabixIllustrator * illustrator) {

}

extern "C"
void destroyParabixIllustrator(ParabixIllustrator * illustrator) {
    // delete illustrator;
}


}
