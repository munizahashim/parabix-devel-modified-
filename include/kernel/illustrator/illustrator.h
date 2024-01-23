#ifndef ILLUSTRATOR_H
#define ILLUSTRATOR_H

#include <kernel/core/kernel.h>
#include <kernel/pipeline/pipeline_builder.h>

namespace kernel {

class ParabixIllustrator {
    using ProgramBuilderRef = const std::unique_ptr<kernel::ProgramBuilder> &;
public:
    ParabixIllustrator(unsigned displayWidth) :  mDisplayWidth(displayWidth), mMaxStreamNameSize(0) {}

    //
    void registerIllustrator(const void * stateObject, const char * displayName);


    void captureByteData(ProgramBuilderRef P, std::string streamName, StreamSet * byteData, char nonASCIIsubstitute = '.');
    void captureBitstream(ProgramBuilderRef P, std::string streamName, StreamSet * bitstream, char zeroCh = '.', char oneCh = '1');
    void captureBixNum(ProgramBuilderRef P, std::string streamName, StreamSet * bixnum, char hexBase = 'A');

    void appendStreamText(unsigned streamNo, std::string streamText);
    void displayAllCapturedData();

protected:
    unsigned addStream(std::string streamName);

private:
    Scalar * mIllustrator;
    std::vector<std::string> mStreamNames;
    std::vector<std::string> mStreamData;
    unsigned mDisplayWidth;
    unsigned mMaxStreamNameSize;
};

// Each kernel can verify that the display Name of every illustrated value is locally unique but since multiple instances
// of a kernel can be instantiated, we also need the address of the state object to identify each value. Additionally, the
// presence of family kernels means we cannot guarantee that all kernels will be compiled at the same time so we cannot
// number the illustrated values at compile time.
void registerIllustrator(ParabixIllustrator * illustrator, const char * kernelName, const char * streamName, const void * stateObject);

}

#endif // ILLUSTRATOR_H
