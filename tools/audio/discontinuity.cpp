#include <cstdio>
#include <vector>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>
#include <re/adt/re_name.h>
#include <re/adt/re_re.h>
#include <kernel/core/kernel_builder.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/io/source_kernel.h>
#include <kernel/io/stdout_kernel.h>
#include <kernel/core/streamsetptr.h>
#include <kernel/scan/scanmatchgen.h>
#include <kernel/streamutils/stream_select.h>
#include <string>
#include <toolchain/toolchain.h>
#include <fcntl.h>
#include <iostream>
#include <kernel/pipeline/driver/cpudriver.h>
#include <audio/audio.h>
#include <audio/stream_manipulation.h>
#include <iostream>

using namespace kernel;
using namespace llvm;
using namespace codegen;
using namespace audio;

#define SHOW_STREAM(name)           \
    if (codegen::EnableIllustrator) \
    P->captureBitstream(#name, name)
#define SHOW_BIXNUM(name)           \
    if (codegen::EnableIllustrator) \
    P->captureBixNum(#name, name)
#define SHOW_BYTES(name)            \
    if (codegen::EnableIllustrator) \
    P->captureByteData(#name, name)

static cl::OptionCategory DemoOptions("Demo Options", "Demo control options.");
static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input file>"), cl::Required, cl::cat(DemoOptions));
static cl::opt<int> threshold("t", cl::desc("Difference threshold"), cl::Required, cl::cat(DemoOptions));

typedef void (*PipelineFunctionType)(StreamSetPtr & marker_1, StreamSetPtr & marker_2, int32_t fd);
PipelineFunctionType generatePipeline(CPUDriver &pxDriver, const unsigned int& threshold, const unsigned int &numChannels, const unsigned int &bitsPerSample)
{
    std::vector<StreamSet *> Makers = {pxDriver.CreateStreamSet(1,1), pxDriver.CreateStreamSet(1,1)};

    auto &b = pxDriver.getBuilder();
    auto P = pxDriver.makePipelineWithIO({}, {Bind("Marker1", Makers[0], ReturnedBuffer(1)), Bind("Marker2", Makers[1], ReturnedBuffer(1))}, 
                                             {Binding{b.getInt32Ty(), "inputFileDecriptor"}});
    Scalar * const fileDescriptor = P->getInputScalar("inputFileDecriptor");

    std::vector<StreamSet *> ChannelSampleStreams(numChannels);
    for (unsigned i=0;i<numChannels;++i)
    {
        ChannelSampleStreams[i] = P->CreateStreamSet(1,bitsPerSample);
    }
    ParseAudioBuffer(P, fileDescriptor, numChannels, bitsPerSample, ChannelSampleStreams);

    std::vector<StreamSet *> OutputStreams(numChannels);

    for (unsigned i = 0; i < numChannels; ++i)
    {
        StreamSet *BasisBits = P->CreateStreamSet(bitsPerSample);
        S2P(P, bitsPerSample, ChannelSampleStreams[i], BasisBits);
        //SHOW_BIXNUM(BasisBits);
        P->CreateKernelCall<DiscontinuityKernel>(BasisBits, threshold, Makers[i]);
        
        SHOW_STREAM(Makers[i]);
    }
    return reinterpret_cast<PipelineFunctionType>(P->compile());
}

int main(int argc, char *argv[])
{
    codegen::ParseCommandLineOptions(argc, argv, {&DemoOptions, codegen::codegen_flags()});

    CPUDriver driver("demo");
    const int fd = open(inputFile.c_str(), O_RDONLY);
    unsigned int sampleRate = 0, numChannels = 1, bitsPerSample = 16, numSamples = 0;
    try
    {
        readWAVHeader(fd, numChannels, sampleRate, bitsPerSample, numSamples);
        lseek(fd, 44, SEEK_SET);
        std::cout << "numChannels: " << numChannels << ", sampleRate: " << sampleRate << ", bitsPerSample: " << bitsPerSample << ", numSamples: " << numSamples << "\n";
    }
    catch (const std::exception &e)
    {
        llvm::errs() << "Warning: cannot parse " << inputFile << " WAV header for processing. Processing file as text.\n";
        lseek(fd, 0, SEEK_SET);
    }

    auto fn = generatePipeline(driver, threshold, numChannels, bitsPerSample);
    StreamSetPtr wavStream, wavStream1;
    fn(wavStream, wavStream1, fd);
    close(fd);
    return 0;
}