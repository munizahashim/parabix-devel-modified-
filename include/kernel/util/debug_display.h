/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <kernel/core/kernel.h>
#include <kernel/pipeline/pipeline_builder.h>
#include <pablo/pablo_kernel.h>
#include <fstream>

namespace kernel {

/**
 * A debug kernel which displays the contents of a streamset to stderr using
 * either `PrintRegister` or `PrintInt` depending of the field width of `s`.
 */
class DebugDisplayKernel : public MultiBlockKernel {
    using KernelBuilder & = KernelBuilder &;
public:
    DebugDisplayKernel(KernelBuilder & b, llvm::StringRef name, StreamSet * s);
    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;
private:
    llvm::StringRef mName;
    uint32_t        mFW;
    uint32_t        mSCount;
};

namespace util {

/**
 * Displays the contents of an arbitrary streamset to stderr.
 * 
 * If the field width of the stream is 1, then stream stream will be printed as
 * bitblocks using `PrintRegister`. Otherwise, the stream will be printed as a
 * sequences of integers using `PrintInt`. For streamsets with more than one
 * stream, the index of the stream being shown will be displayed along with the
 * provided name.
 * 
 * Usage:
 *  The primary use for this kernel is to display the stream outputs of kernels
 *  when building pipelines. See also stream_select.h for extracting specific
 *  streams from stream sets.
 * 
 *      using namespace kernel;
 *      // -- snip --
 *      std::unique_ptr<ProgramBuilder> P = ...;
 *      StreamSet * SomeStream = ...;
 *      util::DebugDisplay(P, "some_stream", SomeStream);
 */
inline void DebugDisplay(const std::unique_ptr<ProgramBuilder> & P, llvm::StringRef name, StreamSet * s) {
    P->CreateKernelCall<DebugDisplayKernel>(name, s);
}

} // namespace kernel::util

/*
  Displays the contents of multiple aligned streams to stderr.

  A ParabixIllustrator object is initially created with a specified
  display width.   All stream data will be displayed using this width.  Ex:

  ParabixIllustrator illustrator(50);  // an illustrator with width 50.
 
  To include an illustrator object in a Parabix pipeline, it should be passed
  in as a pipeline parameter:   Binding{b.getIntAddrTy(), "illustratorAddr"}
  Then this parameter should be registered in the illustrator object for
  code generation.
  Scalar * callback_obj = P->getInputScalar("illustrator");
  illustrator.registerIllustrator(callback_obj);

  Byte streams or bit streams may be added for display.   Byte streams
  are displayed using printable ASCII characters only, with a substitution
  character used for nonprintable characters.

  For example, with a program builder P and StreamSet ByteData:
  illustrator.captureByteData(P, "bytedata", ByteData);

  Bit streams are printed in left to right order, with one bits shown
  using the '1' character and zero bits shown using the '.' character
  by default.

  To capture aligned bit stream data:
  illustrator.captureBitstream(P, "bitstream", BitStream);

  Streamsets representing bixnums of up to 4 digits may be captured
  and displayed in hexadecimal notation.

  illustrator.captureBixNum(P, "bixnum", BixNum);

  As many bytestreams and bitstreams as desired may be captured.

  Upon completion of pipeline processing, all captured data is emitted
  to stderr with the call:
  illustrator.displayAllCapturedData();
 */

class ParabixIllustrator {
    using ProgramBuilderRef = const std::unique_ptr<kernel::ProgramBuilder> &;
public:
    ParabixIllustrator(unsigned displayWidth) :  mDisplayWidth(displayWidth), mMaxStreamNameSize(0) {}

    void registerIllustrator(Scalar * illustrator);

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

} // namespace kernel
