#include <grep/grep_toolchain.h>
#include <llvm/Support/CommandLine.h>

using namespace llvm;

constexpr auto DefaultByteCClimit = 6;

namespace grep {

int Threads;
static cl::opt<int, true> OptThreads("t", cl::location(Threads),
                                     cl::desc("Total number of threads."), cl::init(2));

bool UnicodeIndexing;
static cl::opt<bool, true> OptUnicodeIndexing("UnicodeIndexing", cl::location(UnicodeIndexing),
                                              cl::desc("Enable CC multiplexing and Unicode indexing."), cl::init(false));

bool PropertyKernels;
static cl::opt<bool, true> OptPropertyKernels("enable-property-kernels", cl::location(PropertyKernels),
                                              cl::desc("Enable Unicode property kernels."), cl::init(true));

bool MultithreadedSimpleRE;
static cl::opt<bool, true> OptMultithreadedSimpleRE("enable-simple-RE-kernels", cl::location(MultithreadedSimpleRE),
                                                    cl::desc("Enable individual CC kernels for simple REs."), cl::init(false));

int ScanMatchBlocks;
static cl::opt<int, true> OptScanMatchBlocks("scanmatch-blocks", cl::location(ScanMatchBlocks),
                                             cl::desc("Scanmatch blocks per stride"), cl::init(4));

int MatchCoordinateBlocks;
static cl::opt<int, true> OptMatchCoordinateBlocks("match-coordinates", cl::location(MatchCoordinateBlocks),
                                                   cl::desc("Enable experimental MatchCoordinates kernels with a given number of blocks per stride"), cl::init(0));

int FileBatchSegments;
static cl::opt<int, true> OptFileBatchSegments("file-batch-segments", cl::location(FileBatchSegments),
                                                   cl::desc("Max total size (as a multiple of segment size) for processing small files as a batch"), cl::init(4));

unsigned ByteCClimit;
static cl::opt<unsigned, true> OptByteCClimit("byte-CC-limit", cl::location(ByteCClimit),
                                              cl::desc("Max number of CCs for byte CC pipeline."), cl::init(DefaultByteCClimit));
bool TraceFiles;
static cl::opt<bool, true> OptTraceFiles("TraceFiles", cl::location(TraceFiles),
                                         cl::desc("Report files as they are opened."), cl::init(false));
bool ShowExternals;
static cl::opt<bool, true> OptShowExternals("ShowExternals", cl::location(ShowExternals),
                                         cl::desc("Show externals as they are declared."), cl::init(false));

bool UseByteFilterByMask;
static cl::opt<bool, true> OptUseByteFilterByMask("UseByteFilterByMask", cl::location(UseByteFilterByMask),
                                         cl::desc("Use ByteFilterByMask."), cl::init(true));

bool UseNestedColourizationPipeline;
static cl::opt<bool, true> OptUsePipelinedColourization("UseNestedColourizationPipeline", cl::location(UseNestedColourizationPipeline),
                                         cl::desc("Use a nested pipeline for colourization."), cl::init(true));
}
