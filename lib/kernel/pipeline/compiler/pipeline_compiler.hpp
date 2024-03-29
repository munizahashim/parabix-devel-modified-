#ifndef PIPELINE_COMPILER_HPP
#define PIPELINE_COMPILER_HPP

#include "config.h"

#include <kernel/pipeline/pipeline_kernel.h>
#include <kernel/core/kernel_compiler.h>

#include "common/common.hpp"
#include "common/graphs.h"

#include <boost/multi_array.hpp>

using namespace boost;
using namespace boost::adaptors;
using boost::container::flat_set;
using boost::container::flat_map;
using namespace llvm;
using IDISA::FixedVectorType;

#include "analysis/pipeline_analysis.hpp"

// TODO: merge Cycle counter and PAPI?

namespace kernel {

enum CycleCounter {
  KERNEL_SYNCHRONIZATION            = 0
  , PARTITION_JUMP_SYNCHRONIZATION  = 1
  , BUFFER_EXPANSION                = 2
  , BUFFER_COPY                     = 3
  , KERNEL_EXECUTION                = 4
  , TOTAL_TIME                      = 5
  // ----------------------
  , SQ_SUM_TOTAL_TIME               = 6
  , NUM_OF_INVOCATIONS              = 7
  // ----------------------
  , FULL_PIPELINE_TIME              = 8
  // ----------------------
  , NUM_OF_KERNEL_CYCLE_COUNTERS    = 8
  , TOTAL_NUM_OF_CYCLE_COUNTERS    = 9
};

#ifdef ENABLE_PAPI
enum PAPIKernelCounter {
  PAPI_KERNEL_SYNCHRONIZATION = 0
  , PAPI_PARTITION_JUMP_SYNCHRONIZATION = 1
  , PAPI_BUFFER_EXPANSION = 2
  , PAPI_BUFFER_COPY = 3
  , PAPI_KERNEL_EXECUTION = 4
  , PAPI_KERNEL_TOTAL = 5
  // ------------------
  , NUM_OF_PAPI_KERNEL_COUNTERS = 6
  // ------------------
  , PAPI_FULL_PIPELINE_TIME = 6
  // ------------------
  , NUM_OF_PAPI_COUNTERS = 7
};
#endif

const static std::string BASE_THREAD_LOCAL_STREAMSET_MEMORY = "LSM";

const static std::string BASE_THREAD_LOCAL_STREAMSET_MEMORY_BYTES = "LSMb";

const static std::string EXPECTED_NUM_OF_STRIDES_MULTIPLIER = "EnSM";

const static std::string ZERO_EXTENDED_BUFFER = "ZeB";
const static std::string ZERO_EXTENDED_SPACE = "ZeS";

const static std::string KERNEL_THREAD_LOCAL_SUFFIX = ".KTL";
const static std::string NEXT_LOGICAL_SEGMENT_NUMBER = "@NLSN";
#ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
const static std::string NESTED_LOGICAL_SEGMENT_NUMBER_PREFIX = "!NLSN";
#endif

const static std::string MINIMUM_NUM_OF_THREADS = "MIN.T";
const static std::string MAXIMUM_NUM_OF_THREADS = "MAX.T";
const static std::string BUFFER_SEGMENT_LENGTH = "BSL";
const static std::string DYNAMIC_MULTITHREADING_SEGMENT_PERIOD = "SEG.C";
const static std::string DYNAMIC_MULTITHREADING_ADDITIONAL_THREAD_SYNCHRONIZATION_THRESHOLD = "TSTA";
const static std::string DYNAMIC_MULTITHREADING_REMOVE_THREAD_SYNCHRONIZATION_THRESHOLD = "TSTR";

#define SYNC_LOCK_FULL 0U
#define SYNC_LOCK_PRE_INVOCATION 1U
#define SYNC_LOCK_POST_INVOCATION 2U

const static std::array<std::string, 3> LOGICAL_SEGMENT_SUFFIX = { ".LSN", ".LSNs", ".LSNt" };

const static std::string INTERNALLY_SYNCHRONIZED_SUB_SEGMENT_SUFFIX = ".ISS";

const static std::string DEBUG_FD = ".DFd";

const static std::array<std::string, 2> OPT_BR_INFIX = { ".0", ".1" };

const static std::string SCALED_SLIDING_WINDOW_SIZE_PREFIX = "@SWS";

const static std::string TERMINATION_PREFIX = "@TERM";
const static std::string CONSUMER_TERMINATION_COUNT_PREFIX = "@PTC";
const static std::string ITEM_COUNT_SUFFIX = ".IN";
const static std::string STATE_FREE_INTERNAL_ITEM_COUNT_SUFFIX = ".SIN";
const static std::string INTERNALLY_SYNCHRONIZED_INTERNAL_ITEM_COUNT_SUFFIX = ".ISIN";
const static std::string DEFERRED_ITEM_COUNT_SUFFIX = ".DC";
const static std::string CONSUMED_ITEM_COUNT_PREFIX = "@CON";
const static std::string TRANSITORY_CONSUMED_ITEM_COUNT_PREFIX = "@TCN";

const static std::string REPEATING_STREAMSET_HANDLE_PREFIX = "@RSS.";
const static std::string REPEATING_STREAMSET_LENGTH_PREFIX = "@RSSL.";
const static std::string REPEATING_STREAMSET_MALLOCED_DATA_PREFIX = "@RSSD.";

const static std::string STATISTICS_CYCLE_COUNT_SUFFIX = ".SCy";
const static std::string STATISTICS_CYCLE_COUNT_SQUARE_SUM_SUFFIX = ".SCY";
const static std::string STATISTICS_CYCLE_COUNT_TOTAL = "T" + STATISTICS_CYCLE_COUNT_SUFFIX;

#ifdef ENABLE_PAPI
const static std::string STATISTICS_PAPI_COUNT_ARRAY_SUFFIX = ".PCS";
const static std::string STATISTICS_PAPI_TOTAL_COUNT_ARRAY = "!PCS";
const static std::string STATISTICS_PAPI_EVENT_SET_CODE = "PES";
const static std::string STATISTICS_PAPI_EVENT_SET_LIST = "PESL";
#endif

const static std::string STATISTICS_BLOCKING_IO_SUFFIX = ".SBY";
const static std::string STATISTICS_BLOCKING_IO_HISTORY_SUFFIX = ".SHY";
const static std::string STATISTICS_BUFFER_EXPANSION_SUFFIX = ".SBX";
const static std::string STATISTICS_BUFFER_EXPANSION_TEMP_STACK = "@SBTS";
const static std::string STATISTICS_STRIDES_PER_SEGMENT_SUFFIX = ".SSPS";
const static std::string STATISTICS_PRODUCED_ITEM_COUNT_SUFFIX = ".SPIC";
const static std::string STATISTICS_UNCONSUMED_ITEM_COUNT_SUFFIX = ".SUIC";

const static std::string STATISTICS_TRANSFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX = ".TICH";
const static std::string STATISTICS_DEFERRED_ITEM_COUNT_HISTOGRAM_SUFFIX = ".TDCH";

const static std::string STATISTICS_DYNAMIC_MULTITHREADING_STATE_DATA = "@SDMSD";
const static std::string STATISTICS_DYNAMIC_MULTITHREADING_STATE_CURRENT = "@SDMSC";

const static std::string LAST_GOOD_VIRTUAL_BASE_ADDRESS = ".LGA";

const static std::string PENDING_FREEABLE_BUFFER_ADDRESS = ".PFA";
const static std::string PENDING_FREEABLE_BUFFER_CAPACITY = ".PFC";

using ArgVec = Vec<Value *, 64>;

using ThreadLocalScalarAccumulationRule = Kernel::ThreadLocalScalarAccumulationRule;

using BufferPortMap = flat_set<std::pair<unsigned, unsigned>>;

using PartitionJumpPhiOutMap = flat_map<std::pair<unsigned, unsigned>, Value *>;

using PartitionPhiNodeTable = multi_array<PHINode *, 2>;

enum HistogramReportType { TransferredItems, DeferredItems };

class PipelineCompiler final : public KernelCompiler, public PipelineCommonGraphFunctions {

    template<typename T> friend struct OutputPortVec;

    enum { WITH_OVERFLOW = 0, WITHOUT_OVERFLOW = 1};
    using OverflowItemCounts = Vec<std::array<Value *, 2>, 8>;

public:

    PipelineCompiler(BuilderRef b, PipelineKernel * const pipelineKernel);

    void generateImplicitKernels(BuilderRef b);
    void addPipelineKernelProperties(BuilderRef b);
    void constructStreamSetBuffers(BuilderRef b) override;
    void generateInitializeMethod(BuilderRef b);
    void generateAllocateSharedInternalStreamSetsMethod(BuilderRef b, Value * const expectedNumOfStrides);
    void generateInitializeThreadLocalMethod(BuilderRef b);
    void generateAllocateThreadLocalInternalStreamSetsMethod(BuilderRef b, Value * expectedNumOfStrides);
    void generateKernelMethod(BuilderRef b);
    void generateFinalizeMethod(BuilderRef b);
    void generateFinalizeThreadLocalMethod(BuilderRef b);
    std::vector<Value *> getFinalOutputScalars(BuilderRef b) override;
    void runOptimizationPasses(BuilderRef b);
    void bindAdditionalInitializationArguments(BuilderRef b, ArgIterator & arg, const ArgIterator & arg_end) override;
    static void linkPThreadLibrary(BuilderRef b);
    #ifdef ENABLE_PAPI
    static void linkPAPILibrary(BuilderRef b);
    #endif

    unsigned getCacheLineGroupId(const unsigned kernelId) const {
        if (codegen::DebugOptionIsSet(codegen::DisableCacheAlignedKernelStructs)) {
            return 0;
        }
        #ifdef GROUP_SHARED_KERNEL_STATE_INTO_CACHE_LINE_ALIGNED_REGIONS
        // if (mNumOfThreads > 1) {
            return kernelId;
        // }
        #endif
        return 0;
    }

private:

    PipelineCompiler(PipelineKernel * const pipelineKernel, PipelineAnalysis && P);

// internal pipeline state construction functions

public:

    void addInternalKernelProperties(BuilderRef b, const unsigned kernelId, const bool isRoot);
    void generateSingleThreadKernelMethod(BuilderRef b);
    void generateMultiThreadKernelMethod(BuilderRef b);

// main doSegment functions

    void start(BuilderRef b);
    void setActiveKernel(BuilderRef b, const unsigned index, const bool allowThreadLocal, const bool getCommonThreadLocal = false);
    void executeKernel(BuilderRef b);
    void end(BuilderRef b);

// internal pipeline functions

    LLVM_READNONE StructType * getThreadStuctType(BuilderRef b, const std::vector<Value *> & props) const;
    void writeThreadStructObject(BuilderRef b, StructType * const threadStateTy, Value * threadState, Value * const shared, Value * const threadLocal, const std::vector<Value *> & props, Value * const threadNum, Value * const numOfThreads) const;
    void readThreadStuctObject(BuilderRef b, StructType * const threadStateTy, Value * threadState);
    void deallocateThreadState(BuilderRef b, Value * const threadState);

    void allocateThreadLocalState(BuilderRef b, Value * const localState, Value * const threadId = nullptr);
    void deallocateThreadLocalState(BuilderRef b, Value * const localState);
    Value * readTerminationSignalFromLocalState(BuilderRef b, StructType * const threadStateTy, Value * const threadState) const;
    void writeTerminationSignalToLocalState(BuilderRef b, StructType * const threadStateTy, Value * const threadState, Value * const terminated) const;

    std::vector<llvm::Value *> storeDoSegmentState() const;
    void readDoSegmentState(BuilderRef b, StructType * const threadStructTy, Value * const propertyState);
    void restoreDoSegmentState(const std::vector<llvm::Value *> & S);

    inline Value * isProcessThread(BuilderRef b, StructType * const threadStateTy, Value * const threadState) const;
    void updateExternalProducedItemCounts(BuilderRef b);
    void writeMaximumStrideLengthMetadata(BuilderRef b) const;

// partitioning codegen functions

    void makePartitionEntryPoints(BuilderRef b);
    void branchToInitialPartition(BuilderRef b);
    BasicBlock * getPartitionExitPoint(BuilderRef b);
    void checkForPartitionEntry(BuilderRef b);

    void determinePartitionStrideRateScalingFactor();

    void writePartitionEntryIOGuard(BuilderRef b);
    Value * calculatePartitionSegmentLength(BuilderRef b);

    void loadLastGoodVirtualBaseAddressesOfUnownedBuffersInPartition(BuilderRef b) const;

    void phiOutPartitionItemCounts(BuilderRef b, const unsigned kernel, const unsigned targetPartitionId, const bool fromKernelEntryBlock);
    void phiOutPartitionStatusFlags(BuilderRef b, const unsigned targetPartitionId, const bool fromKernelEntry);

    void phiOutPartitionStateAndReleaseSynchronizationLocks(BuilderRef b, const unsigned targetKernelId, const unsigned targetPartitionId, const bool fromKernelEntryBlock, Value * const afterFirstSegNo);

    void acquirePartitionSynchronizationLock(BuilderRef b, const unsigned firstKernelInTargetPartition, Value * const segNo);
    void releaseAllSynchronizationLocksFor(BuilderRef b, const unsigned kernel);

    void writeInitiallyTerminatedPartitionExit(BuilderRef b);
    void checkForPartitionExit(BuilderRef b);

    void ensureAnyExternalProcessedAndProducedCountsAreUpdated(BuilderRef b, const unsigned targetKernelId, const bool fromKernelEntry);

// flow control functions

    void addSegmentLengthSlidingWindowKernelProperties(BuilderRef b, const size_t kernelId, const size_t groupId);
    void initializeInitialSlidingWindowSegmentLengths(BuilderRef b, Value * const segmentLengthScalingFactor);
    void initializeFlowControl(BuilderRef b);
    void detemineMaximumNumberOfStrides(BuilderRef b);
    void updateNextSlidingWindowSize(BuilderRef b, Value * const maxNumOfStrides, Value * const actualNumOfStrides);

// inter-kernel codegen functions

    void readAvailableItemCounts(BuilderRef b);
    void readProcessedItemCounts(BuilderRef b);
    void readProducedItemCounts(BuilderRef b);

    void initializeKernelLoopEntryPhis(BuilderRef b);
    void initializeKernelCheckOutputSpacePhis(BuilderRef b);
    void initializeKernelTerminatedPhis(BuilderRef b);
    void initializeJumpToNextUsefulPartitionPhis(BuilderRef b);
    void initializeKernelInsufficientIOExitPhis(BuilderRef b);
    void initializeKernelLoopExitPhis(BuilderRef b);
    void initializeKernelExitPhis(BuilderRef b);
    void checkForSufficientIO(BuilderRef b);
    void determineNumOfLinearStrides(BuilderRef b);
    void checkForSufficientInputData(BuilderRef b, const BufferPort & inputPort, const unsigned streamSet);
    void checkForSufficientOutputSpace(BuilderRef b, const BufferPort & outputPort, const unsigned streamSet);
    void ensureSufficientOutputSpace(BuilderRef b, const BufferPort & port, const unsigned streamSet);

    Value * calculateTransferableItemCounts(BuilderRef b, Value * const numOfLinearStrides);

    enum class InputExhaustionReturnType {
        Conjunction, Disjunction
    };

    Value * checkIfInputIsExhausted(BuilderRef b, InputExhaustionReturnType returnValType);
    Value * hasMoreInput(BuilderRef b);

    struct FinalItemCount {
        Value * minFixedRateFactor;
        Value * partialPartitionStrides;
    };

    void calculateFinalItemCounts(BuilderRef b, Vec<Value *> & accessibleItems, Vec<Value *> & writableItems, Value *& minFixedRateFactor, Value *& finalStrideRemainder);

    Value * revertTransitiveAddCalculation(BuilderRef b, const ProcessingRate &rate, Value * expectedItemCount, Value * rejectedTerminationSignal);

    void zeroInputAfterFinalItemCount(BuilderRef b, const Vec<Value *> & accessibleItems, Vec<Value *> & inputBaseAddresses);
    void freeZeroedInputBuffers(BuilderRef b);

    Value * allocateLocalZeroExtensionSpace(BuilderRef b, BasicBlock * const insertBefore) const;

    void writeKernelCall(BuilderRef b);
    void buildKernelCallArgumentList(BuilderRef b, ArgVec & args);
    void updateProcessedAndProducedItemCounts(BuilderRef b);
    void writeInternalProcessedAndProducedItemCounts(BuilderRef b, const bool atTermination);
    void readAndUpdateInternalProcessedAndProducedItemCounts(BuilderRef b);
    void readReturnedOutputVirtualBaseAddresses(BuilderRef b) const;
    Value * addVirtualBaseAddressArg(BuilderRef b, const StreamSetBuffer * buffer, ArgVec & args);

    void normalCompletionCheck(BuilderRef b);

    void writeInsufficientIOExit(BuilderRef b);
    void writeJumpToNextPartition(BuilderRef b);

    void computeFullyProcessedItemCounts(BuilderRef b, Value * const terminated);
    void computeFullyProducedItemCounts(BuilderRef b, Value * const terminated);

    void updateKernelExitPhisAfterInitiallyTerminated(BuilderRef b);
    void updatePhisAfterTermination(BuilderRef b);

    void clearUnwrittenOutputData(BuilderRef b);

    void computeMinimumConsumedItemCounts(BuilderRef b);
    void writeConsumedItemCounts(BuilderRef b);
    void recordFinalProducedItemCounts(BuilderRef b);
    void writeUpdatedItemCounts(BuilderRef b);

    void writeOutputScalars(BuilderRef b, const size_t index, std::vector<Value *> & args);
    void initializeScalarValues(BuilderRef b);
    Value * getScalar(BuilderRef b, const size_t index);

// intra-kernel codegen functions

    Value * getInputStrideLength(BuilderRef b, const BufferPort &inputPort, const StringRef location);
    Value * getOutputStrideLength(BuilderRef b, const BufferPort &outputPort, const StringRef location);
    Value * calculateStrideLength(BuilderRef b, const BufferPort & port, Value * const previouslyTransferred, Value * const strideIndex, StringRef location);
    Value * calculateNumOfLinearItems(BuilderRef b, const BufferPort &port, Value * const adjustment, StringRef location);
    Value * getAccessibleInputItems(BuilderRef b, const BufferPort & inputPort, const bool useOverflow = true);
    Value * getNumOfAccessibleStrides(BuilderRef b, const BufferPort & inputPort, Value * const numOfLinearStrides);
    Value * getWritableOutputItems(BuilderRef b, const BufferPort & outputPort, const bool force = false);
    Value * getNumOfWritableStrides(BuilderRef b, const BufferPort & port, Value * const numOfLinearStrides);
    Value * addLookahead(BuilderRef b, const BufferPort & inputPort, Value * const itemCount) const;
    Value * subtractLookahead(BuilderRef b, const BufferPort & inputPort, Value * const itemCount);

    unsigned getPopCountStepSize(const StreamSetPort inputRefPort) const;
    Value * getPartialSumItemCount(BuilderRef b, const BufferPort &port, Value * const previouslyTransferred, Value * const offset, StringRef location) const;
    Value * getMaximumNumOfPartialSumStrides(BuilderRef b, const BufferPort &port, Value * const numOfLinearStrides);
    void splatMultiStepPartialSumValues(BuilderRef b);

// termination codegen functions

    void addTerminationProperties(BuilderRef b, const size_t kernel, const size_t groupId);
    Value * hasKernelTerminated(BuilderRef b, const size_t kernel, const bool normally = false) const;
    Value * isClosed(BuilderRef b, const StreamSetPort inputPort, const bool normally = false) const;
    Value * isClosed(BuilderRef b, const unsigned streamSet, const bool normally = false) const;
    unsigned getTerminationSignalIndex(const unsigned consumer) const;
    Value * isClosedNormally(BuilderRef b, const StreamSetPort inputPort) const;
    bool kernelCanTerminateAbnormally(const unsigned kernel) const;
    void checkIfKernelIsAlreadyTerminated(BuilderRef b);
    void checkPropagatedTerminationSignals(BuilderRef b);
    Value * readTerminationSignal(BuilderRef b, const unsigned kernelId);
    void writeTerminationSignal(BuilderRef b, const unsigned kernelId, Value * const signal) const;
    Value * hasPipelineTerminated(BuilderRef b);
    void signalAbnormalTermination(BuilderRef b);
    LLVM_READNONE static Constant * getTerminationSignal(BuilderRef b, const TerminationSignal type);

    void readCountableItemCountsAfterAbnormalTermination(BuilderRef b);
    void propagateTerminationSignal(BuilderRef b);
    void verifyPostInvocationTerminationSignal(BuilderRef b);

// consumer codegen functions

    unsigned getTruncatedStreamSetSourceId(const unsigned streamSet) const;
    void addConsumerKernelProperties(BuilderRef b, const unsigned producer);
    void writeTransitoryConsumedItemCount(BuilderRef b, const unsigned streamSet, Value * const produced);
    void readExternalConsumerItemCounts(BuilderRef b);
    void readConsumedItemCounts(BuilderRef b);
    Value * readConsumedItemCount(BuilderRef b, const size_t streamSet);
    void setConsumedItemCount(BuilderRef b, const size_t streamSet, Value * consumed, const unsigned slot) const;
    void updateExternalConsumedItemCounts(BuilderRef b);
    void zeroAnySkippedTransitoryConsumedItemCountsUntil(BuilderRef b, const unsigned targetKernelId);

// buffer management codegen functions

    void addBufferHandlesToPipelineKernel(BuilderRef b, const unsigned index, const unsigned groupId);
    void allocateOwnedBuffers(BuilderRef b, Value * const expectedNumOfStrides, Value * const expectedSourceOutputSize, const bool nonLocal);
    void loadInternalStreamSetHandles(BuilderRef b, const bool nonLocal);
    void remapThreadLocalBufferMemory(BuilderRef b);
    void releaseOwnedBuffers(BuilderRef b);
    void freePendingFreeableDynamicBuffers(BuilderRef b);
    void resetInternalBufferHandles();
    void loadLastGoodVirtualBaseAddressesOfUnownedBuffers(BuilderRef b, const size_t kernelId) const;

    Rational getReturnedBufferScaleFactor(const size_t streamSet) const;

    Value * getVirtualBaseAddress(BuilderRef b, const BufferPort & rateData, const BufferNode & bn, Value * position, const bool prefetch, const bool write) const;
    void getInputVirtualBaseAddresses(BuilderRef b, Vec<Value *> & baseAddresses) const;
    void getZeroExtendedInputVirtualBaseAddresses(BuilderRef b, const Vec<Value *> & baseAddresses, Value * const zeroExtensionSpace, Vec<Value *> & zeroExtendedVirtualBaseAddress) const;

// repeating streamset functions

    using InternallyGeneratedStreamSetMap = flat_map<Value *, std::pair<Value *, Value>>;

    void generateGlobalDataForRepeatingStreamSet(BuilderRef b, const unsigned streamSet, Value * const expectedNumOfStrides);
    void addRepeatingStreamSetBufferProperties(BuilderRef b);
    void deallocateRepeatingBuffers(BuilderRef b);
    void generateMetaDataForRepeatingStreamSets(BuilderRef b);
    Constant * getGuaranteedRepeatingStreamSetLength(BuilderRef b, const unsigned streamSet) const;
    void bindRepeatingStreamSetInitializationArguments(BuilderRef b, ArgIterator & arg, const ArgIterator & arg_end) const;
    void addRepeatingStreamSetInitializationArguments(const unsigned kernelId, ArgVec & args) const;

// prefetch instructions

    void prefetchAtLeastThreeCacheLinesFrom(BuilderRef b, Value * const addr, const bool write) const;

// cycle counter functions

    void addCycleCounterProperties(BuilderRef b, const unsigned kernel, const bool isRoot);

    bool trackCycleCounter(const CycleCounter type) const;
    void startCycleCounter(BuilderRef b, const CycleCounter type);
    void startCycleCounter(BuilderRef b, const std::initializer_list<CycleCounter> types);
    void updateCycleCounter(BuilderRef b, const unsigned kernelId, const CycleCounter type) const;
    void updateCycleCounter(BuilderRef b, const unsigned kernelId, Value * const cond, const CycleCounter ifTrue, const CycleCounter ifFalse) const;
    void updateTotalCycleCounterTime(BuilderRef b) const;

    static void linkInstrumentationFunctions(BuilderRef b);

    void recordBlockingIO(BuilderRef b, const StreamSetPort port) const;

    void printOptionalCycleCounter(BuilderRef b);
    StreamSetPort selectPrincipleCycleCountBinding(const unsigned kernel) const;
    void printOptionalBlockingIOStatistics(BuilderRef b);


    void initializeBufferExpansionHistory(BuilderRef b) const;
    void recordBufferExpansionHistory(BuilderRef b, const unsigned streamSet, const BufferNode & bn, const BufferPort & port, const StreamSetBuffer * const buffer) const;
    void printOptionalBufferExpansionHistory(BuilderRef b);

    void initializeStridesPerSegment(BuilderRef b) const;
    void recordStridesPerSegment(BuilderRef b, unsigned kernelId, Value * const totalStrides) const;
    void concludeStridesPerSegmentRecording(BuilderRef b) const;
    void printOptionalStridesPerSegment(BuilderRef b) const;
    void printOptionalBlockedIOPerSegment(BuilderRef b) const;

    void addProducedItemCountDeltaProperties(BuilderRef b, unsigned kernel) const;
    void recordProducedItemCountDeltas(BuilderRef b) const;
    void printProducedItemCountDeltas(BuilderRef b) const;

    void addUnconsumedItemCountProperties(BuilderRef b, unsigned kernel) const;
    void recordUnconsumedItemCounts(BuilderRef b);
    void printUnconsumedItemCounts(BuilderRef b) const;

    void addItemCountDeltaProperties(BuilderRef b, const unsigned kernel, const StringRef suffix) const;

    void recordItemCountDeltas(BuilderRef b, const Vec<Value *> & current, const Vec<Value *> & prior, const StringRef suffix) const;

    void printItemCountDeltas(BuilderRef b, const StringRef title, const StringRef suffix) const;

// internal optimization passes

    void simplifyPhiNodes(Module * const m) const;
    void replacePhiCatchWithCurrentBlock(BuilderRef b, BasicBlock *& toReplace, BasicBlock * const phiContainer);

// synchronization functions

    void obtainCurrentSegmentNumber(BuilderRef b, BasicBlock * const entryBlock);
    void incrementCurrentSegNo(BuilderRef b, BasicBlock * const exitBlock);
    void acquireSynchronizationLock(BuilderRef b, const unsigned kernelId, const unsigned lockType, Value * const segNo);
    void releaseSynchronizationLock(BuilderRef b, const unsigned kernelId, const unsigned lockType, Value * const segNo);
    Value * getSynchronizationLockPtrForKernel(BuilderRef b, const unsigned kernelId, const unsigned lockType) const;
    inline LLVM_READNONE bool isMultithreaded() const;
    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    Value * obtainNextSegmentNumber(BuilderRef b);
    #endif

// family functions

    void addFamilyKernelProperties(BuilderRef b, const unsigned kernelId, const unsigned groupId) const;
    void bindFamilyInitializationArguments(BuilderRef b, ArgIterator & arg, const ArgIterator & arg_end) const;
    void addFamilyCallInitializationArguments(BuilderRef b, const unsigned kernelId, ArgVec & args) const;

// thread local functions

    Value * getThreadLocalHandlePtr(BuilderRef b, const unsigned kernelIndex, const bool commonThreadLocal = false) const;

// optimization branch functions
    bool isEitherOptimizationBranchKernelInternallySynchronized() const;
    Value * checkOptimizationBranchSpanLength(BuilderRef b, Value * const numOfLinearStrides);

// papi instrumentation functions
#ifdef ENABLE_PAPI
    ArrayType * getPAPIEventCounterType(BuilderRef b) const;
    void addPAPIEventCounterPipelineProperties(BuilderRef b);
    void addPAPIEventCounterKernelProperties(BuilderRef b, const unsigned kernel, const bool isRoot);
    void initializePAPI(BuilderRef b) const;
    void registerPAPIThread(BuilderRef b) const;
    void getPAPIEventSet(BuilderRef b);
    void createPAPIMeasurementArrays(BuilderRef b);
    void readPAPIMeasurement(BuilderRef b, Value * const measurementArray) const;
    void startPAPIMeasurement(BuilderRef b, const PAPIKernelCounter measurementType) const;
    void startPAPIMeasurement(BuilderRef b, const std::initializer_list<PAPIKernelCounter> types) const;
    void accumPAPIMeasurementWithoutReset(BuilderRef b, const size_t kernelId, const PAPIKernelCounter measurementType) const;
    void accumPAPIMeasurementWithoutReset(BuilderRef b, const size_t kernelId, Value * const cond, const PAPIKernelCounter ifTrue, const PAPIKernelCounter ifFalse) const;
    void recordTotalPAPIMeasurement(BuilderRef b) const;
    void unregisterPAPIThread(BuilderRef b) const;
    void startPAPI(BuilderRef b);
    void stopPAPI(BuilderRef b);
    void printPAPIReportIfRequested(BuilderRef b);
    void checkPAPIRetValAndExitOnError(BuilderRef b, StringRef source, const int expected, Value * const retVal) const;

#endif

// histogram functions

    bool recordsAnyHistogramData() const;
    void addHistogramProperties(BuilderRef b, const size_t kernelId, const size_t groupId);
    void freeHistogramProperties(BuilderRef b);
    void updateTransferredItemsForHistogramData(BuilderRef b);
    void printHistogramReport(BuilderRef b, HistogramReportType type) const;

    static void linkHistogramFunctions(BuilderRef b);

// dynamic multithreading functions

    void addDynamicThreadingReportProperties(BuilderRef b, const unsigned groupId);
    void initDynamicThreadingReportProperties(BuilderRef b);
    void recordDynamicThreadingState(BuilderRef b, Value * segNo, Value * currentSyncOverhead, Value * currentNumOfThreads) const;
    void printDynamicThreadingReport(BuilderRef b) const;
    static void linkDynamicThreadingReport(BuilderRef b);

// debug message functions

    #ifdef PRINT_DEBUG_MESSAGES
    void debugInit(BuilderRef b);
    template <typename ... Args>
    void debugPrint(BuilderRef b, Twine format, Args ...args) const;
    #endif

// misc. functions

    Value * getFamilyFunctionFromKernelState(BuilderRef b, Type * const type, const std::string &suffix) const;
    Value * callKernelInitializeFunction(BuilderRef b, const ArgVec & args) const;
    std::pair<Value *, FunctionType *> getKernelAllocateSharedInternalStreamSetsFunction(BuilderRef b) const;
    void callKernelInitializeThreadLocalFunction(BuilderRef b) const;
    std::pair<Value *, FunctionType *> getKernelAllocateThreadLocalInternalStreamSetsFunction(BuilderRef b) const;
    std::pair<Value *, FunctionType *> getKernelDoSegmentFunction(BuilderRef b) const;
    Value * callKernelExpectedSourceOutputSizeFunction(BuilderRef b, ArrayRef<Value *> args) const;
    Value * callKernelFinalizeThreadLocalFunction(BuilderRef b, const SmallVector<Value *, 2> & args) const;
    Value * callKernelFinalizeFunction(BuilderRef b, const SmallVector<Value *, 1> & args) const;

    LLVM_READNONE std::string makeKernelName(const size_t kernelIndex) const;
    LLVM_READNONE std::string makeBufferName(const size_t kernelIndex, const StreamSetPort port) const;

    using PipelineCommonGraphFunctions::getReference;

    const StreamSetPort getReference(const StreamSetPort port) const;

    using PipelineCommonGraphFunctions::getInputBufferVertex;
    using PipelineCommonGraphFunctions::getInputBuffer;
    using PipelineCommonGraphFunctions::getInputBinding;

    unsigned getInputBufferVertex(const StreamSetPort inputPort) const;
    StreamSetBuffer * getInputBuffer(const StreamSetPort inputPort) const;
    const Binding & getInputBinding(const StreamSetPort inputPort) const;

    using PipelineCommonGraphFunctions::getOutputBufferVertex;
    using PipelineCommonGraphFunctions::getOutputBuffer;
    using PipelineCommonGraphFunctions::getOutputBinding;

    unsigned getOutputBufferVertex(const StreamSetPort outputPort) const;
    StreamSetBuffer * getOutputBuffer(const StreamSetPort outputPort) const;
    const Binding & getOutputBinding(const StreamSetPort outputPort) const;

    using PipelineCommonGraphFunctions::getBinding;

    const Binding & getBinding(const StreamSetPort port) const;

    void clearInternalStateForCurrentKernel();
    void initializeKernelAssertions(BuilderRef b);

    bool hasAtLeastOneNonGreedyInput() const;
    bool hasAnyGreedyInput(const unsigned kernelId) const;
    bool isDataParallel(const size_t kernel) const;
    bool isCurrentKernelStateFree() const;
    bool hasPrincipalInputRate() const;

protected:

    SimulationAllocator                         mAllocator;

    const bool                                  CheckAssertions;
    const bool                                  mTraceProcessedProducedItemCounts;
    const bool                                  mTraceDynamicBuffers;
    const bool                                  mTraceIndividualConsumedItemCounts;
    const bool                                  mGenerateTransferredItemCountHistogram;
    const bool                                  mGenerateDeferredItemCountHistogram;
    const bool                                  mIsNestedPipeline;
    const bool                                  mUseDynamicMultithreading;

    const LengthAssertions &                    mLengthAssertions;

    // analysis state
    static constexpr unsigned                   PipelineInput = 0;
    static constexpr unsigned                   FirstKernel = 1;
    const unsigned                              LastKernel;
    const unsigned                              PipelineOutput;
    const unsigned                              FirstStreamSet;
    const unsigned                              LastStreamSet;
    const unsigned                              FirstBinding;
    const unsigned                              LastBinding;
    const unsigned                              FirstCall;
    const unsigned                              LastCall;
    const unsigned                              FirstScalar;
    const unsigned                              LastScalar;
    const unsigned                              PartitionCount;

    #ifdef ENABLE_PAPI
    const unsigned                              NumOfPAPIEvents;
    #else
    constexpr static unsigned                   NumOfPAPIEvents = 0;
    #endif

    const size_t                                RequiredThreadLocalStreamSetMemory;

    const bool                                  PipelineHasTerminationSignal;
    const bool                                  HasZeroExtendedStream;
    const bool                                  EnableCycleCounter;
    const bool                                  TraceIO;
    const bool                                  TraceUnconsumedItemCounts;
    const bool                                  TraceProducedItemCounts;
    const bool                                  TraceDynamicMultithreading;

    const KernelIdVector                        KernelPartitionId;
    const KernelIdVector                        FirstKernelInPartition;
    const std::vector<unsigned>                 StrideStepLength;
    const std::vector<unsigned>                 MinimumNumOfStrides;
    const std::vector<unsigned>                 MaximumNumOfStrides;
    const std::vector<Rational>                 PartitionRootStridesPerThreadLocalPage;
    const std::vector<Rational>                 NumOfPartitionOverflowVectors;
    const RelationshipGraph                     mStreamGraph;
    const RelationshipGraph                     mScalarGraph;
    const BufferGraph                           mBufferGraph;
    const std::vector<unsigned>                 PartitionJumpTargetId;
    const ConsumerGraph                         mConsumerGraph;
    const PartialSumStepFactorGraph             mPartialSumStepFactorGraph;
    const TerminationChecks                     mTerminationCheck;
    const TerminationPropagationGraph           mTerminationPropagationGraph;
    const InternallyGeneratedStreamSetGraph     mInternallyGeneratedStreamSetGraph;
    const BitVector                             HasTerminationSignal;
    const FamilyScalarGraph                     mFamilyScalarGraph;


    // pipeline state
    unsigned                                    mKernelId = 0;
    const Kernel *                              mKernel = nullptr;
    Value *                                     mKernelSharedHandle = nullptr;
    Value *                                     mKernelThreadLocalHandle = nullptr;
    Value *                                     mKernelCommonThreadLocalHandle = nullptr;
    Value *                                     mSegNo = nullptr;
    Value *                                     mNumOfFixedThreads = nullptr;
    #ifdef USE_PARTITION_GUIDED_SYNCHRONIZATION_VARIABLE_REGIONS
    Value *                                     mBaseSegNo = nullptr;
    PHINode *                                   mPartitionExitSegNoPhi = nullptr;
    bool                                        mUsingNewSynchronizationVariable = false;
    unsigned                                    mCurrentNestedSynchronizationVariable = 0;
    #endif
    PHINode *                                   mMadeProgressInLastSegment = nullptr;
    Value *                                     mPipelineProgress = nullptr;
    Value *                                     mThreadLocalMemorySizePtr = nullptr;

    BasicBlock *                                mPipelineLoop = nullptr;
    BasicBlock *                                mKernelLoopStart = nullptr;
    BasicBlock *                                mKernelLoopEntry = nullptr;
    BasicBlock *                                mKernelCheckOutputSpace = nullptr;
    BasicBlock *                                mKernelLoopCall = nullptr;
    BasicBlock *                                mKernelCompletionCheck = nullptr;
    BasicBlock *                                mKernelInitiallyTerminated = nullptr;
    BasicBlock *                                mKernelInitiallyTerminatedExit = nullptr;
    BasicBlock *                                mKernelTerminated = nullptr;
    BasicBlock *                                mKernelInsufficientInput = nullptr;
    BasicBlock *                                mKernelJumpToNextUsefulPartition = nullptr;
    BasicBlock *                                mKernelLoopExit = nullptr;
    BasicBlock *                                mKernelLoopExitPhiCatch = nullptr;
    BasicBlock *                                mKernelExit = nullptr;
    BasicBlock *                                mPipelineEnd = nullptr;
    BasicBlock *                                mRethrowException = nullptr;

    Value *                                     mThreadLocalStreamSetBaseAddress = nullptr;
    Value *                                     mExpectedNumOfStridesMultiplier = nullptr;
    Value *                                     mThreadLocalSizeMultiplier = nullptr;

    Vec<AllocaInst *, 16>                       mAddressableItemCountPtr;
    Vec<AllocaInst *, 16>                       mVirtualBaseAddressPtr;
    Vec<AllocaInst *, 4>                        mTruncatedInputBuffer;
    FixedVector<PHINode *>                      mInitiallyAvailableItemsPhi;
    FixedVector<Value *>                        mLocallyAvailableItems;

    FixedVector<Value *>                        mScalarValue;
    BitVector                                   mIsStatelessKernel;
    BitVector                                   mIsInternallySynchronized;

    // partition state
    FixedVector<BasicBlock *>                   mPartitionEntryPoint;
    unsigned                                    mCurrentPartitionId = 0;
    unsigned                                    mCurrentPartitionRoot = 0;
    unsigned                                    LastKernelInPartition = 0;

    Rational                                    mPartitionStrideRateScalingFactor;

    Value *                                     mFinalPartitionSegment = nullptr;
    PHINode *                                   mFinalPartitionSegmentAtLoopExitPhi = nullptr;
    PHINode *                                   mFinalPartitionSegmentAtExitPhi = nullptr;
    PHINode *                                   mFinalPartialStrideFixedRateRemainderPhi = nullptr;
    PHINode *                                   mFinalPartialStrideFixedRateRemainderAtTerminationPhi = nullptr;



    Value *                                     mNumOfPartitionStrides = nullptr;

    BasicBlock *                                mCurrentPartitionEntryGuard = nullptr;
    BasicBlock *                                mNextPartitionEntryPoint = nullptr;
    FixedVector<Value *>                        mKernelTerminationSignal;
    FixedVector<Value *>                        mInitialConsumedItemCount;

    PartitionPhiNodeTable                       mPartitionProducedItemCountPhi;
    PartitionPhiNodeTable                       mPartitionConsumedItemCountPhi;
    PartitionPhiNodeTable                       mPartitionTerminationSignalPhi;
    FixedVector<PHINode *>                      mPartitionPipelineProgressPhi;

    // optimization branch
    PHINode *                                   mOptimizationBranchPriorScanStatePhi = nullptr;
    Value *                                     mOptimizationBranchSelectedBranch = nullptr;

    // kernel state
    Value *                                     mInitialTerminationSignal = nullptr;
    Value *                                     mInitiallyTerminated = nullptr;
    Value *                                     mMaximumNumOfStrides = nullptr;
    PHINode *                                   mMaximumNumOfStridesAtLoopExitPhi = nullptr;
    PHINode *                                   mMaximumNumOfStridesAtJumpPhi = nullptr;
    PHINode *                                   mMaximumNumOfStridesAtExitPhi = nullptr;
    Value *                                     mThreadLocalScalingFactor = nullptr;
    PHINode *                                   mCurrentNumOfStridesAtLoopEntryPhi = nullptr;
    PHINode *                                   mCurrentNumOfStridesAtTerminationPhi = nullptr;
    Value *                                     mUpdatedNumOfStrides = nullptr;
    PHINode *                                   mTotalNumOfStridesAtLoopExitPhi = nullptr;
    PHINode *                                   mAnyProgressedAtLoopExitPhi = nullptr;
    PHINode *                                   mAnyProgressedAtExitPhi = nullptr;
    PHINode *                                   mAlreadyProgressedPhi = nullptr;
    PHINode *                                   mExecutedAtLeastOnceAtLoopEntryPhi = nullptr;
    PHINode *                                   mTerminatedSignalPhi = nullptr;
    PHINode *                                   mTerminatedAtLoopExitPhi = nullptr;
    PHINode *                                   mTerminatedAtExitPhi = nullptr;
    PHINode *                                   mTotalNumOfStridesAtExitPhi = nullptr;
    Value *                                     mNumOfLinearStrides = nullptr;
    Value *                                     mPotentialSegmentLength = nullptr;
    PHINode *                                   mPotentialSegmentLengthAtTerminationPhi = nullptr;
    PHINode *                                   mPotentialSegmentLengthAtLoopExitPhi = nullptr;
    Value *                                     mCurrentNumOfLinearStrides = nullptr;
    Value *                                     mHasZeroExtendedInput = nullptr;
    Value *                                     mInternallySynchronizedSubsegmentNumber = nullptr;
    PHINode *                                   mNumOfLinearStridesPhi = nullptr;
    PHINode *                                   mFixedRateFactorPhi = nullptr;
    Value *                                     mCurrentFixedRateFactor = nullptr;
    PHINode *                                   mIsFinalInvocationPhi = nullptr;
    Value *                                     mIsFinalInvocation = nullptr;
    Value *                                     mHasMoreInput = nullptr;
    PHINode *                                   mStrideStepSizeAtLoopEntryPhi = nullptr;
    Value *                                     mStrideStepSize = nullptr;
    Value *                                     mAnyClosed = nullptr;
    Value *                                     mPrincipalFixedRateFactor = nullptr;
    Value *                                     mHasExhaustedClosedInput = nullptr;
    Rational                                    mFixedRateLCM;
    Value *                                     mTerminatedExplicitly = nullptr;
    Value *                                     mBranchToLoopExit = nullptr;

    bool                                        mKernelIsInternallySynchronized = false;
    bool                                        mKernelCanTerminateEarly = false;
    bool                                        mHasPrincipalInput = false;
    bool                                        mRecordHistogramData = false;
    bool                                        mIsPartitionRoot = false;
    bool                                        mIsOptimizationBranch = false;
    bool                                        mMayHaveInsufficientIO = false;
    bool                                        mExecuteStridesIndividually = false;
    bool                                        mCurrentKernelIsStateFree = false;
    bool                                        mAllowDataParallelExecution = false;
    bool                                        mHasPrincipalInputRate = false;

    unsigned                                    mNumOfTruncatedInputBuffers = 0;

    InputPortVector<Value *>                    mInitiallyProcessedItemCount; // *before* entering the kernel
    InputPortVector<Value *>                    mInitiallyProcessedDeferredItemCount;
    InputPortVector<PHINode *>                  mAlreadyProcessedPhi; // entering the segment loop
    InputPortVector<PHINode *>                  mAlreadyProcessedDeferredPhi;

    InputPortVector<Value *>                    mIsInputZeroExtended;
    InputPortVector<PHINode *>                  mInputVirtualBaseAddressPhi;
    InputPortVector<Value *>                    mFirstInputStrideLength;

    InputPortVector<Value *>                    mInternalAccessibleInputItems;
    InputPortVector<PHINode *>                  mLinearInputItemsPhi;
    InputPortVector<Value *>                    mReturnedProcessedItemCountPtr; // written by the kernel
    InputPortVector<Value *>                    mProcessedItemCountPtr; // exiting the segment loop
    InputPortVector<Value *>                    mProcessedItemCount;
    InputPortVector<PHINode *>                  mProcessedItemCountAtTerminationPhi;
    InputPortVector<Value *>                    mProcessedDeferredItemCountPtr;
    InputPortVector<Value *>                    mProcessedDeferredItemCount;

    InputPortVector<Value *>                    mExhaustedInputPort;
    InputPortVector<PHINode *>                  mExhaustedInputPortPhi;

    InputPortVector<PHINode *>                  mCurrentProcessedItemCountPhi;
    InputPortVector<PHINode *>                  mCurrentProcessedDeferredItemCountPhi;
    InputPortVector<Value *>                    mCurrentLinearInputItems;

    InputPortVector<PHINode *>                  mConsumedItemCountsAtLoopExitPhi; // exiting the kernel
    InputPortVector<PHINode *>                  mUpdatedProcessedPhi; // exiting the kernel
    InputPortVector<PHINode *>                  mUpdatedProcessedDeferredPhi;
    InputPortVector<Value *>                    mFullyProcessedItemCount; // *after* exiting the kernel

    FixedVector<Value *>                        mInitiallyProducedItemCount; // *before* entering the kernel
    FixedVector<Value *>                        mInitiallyProducedDeferredItemCount;
    OutputPortVector<PHINode *>                 mAlreadyProducedPhi; // entering the segment loop
    OutputPortVector<Value *>                   mAlreadyProducedDelayedPhi;
    OutputPortVector<PHINode *>                 mAlreadyProducedDeferredPhi;
    OutputPortVector<Value *>                   mFirstOutputStrideLength;

    OutputPortVector<Value *>                   mInternalWritableOutputItems;
    OutputPortVector<PHINode *>                 mLinearOutputItemsPhi;
    OutputPortVector<Value *>                   mReturnedOutputVirtualBaseAddressPtr; // written by the kernel
    OutputPortVector<Value *>                   mReturnedProducedItemCountPtr; // written by the kernel
    OutputPortVector<Value *>                   mProducedItemCountPtr; // exiting the segment loop
    OutputPortVector<Value *>                   mProducedItemCount;
    OutputPortVector<Value *>                   mProducedDeferredItemCountPtr;
    OutputPortVector<Value *>                   mProducedDeferredItemCount;

    OutputPortVector<PHINode *>                 mCurrentProducedItemCountPhi;
    OutputPortVector<PHINode *>                 mCurrentProducedDeferredItemCountPhi;
    OutputPortVector<Value *>                   mCurrentLinearOutputItems;

    OutputPortVector<PHINode *>                 mProducedAtJumpPhi;
    OutputPortVector<PHINode *>                 mProducedAtTerminationPhi; // exiting after termination
    OutputPortVector<Value *>                   mProducedAtTermination;
    OutputPortVector<PHINode *>                 mUpdatedProducedPhi; // exiting the kernel
    OutputPortVector<PHINode *>                 mUpdatedProducedDeferredPhi;
    OutputPortVector<PHINode *>                 mFullyProducedItemCount; // *after* exiting the kernel


    // cycle counter state
    FixedVector<PHINode *>                      mPartitionStartTimePhi;
    FixedArray<Value *, TOTAL_NUM_OF_CYCLE_COUNTERS>  mCycleCounters;

    // dynamic multithreading cycle counter state
//    Value *                                     mFullSegmentStartTime = nullptr;
    Value *                                     mAccumulatedSynchronizationTimePtr = nullptr;

    // papi counter state
    #ifdef ENABLE_PAPI
    //SmallVector<int, 8>                         PAPIEventList;
    Value *                                     PAPIEventSet = nullptr;
    Value *                                     PAPIEventSetVal = nullptr;
    FixedArray<Value *, NUM_OF_PAPI_COUNTERS>   PAPIEventCounterArray;
    Value *                                     PAPITempMeasurementArray = nullptr;
    #endif

    // debug state
    Value *                                     mThreadId = nullptr;
    Value *                                     mDebugFileName = nullptr;
    Value *                                     mDebugFdPtr = nullptr;
    Value *                                     mDebugActualNumOfStrides;
    Value *                                     mCurrentKernelName = nullptr;    
    FixedVector<Value *>                        mKernelName;


    #ifndef NDEBUG
    FunctionType *                              mKernelDoSegmentFunctionType = nullptr;
    #endif

    // misc.

    OwningVector<Kernel>                        mInternalKernels;
    OwningVector<Binding>                       mInternalBindings;
    OwningVector<StreamSetBuffer>               mInternalBuffers;

};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
inline PipelineCompiler::PipelineCompiler(BuilderRef b, PipelineKernel * const pipelineKernel)
: PipelineCompiler(pipelineKernel, PipelineAnalysis::analyze(b, pipelineKernel)) {
    // Use a delegating constructor to compute the pipeline graph data once and pass it to
    // the compiler. Although a const function attribute ought to suffice, gcc 8.2 does not
    // resolve it correctly and clang requires -O2 or better.
}

#define TRANSFERRED_ITEMS \
    DebugOptionIsSet(codegen::GenerateTransferredItemCountHistogram)

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief constructor
 ** ------------------------------------------------------------------------------------------------------------- */
inline PipelineCompiler::PipelineCompiler(PipelineKernel * const pipelineKernel, PipelineAnalysis && P)
: KernelCompiler(pipelineKernel)
, PipelineCommonGraphFunctions(mStreamGraph, mBufferGraph)
#ifdef FORCE_PIPELINE_ASSERTIONS
, CheckAssertions(true)
#else
, CheckAssertions(codegen::DebugOptionIsSet(codegen::EnableAsserts) || codegen::DebugOptionIsSet(codegen::EnablePipelineAsserts))
#endif
, mTraceProcessedProducedItemCounts(P.mTraceProcessedProducedItemCounts)
, mTraceDynamicBuffers(codegen::DebugOptionIsSet(codegen::TraceDynamicBuffers))
, mTraceIndividualConsumedItemCounts(P.mTraceIndividualConsumedItemCounts)
, mGenerateTransferredItemCountHistogram(DebugOptionIsSet(codegen::GenerateTransferredItemCountHistogram))
, mGenerateDeferredItemCountHistogram(DebugOptionIsSet(codegen::GenerateDeferredItemCountHistogram))
, mIsNestedPipeline(P.IsNestedPipeline)
, mUseDynamicMultithreading(codegen::EnableDynamicMultithreading && !P.IsNestedPipeline)
, mLengthAssertions(pipelineKernel->getLengthAssertions())
, LastKernel(P.LastKernel)
, PipelineOutput(P.PipelineOutput)
, FirstStreamSet(P.FirstStreamSet)
, LastStreamSet(P.LastStreamSet)
, FirstBinding(P.FirstBinding)
, LastBinding(P.LastBinding)
, FirstCall(P.FirstCall)
, LastCall(P.LastCall)
, FirstScalar(P.FirstScalar)
, LastScalar(P.LastScalar)
, PartitionCount(P.PartitionCount)
#ifdef ENABLE_PAPI
, NumOfPAPIEvents([&]() -> unsigned {
    const auto & S = codegen::PapiCounterOptions;
    if (S.compare(codegen::OmittedOption) == 0) {
        return 0;
    } else {
        return std::count_if(S.begin(), S.end(), [](std::string::value_type c){return c == ',';}) + 1;
    }
}())
#endif
, RequiredThreadLocalStreamSetMemory(P.RequiredThreadLocalStreamSetMemory)
, PipelineHasTerminationSignal(pipelineKernel->canSetTerminateSignal())
, HasZeroExtendedStream(P.HasZeroExtendedStream)
, EnableCycleCounter(DebugOptionIsSet(codegen::EnableCycleCounter))
, TraceIO(DebugOptionIsSet(codegen::EnableBlockingIOCounter) || DebugOptionIsSet(codegen::TraceBlockedIO))
, TraceUnconsumedItemCounts(DebugOptionIsSet(codegen::TraceUnconsumedItemCounts))
, TraceProducedItemCounts(DebugOptionIsSet(codegen::TraceProducedItemCounts))
, TraceDynamicMultithreading(mUseDynamicMultithreading && DebugOptionIsSet(codegen::TraceDynamicMultithreading))

, KernelPartitionId(std::move(P.KernelPartitionId))
, FirstKernelInPartition(std::move(P.FirstKernelInPartition))
, StrideStepLength(std::move(P.StrideRepetitionVector))
, MinimumNumOfStrides(std::move(P.MinimumNumOfStrides))
, MaximumNumOfStrides(std::move(P.MaximumNumOfStrides))
, PartitionRootStridesPerThreadLocalPage(std::move(P.PartitionRootStridesPerThreadLocalPage))
, NumOfPartitionOverflowVectors(std::move(P.NumOfPartialOverflowStridesPerPartitionRootStride))
, mStreamGraph(std::move(P.mStreamGraph))
, mScalarGraph(std::move(P.mScalarGraph))
, mBufferGraph(std::move(P.mBufferGraph))

, PartitionJumpTargetId(std::move(P.PartitionJumpTargetId))
, mConsumerGraph(std::move(P.mConsumerGraph))
, mPartialSumStepFactorGraph(std::move(P.mPartialSumStepFactorGraph))
, mTerminationCheck(std::move(P.mTerminationCheck))
, mTerminationPropagationGraph(std::move(P.mTerminationPropagationGraph))
, mInternallyGeneratedStreamSetGraph(std::move(P.mInternallyGeneratedStreamSetGraph))

, HasTerminationSignal(std::move(P.HasTerminationSignal))

, mFamilyScalarGraph(std::move(P.mFamilyScalarGraph))

, mInitiallyAvailableItemsPhi(FirstStreamSet, LastStreamSet, mAllocator)
, mLocallyAvailableItems(FirstStreamSet, LastStreamSet, mAllocator)

, mScalarValue(FirstKernel, LastScalar, mAllocator)
, mIsStatelessKernel(PipelineOutput - PipelineInput + 1)
, mIsInternallySynchronized(PipelineOutput - PipelineInput + 1)

, mPartitionEntryPoint(PartitionCount + 1, mAllocator)

, mKernelTerminationSignal(FirstKernel, LastKernel, mAllocator)
, mInitialConsumedItemCount(FirstStreamSet, LastStreamSet, mAllocator)

, mPartitionProducedItemCountPhi(extents[PartitionCount][LastStreamSet - FirstStreamSet + 1])
, mPartitionConsumedItemCountPhi(extents[PartitionCount][LastStreamSet - FirstStreamSet + 1])
, mPartitionTerminationSignalPhi(extents[PartitionCount][LastKernel - FirstKernel + 1])
, mPartitionPipelineProgressPhi(PartitionCount, mAllocator)

, mInitiallyProcessedItemCount(P.MaxNumOfInputPorts, mAllocator)
, mInitiallyProcessedDeferredItemCount(P.MaxNumOfInputPorts, mAllocator)
, mAlreadyProcessedPhi(P.MaxNumOfInputPorts, mAllocator)
, mAlreadyProcessedDeferredPhi(P.MaxNumOfInputPorts, mAllocator)
, mIsInputZeroExtended(P.MaxNumOfInputPorts, mAllocator)
, mInputVirtualBaseAddressPhi(P.MaxNumOfInputPorts, mAllocator)
, mFirstInputStrideLength(P.MaxNumOfInputPorts, mAllocator)
, mInternalAccessibleInputItems(P.MaxNumOfInputPorts, mAllocator)
, mLinearInputItemsPhi(P.MaxNumOfInputPorts, mAllocator)
, mReturnedProcessedItemCountPtr(P.MaxNumOfInputPorts, mAllocator)
, mProcessedItemCountPtr(P.MaxNumOfInputPorts, mAllocator)
, mProcessedItemCount(P.MaxNumOfInputPorts, mAllocator)
, mProcessedItemCountAtTerminationPhi(P.MaxNumOfInputPorts, mAllocator)
, mProcessedDeferredItemCountPtr(P.MaxNumOfInputPorts, mAllocator)
, mProcessedDeferredItemCount(P.MaxNumOfInputPorts, mAllocator)
, mExhaustedInputPort(P.MaxNumOfInputPorts, mAllocator)
, mExhaustedInputPortPhi(P.MaxNumOfInputPorts, mAllocator)
, mCurrentProcessedItemCountPhi(P.MaxNumOfInputPorts, mAllocator)
, mCurrentProcessedDeferredItemCountPhi(P.MaxNumOfInputPorts, mAllocator)
, mCurrentLinearInputItems(P.MaxNumOfInputPorts, mAllocator)
, mConsumedItemCountsAtLoopExitPhi(P.MaxNumOfInputPorts, mAllocator)
, mUpdatedProcessedPhi(P.MaxNumOfInputPorts, mAllocator)
, mUpdatedProcessedDeferredPhi(P.MaxNumOfInputPorts, mAllocator)
, mFullyProcessedItemCount(P.MaxNumOfInputPorts, mAllocator)

, mInitiallyProducedItemCount(FirstStreamSet, LastStreamSet, mAllocator)
, mInitiallyProducedDeferredItemCount(FirstStreamSet, LastStreamSet, mAllocator)

, mAlreadyProducedPhi(P.MaxNumOfOutputPorts, mAllocator)
, mAlreadyProducedDelayedPhi(P.MaxNumOfOutputPorts, mAllocator)
, mAlreadyProducedDeferredPhi(P.MaxNumOfOutputPorts, mAllocator)
, mFirstOutputStrideLength(P.MaxNumOfOutputPorts, mAllocator)
, mInternalWritableOutputItems(P.MaxNumOfOutputPorts, mAllocator)
, mLinearOutputItemsPhi(P.MaxNumOfOutputPorts, mAllocator)
, mReturnedOutputVirtualBaseAddressPtr(P.MaxNumOfOutputPorts, mAllocator)
, mReturnedProducedItemCountPtr(P.MaxNumOfOutputPorts, mAllocator)
, mProducedItemCountPtr(P.MaxNumOfOutputPorts, mAllocator)
, mProducedItemCount(P.MaxNumOfOutputPorts, mAllocator)
, mProducedDeferredItemCountPtr(P.MaxNumOfOutputPorts, mAllocator)
, mProducedDeferredItemCount(P.MaxNumOfOutputPorts, mAllocator)

, mCurrentProducedItemCountPhi(P.MaxNumOfOutputPorts, mAllocator)
, mCurrentProducedDeferredItemCountPhi(P.MaxNumOfOutputPorts, mAllocator)
, mCurrentLinearOutputItems(P.MaxNumOfOutputPorts, mAllocator)

, mProducedAtJumpPhi(P.MaxNumOfOutputPorts, mAllocator)
, mProducedAtTerminationPhi(P.MaxNumOfOutputPorts, mAllocator)
, mProducedAtTermination(P.MaxNumOfOutputPorts, mAllocator)
, mUpdatedProducedPhi(P.MaxNumOfOutputPorts, mAllocator)
, mUpdatedProducedDeferredPhi(P.MaxNumOfOutputPorts, mAllocator)
, mFullyProducedItemCount(P.MaxNumOfOutputPorts, mAllocator)

, mPartitionStartTimePhi(PartitionCount, mAllocator)

, mKernelName(PipelineInput, LastKernel, mAllocator)

, mInternalKernels(std::move(P.mInternalKernels))
, mInternalBindings(std::move(P.mInternalBindings))
, mInternalBuffers(std::move(P.mInternalBuffers))
{

}

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief getItemWidth
 ** ------------------------------------------------------------------------------------------------------------- */
inline LLVM_READNONE unsigned getItemWidth(const Type * ty ) {
    if (LLVM_LIKELY(isa<ArrayType>(ty))) {
        ty = ty->getArrayElementType();
    }
    return cast<IntegerType>(cast<FixedVectorType>(ty)->getElementType())->getBitWidth();
}

bool PipelineCompiler::isMultithreaded() const {
    return true; // mNumOfThreads != 1 || mIsNestedPipeline;
}

} // end of namespace

#include "debug_messages.hpp"

#endif // PIPELINE_COMPILER_HPP
