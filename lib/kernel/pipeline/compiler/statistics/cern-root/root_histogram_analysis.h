#include <stdlib.h>

// By default, CERN ROOT requires RTTI but this is disabled for LLVM. To avoid the complication of mixing RTTI modes,
// we isolate all of the ROOT utilities into a seperate library.

struct HistogramPortListEntry {
    uint64_t ItemCount;
    uint64_t Frequency;
    HistogramPortListEntry * Next;
};

struct HistogramPortData {
    uint32_t PortType;
    uint32_t PortNum;
    const char * BindingName;
    uint64_t Size;
    void * Data; // if Size = 0, this points to a HistogramPortListEntry; otherwise its an 64-bit array of length size.
};

struct HistogramKernelData {
    uint32_t Id;
    uint32_t NumOfPorts;
    const char * KernelName;
    HistogramPortData * PortData;
};

#ifdef ENABLE_CERN_ROOT
extern "C" {

void cern_root_analyze_histogram_data(const HistogramKernelData * const data, const uint64_t numOfKernels, uint32_t reportType);

}
#endif
