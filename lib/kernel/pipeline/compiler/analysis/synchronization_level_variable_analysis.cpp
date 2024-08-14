#include "pipeline_analysis.hpp"

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief identifySynchronizationVariableLevels
 ** ------------------------------------------------------------------------------------------------------------- */
void PipelineAnalysis::identifySynchronizationVariableLevels() {
    SynchronizationVariableNumber.resize(PartitionCount, 0);
    if (codegen::EnableJumpGuidedSynchronizationVariables && !IsNestedPipeline) {
        unsigned currentSyncNumber = 0;
        for (size_t partId = 2; partId < (PartitionCount - 1); partId++) {
            if (PartitionJumpTargetId[partId - 1] == (PartitionCount - 1)) {
                ++currentSyncNumber;
            }
            SynchronizationVariableNumber[partId] = currentSyncNumber;
        }
    }
}

}
