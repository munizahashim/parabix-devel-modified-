/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <initializer_list>
#include <kernel/core/kernel.h>

namespace kernel {

class ErrorMonitorKernel final : public MultiBlockKernel {
public:
    using IOStreamBindings = std::initializer_list<std::pair<StreamSet *, StreamSet *>>;

    ErrorMonitorKernel(KernelBuilder & b, StreamSet * error, IOStreamBindings bindings);
private:
    std::string mName;
    std::vector<std::pair<std::string, std::string>> mStreamNames;
    std::size_t mNextNameId = 0;

    void generateMultiBlockLogic(KernelBuilder & b, llvm::Value * const numOfStrides) override;

    std::pair<std::string, std::string> generateNewStreamSetNames() {
        std::size_t id = mNextNameId++;
        return {"in_" + std::to_string(id), "out_" + std::to_string(id)};
    }

    template<typename FuncTy>
    void foreachMonitoredStreamSet(FuncTy fn) {
        for (auto const & binding : mStreamNames) {
            fn(binding.first, binding.second);
        }
    }
};

} // namespace kernel

