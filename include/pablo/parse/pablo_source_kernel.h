/*
 *  Part of the Parabix Project, under the Open Software License 3.0.
 *  SPDX-License-Identifier: OSL-3.0
 */

#pragma once

#include <memory>
#include <vector>
#include <pablo/pablo_kernel.h>

namespace pablo {

namespace parse {
    class PabloParser;
    class SourceFile;
}

class PabloSourceKernel final : public PabloKernel {
public:
    PabloSourceKernel(LLVMTypeSystemInterface & ts,
                      std::shared_ptr<parse::PabloParser> & parser,
                      std::shared_ptr<parse::SourceFile> & sourceFile,
                      std::string const & kernelName,
                      kernel::Bindings inputStreamBindings = {},
                      kernel::Bindings outputStreamBindings = {},
                      kernel::Bindings inputScalarBindings = {},
                      kernel::Bindings outputScalarBindings = {});

    PabloSourceKernel(LLVMTypeSystemInterface & ts,
                      std::shared_ptr<parse::PabloParser> & parser,
                      std::string const & sourceFile,
                      std::string const & kernelName,
                      kernel::Bindings inputStreamBindings,
                      kernel::Bindings outputStreamBindings,
                      kernel::Bindings inputScalarBindings,
                      kernel::Bindings outputScalarBindings);

private:
    void generatePabloMethod() override;

    std::shared_ptr<parse::PabloParser>     mParser;
    std::shared_ptr<parse::SourceFile>      mSource;
    std::string                             mKernelName;
};

} // namespace pablo
