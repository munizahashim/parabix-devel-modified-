#ifndef PROGRAM_BUILDER_H
#define PROGRAM_BUILDER_H

#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/pipeline/driver/driver.h>

namespace kernel {

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief ProgramBuilder
 ** ------------------------------------------------------------------------------------------------------------- */
class ProgramBuilder : public PipelineBuilder {
    friend class ::BaseDriver;
public:

    void * compile();

    Kernel * makeKernel() override;

    ProgramBuilder(BaseDriver & driver, PipelineKernel * const kernel);

private:

    void * compileKernel(Kernel * const kernel);
};

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief ProgramBuilder
 ** ------------------------------------------------------------------------------------------------------------- */
template<typename FunctionType>
class TypedProgramBuilder final : public ProgramBuilder {
    friend class ::BaseDriver;
public:

    FunctionType compile() {
        return reinterpret_cast<FunctionType>(ProgramBuilder::compile());
    }

    TypedProgramBuilder(BaseDriver & driver, PipelineKernel * const kernel)
    : ProgramBuilder(driver, kernel) {

    }

};

}

#endif // PROGRAM_BUILDER_H
