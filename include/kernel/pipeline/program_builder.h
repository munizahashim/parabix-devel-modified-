#ifndef PROGRAM_BUILDER_H
#define PROGRAM_BUILDER_H

#include <kernel/pipeline/pipeline_builder.h>
#include <kernel/pipeline/driver/driver.h>
#include <kernel/core/streamsetptr.h>
#include <type_traits>

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

struct streamset_t {

    template<typename... Attrs>
    constexpr streamset_t(int elementCount, int fieldWidth, Attrs &&... attrs)
    : ElementCount(elementCount)
    , FieldWidth(fieldWidth)
    , AnnotatedProcessingRate(kernel::detail::annotateProcessingRate<Attrs...>(std::forward<Attrs>(attrs)...)) {

    }

    const int ElementCount;
    const int FieldWidth;
    kernel::detail::AnnotatedProcessingRate AnnotatedProcessingRate;
};

template <typename T>
struct Input {

    Input(std::string && name)
    : Name(std::move(name)) {

    }

    std::string Name;
};

template <>
struct Input<streamset_t> {

    template<typename ... ValueArgs>
    Input(std::string && name, ValueArgs &&... args)
    : Name(std::move(name))
    , Value(std::forward<ValueArgs>(args)...) {

    }

    std::string Name;
    streamset_t Value;
};


template <typename T>
struct Output {

    explicit Output(std::string name)
    : Name(std::move(name)) {

    }

    std::string Name;
};

template <>
struct Output<streamset_t>  {

    template<typename ... ValueArgs>
    explicit Output(std::string name, ValueArgs &&... args)
    : Name(std::move(name))
    , Value(std::forward<ValueArgs>(args)...) {

    }

    std::string Name;
    streamset_t Value;
};

namespace { /* anonymous */

template <typename T, typename U>
struct tuple_cons {
    using type = typename std::tuple<T, U>;
};

template <typename T, typename... Us>
struct tuple_cons<T, std::tuple<Us...>> {
    using type = typename std::tuple<T, Us...>;
};

// void types represent non-existant type values; filter them whenever possible
// so the tuple reduces to a single void type if and only if no types exist
template <typename T>
struct tuple_cons<T, void> {
    using type = T;
};

template <typename T>
struct tuple_cons<void, T> {
    using type = T;
};

template <typename T>
struct tuple_cons<T, std::tuple<>> {
    using type = T;
};

template<typename R, typename T>
struct make_function_decl {
    using type = R(T);
};

template<typename R>
struct make_function_decl<R, void> {
    using type = R();
};

template<typename R, typename ...Ts>
struct make_function_decl<R, std::tuple<Ts...>> {
    using type = R(Ts...);
};

} /* end of anonymous namespace */

/** ------------------------------------------------------------------------------------------------------------- *
 * @brief ProgramBuilder
 ** ------------------------------------------------------------------------------------------------------------- */
template<typename ... Args>
class TypedProgramBuilder final : public ProgramBuilder {
    friend class ::BaseDriver;

    template<typename ... Rest>
    struct extract_args {
        using ReturnType = void;
        using ParamsType = void;
        static constexpr auto InputScalarCount = 0;
        static constexpr auto OutputScalarCount = 0;
        static constexpr auto InputStreamSetCount = 0;
        static constexpr auto OutputStreamSetCount = 0;
    };

    template<typename InputType>
    struct extract_args<Input<InputType>> {
        using ReturnType = void;
        using ParamsType = const InputType;
        static constexpr auto InputScalarCount = 1;
        static constexpr auto OutputScalarCount = 0;
        static constexpr auto InputStreamSetCount = 0;
        static constexpr auto OutputStreamSetCount = 0;
    };

    template<typename OutputType>
    struct extract_args<Output<OutputType>> {
        using Type = OutputType;
        using ReturnType = OutputType;
        using ParamsType = void;
        static constexpr auto InputScalarCount = 0;
        static constexpr auto OutputScalarCount = 1;
        static constexpr auto InputStreamSetCount = 0;
        static constexpr auto OutputStreamSetCount = 0;
    };

    template<>
    struct extract_args<Input<streamset_t>> {
        using ReturnType = void;
        using ParamsType = const StreamSetPtr &;
        static constexpr auto InputScalarCount = 0;
        static constexpr auto OutputScalarCount = 0;
        static constexpr auto InputStreamSetCount = 1;
        static constexpr auto OutputStreamSetCount = 0;
    };

    template<>
    struct extract_args<Output<streamset_t>> {
        using ReturnType = void;
        using ParamsType = StreamSetPtr &;
        static constexpr auto InputScalarCount = 0;
        static constexpr auto OutputScalarCount = 0;
        static constexpr auto InputStreamSetCount = 0;
        static constexpr auto OutputStreamSetCount = 1;
    };

    template<typename InputType, typename ... Rest>
    struct extract_args<Input<InputType>, Rest...> {
        using Type = InputType;
        using ReturnType = typename extract_args<Rest...>::ReturnType;
        using ParamsType = typename tuple_cons<Type, typename extract_args<Rest...>::ParamsType>::type;
        static constexpr auto InputScalarCount = extract_args<Rest...>::InputScalarCount + 1;
        static constexpr auto OutputScalarCount = extract_args<Rest...>::OutputScalarCount;
        static constexpr auto InputStreamSetCount = extract_args<Rest...>::InputStreamSetCount;
        static constexpr auto OutputStreamSetCount = extract_args<Rest...>::OutputStreamSetCount;
    };

    template<typename OutputType, typename ... Rest>
    struct extract_args<Output<OutputType>, Rest...> {
        using Type = OutputType;
        using ReturnType = typename tuple_cons<Type, typename extract_args<Rest...>::ReturnType>::type;
        using ParamsType = typename extract_args<Rest...>::ParamsType;
        static constexpr auto InputScalarCount = extract_args<Rest...>::InputScalarCount;
        static constexpr auto OutputScalarCount = extract_args<Rest...>::OutputScalarCount + 1;
        static constexpr auto InputStreamSetCount = extract_args<Rest...>::InputStreamSetCount;
        static constexpr auto OutputStreamSetCount = extract_args<Rest...>::OutputStreamSetCount;
    };

    template<typename ... Rest>
    struct extract_args<Input<streamset_t>, Rest...> {
        using Type = const StreamSetPtr &;
        using ReturnType = typename extract_args<Rest...>::ReturnType;
        using ParamsType = typename tuple_cons<Type, typename extract_args<Rest...>::ParamsType>::type;
        static constexpr auto InputScalarCount = extract_args<Rest...>::InputScalarCount;
        static constexpr auto OutputScalarCount = extract_args<Rest...>::OutputScalarCount;
        static constexpr auto InputStreamSetCount = extract_args<Rest...>::InputStreamSetCount + 1;
        static constexpr auto OutputStreamSetCount = extract_args<Rest...>::OutputStreamSetCount;
    };

    template<typename ... Rest>
    struct extract_args<Output<streamset_t>, Rest...> {
        using Type = StreamSetPtr &;
        // despite being an output argument, references to the output streamset
        // values are passed as input arguments
        using ReturnType = typename extract_args<Rest...>::ReturnType;
        using ParamsType = typename tuple_cons<Type, typename extract_args<Rest...>::ParamsType>::type;
        static constexpr auto InputScalarCount = extract_args<Rest...>::InputScalarCount;
        static constexpr auto OutputScalarCount = extract_args<Rest...>::OutputScalarCount;
        static constexpr auto InputStreamSetCount = extract_args<Rest...>::InputStreamSetCount;
        static constexpr auto OutputStreamSetCount = extract_args<Rest...>::OutputStreamSetCount + 1;
    };

    using ParamsType = typename extract_args<Args...>::ParamsType;

    template<typename InputType>
    inline static void append_arg(BaseDriver & driver,
                           Bindings & inputScalars, Bindings & outputScalars,
                           Bindings & inputStreamSets, Bindings & outputStreamSets,
                           Input<InputType> && f) {
        llvm::Type * const ty = llvm::TypeBuilder<InputType, false>::get(driver.getContext());
        inputScalars.emplace_back(ty, f.Name, driver.CreateScalar(ty));
    }

    template<typename OutputType>
    inline static void append_arg(BaseDriver & driver,
                           Bindings & inputScalars, Bindings & outputScalars,
                           Bindings & inputStreamSets, Bindings & outputStreamSets,
                           Output<OutputType> && f) {
        llvm::Type * const ty = llvm::TypeBuilder<OutputType, false>::get(driver.getContext());
        outputScalars.emplace_back(ty, f.Name, driver.CreateScalar(ty));
    }

    template<>
    inline static void append_arg(BaseDriver & driver,
                           Bindings & inputScalars, Bindings & outputScalars,
                           Bindings & inputStreamSets, Bindings & outputStreamSets,
                           Input<streamset_t> && f) {
        inputStreamSets.emplace_back(std::move(f.Name),
                                     driver.CreateStreamSet(f.Value.ElementCount, f.Value.FieldWidth),
                                     std::move(f.Value.AnnotatedProcessingRate));
    }

    template<>
    inline static void append_arg(BaseDriver & driver,
                           Bindings & inputScalars, Bindings & outputScalars,
                           Bindings & inputStreamSets, Bindings & outputStreamSets,
                           Output<streamset_t> && f) {
        outputStreamSets.emplace_back(std::move(f.Name),
                                      driver.CreateStreamSet(f.Value.ElementCount, f.Value.FieldWidth),
                                      std::move(f.Value.AnnotatedProcessingRate));
    }

    template<typename... Fs>
    inline static void append_args(BaseDriver & driver,
                            Bindings & inputScalars, Bindings & outputScalars,
                            Bindings & inputStreamSets, Bindings & outputStreamSets,
                            Fs &&... fs);

    template<>
    inline static void append_args(BaseDriver & driver,
                            Bindings & inputScalars, Bindings & outputScalars,
                            Bindings & inputStreamSets, Bindings & outputStreamSets) {
        /* do nothing */
    }

    template<typename F, typename... Fs>
    inline static void append_args(BaseDriver & driver,
                            Bindings & inputScalars, Bindings & outputScalars,
                            Bindings & inputStreamSets, Bindings & outputStreamSets,
                            F f, Fs &&... fs) {
        append_arg(driver, inputScalars, outputScalars, inputStreamSets, outputStreamSets, std::forward<F>(f));
        append_args(driver, inputScalars, outputScalars, inputStreamSets, outputStreamSets, std::forward<Fs>(fs)...);
    }

public:
    using ReturnType = typename extract_args<Args...>::ReturnType;
    using FunctionDeclType =  typename make_function_decl<ReturnType, ParamsType>::type *;

    FunctionDeclType compile() {
        return reinterpret_cast<FunctionDeclType>(ProgramBuilder::compile());
    }

    inline TypedProgramBuilder(BaseDriver & driver, Args &&... args)
    : ProgramBuilder(driver, constructKernel(driver, std::forward<Args>(args)...)) {

    }

    TypedProgramBuilder(TypedProgramBuilder &&) = default;

private:

    static PipelineKernel * constructKernel(BaseDriver & driver, Args &&... args) {
        Bindings scalar_inputs{};
        constexpr auto inputScalarCount = extract_args<Args...>::InputScalarCount;
        if constexpr (inputScalarCount > 0) scalar_inputs.reserve(inputScalarCount);

        Bindings scalar_outputs{};
        constexpr auto outputScalarCount = extract_args<Args...>::OutputScalarCount;
        if constexpr (outputScalarCount > 0) scalar_outputs.reserve(outputScalarCount);

        Bindings stream_inputs{};
        constexpr auto inputStreamSetCount = extract_args<Args...>::InputStreamSetCount;
        if constexpr (inputStreamSetCount > 0) stream_inputs.reserve(inputStreamSetCount);

        Bindings stream_outputs{};
        constexpr auto outputStreamSetCount = extract_args<Args...>::OutputStreamSetCount;
        if constexpr (outputStreamSetCount > 0) stream_outputs.reserve(outputStreamSetCount);

        append_args(driver, scalar_inputs, scalar_outputs, stream_inputs, stream_outputs, std::forward<Args>(args)...);

        assert (scalar_inputs.size() == inputScalarCount);
        assert (scalar_outputs.size() == outputScalarCount);
        assert (stream_inputs.size() == inputStreamSetCount);
        assert (stream_outputs.size() == outputStreamSetCount);

        PipelineKernel * const pipeline =
            new PipelineKernel(driver,
                               std::move(stream_inputs), std::move(stream_outputs),
                               std::move(scalar_inputs), std::move(scalar_outputs));
        return pipeline;
    }

};

template<typename ... Args>
TypedProgramBuilder<Args...> CreatePipeline(BaseDriver & driver, Args... args) {
    return TypedProgramBuilder<Args...>{driver, std::forward<Args>(args)...};
}

}

#endif // PROGRAM_BUILDER_H
