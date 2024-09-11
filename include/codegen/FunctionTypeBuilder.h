#pragma once

#include <codegen/TypeBuilder.h>

// NOTE: Currently, LLVM TypeBuilder can deduce FuntionTypes for up to 5 arguments. The following
// templates have no limit but should be deprecated if the TypeBuilder ever supports n-ary functions.

namespace {

// primary template handles types that have no nested ::type member:
template<class, class = void>
struct is_known_type : std::false_type {};

// specialization recognizes types that do have a nested ::type member:
template<class T>
struct is_known_type<T, std::void_t<decltype(T::get)>> : std::true_type {};

template<unsigned i, typename... Args>
struct ParameterTypeBuilder;


template<unsigned i, typename A1, typename... An>
struct ParameterTypeBuilder<i, A1, An...> {
    static_assert(is_known_type<llvm::TypeBuilder<A1, false>>::value, "unknown parameter type");
    static void get(llvm::LLVMContext & C, llvm::Type ** params) noexcept {
        params[i] = llvm::TypeBuilder<A1, false>::get(C);
        ParameterTypeBuilder<i + 1, An...>::get(C, params);
    }
};

template<unsigned i, typename A>
struct ParameterTypeBuilder<i, A> {
    static void get(llvm::LLVMContext & C, llvm::Type ** params) noexcept {
        params[i] = llvm::TypeBuilder<A, false>::get(C);
    }
};

template<unsigned i>
struct ParameterTypeBuilder<i> {
    static void get(llvm::LLVMContext &, llvm::Type **) noexcept {
        /* do nothing */
    }
};

}

template<typename T>
struct FunctionTypeBuilder;

template<typename R, typename... Args>
struct FunctionTypeBuilder<R(Args...)> {
    static_assert(is_known_type<llvm::TypeBuilder<R, false>>::value, "unknown return type");
    static llvm::FunctionType * get(llvm::LLVMContext & C) noexcept {
        llvm::Type * params[sizeof...(Args)];
        ParameterTypeBuilder<0, Args...>::get(C, params);
        return llvm::FunctionType::get(llvm::TypeBuilder<R, false>::get(C), params, false);
    }
};

template<typename R, typename... Args>
struct FunctionTypeBuilder<R(Args...) noexcept> {
    static_assert(is_known_type<llvm::TypeBuilder<R, false>>::value, "unknown return type");
    static llvm::FunctionType * get(llvm::LLVMContext & C) noexcept {
        llvm::Type * params[sizeof...(Args)];
        ParameterTypeBuilder<0, Args...>::get(C, params);
        return llvm::FunctionType::get(llvm::TypeBuilder<R, false>::get(C), params, false);
    }
};

template<typename R>
struct FunctionTypeBuilder<R()> {
    static_assert(is_known_type<llvm::TypeBuilder<R, false>>::value, "unknown return type");
    static llvm::FunctionType * get(llvm::LLVMContext & C) noexcept {
        return llvm::FunctionType::get(llvm::TypeBuilder<R, false>::get(C), false);
    }
};


template<typename R>
struct FunctionTypeBuilder<R() noexcept> {
    static_assert(is_known_type<llvm::TypeBuilder<R, false>>::value, "unknown return type");
    static llvm::FunctionType * get(llvm::LLVMContext & C) noexcept {
        return llvm::FunctionType::get(llvm::TypeBuilder<R, false>::get(C), false);
    }
};
