#pragma once

#if 0

#include <random>
#include <array>


class xoroshiro128 {

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:

    using state_type = std::array<uint64_t, 4>;
    using result_type = std::default_random_engine::result_type;

    static constexpr result_type (min)() { return 0; }
    static constexpr result_type (max)() { return UINT32_MAX; }
    friend bool operator==(xoroshiro128 const &, xoroshiro128 const &);
    friend bool operator!=(xoroshiro128 const &, xoroshiro128 const &);

    xoroshiro128() : _state({ 0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c }) {

    }

    explicit xoroshiro128(std::random_device &rd) {
        seed(rd);
    }

    void seed(std::random_device &rd) {
        for (unsigned i = 0; i < 4; ++i) {
            _state[i] = rd();
        }
    }

    inline result_type operator()() {
        const uint64_t result = rotl(_state[0] + _state[3], 23) + _state[0];
        const uint64_t t = _state[1] << 17;
        _state[2] ^= _state[0];
        _state[3] ^= _state[1];
        _state[1] ^= _state[2];
        _state[0] ^= _state[3];
        _state[2] ^= t;
        _state[3] = rotl(_state[3], 45);
        return result;
    }

    inline void discard(unsigned long long n) {
        for (unsigned long long i = 0; i < n; ++i)
            operator()();
    }

private:
    state_type _state;
};

bool operator==(xoroshiro128 const &lhs, xoroshiro128 const &rhs)
{
    return lhs._state == rhs._state;
}
bool operator!=(xoroshiro128 const &lhs, xoroshiro128 const &rhs)
{
    return lhs._state != rhs._state;
}

#endif

