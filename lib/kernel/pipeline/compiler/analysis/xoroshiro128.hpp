#ifndef XORSHIFT128_HPP
#define XORSHIFT128_HPP

#include <random>
#include <array>


class xoroshiro128 {

    constexpr std::uint32_t rotl(const std::uint32_t x, const int s) noexcept {
        return (x << s) | (x >> (32 - s));
    }

public:

    using state_type = std::array<std::uint32_t, 4>;
    using result_type = uint32_t;

    static constexpr result_type (min)() { return 0; }
    static constexpr result_type (max)() { return UINT32_MAX; }
    friend bool operator==(xoroshiro128 const &, xoroshiro128 const &);
    friend bool operator!=(xoroshiro128 const &, xoroshiro128 const &);

    xoroshiro128()
    : _state({0xB0F3D4F1, 0x51723f59, 0x2fb5c3e4, 0xf6e3f1c9 }) {

    }

    explicit xoroshiro128(std::random_device &rd) {
        seed(rd);
    }

    void seed(std::random_device &rd) {
        for (unsigned i = 0; i < 4; ++i) {
            _state[i] = rd();
        }
    }

    result_type operator()() {
        const uint64_t result = rotl(_state[1] * 5, 7) * 9;
        const uint64_t t = _state[1] << 17;
        _state[2] ^= _state[0];
        _state[3] ^= _state[1];
        _state[1] ^= _state[2];
        _state[0] ^= _state[3];
        _state[2] ^= t;
        _state[3] = rotl(_state[3], 45);
        return result;
    }

    void discard(unsigned long long n)
    {
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


#endif // XORSHIFT128_HPP
