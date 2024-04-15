#pragma once

#include <assert.h>
#include <limits>
#include <cmath>

namespace kernel {

struct Binding;

struct ProcessingRateProbabilityDistribution {
    enum DistributionTypeId {
        Uniform
        , Gamma
        , Normal
        , Maximum
    };

    DistributionTypeId getTypeId() const {
        return TypeId;
    }

    float getAlpha() const {
        assert (TypeId == DistributionTypeId::Gamma);
        return A;
    }

    float getBeta() const {
        assert (TypeId == DistributionTypeId::Gamma);
        return B;
    }

    float getMean() const {
        assert (TypeId == DistributionTypeId::Normal);
        return A;
    }

    float getStdDev() const {
        assert (TypeId == DistributionTypeId::Normal);
        return B;
    }

    float getSkew() const {
        assert (TypeId == DistributionTypeId::Normal);
        return C;
    }

    ProcessingRateProbabilityDistribution ( const ProcessingRateProbabilityDistribution & ) = default;

    ProcessingRateProbabilityDistribution & operator= ( const ProcessingRateProbabilityDistribution & ) = default;

    bool operator == (const ProcessingRateProbabilityDistribution & other) const {
        if (TypeId != other.TypeId) return false;
        if (std::fabs(A - other.A) > std::numeric_limits<float>::epsilon()) return false;
        if (std::fabs(B - other.B) > std::numeric_limits<float>::epsilon()) return false;
        if (std::fabs(C - other.C) > std::numeric_limits<float>::epsilon()) return false;
        return true;
    }

    bool operator != (const ProcessingRateProbabilityDistribution & other) const {
        return !operator==(other);
    }

    bool operator < (const ProcessingRateProbabilityDistribution & other) const {
        if (other.TypeId == Maximum) return true;
        return false;
    }

protected:

    friend struct Binding;
    friend ProcessingRateProbabilityDistribution UniformDistribution();
    friend ProcessingRateProbabilityDistribution GammaDistribution(const float alpha, const float beta);
    friend ProcessingRateProbabilityDistribution NormalDistribution(const float mean, const float stddev);
    friend ProcessingRateProbabilityDistribution SkewNormalDistribution(const float mean, const float stddev, const float skewness);
    friend ProcessingRateProbabilityDistribution MaximumDistribution();

    ProcessingRateProbabilityDistribution(const DistributionTypeId typeId = DistributionTypeId::Uniform,
                                          const float a = 0, const float b = 0, const float c = 0)
    : TypeId(typeId), A(a), B(b), C(c) {

    }
private:
    DistributionTypeId TypeId;
    float A;
    float B;
    float C;
};

inline ProcessingRateProbabilityDistribution UniformDistribution() {
    return ProcessingRateProbabilityDistribution(ProcessingRateProbabilityDistribution::DistributionTypeId::Uniform, 0.0f, 0.0f, 0);
}

inline ProcessingRateProbabilityDistribution GammaDistribution(const float alpha, const float beta) {
    return ProcessingRateProbabilityDistribution(ProcessingRateProbabilityDistribution::DistributionTypeId::Gamma, alpha, beta, 0);
}

inline ProcessingRateProbabilityDistribution NormalDistribution(const float mean, const float stddev) {
    return ProcessingRateProbabilityDistribution(ProcessingRateProbabilityDistribution::DistributionTypeId::Normal, mean, stddev, 0);
}

inline ProcessingRateProbabilityDistribution SkewNormalDistribution(const float mean, const float stddev, const float skewness) {
    return ProcessingRateProbabilityDistribution(ProcessingRateProbabilityDistribution::DistributionTypeId::Normal, mean, stddev, skewness);
}

inline ProcessingRateProbabilityDistribution MaximumDistribution() {
    return ProcessingRateProbabilityDistribution(ProcessingRateProbabilityDistribution::DistributionTypeId::Maximum, 0.0f, 0.0f, 0);
}

}

