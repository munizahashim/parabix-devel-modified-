#ifndef PROCESSING_RATE_PROBABILITY_FUNCTION_H
#define PROCESSING_RATE_PROBABILITY_FUNCTION_H

#include <assert.h>

namespace kernel {

struct Binding;

struct ProcessingRateProbabilityDistribution {
    enum DistributionTypeId {
        Uniform
        , Gamma
        , Normal
    };

    DistributionTypeId getTypeId() const {
        return TypeId;
    }

    float getMin() const {
        assert (TypeId == DistributionTypeId::Uniform);
        return A;
    }

    float getMax() const {
        assert (TypeId == DistributionTypeId::Uniform);
        return B;
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
        assert (TypeId == DistributionTypeId::Uniform);
        return A;
    }

    float getStdDev() const {
        assert (TypeId == DistributionTypeId::Uniform);
        return B;
    }

    float getSkew() const {
        assert (TypeId == DistributionTypeId::Uniform);
        return C;
    }

protected:

    friend struct Binding;
    friend ProcessingRateProbabilityDistribution UniformDistribution(const float min, const float max);
    friend ProcessingRateProbabilityDistribution GammaDistribution(const float alpha, const float beta);
    friend ProcessingRateProbabilityDistribution NormalDistribution(const float mean, const float stddev);
    friend ProcessingRateProbabilityDistribution SkewNormalDistribution(const float mean, const float stddev, const float skewness);

    ProcessingRateProbabilityDistribution(const DistributionTypeId typeId = DistributionTypeId::Uniform,
                                          const float a = 0, const float b = 0, const float c = 0)
    : TypeId(typeId), A(a), B(b), C(c) {

    }
private:
    const DistributionTypeId TypeId;
    const float A;
    const float B;
    const float C;
};

ProcessingRateProbabilityDistribution UniformDistribution(const float min = 0, const float max = 1) {
    return ProcessingRateProbabilityDistribution(ProcessingRateProbabilityDistribution::DistributionTypeId::Uniform, min, max, 0);
}

ProcessingRateProbabilityDistribution GammaDistribution(const float alpha, const float beta) {
    return ProcessingRateProbabilityDistribution(ProcessingRateProbabilityDistribution::DistributionTypeId::Gamma, alpha, beta, 0);
}

ProcessingRateProbabilityDistribution NormalDistribution(const float mean, const float stddev) {
    return ProcessingRateProbabilityDistribution(ProcessingRateProbabilityDistribution::DistributionTypeId::Normal, mean, stddev, 0);
}

ProcessingRateProbabilityDistribution SkewNormalDistribution(const float mean, const float stddev, const float skewness) {
    return ProcessingRateProbabilityDistribution(ProcessingRateProbabilityDistribution::DistributionTypeId::Normal, mean, stddev, skewness);
}

}

#endif // PROCESSING_RATE_PROBABILITY_FUNCTION_H
