#pragma once

#include "generators/GeneratedMISInstance.hpp"

struct ErdosRenyiConfig
{
    int n;
    double p;
    unsigned int seed;
};

class ErdosRenyiGenerator
{
public:
    // Throws std::invalid_argument unless n >= 0 and 0 <= p <= 1.
    static void validate(const ErdosRenyiConfig& config);

    GeneratedMISInstance generate(
        const ErdosRenyiConfig& config
    ) const;
};
