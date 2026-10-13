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
    GeneratedMISInstance generate(
        const ErdosRenyiConfig& config
    ) const;
};
