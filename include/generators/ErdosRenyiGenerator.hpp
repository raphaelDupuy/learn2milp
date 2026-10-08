#pragma once

#include "problems/MIS.hpp"

struct ErdosRenyiConfig
{
    int n;
    double p;
    unsigned int seed;
};

class ErdosRenyiGenerator
{
public:
    MISInstance generate(const ErdosRenyiConfig& config) const;
};
