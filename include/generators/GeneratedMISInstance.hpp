#pragma once

#include <string>

#include "problems/MIS.hpp"

struct MISGenerationMetadata
{
    std::string family;
    unsigned int seed;

    int n;
    double p;

    std::string generatorVersion;
};

struct GeneratedMISInstance
{
    MISInstance graph;
    MISGenerationMetadata metadata;
};
