
#pragma once

#include <string>
#include <vector>

#include "problems/MIS.hpp"

struct MISResult
{
    bool hasSolution = false;

    std::vector<int> selectedVertices;

    double objectiveValue = 0.0;
    double solvingTimeSeconds = 0.0;

    long long exploredNodes = 0;

    std::string status;
    std::string errorMessage;
};

bool solveMIS(
    const MISInstance& instance,
    MISResult& result
);
