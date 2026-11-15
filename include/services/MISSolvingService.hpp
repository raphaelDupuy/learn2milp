
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

/*
 * Solves the instance to optimality with SCIP as shipped (its own
 * branching rules and settings, no time limit). To choose the branching
 * rule, the settings or a limit, use benchmarkMIS()
 * (services/MISBenchmarkService.hpp), which this function calls.
 */
bool solveMIS(
    const MISInstance& instance,
    MISResult& result
);
