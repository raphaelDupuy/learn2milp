
#include "services/MISSolvingService.hpp"

#include <cmath>
#include <iostream>
#include <string>

bool checkMIS(
    const std::string& testName,
    const MISInstance& instance,
    double expectedObjective
)
{
    MISResult result;

    if (!solveMIS(instance, result))
    {
        std::cerr << "[FAIL] " << testName
                  << ": " << result.errorMessage << '\n';
        return false;
    }

    if (!result.hasSolution)
    {
        std::cerr << "[FAIL] " << testName
                  << ": no feasible solution found\n";
        return false;
    }

    if (std::abs(result.objectiveValue - expectedObjective) > 1e-6)
    {
        std::cerr << "[FAIL] " << testName
                  << ": expected objective "
                  << expectedObjective
                  << ", got " << result.objectiveValue << '\n';
        return false;
    }

    if (result.status != "OPTIMAL")
    {
        std::cerr << "[FAIL] " << testName
                  << ": expected OPTIMAL status, got "
                  << result.status << '\n';
        return false;
    }

    std::cout << "[PASS] " << testName
              << " (objective = " << result.objectiveValue << ")\n";

    return true;
}

int main()
{
    int failures = 0;

    // Graph with four isolated vertices: optimum = 4.
    MISInstance emptyGraph{
        4,
        {}
    };

    if (!checkMIS("Empty graph", emptyGraph, 4.0))
        ++failures;

    // Path 0--1--2: optimum = 2.
    MISInstance pathGraph{
        3,
        {{0, 1}, {1, 2}}
    };

    if (!checkMIS("Path graph", pathGraph, 2.0))
        ++failures;

    // Triangle: optimum = 1.
    MISInstance triangleGraph{
        3,
        {{0, 1}, {0, 2}, {1, 2}}
    };

    if (!checkMIS("Triangle graph", triangleGraph, 1.0))
        ++failures;

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS solving tests passed.\n";
    return 0;
}
