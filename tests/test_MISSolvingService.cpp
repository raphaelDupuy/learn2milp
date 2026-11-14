
#include "services/MISSolvingService.hpp"

#include "services/SCIPStatusName.hpp"

#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

// An invalid instance must be refused with an explanation, before any
// model is built.
bool checkRejected(
    const std::string& testName,
    const MISInstance& instance
)
{
    MISResult result;

    if (solveMIS(instance, result))
    {
        std::cerr << "[FAIL] " << testName
                  << ": invalid instance was solved\n";
        return false;
    }

    if (result.errorMessage.rfind("Invalid MIS instance: ", 0) != 0
        || result.hasSolution)
    {
        std::cerr << "[FAIL] " << testName
                  << ": unexpected error '" << result.errorMessage << "'\n";
        return false;
    }

    std::cout << "[PASS] " << testName << '\n';
    return true;
}

// The model builder itself must not index outside the variable array.
bool checkModelBuilderRejects(
    const std::string& testName,
    const MISInstance& instance
)
{
    SCIP* scip = nullptr;

    if (SCIPcreate(&scip) != SCIP_OKAY
        || SCIPincludeDefaultPlugins(scip) != SCIP_OKAY)
    {
        std::cerr << "[FAIL] " << testName << ": cannot create SCIP\n";
        return false;
    }

    std::vector<SCIP_VAR*> vars;
    const SCIP_RETCODE retcode = buildMISModel(scip, instance, vars);
    const bool nothingCreated = SCIPgetStage(scip) == SCIP_STAGE_INIT;

    SCIPfree(&scip);

    if (retcode != SCIP_INVALIDDATA || !nothingCreated || !vars.empty())
    {
        std::cerr << "[FAIL] " << testName
                  << ": expected SCIP_INVALIDDATA and no problem\n";
        return false;
    }

    std::cout << "[PASS] " << testName << '\n';
    return true;
}

// Every SCIP status has its own name; only UNKNOWN is named "UNKNOWN".
bool checkStatusNames()
{
    const SCIP_STATUS statuses[] = {
        SCIP_STATUS_UNKNOWN,
        SCIP_STATUS_OPTIMAL,
        SCIP_STATUS_INFEASIBLE,
        SCIP_STATUS_UNBOUNDED,
        SCIP_STATUS_INFORUNBD,
        SCIP_STATUS_USERINTERRUPT,
        SCIP_STATUS_TERMINATE,
        SCIP_STATUS_NODELIMIT,
        SCIP_STATUS_TOTALNODELIMIT,
        SCIP_STATUS_STALLNODELIMIT,
        SCIP_STATUS_TIMELIMIT,
        SCIP_STATUS_MEMLIMIT,
        SCIP_STATUS_GAPLIMIT,
        SCIP_STATUS_PRIMALLIMIT,
        SCIP_STATUS_DUALLIMIT,
        SCIP_STATUS_SOLLIMIT,
        SCIP_STATUS_BESTSOLLIMIT,
        SCIP_STATUS_RESTARTLIMIT
    };

    std::set<std::string> names;
    bool unknownOnlyForUnknown = true;

    for (const SCIP_STATUS status : statuses)
    {
        const std::string name = getSCIPStatusName(status);
        names.insert(name);

        if ((name == "UNKNOWN") != (status == SCIP_STATUS_UNKNOWN))
            unknownOnlyForUnknown = false;
    }

    if (names.size() != sizeof(statuses) / sizeof(statuses[0])
        || !unknownOnlyForUnknown
        || getSCIPStatusName(SCIP_STATUS_PRIMALLIMIT) != "PRIMALLIMIT")
    {
        std::cerr << "[FAIL] SCIP status names are not distinct\n";
        return false;
    }

    std::cout << "[PASS] SCIP status names\n";
    return true;
}

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

    // Invalid instances: endpoint outside the vertex range, negative
    // endpoint, self-loop, duplicate edge, edge not written as u < v.
    if (!checkRejected("Endpoint out of range", MISInstance{3, {{0, 3}}}))
        ++failures;

    if (!checkRejected("Negative endpoint", MISInstance{3, {{-1, 2}}}))
        ++failures;

    if (!checkRejected("Self-loop", MISInstance{3, {{1, 1}}}))
        ++failures;

    if (!checkRejected(
            "Duplicate edge", MISInstance{3, {{0, 1}, {0, 1}}}))
        ++failures;

    if (!checkRejected("Reversed edge", MISInstance{3, {{2, 1}}}))
        ++failures;

    if (!checkRejected("Negative vertex count", MISInstance{-1, {}}))
        ++failures;

    if (!checkModelBuilderRejects(
            "Model builder rejects out-of-range endpoint",
            MISInstance{3, {{0, 3}}}))
        ++failures;

    if (!checkModelBuilderRejects(
            "Model builder rejects negative vertex count",
            MISInstance{-1, {}}))
        ++failures;

    if (!checkStatusNames())
        ++failures;

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS solving tests passed.\n";
    return 0;
}
