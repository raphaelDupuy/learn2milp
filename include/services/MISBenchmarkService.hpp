#pragma once

#include <string>

#include "problems/MIS.hpp"

/*
 * Measures how SCIP solves one MIS instance with a given branching rule.
 *
 * This is the evaluation side of the branching study: the same function
 * measures SCIP's own rules (the baselines) and, later, learned rules.
 */

enum class MISBenchmarkSettings
{
    // Branching study settings (services/MISSolverSettings.hpp): the
    // conditions in which the strong branching samples are collected.
    Study,

    // SCIP as shipped, with no parameter changed: an outside reference.
    Default
};

struct MISBenchmarkConfig
{
    /*
     * Name of the branching rule to give the highest priority to:
     *
     * - a rule shipped with SCIP, e.g. "relpscost" (SCIP's default),
     *   "pscost" or "vanillafullstrong";
     * - "misfullstrong", the full strong branching expert that the data
     *   collection records;
     * - empty to keep SCIP's own priorities.
     */
    std::string branchingRule;

    MISBenchmarkSettings settings = MISBenchmarkSettings::Study;

    double timeLimitSeconds = 3600.0;

    // SCIP randomization/randomseedshift.
    unsigned int scipSeed = 0;
};

struct MISBenchmarkResult
{
    std::string status;

    // True when the instance was solved to proven optimality.
    bool solved = false;

    long long exploredNodes = 0;
    long long lpIterations = 0;
    double solvingTimeSeconds = 0.0;

    // Bounds on the size of a maximum independent set.
    bool hasSolution = false;
    double primalBound = 0.0;
    double dualBound = 0.0;
};

// Text names of the settings, as used on the command line and in result
// tables: "study" and "default".
const char* misBenchmarkSettingsName(MISBenchmarkSettings settings);

bool parseMISBenchmarkSettings(
    const std::string& name,
    MISBenchmarkSettings& settings
);

// True if the name designates a branching rule that benchmarkMIS() can
// select: "misfullstrong" or a rule shipped with SCIP.
bool isMISBenchmarkRuleAvailable(const std::string& branchingRule);

/*
 * Solves the instance once. Fails, with an explanation in errorMessage,
 * if the instance is invalid, the branching rule is unknown or SCIP
 * reports an error. Reaching the time limit is not a failure.
 */
bool benchmarkMIS(
    const MISInstance& instance,
    const MISBenchmarkConfig& config,
    MISBenchmarkResult& result,
    std::string& errorMessage
);
