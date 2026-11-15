#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "problems/MIS.hpp"
#include "services/MISSolverLimits.hpp"

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
     * - "mislearned", the rule driven by the model of learnedModelPath;
     * - empty to keep SCIP's own priorities.
     */
    std::string branchingRule;

    // Model file of the "mislearned" rule (io/MISBranchingModelIO.hpp).
    // Must be empty for every other rule.
    std::filesystem::path learnedModelPath;

    MISBenchmarkSettings settings = MISBenchmarkSettings::Study;

    double timeLimitSeconds = MIS_DEFAULT_TIME_LIMIT_SECONDS;

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

    // Vertices of the best solution found, in increasing order; empty
    // when there is no solution.
    std::vector<int> selectedVertices;
};

// Text names of the settings, as used on the command line and in result
// tables: "study" and "default".
const char* misBenchmarkSettingsName(MISBenchmarkSettings settings);

bool parseMISBenchmarkSettings(
    const std::string& name,
    MISBenchmarkSettings& settings
);

// True if the name designates a branching rule that benchmarkMIS() can
// select: "misfullstrong", "mislearned" or a rule shipped with SCIP.
bool isMISBenchmarkRuleAvailable(const std::string& branchingRule);

/*
 * Checks a branching rule and its model before any run, as benchmarkMIS()
 * does: the rule must be available, the mislearned rule needs a model
 * file that can be loaded, and the other rules take none. Lets a caller
 * refuse a request before solving anything.
 */
bool validateMISBenchmarkRule(
    const std::string& branchingRule,
    const std::filesystem::path& learnedModelPath,
    std::string& errorMessage
);

/*
 * Solves the instance once. Fails, with an explanation in errorMessage,
 * if the instance is invalid, the branching rule is unknown, the model
 * cannot be loaded or SCIP reports an error. Reaching the time limit is not a failure.
 */
bool benchmarkMIS(
    const MISInstance& instance,
    const MISBenchmarkConfig& config,
    MISBenchmarkResult& result,
    std::string& errorMessage
);

/*
 * Shifted geometric mean, exp(mean(log(x + shift))) - shift (0 for no
 * value): it limits the weight of both the very easy and the very hard
 * instances. Node counts are summarized with a shift of 100, solving
 * times with a shift of 1 second.
 */
double misShiftedGeometricMean(const std::vector<double>& values, double shift);

constexpr double MIS_NODES_SHIFT = 100.0;
constexpr double MIS_TIME_SHIFT_SECONDS = 1.0;

// Runs of one rule, summarized as at the end of a benchmark.
struct MISRuleSummary
{
    std::size_t runs = 0;
    std::size_t solved = 0;
    std::vector<double> nodes;
    std::vector<double> times;

    void add(const MISBenchmarkResult& result);

    // Shifted geometric means of the node counts and solving times.
    double meanNodes() const;
    double meanTimeSeconds() const;
};
