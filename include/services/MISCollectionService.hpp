#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "io/MISStrongBranchingSampleIO.hpp"
#include "problems/MIS.hpp"
#include "services/MISSolverLimits.hpp"

/*
 * Strong branching data collection on one MIS instance.
 *
 * SCIP solves the instance with the MIS full strong branching rule, which
 * records one sample per branching decision. The solver settings are the
 * branching study settings of services/MISSolverSettings.hpp, also used
 * to evaluate branching rules.
 *
 * Three files are produced next to each other:
 *
 *   <prefix>.sb.nodes.csv       one row per sample
 *   <prefix>.sb.candidates.csv  one row per branching candidate
 *   <prefix>.sb.meta            collection metadata
 *
 * The CSV tables and the file suffixes (MIS_SB_*_SUFFIX) are described
 * in io/MISStrongBranchingSampleIO.hpp. For
 * given instance, limits, seed and policy model they are reproducible
 * byte for byte, unless the time limit is what stopped the run. The metadata file is
 * written last: its presence means that the two tables are complete.
 *
 * By default the search follows the expert: the rule branches on the
 * variable strong branching ranks first. With a policy model, the search
 * follows the model instead (the candidate the mislearned rule would
 * choose), while every visited node is still labelled by strong
 * branching. The samples then describe the nodes the model leads to,
 * which is what a model trained on them will face inside the solver
 * (on-policy collection, as in DAgger). Nodes that strong branching
 * proves prunable are still pruned.
 */

struct MISCollectionConfig
{
    // Identifies the instance in the metadata file (e.g. its file name).
    std::string instanceName;

    // Limits on the solving process. A run stopped by a limit still
    // produces a valid, but truncated, dataset.
    double timeLimitSeconds = MIS_DEFAULT_TIME_LIMIT_SECONDS;
    long long nodeLimit = -1; // -1: no limit

    // SCIP randomization/randomseedshift.
    unsigned int scipSeed = 0;

    // Branching model driving the search (see above). Empty: the expert.
    std::filesystem::path policyModelPath;
};

struct MISCollectionResult
{
    std::filesystem::path nodesPath;
    std::filesystem::path candidatesPath;
    std::filesystem::path metadataPath;

    std::string status;

    // True when the search tree was fully explored.
    bool complete = false;

    long long exploredNodes = 0;
    std::size_t numberOfSamples = 0;
    std::size_t numberOfCandidates = 0;

    // With a policy model: branching decisions taken by the model, and
    // those where it chose a candidate with the expert's best score.
    std::size_t policyDecisions = 0;
    std::size_t policyAgreements = 0;

    double solvingTimeSeconds = 0.0;

    // Bounds on the size of a maximum independent set.
    bool hasSolution = false;
    double primalBound = 0.0;
    double dualBound = 0.0;
};

/*
 * Collects strong branching samples for one instance.
 *
 * Existing output files are never overwritten. On failure, no output
 * file is left behind and errorMessage describes the problem.
 */
bool collectMISStrongBranchingSamples(
    const MISInstance& instance,
    const MISCollectionConfig& config,
    const std::filesystem::path& outputPrefix,
    MISCollectionResult& result,
    std::string& errorMessage
);
