#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "problems/MIS.hpp"

/*
 * Strong branching data collection on one MIS instance.
 *
 * SCIP solves the instance with the MIS full strong branching rule, which
 * records one sample per branching decision. The solver settings are the
 * ones used for both data collection and evaluation in this project:
 *
 * - presolving enabled;
 * - restarts disabled (presolving/maxrestarts = 0);
 * - cutting planes at the root node only (separating/maxrounds = 0).
 *
 * Three files are produced next to each other:
 *
 *   <prefix>.sb.nodes.csv       one row per sample
 *   <prefix>.sb.candidates.csv  one row per branching candidate
 *   <prefix>.sb.meta            collection metadata
 *
 * The CSV tables are described in io/MISStrongBranchingSampleIO.hpp. For
 * given instance, limits and seed they are reproducible byte for byte,
 * unless the time limit is what stopped the run. The metadata file is
 * written last: its presence means that the two tables are complete.
 */

struct MISCollectionConfig
{
    // Identifies the instance in the metadata file (e.g. its file name).
    std::string instanceName;

    // Limits on the solving process. A run stopped by a limit still
    // produces a valid, but truncated, dataset.
    double timeLimitSeconds = 3600.0;
    long long nodeLimit = -1; // -1: no limit

    // SCIP randomization/randomseedshift.
    unsigned int scipSeed = 0;
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

    double solvingTimeSeconds = 0.0;

    // Bounds on the size of a maximum independent set.
    bool hasSolution = false;
    double primalBound = 0.0;
    double dualBound = 0.0;
};

// Suffixes appended to the output prefix.
extern const char* const MIS_SB_NODES_SUFFIX;
extern const char* const MIS_SB_CANDIDATES_SUFFIX;
extern const char* const MIS_SB_METADATA_SUFFIX;

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
