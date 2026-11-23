#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "generators/TaillardGenerator.hpp"

/*
 * Generation of JSSP instance files: the implementation behind the
 * jssp-generator command line tool.
 *
 * An instance is written as two files: PATH (the instance, see
 * io/JSSPInstanceIO.hpp) and PATH.meta (its generator and parameters, see
 * io/InstanceMetadataIO.hpp). Existing files are never overwritten, and a
 * failed generation leaves no file behind.
 */

struct JSSPSingleGenerationResult
{
    std::filesystem::path instancePath;
    std::filesystem::path metadataPath;

    int jobs = 0;
    int machines = 0;
    int lowerBound = 0;       // jsspOneMachineBound()
    int dispatchMakespan = 0; // jsspDispatchMakespan()
};

bool generateSingleJSSP(
    const TaillardConfig& config,
    const std::filesystem::path& outputPath,
    JSSPSingleGenerationResult& result,
    std::string& errorMessage
);

/*
 * A batch holds one instance per combination of the lists: every number
 * of jobs, every number of machines, every seed, in this order, the two
 * seeds of Taillard's generator being derived from each seed
 * (taillardConfigFromSeed()). No list may be empty or contain a value
 * twice.
 *
 * The instances are written to outputDirectory (created if needed) and
 * named after taillardInstanceId(). A manifest, outputDirectory/manifest.csv,
 * lists them with the columns
 *
 *   instance_id,file,generator_type,generator_version,seed,jobs,machines,
 *   time_seed,machine_seed,lower_bound,dispatch_makespan
 *
 * and is written last: its presence means that the batch is complete. A
 * directory that already holds a manifest, a temporary manifest or one of
 * the instances is refused.
 */
struct JSSPBatchConfig
{
    std::vector<int> jobsValues;
    std::vector<int> machinesValues;
    std::vector<unsigned int> seeds;

    std::filesystem::path outputDirectory;
};

struct JSSPBatchProgress
{
    std::size_t index = 0; // 1-based
    std::size_t total = 0;
    std::string fileName;
    int lowerBound = 0;
    int dispatchMakespan = 0;
};

// Called after each instance is written.
using JSSPBatchProgressCallback = std::function<void(const JSSPBatchProgress&)>;

struct JSSPBatchResult
{
    std::size_t numberOfInstances = 0;
    std::filesystem::path manifestPath;
};

// At most this many instances per batch.
constexpr std::size_t JSSP_MAX_BATCH_SIZE = 100000;

bool generateBatchJSSP(
    const JSSPBatchConfig& config,
    JSSPBatchResult& result,
    std::string& errorMessage,
    const JSSPBatchProgressCallback& progress = {}
);
