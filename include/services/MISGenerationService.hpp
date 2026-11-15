#pragma once

#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "generators/MISGenerator.hpp"

/*
 * Generation of MIS instance files: the single implementation behind the
 * mis-generator command line tool and the terminal interface.
 *
 * An instance is written as two files: PATH (the graph, see
 * io/MISInstanceIO.hpp) and PATH.meta (its generator and parameters, see
 * io/InstanceMetadataIO.hpp). Existing files are never overwritten, and a
 * failed generation leaves no file behind.
 */

struct MISSingleGenerationResult
{
    std::filesystem::path graphPath;
    std::filesystem::path metadataPath;

    int numberOfVertices = 0;
    std::size_t numberOfEdges = 0;
    unsigned int seed = 0;
};

bool generateSingleMIS(
    const MISGeneratorConfig& config,
    const std::filesystem::path& outputPath,
    MISSingleGenerationResult& result,
    std::string& errorMessage
);

/*
 * A batch holds one instance per combination of the parameter lists, of
 * one graph family: every n, every p (Erdos-Renyi) or k
 * (Barabasi-Albert), every seed, in this order. The list of the other
 * family must be empty, and no list may contain a value twice.
 *
 * The instances are written to outputDirectory (created if needed) and
 * named after misInstanceId(). A manifest, outputDirectory/manifest.csv,
 * lists them with the columns
 *
 *   instance_id,file,generator_type,generator_version,seed,
 *   <parameters of the family>,m
 *
 * where the parameters are those of misManifestParameterNames() and m is
 * the number of edges. The manifest is written last: its presence means
 * that the batch is complete. A directory that already holds a manifest,
 * a temporary manifest or one of the instances is refused.
 */
struct MISBatchConfig
{
    MISGraphFamily family = MISGraphFamily::ErdosRenyi;

    std::vector<int> nValues;
    std::vector<double> pValues; // Erdos-Renyi only
    std::vector<int> kValues;    // Barabasi-Albert only
    std::vector<unsigned int> seeds;

    std::filesystem::path outputDirectory;
};

struct MISBatchProgress
{
    std::size_t index = 0; // 1-based
    std::size_t total = 0;
    std::string fileName;
    std::size_t numberOfEdges = 0;
};

// Called after each instance is written.
using MISBatchProgressCallback = std::function<void(const MISBatchProgress&)>;

struct MISBatchResult
{
    std::size_t numberOfInstances = 0;
    std::filesystem::path manifestPath;
};

// At most this many instances per batch.
constexpr std::size_t MIS_MAX_BATCH_SIZE = 100000;

bool generateBatchMIS(
    const MISBatchConfig& config,
    MISBatchResult& result,
    std::string& errorMessage,
    const MISBatchProgressCallback& progress = {}
);
