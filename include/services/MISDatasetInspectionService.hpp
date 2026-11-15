#pragma once

#include <cstddef>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

/*
 * Summary of one directory of the branching study: the MIS instances
 * (NAME.mis with NAME.mis.meta) and the strong branching collections
 * (NAME.sb.meta with its two tables, see services/MISCollectionService.hpp)
 * it holds. Subdirectories are not visited.
 */

// Smallest, median and largest of a set of values (all 0 when empty).
struct MISValueRange
{
    double minimum = 0.0;
    double median = 0.0;
    double maximum = 0.0;
};

MISValueRange misValueRange(std::vector<double> values);

struct MISInstanceSetSummary
{
    std::size_t numberOfInstances = 0;

    // Number of instances of each generator_type.
    std::map<std::string, std::size_t> instancesPerGenerator;

    MISValueRange vertices;
    MISValueRange edges;
};

struct MISCollectionSetSummary
{
    std::size_t numberOfCollections = 0;

    // Collections stopped by a limit: their search tree is truncated.
    std::size_t truncatedCollections = 0;

    // Number of collections of each policy: "expert", or "model FILE"
    // for on-policy samples.
    std::map<std::string, std::size_t> collectionsPerPolicy;

    long long exploredNodes = 0;
    long long samples = 0;
    long long candidates = 0;

    /*
     * Samples a model is trained on: the node was branched, the expert's
     * choice is valid and strong branching evaluated every candidate (as
     * ml/branching_data.py). Counted from the tables, except for the
     * collections in an older format, which are left out and counted in
     * collectionsWithoutUsableCount.
     */
    long long usableSamples = 0;
    std::size_t collectionsWithoutUsableCount = 0;

    // On-policy collections: decisions taken by the model and how many
    // of them agree with the expert.
    long long policyDecisions = 0;
    long long policyAgreements = 0;

    MISValueRange samplesPerCollection;

    double totalSolvingTimeSeconds = 0.0;
    MISValueRange solvingTimeSeconds;
};

struct MISDirectorySummary
{
    MISInstanceSetSummary instances;
    MISCollectionSetSummary collections;

    // Instances and collections that cannot be read (an older format, a
    // damaged file), each as "FILE: reason". They are left out of the
    // summaries above.
    std::vector<std::string> skippedFiles;
};

/*
 * Fails, with an explanation in errorMessage, only if the directory
 * itself cannot be read.
 */
bool inspectMISDirectory(
    const std::filesystem::path& directory,
    MISDirectorySummary& summary,
    std::string& errorMessage
);
