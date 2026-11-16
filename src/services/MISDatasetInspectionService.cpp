#include "services/MISDatasetInspectionService.hpp"

#include "io/CsvReader.hpp"
#include "io/MISInstanceIO.hpp"
#include "io/MISStrongBranchingSampleIO.hpp"
#include "services/MISFeaturizationService.hpp"
#include "services/MISInstanceLoadingService.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace
{

std::ifstream openInput(const std::filesystem::path& path)
{
    std::ifstream input(path);

    if (!input)
        throw std::runtime_error("Cannot open " + path.string());

    return input;
}

/*
 * Collection metadata: a header "MIS_SB_DATASET VERSION", then one
 * "key value" line per field, text values being quoted.
 */
struct CollectionMetadata
{
    int version = 0;
    std::map<std::string, std::string> fields;

    bool has(const std::string& key) const
    {
        return fields.count(key) > 0;
    }

    long long integer(const std::string& key, const std::string& file) const
    {
        long long value = 0;
        const auto found = fields.find(key);

        if (found == fields.end() || !parseCsvInteger(found->second, value))
        {
            throw std::runtime_error(
                file + ": missing or invalid " + key
            );
        }

        return value;
    }

    double real(const std::string& key, const std::string& file) const
    {
        double value = 0.0;
        const auto found = fields.find(key);

        if (found == fields.end() || !parseCsvReal(found->second, value))
        {
            throw std::runtime_error(
                file + ": missing or invalid " + key
            );
        }

        return value;
    }
};

CollectionMetadata readCollectionMetadata(const std::filesystem::path& path)
{
    std::ifstream input = openInput(path);
    CollectionMetadata metadata;

    std::string line;
    std::string format;

    if (!std::getline(input, line)
        || !(std::istringstream(line) >> format >> metadata.version)
        || format != "MIS_SB_DATASET")
    {
        throw std::runtime_error(
            path.string() + ": not a strong branching collection"
        );
    }

    while (std::getline(input, line))
    {
        if (line.empty())
            continue;

        std::istringstream stream(line);
        std::string key;
        std::string value;

        stream >> key >> std::ws;

        if (stream.peek() == '"')
            stream >> std::quoted(value);
        else
            stream >> value;

        if (key.empty() || stream.fail())
            throw std::runtime_error(path.string() + ": malformed line");

        metadata.fields[key] = value;
    }

    return metadata;
}

MISInstanceSetSummary inspectInstances(
    const std::filesystem::path& directory,
    std::vector<std::string>& skippedFiles
)
{
    MISInstanceSetSummary summary;
    std::vector<double> vertices;
    std::vector<double> edges;

    for (const auto& path : listMISInstances(directory))
    {
        LoadedMISInstance loaded;
        std::string error;

        if (!loadMISInstance(path, loaded, error))
        {
            skippedFiles.push_back(path.string() + ": " + error);
            continue;
        }

        ++summary.numberOfInstances;
        ++summary.instancesPerGenerator[loaded.metadata.generatorType];
        vertices.push_back(loaded.instance.n);
        edges.push_back(static_cast<double>(loaded.instance.edges.size()));
    }

    summary.vertices = misValueRange(vertices);
    summary.edges = misValueRange(edges);
    return summary;
}

// What one collection adds to the summary.
struct CollectionFigures
{
    long long samples = 0;
    long long candidates = 0;
    long long exploredNodes = 0;
    bool truncated = false;
    bool hasSolvingTime = false;
    double solvingTimeSeconds = 0.0;
    std::string policy = "expert";
    long long policyDecisions = 0;
    long long policyAgreements = 0;
    bool countable = false;
    long long usableSamples = 0;
};

// Throws std::runtime_error for a malformed collection.
CollectionFigures readCollection(const std::filesystem::path& path)
{
    const std::string file = path.string();
    const CollectionMetadata metadata = readCollectionMetadata(path);
    CollectionFigures figures;

    figures.samples = metadata.integer("number_of_samples", file);
    figures.candidates = metadata.integer("number_of_candidates", file);

    if (metadata.has("explored_nodes"))
        figures.exploredNodes = metadata.integer("explored_nodes", file);

    figures.truncated = metadata.has("complete")
        && metadata.integer("complete", file) == 0;

    if (metadata.has("solving_time_seconds"))
    {
        figures.hasSolvingTime = true;
        figures.solvingTimeSeconds =
            metadata.real("solving_time_seconds", file);
    }

    if (metadata.has("policy_model"))
    {
        figures.policy = "model " + metadata.fields.at("policy_model");
        figures.policyDecisions = metadata.integer("policy_decisions", file);
        figures.policyAgreements =
            metadata.integer("policy_expert_agreements", file);
    }

    // NAME.sb.meta -> NAME: the prefix of the two tables.
    const std::string prefix = file.substr(
        0, file.size() - std::string(MIS_SB_METADATA_SUFFIX).size());
    std::string error;

    if (!countUsableMISStrongBranchingSamples(
            prefix + MIS_SB_NODES_SUFFIX,
            prefix + MIS_SB_CANDIDATES_SUFFIX,
            figures.usableSamples, figures.countable, error))
    {
        throw std::runtime_error(error);
    }

    return figures;
}

MISCollectionSetSummary inspectCollections(
    const std::filesystem::path& directory,
    std::vector<std::string>& skippedFiles
)
{
    MISCollectionSetSummary summary;
    std::vector<double> samples;
    std::vector<double> times;

    for (const auto& path : listMISCollections(directory))
    {
        CollectionFigures figures;

        try
        {
            figures = readCollection(path);
        }
        catch (const std::exception& exception)
        {
            const std::string message = exception.what();

            // Messages from the metadata reader already name the file.
            skippedFiles.push_back(
                message.rfind(path.string(), 0) == 0
                ? message : path.string() + ": " + message);
            continue;
        }

        ++summary.numberOfCollections;
        summary.samples += figures.samples;
        summary.candidates += figures.candidates;
        summary.exploredNodes += figures.exploredNodes;
        summary.truncatedCollections += figures.truncated ? 1 : 0;
        summary.policyDecisions += figures.policyDecisions;
        summary.policyAgreements += figures.policyAgreements;
        ++summary.collectionsPerPolicy[figures.policy];
        samples.push_back(static_cast<double>(figures.samples));

        if (figures.hasSolvingTime)
        {
            summary.totalSolvingTimeSeconds += figures.solvingTimeSeconds;
            times.push_back(figures.solvingTimeSeconds);
        }

        if (figures.countable)
            summary.usableSamples += figures.usableSamples;
        else
            ++summary.collectionsWithoutUsableCount;
    }

    summary.samplesPerCollection = misValueRange(samples);
    summary.solvingTimeSeconds = misValueRange(times);
    return summary;
}

} // namespace

MISValueRange misValueRange(std::vector<double> values)
{
    MISValueRange range;

    if (values.empty())
        return range;

    std::sort(values.begin(), values.end());

    const std::size_t middle = values.size() / 2;

    range.minimum = values.front();
    range.maximum = values.back();
    range.median = values.size() % 2 == 1
        ? values[middle]
        : (values[middle - 1] + values[middle]) / 2.0;

    return range;
}

bool inspectMISDirectory(
    const std::filesystem::path& directory,
    MISDirectorySummary& summary,
    std::string& errorMessage
)
{
    try
    {
        std::error_code error;

        if (!std::filesystem::is_directory(directory, error))
            throw std::runtime_error("Not a directory: " + directory.string());

        MISDirectorySummary inspected;
        inspected.instances =
            inspectInstances(directory, inspected.skippedFiles);
        inspected.collections =
            inspectCollections(directory, inspected.skippedFiles);

        summary = inspected;
        errorMessage.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        errorMessage = exception.what();
        return false;
    }
}
