#include "services/MISDatasetInspectionService.hpp"

#include "generators/MISGenerator.hpp"
#include "services/MISCollectionService.hpp"
#include "services/MISFeaturizationService.hpp"
#include "services/MISGenerationService.hpp"
#include "services/MISInstanceLoadingService.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

int failures = 0;

void check(bool condition, const std::string& testName)
{
    if (condition)
    {
        std::cout << "[PASS] " << testName << '\n';
    }
    else
    {
        std::cerr << "[FAIL] " << testName << '\n';
        ++failures;
    }
}

bool sameRange(const MISValueRange& range, double min, double med, double max)
{
    return range.minimum == min && range.median == med && range.maximum == max;
}

void writeFile(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream(path) << text;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " WORK_DIR\n";
        return 2;
    }

    const std::filesystem::path workDir(argv[1]);
    std::filesystem::remove_all(workDir);

    const std::filesystem::path instances = workDir / "instances";
    const std::filesystem::path samples = workDir / "samples";
    std::filesystem::create_directories(instances);
    std::filesystem::create_directories(samples);

    // Value ranges.
    check(sameRange(misValueRange({}), 0, 0, 0), "Empty range is zero");
    check(sameRange(misValueRange({3, 1, 2}), 1, 2, 3), "Odd count median");
    check(sameRange(misValueRange({4, 1, 3, 2}), 1, 2.5, 4),
        "Even count median");

    // Two instances of each family, then strong branching collections on
    // the Erdos-Renyi ones.
    const std::vector<MISGeneratorConfig> configs{
        ErdosRenyiConfig{100, 0.10, 1},
        ErdosRenyiConfig{100, 0.10, 2},
        BarabasiAlbertConfig{40, 3, 1},
    };

    std::string error;

    for (const MISGeneratorConfig& config : configs)
    {
        MISSingleGenerationResult generated;

        check(generateSingleMIS(config,
                  instances / (misInstanceId(config) + ".mis"),
                  generated, error),
            "Instance is generated: " + error);
    }

    std::size_t expectedSamples = 0;
    std::size_t expectedCandidates = 0;
    std::size_t expectedUsable = 0;

    for (const unsigned int seed : {1U, 2U})
    {
        const std::string name =
            misInstanceId(ErdosRenyiConfig{100, 0.10, seed});

        LoadedMISInstance loaded;
        loadMISInstance(instances / (name + ".mis"), loaded, error);

        MISCollectionConfig config;
        config.instanceName = name + ".mis";

        MISCollectionResult collected;
        check(collectMISStrongBranchingSamples(
                  loaded.instance, config, samples / name, collected, error),
            "Collection succeeds: " + error);

        // The featurizer's count of usable samples is the reference.
        MISFeaturizationResult featurized;
        check(featurizeMISStrongBranchingSamples(
                  loaded.instance, collected.nodesPath,
                  collected.candidatesPath, workDir / (name + ".features.csv"),
                  featurized, error),
            "Featurization succeeds: " + error);

        expectedSamples += collected.numberOfSamples;
        expectedCandidates += collected.numberOfCandidates;
        expectedUsable += featurized.numberOfUsableSamples;
    }

    MISDirectorySummary summary;

    check(inspectMISDirectory(instances, summary, error),
        "Instance directory is inspected: " + error);
    check(summary.instances.numberOfInstances == 3
            && summary.instances.instancesPerGenerator.at("erdos_renyi") == 2
            && summary.instances.instancesPerGenerator.at("barabasi_albert")
                == 1
            && sameRange(summary.instances.vertices, 40, 100, 100)
            && summary.instances.edges.minimum == 3 * (40 - 3)
            && summary.collections.numberOfCollections == 0,
        "Instances are counted per family with their sizes");

    check(inspectMISDirectory(samples, summary, error),
        "Sample directory is inspected: " + error);

    const MISCollectionSetSummary& collections = summary.collections;

    check(summary.instances.numberOfInstances == 0
            && collections.numberOfCollections == 2
            && collections.collectionsPerPolicy.at("expert") == 2
            && collections.truncatedCollections == 0,
        "Collections are counted per policy");
    check(collections.samples == static_cast<long long>(expectedSamples)
            && collections.candidates
                == static_cast<long long>(expectedCandidates),
        "Samples and candidates add up the collections");
    check(collections.usableSamples == static_cast<long long>(expectedUsable)
            && collections.collectionsWithoutUsableCount == 0
            && expectedUsable > 0,
        "Usable samples agree with the featurizer ("
            + std::to_string(expectedUsable) + ")");
    check(collections.totalSolvingTimeSeconds > 0.0
            && collections.exploredNodes > 0,
        "Solving times and nodes are read");

    // A collection whose tables lack the columns of the usable samples.
    const std::filesystem::path old = workDir / "old";
    std::filesystem::create_directories(old);
    writeFile(old / "a.sb.meta",
        "MIS_SB_DATASET 1\nnumber_of_samples 5\nnumber_of_candidates 20\n");
    writeFile(old / "a.sb.nodes.csv", "sample_index,node_id\n");
    writeFile(old / "a.sb.candidates.csv", "sample_index,vertex_id\n");

    check(inspectMISDirectory(old, summary, error)
            && summary.collections.samples == 5
            && summary.collections.collectionsWithoutUsableCount == 1,
        "Older collections are summarized without usable samples");

    // Unreadable files are skipped and reported; the others are summed.
    writeFile(old / "b.sb.meta", "SOMETHING_ELSE 2\n");
    writeFile(old / "c.mis", "not an instance\n");
    check(inspectMISDirectory(old, summary, error)
            && summary.collections.numberOfCollections == 1
            && summary.collections.samples == 5
            && summary.instances.numberOfInstances == 0
            && summary.skippedFiles.size() == 2
            && summary.skippedFiles[0].find("c.mis") != std::string::npos
            && summary.skippedFiles[1].find("b.sb.meta") != std::string::npos,
        "Unreadable files are skipped and reported with their names");
    check(inspectMISDirectory(samples, summary, error)
            && summary.skippedFiles.empty(),
        "Nothing is skipped in a valid directory");

    check(!inspectMISDirectory(workDir / "missing", summary, error),
        "Missing directory is reported");

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All dataset inspection tests passed.\n";
    return 0;
}
