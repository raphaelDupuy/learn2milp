/*
 * Tests of the MIS generation service: file names, manifest, progress
 * reports, validation before any file is written, and the guarantees on
 * existing and partial files. The mis-generator command line tool is a
 * front end of this service (see test_mis_generator*.sh).
 */

#include "services/MISGenerationService.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "io/MISInstanceIO.hpp"

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

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}

std::vector<std::string> readLines(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::vector<std::string> lines;
    std::string line;

    while (std::getline(file, line))
        lines.push_back(line);

    return lines;
}

std::size_t countFiles(
    const std::filesystem::path& directory,
    const std::string& extension
)
{
    if (!std::filesystem::exists(directory))
        return 0;

    std::size_t count = 0;

    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if (entry.path().extension() == extension)
            ++count;
    }

    return count;
}

void testErdosRenyiBatch(const std::filesystem::path& workDir)
{
    MISBatchConfig config;
    config.nValues = {10, 12};
    config.pValues = {0.1, 0.2, 0.05};
    config.seeds = {1, 2};
    config.outputDirectory = workDir / "erdos_renyi";

    std::vector<MISBatchProgress> reports;
    MISBatchResult result;
    std::string error;

    check(
        generateBatchMIS(
            config, result, error,
            [&reports](const MISBatchProgress& progress) {
                reports.push_back(progress);
            }),
        "Erdos-Renyi batch is generated: " + error
    );

    // Names use the shortest text of each probability.
    bool named = true;

    for (const char* expected : {
            "erdos_renyi_n10_p0_1_seed1",
            "erdos_renyi_n10_p0_2_seed2",
            "erdos_renyi_n12_p0_1_seed2",
            "erdos_renyi_n12_p0_2_seed1",
            "erdos_renyi_n12_p0_05_seed1"})
    {
        const auto graph =
            config.outputDirectory / (std::string(expected) + ".mis");

        named = named
            && std::filesystem::exists(graph)
            && std::filesystem::exists(graph.string() + ".meta");
    }

    check(named, "Instances are named after n, p and the seed");

    check(
        result.numberOfInstances == 12
        && countFiles(config.outputDirectory, ".mis") == 12
        && result.manifestPath == config.outputDirectory / "manifest.csv",
        "Every (n, p, seed) combination is generated"
    );

    const auto manifest = readLines(result.manifestPath);

    check(
        manifest.size() == 13
        && manifest[0]
            == "instance_id,file,generator_type,generator_version,seed,n,p,m"
        && manifest[1].rfind(
            "erdos_renyi_n10_p0_1_seed1,erdos_renyi_n10_p0_1_seed1.mis,"
            "erdos_renyi,1.0,1,10,0.10000000000000001,", 0) == 0,
        "Manifest lists the instances in generation order"
    );

    bool reported = reports.size() == 12;

    for (std::size_t i = 0; reported && i < reports.size(); ++i)
    {
        reported = reports[i].index == i + 1 && reports[i].total == 12;
    }

    check(
        reported
        && reports.front().fileName == "erdos_renyi_n10_p0_1_seed1.mis"
        && manifest[1].substr(manifest[1].rfind(',') + 1)
            == std::to_string(reports.front().numberOfEdges),
        "Progress is reported once per instance, with its edge count"
    );

    // The same batch elsewhere is identical, byte for byte.
    MISBatchConfig copy = config;
    copy.outputDirectory = workDir / "erdos_renyi_copy";

    check(
        generateBatchMIS(copy, result, error)
        && readFile(copy.outputDirectory / "manifest.csv")
            == readFile(config.outputDirectory / "manifest.csv")
        && readFile(copy.outputDirectory / "erdos_renyi_n12_p0_2_seed1.mis")
            == readFile(
                config.outputDirectory / "erdos_renyi_n12_p0_2_seed1.mis"),
        "A batch is reproducible"
    );

    // A complete batch is never touched again.
    const std::string before = readFile(result.manifestPath);

    check(
        !generateBatchMIS(config, result, error)
        && readFile(config.outputDirectory / "manifest.csv") == before,
        "A directory with a manifest is refused"
    );
}

void testBarabasiAlbertBatch(const std::filesystem::path& workDir)
{
    MISBatchConfig config;
    config.family = MISGraphFamily::BarabasiAlbert;
    config.nValues = {20, 30};
    config.kValues = {2, 3};
    config.seeds = {1, 2};
    config.outputDirectory = workDir / "barabasi_albert";

    MISBatchResult result;
    std::string error;

    check(
        generateBatchMIS(config, result, error),
        "Barabasi-Albert batch is generated: " + error
    );

    const auto manifest = readLines(result.manifestPath);

    check(
        result.numberOfInstances == 8
        && manifest.size() == 9
        && manifest[0]
            == "instance_id,file,generator_type,generator_version,seed,n,k,m"
        && manifest[1]
            == "barabasi_albert_n20_k2_seed1,barabasi_albert_n20_k2_seed1.mis,"
               "barabasi_albert,1.0,1,20,2,36"
        && manifest[8]
            == "barabasi_albert_n30_k3_seed2,barabasi_albert_n30_k3_seed2.mis,"
               "barabasi_albert,1.0,2,30,3,81",
        "Barabasi-Albert manifest lists n, k and the k (n - k) edges"
    );

    MISInstance instance;
    std::ifstream file(config.outputDirectory / "barabasi_albert_n30_k3_seed2.mis");
    std::string readError;

    check(
        readMISInstance(file, instance, readError)
        && instance.n == 30 && instance.edges.size() == 81,
        "Barabasi-Albert instances are readable"
    );
}

void testBatchValidation(const std::filesystem::path& workDir)
{
    MISBatchResult result;
    std::string error;

    const auto refused = [&](MISBatchConfig config, const std::string& name)
    {
        config.outputDirectory = workDir / ("refused_" + name);

        check(
            !generateBatchMIS(config, result, error) && !error.empty()
            && !std::filesystem::exists(config.outputDirectory),
            "Refused before anything is written: " + name
        );
    };

    MISBatchConfig erdosRenyi;
    erdosRenyi.nValues = {10};
    erdosRenyi.pValues = {0.1};
    erdosRenyi.seeds = {1};

    MISBatchConfig barabasiAlbert;
    barabasiAlbert.family = MISGraphFamily::BarabasiAlbert;
    barabasiAlbert.nValues = {10, 20};
    barabasiAlbert.kValues = {2};
    barabasiAlbert.seeds = {1};

    MISBatchConfig config = erdosRenyi;
    config.kValues = {2};
    refused(config, "k values in an Erdos-Renyi batch");

    config = barabasiAlbert;
    config.pValues = {0.1};
    refused(config, "p values in a Barabasi-Albert batch");

    config = barabasiAlbert;
    config.kValues.clear();
    refused(config, "Barabasi-Albert batch without k");

    config = barabasiAlbert;
    config.kValues = {2, 15};
    refused(config, "k not smaller than one of the n");

    config = erdosRenyi;
    config.seeds = {1, 1};
    refused(config, "duplicate seed");

    config = erdosRenyi;
    config.pValues = {1.5};
    refused(config, "probability above 1");

    config = erdosRenyi;
    config.outputDirectory.clear();

    check(
        !generateBatchMIS(config, result, error),
        "Batch without output directory is refused"
    );

    // An instance already present is preserved, and nothing is added.
    const auto occupied = workDir / "occupied";
    std::filesystem::create_directories(occupied);
    {
        std::ofstream(occupied / "erdos_renyi_n10_p0_1_seed1.mis") << "keep\n";
    }

    config = erdosRenyi;
    config.nValues = {8, 10};
    config.outputDirectory = occupied;

    check(
        !generateBatchMIS(config, result, error)
        && readFile(occupied / "erdos_renyi_n10_p0_1_seed1.mis") == "keep\n"
        && countFiles(occupied, ".mis") == 1
        && !std::filesystem::exists(occupied / "manifest.csv"),
        "Existing instance files are never overwritten"
    );
}

void testSingle(const std::filesystem::path& workDir)
{
    std::filesystem::create_directories(workDir / "single");

    const auto path = workDir / "single" / "graph.mis";
    MISSingleGenerationResult result;
    std::string error;

    check(
        generateSingleMIS(BarabasiAlbertConfig{50, 4, 9}, path, result, error)
        && result.graphPath == path
        && result.metadataPath == workDir / "single" / "graph.mis.meta"
        && result.numberOfVertices == 50
        && result.numberOfEdges == 184
        && result.seed == 9
        && readFile(result.metadataPath).find(
            "generator_type \"barabasi_albert\"") != std::string::npos,
        "Single instance is generated and described: " + error
    );

    const std::string before = readFile(path);

    check(
        !generateSingleMIS(ErdosRenyiConfig{10, 0.5, 1}, path, result, error)
        && readFile(path) == before,
        "Single generation never overwrites a file"
    );

    const auto missing = workDir / "missing_directory" / "graph.mis";

    check(
        !generateSingleMIS(ErdosRenyiConfig{10, 0.5, 1}, missing, result, error)
        && !std::filesystem::exists(missing)
        && !std::filesystem::exists(missing.string() + ".meta"),
        "A failed generation leaves no file behind"
    );

    const auto invalid = workDir / "single" / "invalid.mis";

    check(
        !generateSingleMIS(BarabasiAlbertConfig{5, 5, 1}, invalid, result, error)
        && !std::filesystem::exists(invalid),
        "Invalid parameters are refused"
    );
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
    std::filesystem::create_directories(workDir);

    testErdosRenyiBatch(workDir);
    testBarabasiAlbertBatch(workDir);
    testBatchValidation(workDir);
    testSingle(workDir);

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS generation service tests passed.\n";
    return 0;
}
