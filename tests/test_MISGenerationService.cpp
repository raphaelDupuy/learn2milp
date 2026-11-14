/*
 * The batch generation service must name instances exactly like the
 * mis-generator command line tool (see test_mis_generator_batch.sh): the
 * same instance must not have two different names.
 */

#include "services/MISGenerationService.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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

    MISBatchConfig config;
    config.nValues = {10, 12};
    config.pValues = {0.1, 0.2, 0.05};
    config.seeds = {1, 2};
    config.outputDirectory = workDir;

    std::string error;

    check(generateBatchMIS(config, error), "Batch is generated: " + error);

    // Same names as the command line tool, with the shortest text of
    // each probability.
    bool named = true;

    for (const char* expected : {
            "erdos_renyi_n10_p0_1_seed1",
            "erdos_renyi_n10_p0_2_seed2",
            "erdos_renyi_n12_p0_1_seed2",
            "erdos_renyi_n12_p0_2_seed1",
            "erdos_renyi_n12_p0_05_seed1"})
    {
        const auto graph = workDir / (std::string(expected) + ".mis");

        named = named
            && std::filesystem::exists(graph)
            && std::filesystem::exists(graph.string() + ".meta");
    }

    check(named, "Instances are named after n, p and the seed");

    const std::string manifest = readFile(workDir / "manifest.csv");

    check(
        manifest.find(
            "\nerdos_renyi_n10_p0_1_seed1,erdos_renyi_n10_p0_1_seed1.mis,")
            != std::string::npos,
        "Manifest lists the same names"
    );

    std::size_t instances = 0;

    for (const auto& entry : std::filesystem::directory_iterator(workDir))
    {
        if (entry.path().extension() == ".mis")
            ++instances;
    }

    check(instances == 12, "Every (n, p, seed) combination is generated");

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS generation service tests passed.\n";
    return 0;
}
