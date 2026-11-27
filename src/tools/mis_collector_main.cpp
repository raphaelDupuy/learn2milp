#include "services/MISCollectionService.hpp"
#include "services/MISInstanceLoadingService.hpp"
#include "tools/CommandLine.hpp"

#include <filesystem>
#include <iostream>
#include <ostream>
#include <string>
#include <system_error>

namespace
{
void printUsage(const char* program, std::ostream& out = std::cerr)
{
    out
        << "Usage:\n"
        << "  " << program
        << " --instance PATH --output-dir DIR"
        << " [--time-limit SECONDS] [--node-limit N] [--scip-seed S]"
        << " [--policy-model PATH]\n"
        << "  [--problem mis|dominating]\n\n"
        << "Solves one instance with full strong branching and records\n"
        << "one sample per branching decision.\n\n"
        << "Options:\n"
        << "  --instance    Path of the .mis instance"
        << " (its .meta file must sit next to it)\n"
        << "  --output-dir  Directory receiving NAME.sb.nodes.csv,\n"
        << "                NAME.sb.candidates.csv and NAME.sb.meta\n"
        << "  --time-limit  Solving time limit in seconds (default "
        << MIS_DEFAULT_TIME_LIMIT_SECONDS << ")\n"
        << "  --node-limit  Maximum number of explored nodes"
        << " (default: none)\n"
        << "  --scip-seed   SCIP random seed shift (default 0)\n"
        << "  --problem     Problem posed on the graph: mis, maximum\n"
        << "                independent set (default), or dominating,\n"
        << "                minimum dominating set\n"
        << "  --policy-model  Branching model file: the search follows\n"
        << "                this model instead of the expert, and strong\n"
        << "                branching still labels every visited node\n"
        << "  -h, --help    Show this help and exit\n";
}
}

int main(int argc, char* argv[])
{
    if (isHelpRequested(argc, argv))
    {
        printUsage(argv[0], std::cout);
        return 0;
    }

    std::string instancePath;
    std::string outputDirectory;
    MISCollectionConfig config;

    try
    {
        const CommandLineOptions options(argc, argv, 1, {
            "--instance", "--output-dir", "--time-limit", "--node-limit",
            "--scip-seed", "--policy-model", "--problem"});

        instancePath = options.text("--instance");
        outputDirectory = options.text("--output-dir");
        config.timeLimitSeconds = options.get(
            "--time-limit", parsePositiveDouble, config.timeLimitSeconds);
        config.nodeLimit = options.get(
            "--node-limit", parsePositiveInteger, config.nodeLimit);
        config.scipSeed = options.get(
            "--scip-seed", parseSCIPSeed, config.scipSeed);
        config.problem = options.get(
            "--problem", parseGraphProblem, config.problem);

        if (options.has("--policy-model"))
            config.policyModelPath = options.text("--policy-model");
    }
    catch (const CommandLineError& error)
    {
        return reportCommandLineError(argv[0], error);
    }

    std::string errorMessage;
    LoadedMISInstance loaded;

    if (!loadMISInstance(instancePath, loaded, errorMessage))
    {
        std::cerr << errorMessage << '\n';
        return 1;
    }

    const std::filesystem::path instanceFile(instancePath);

    std::error_code directoryError;
    std::filesystem::create_directories(outputDirectory, directoryError);

    if (directoryError)
    {
        std::cerr << "Cannot create output directory: "
                  << outputDirectory << '\n';
        return 1;
    }

    config.instanceName = instanceFile.filename().string();

    const std::filesystem::path outputPrefix =
        std::filesystem::path(outputDirectory) / instanceFile.stem();

    MISCollectionResult result;

    if (!collectMISStrongBranchingSamples(
            loaded.instance,
            config,
            outputPrefix,
            result,
            errorMessage))
    {
        std::cerr << "Collection failed: " << errorMessage << '\n';
        return 1;
    }

    std::cout
        << "Collected strong branching samples\n"
        << "  instance: " << config.instanceName << '\n'
        << "  problem: " << graphProblemName(config.problem) << '\n'
        << "  status: " << result.status
        << (result.complete ? "" : " (truncated search tree)") << '\n'
        << "  explored nodes: " << result.exploredNodes << '\n'
        << "  samples: " << result.numberOfSamples << '\n'
        << "  candidates: " << result.numberOfCandidates << '\n';

    if (!config.policyModelPath.empty())
    {
        std::cout
            << "  policy decisions: " << result.policyDecisions << '\n'
            << "  agreeing with the expert: "
            << result.policyAgreements << '\n';
    }

    std::cout
        << "  solving time (s): " << result.solvingTimeSeconds << '\n'
        << "  nodes table: " << result.nodesPath.string() << '\n'
        << "  candidates table: " << result.candidatesPath.string() << '\n'
        << "  metadata: " << result.metadataPath.string() << '\n';

    return 0;
}
