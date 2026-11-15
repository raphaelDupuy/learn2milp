#include "io/MISStrongBranchingSampleIO.hpp"
#include "services/MISFeaturizationService.hpp"
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
        << " --instance PATH --samples-dir DIR --output-dir DIR\n\n"
        << "Turns the strong branching samples collected on one instance\n"
        << "into a training table with one row per branching candidate.\n\n"
        << "Options:\n"
        << "  --instance     Path of the .mis instance\n"
        << "  --samples-dir  Directory containing NAME.sb.nodes.csv,\n"
        << "                 NAME.sb.candidates.csv and NAME.sb.meta\n"
        << "  --output-dir   Directory receiving NAME.sb.features.csv\n"
        << "  -h, --help     Show this help and exit\n";
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
    std::string samplesDirectory;
    std::string outputDirectory;

    try
    {
        const CommandLineOptions options(argc, argv, 1, {
            "--instance", "--samples-dir", "--output-dir"});

        instancePath = options.text("--instance");
        samplesDirectory = options.text("--samples-dir");
        outputDirectory = options.text("--output-dir");
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

    const std::string name =
        std::filesystem::path(instancePath).stem().string();
    const std::filesystem::path samplesPrefix =
        std::filesystem::path(samplesDirectory) / name;

    const std::filesystem::path metadataPath(
        samplesPrefix.string() + MIS_SB_METADATA_SUFFIX);

    // The metadata file is written last by the collection: without it,
    // the sample tables may be incomplete.
    if (!std::filesystem::exists(metadataPath))
    {
        std::cerr << "No complete collection found: "
                  << metadataPath.string() << " is missing.\n";
        return 1;
    }

    std::error_code directoryError;
    std::filesystem::create_directories(outputDirectory, directoryError);

    if (directoryError)
    {
        std::cerr << "Cannot create output directory: "
                  << outputDirectory << '\n';
        return 1;
    }

    const std::filesystem::path outputPath =
        std::filesystem::path(outputDirectory)
        / (name + MIS_SB_FEATURES_SUFFIX);

    MISFeaturizationResult result;

    if (!featurizeMISStrongBranchingSamples(
            loaded.instance,
            samplesPrefix.string() + MIS_SB_NODES_SUFFIX,
            samplesPrefix.string() + MIS_SB_CANDIDATES_SUFFIX,
            outputPath,
            result,
            errorMessage))
    {
        std::cerr << "Featurization failed: " << errorMessage << '\n';
        return 1;
    }

    std::cout
        << "Computed branching features\n"
        << "  instance: " << name << '\n'
        << "  samples: " << result.numberOfSamples
        << " (" << result.numberOfUsableSamples << " usable)\n"
        << "  candidates: " << result.numberOfCandidates << '\n'
        << "  feature table: " << outputPath.string() << '\n';

    return 0;
}
