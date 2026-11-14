#include "services/MISCollectionService.hpp"
#include "services/MISInstanceLoadingService.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>

namespace
{
void printUsage(const char* program)
{
    std::cerr
        << "Usage:\n"
        << "  " << program
        << " --instance PATH --output-dir DIR"
        << " [--time-limit SECONDS] [--node-limit N] [--scip-seed S]\n\n"
        << "Solves one MIS instance with full strong branching and records\n"
        << "one sample per branching decision.\n\n"
        << "Options:\n"
        << "  --instance    Path of the .mis instance"
        << " (its .meta file must sit next to it)\n"
        << "  --output-dir  Directory receiving NAME.sb.nodes.csv,\n"
        << "                NAME.sb.candidates.csv and NAME.sb.meta\n"
        << "  --time-limit  Solving time limit in seconds (default 3600)\n"
        << "  --node-limit  Maximum number of explored nodes"
        << " (default: none)\n"
        << "  --scip-seed   SCIP random seed shift (default 0)\n";
}

bool parsePositiveDouble(const std::string& text, double& value)
{
    try
    {
        std::size_t position = 0;
        value = std::stod(text, &position);

        return position == text.size()
            && std::isfinite(value)
            && value > 0.0;
    }
    catch (...)
    {
        return false;
    }
}

bool parsePositiveInteger(const std::string& text, long long& value)
{
    try
    {
        std::size_t position = 0;
        value = std::stoll(text, &position);

        return position == text.size() && value > 0;
    }
    catch (...)
    {
        return false;
    }
}

bool parseSeed(const std::string& text, unsigned int& value)
{
    try
    {
        std::size_t position = 0;

        if (text.empty() || text.front() == '-')
        {
            return false;
        }

        const unsigned long long parsed = std::stoull(text, &position);

        if (position != text.size()
            || parsed > static_cast<unsigned long long>(
                std::numeric_limits<int>::max()))
        {
            return false;
        }

        value = static_cast<unsigned int>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}
}

int main(int argc, char* argv[])
{
    std::string instancePath;
    std::string outputDirectory;
    MISCollectionConfig config;

    bool hasInstance = false;
    bool hasOutputDirectory = false;
    bool hasTimeLimit = false;
    bool hasNodeLimit = false;
    bool hasSeed = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string option = argv[i];

        if (i + 1 >= argc)
        {
            std::cerr << "Missing value for option: " << option << '\n';
            return 2;
        }

        const std::string value = argv[++i];

        if (option == "--instance")
        {
            if (hasInstance || value.empty())
            {
                std::cerr << "Invalid or duplicate --instance value.\n";
                return 2;
            }

            instancePath = value;
            hasInstance = true;
        }
        else if (option == "--output-dir")
        {
            if (hasOutputDirectory || value.empty())
            {
                std::cerr << "Invalid or duplicate --output-dir value.\n";
                return 2;
            }

            outputDirectory = value;
            hasOutputDirectory = true;
        }
        else if (option == "--time-limit")
        {
            if (hasTimeLimit
                || !parsePositiveDouble(value, config.timeLimitSeconds))
            {
                std::cerr << "Invalid or duplicate --time-limit value.\n";
                return 2;
            }

            hasTimeLimit = true;
        }
        else if (option == "--node-limit")
        {
            if (hasNodeLimit
                || !parsePositiveInteger(value, config.nodeLimit))
            {
                std::cerr << "Invalid or duplicate --node-limit value.\n";
                return 2;
            }

            hasNodeLimit = true;
        }
        else if (option == "--scip-seed")
        {
            if (hasSeed || !parseSeed(value, config.scipSeed))
            {
                std::cerr << "Invalid or duplicate --scip-seed value.\n";
                return 2;
            }

            hasSeed = true;
        }
        else
        {
            std::cerr << "Unknown option: " << option << '\n';
            return 2;
        }
    }

    if (!hasInstance || !hasOutputDirectory)
    {
        printUsage(argv[0]);
        return 2;
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
        << "  status: " << result.status
        << (result.complete ? "" : " (truncated search tree)") << '\n'
        << "  explored nodes: " << result.exploredNodes << '\n'
        << "  samples: " << result.numberOfSamples << '\n'
        << "  candidates: " << result.numberOfCandidates << '\n'
        << "  solving time (s): " << result.solvingTimeSeconds << '\n'
        << "  nodes table: " << result.nodesPath.string() << '\n'
        << "  candidates table: " << result.candidatesPath.string() << '\n'
        << "  metadata: " << result.metadataPath.string() << '\n';

    return 0;
}
