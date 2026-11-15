/*
 * mis-generator: command line front end of the MIS generation service
 * (services/MISGenerationService.hpp). This file only reads the
 * arguments; generation, naming and file writing live in the service.
 *
 * Exit codes: 0 on success; for single, 2 for invalid arguments and 1 if
 * the generation fails; for batch, 2 for any failure.
 */

#include "generators/MISGenerator.hpp"
#include "services/MISGenerationService.hpp"
#include "tools/CommandLine.hpp"

#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void printUsage(const char* program, std::ostream& out = std::cerr)
{
    out
        << "Usage:\n"
        << "  " << program
        << " single [--family F] --n N (--p P | --k K) --seed S"
        << " --output PATH\n"
        << "  " << program
        << " batch [--family F] --n-values N1,N2"
        << " (--p-values P1,P2 | --k-values K1,K2)"
        << " --seeds S1,S2 --output-dir DIR\n\n"
        << "Graph families (--family):\n"
        << "  erdos_renyi      G(n, p): every edge with probability p"
        << " (default)\n"
        << "  barabasi_albert  preferential attachment: every new vertex"
        << " is joined\n"
        << "                   to k existing ones (1 <= k < n)\n\n"
        << "Options for single:\n"
        << "  --n       Number of vertices (N >= 0)\n"
        << "  --p       Edge probability (0 <= P <= 1), erdos_renyi\n"
        << "  --k       Edges of each new vertex (K >= 1),"
        << " barabasi_albert\n"
        << "  --seed    Random seed (unsigned integer)\n"
        << "  --output  Output path for the .mis instance\n\n"
        << "Options for batch:\n"
        << "  --n-values   Comma-separated vertex counts\n"
        << "  --p-values   Comma-separated edge probabilities"
        << " (erdos_renyi)\n"
        << "  --k-values   Comma-separated values of k"
        << " (barabasi_albert)\n"
        << "  --seeds      Comma-separated random seeds\n"
        << "  --output-dir Output directory for the batch\n\n"
        << "Existing files are never overwritten.\n"
        << "-h or --help shows this help.\n";
}

// The option that carries the family parameter, and the other one.
const char* familyOption(MISGraphFamily family, bool list)
{
    if (family == MISGraphFamily::ErdosRenyi)
        return list ? "--p-values" : "--p";

    return list ? "--k-values" : "--k";
}

const char* otherFamilyOption(MISGraphFamily family, bool list)
{
    if (family == MISGraphFamily::ErdosRenyi)
        return list ? "--k-values" : "--k";

    return list ? "--p-values" : "--p";
}

// Requires the option of the family parameter and refuses the other one.
void requireFamilyOptions(
    const CommandLineOptions& options,
    MISGraphFamily family,
    bool list
)
{
    if (options.has(otherFamilyOption(family, list)))
    {
        throw CommandLineError(
            std::string("Option ") + otherFamilyOption(family, list)
            + " is not a parameter of " + misGraphFamilyName(family)
            + " graphs."
        );
    }

    if (!options.has(familyOption(family, list)))
    {
        throw CommandLineError(
            std::string("Missing required option: ")
            + familyOption(family, list)
        );
    }
}

int runSingle(int argc, char* argv[])
{
    MISGraphFamily family = MISGraphFamily::ErdosRenyi;
    int n = 0;
    double p = 0.0;
    int k = 0;
    unsigned int seed = 0;
    std::string outputPath;

    try
    {
        const CommandLineOptions options(argc, argv, 2, {
            "--family", "--n", "--p", "--k", "--seed", "--output"});

        family = options.get("--family", parseMISGraphFamily, family);
        requireFamilyOptions(options, family, false);

        n = options.get<int>("--n", parseNonNegativeInt);
        seed = options.get<unsigned int>("--seed", parseSeed);
        outputPath = options.text("--output");

        if (family == MISGraphFamily::ErdosRenyi)
            p = options.get<double>("--p", parseProbability);
        else
            k = options.get<int>("--k", parsePositiveInt);
    }
    catch (const CommandLineError& error)
    {
        return reportCommandLineError(argv[0], error);
    }

    const bool erdosRenyi = family == MISGraphFamily::ErdosRenyi;

    const MISGeneratorConfig config = erdosRenyi
        ? MISGeneratorConfig{ErdosRenyiConfig{n, p, seed}}
        : MISGeneratorConfig{BarabasiAlbertConfig{n, k, seed}};

    try
    {
        validateMISGeneratorConfig(config);
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Invalid parameters: " << exception.what() << '\n';
        return 2;
    }

    MISSingleGenerationResult result;
    std::string errorMessage;

    if (!generateSingleMIS(config, outputPath, result, errorMessage))
    {
        std::cerr << "Generation failed: " << errorMessage << '\n';
        return 1;
    }

    std::cout
        << "Generated MIS instance\n"
        << "  vertices: " << result.numberOfVertices << '\n'
        << "  edges: " << result.numberOfEdges << '\n'
        << "  seed: " << result.seed << '\n'
        << "  graph: " << result.graphPath.string() << '\n'
        << "  metadata: " << result.metadataPath.string() << '\n';

    return 0;
}

int runBatch(int argc, char* argv[])
{
    MISBatchConfig config;

    try
    {
        const CommandLineOptions options(argc, argv, 2, {
            "--family", "--n-values", "--p-values", "--k-values",
            "--seeds", "--output-dir"});

        config.family = options.get(
            "--family", parseMISGraphFamily, config.family);
        requireFamilyOptions(options, config.family, true);

        config.nValues = options.list<int>("--n-values", parseNonNegativeInt);
        config.seeds = options.list<unsigned int>("--seeds", parseSeed);
        config.outputDirectory = options.text("--output-dir");

        if (config.family == MISGraphFamily::ErdosRenyi)
            config.pValues = options.list<double>("--p-values", parseProbability);
        else
            config.kValues = options.list<int>("--k-values", parsePositiveInt);
    }
    catch (const CommandLineError& error)
    {
        return reportCommandLineError(argv[0], error);
    }

    MISBatchResult result;
    std::string errorMessage;

    const bool generated = generateBatchMIS(
        config,
        result,
        errorMessage,
        [](const MISBatchProgress& progress) {
            std::cout
                << '[' << progress.index << '/' << progress.total << "] "
                << progress.fileName
                << " (m=" << progress.numberOfEdges << ")\n";
        }
    );

    if (!generated)
    {
        throw std::runtime_error(errorMessage);
    }

    std::cout
        << "\nBatch completed: " << result.numberOfInstances
        << " instances\n"
        << "Output directory: " << config.outputDirectory.string() << '\n'
        << "Manifest: " << result.manifestPath.string() << '\n';

    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    if (isHelpRequested(argc, argv))
    {
        printUsage(argv[0], std::cout);
        return 0;
    }

    if (argc < 2)
    {
        printUsage(argv[0]);
        return 2;
    }

    const std::string command = argv[1];

    if (command == "single")
    {
        return runSingle(argc, argv);
    }

    if (command == "batch")
    {
        try
        {
            return runBatch(argc, argv);
        }
        catch (const std::exception& exception)
        {
            std::cerr << "Batch generation failed: "
                      << exception.what() << '\n';
            return 2;
        }
    }

    printUsage(argv[0]);
    return 2;
}
