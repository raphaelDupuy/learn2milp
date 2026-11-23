/*
 * jssp-generator: command line front end of the JSSP generation service
 * (services/JSSPGenerationService.hpp). This file only reads the
 * arguments; generation, naming and file writing live in the service.
 *
 * Exit codes: 0 on success; for single, 2 for invalid arguments and 1 if
 * the generation fails; for batch, 2 for any failure.
 */

#include "generators/TaillardGenerator.hpp"
#include "services/JSSPGenerationService.hpp"
#include "tools/CommandLine.hpp"

#include <iostream>
#include <ostream>
#include <string>

namespace
{

void printUsage(const char* program, std::ostream& out = std::cerr)
{
    out
        << "Usage:\n"
        << "  " << program
        << " single --jobs J --machines M"
        << " (--seed S | --time-seed T --machine-seed U) --output PATH\n"
        << "  " << program
        << " batch --jobs-values J1,J2 --machines-values M1,M2"
        << " --seeds S1,S2 --output-dir DIR\n\n"
        << "Instances as in Taillard (1993): every job visits every machine"
        << " once, in a\n"
        << "random order, with durations uniform in [1, 99], drawn with"
        << " Taillard's own\n"
        << "generator from a time seed and a machine seed.\n\n"
        << "Options for single:\n"
        << "  --jobs          Number of jobs (J >= 1)\n"
        << "  --machines      Number of machines (M >= 1)\n"
        << "  --seed          Seed the two seeds of Taillard's generator"
        << " are derived from\n"
        << "  --time-seed     Time seed of Taillard's generator, in"
        << " [1, 2147483646]\n"
        << "  --machine-seed  Machine seed, same range; with --time-seed,"
        << " the published\n"
        << "                  seeds give the benchmarks ta01 to ta80\n"
        << "  --output        Output path for the .jsp instance\n\n"
        << "Options for batch:\n"
        << "  --jobs-values      Comma-separated numbers of jobs\n"
        << "  --machines-values  Comma-separated numbers of machines\n"
        << "  --seeds            Comma-separated seeds (each derives the two"
        << " seeds)\n"
        << "  --output-dir       Output directory for the batch\n\n"
        << "Existing files are never overwritten.\n"
        << "-h or --help shows this help.\n";
}

bool parseTaillardSeed(const std::string& text, int& value)
{
    return parsePositiveInt(text, value) && value <= TAILLARD_SEED_MAX;
}

int runSingle(int argc, char* argv[])
{
    TaillardConfig config;
    std::string outputPath;

    try
    {
        const CommandLineOptions options(argc, argv, 2, {
            "--jobs", "--machines", "--seed", "--time-seed",
            "--machine-seed", "--output"});

        const int jobs = options.get<int>("--jobs", parsePositiveInt);
        const int machines = options.get<int>("--machines", parsePositiveInt);
        outputPath = options.text("--output");

        const bool derived = options.has("--seed");
        const bool given = options.has("--time-seed") || options.has("--machine-seed");

        if (derived == given)
        {
            throw CommandLineError(
                "Give either --seed, or both --time-seed and --machine-seed.");
        }

        if (derived)
        {
            config = taillardConfigFromSeed(
                jobs, machines, options.get<unsigned int>("--seed", parseSeed));
        }
        else
        {
            config.jobs = jobs;
            config.machines = machines;
            config.timeSeed = options.get<int>("--time-seed", parseTaillardSeed);
            config.machineSeed =
                options.get<int>("--machine-seed", parseTaillardSeed);
        }
    }
    catch (const CommandLineError& error)
    {
        return reportCommandLineError(argv[0], error);
    }

    JSSPSingleGenerationResult result;
    std::string errorMessage;

    if (!generateSingleJSSP(config, outputPath, result, errorMessage))
    {
        std::cerr << "Generation failed: " << errorMessage << '\n';
        return 1;
    }

    std::cout
        << "Generated JSSP instance\n"
        << "  jobs: " << result.jobs << '\n'
        << "  machines: " << result.machines << '\n'
        << "  time seed: " << config.timeSeed << '\n'
        << "  machine seed: " << config.machineSeed << '\n'
        << "  lower bound: " << result.lowerBound << '\n'
        << "  dispatch makespan: " << result.dispatchMakespan << '\n'
        << "  instance: " << result.instancePath.string() << '\n'
        << "  metadata: " << result.metadataPath.string() << '\n';

    return 0;
}

int runBatch(int argc, char* argv[])
{
    JSSPBatchConfig config;

    try
    {
        const CommandLineOptions options(argc, argv, 2, {
            "--jobs-values", "--machines-values", "--seeds", "--output-dir"});

        config.jobsValues = options.list<int>("--jobs-values", parsePositiveInt);
        config.machinesValues =
            options.list<int>("--machines-values", parsePositiveInt);
        config.seeds = options.list<unsigned int>("--seeds", parseSeed);
        config.outputDirectory = options.text("--output-dir");
    }
    catch (const CommandLineError& error)
    {
        return reportCommandLineError(argv[0], error);
    }

    JSSPBatchResult result;
    std::string errorMessage;

    const bool generated = generateBatchJSSP(
        config,
        result,
        errorMessage,
        [](const JSSPBatchProgress& progress) {
            std::cout
                << '[' << progress.index << '/' << progress.total << "] "
                << progress.fileName
                << " (lower bound " << progress.lowerBound
                << ", dispatch " << progress.dispatchMakespan << ")\n";
        });

    if (!generated)
    {
        std::cerr << "Batch generation failed: " << errorMessage << '\n';
        return 2;
    }

    std::cout
        << "\nBatch completed: " << result.numberOfInstances << " instances\n"
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
        return runSingle(argc, argv);

    if (command == "batch")
        return runBatch(argc, argv);

    printUsage(argv[0]);
    return 2;
}
