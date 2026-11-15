#include "io/MISBranchingModelIO.hpp"
#include "services/MISBenchmarkService.hpp"
#include "services/MISInstanceLoadingService.hpp"
#include "tools/CommandLine.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace
{
void printUsage(const char* program, std::ostream& out = std::cerr)
{
    out
        << "Usage:\n"
        << "  " << program
        << " --instances DIR --rules R1,R2 --output PATH\n"
        << "      [--seeds S1,S2] [--settings study|default]"
        << " [--time-limit SECONDS]\n"
        << "      [--model PATH]\n\n"
        << "Solves every .mis instance of a directory with each branching\n"
        << "rule and each SCIP seed, one run at a time, and writes one CSV\n"
        << "row per run.\n\n"
        << "Options:\n"
        << "  --instances   Directory containing the .mis instances\n"
        << "  --rules       Comma-separated branching rules: SCIP rules such\n"
        << "                as relpscost, pscost, vanillafullstrong, or\n"
        << "                misfullstrong (the data collection expert),\n"
        << "                or mislearned (the rule driven by --model)\n"
        << "  --model       Model file of the mislearned rule\n"
        << "  --output      Path of the result table (never overwritten)\n"
        << "  --seeds       Comma-separated SCIP seed shifts (default 0)\n"
        << "  --settings    study: settings used for data collection"
        << " (default)\n"
        << "                default: SCIP with no parameter changed\n"
        << "  --time-limit  Time limit per run in seconds (default 3600)\n"
        << "  -h, --help    Show this help and exit\n";
}

struct RuleSummary
{
    std::size_t runs = 0;
    std::size_t solved = 0;
    std::vector<double> nodes;
    std::vector<double> times;
};
}

int main(int argc, char* argv[])
{
    if (isHelpRequested(argc, argv))
    {
        printUsage(argv[0], std::cout);
        return 0;
    }

    std::string instanceDirectory;
    std::string outputPath;
    std::vector<std::string> rules;
    std::vector<unsigned int> seeds{0};
    MISBenchmarkSettings settings = MISBenchmarkSettings::Study;
    double timeLimit = 3600.0;
    std::string modelPath;

    try
    {
        const CommandLineOptions options(argc, argv, 1, {
            "--instances", "--rules", "--output", "--seeds",
            "--settings", "--model", "--time-limit"});

        instanceDirectory = options.text("--instances");
        rules = options.list<std::string>("--rules", parseText);
        outputPath = options.text("--output");
        settings = options.get(
            "--settings", parseMISBenchmarkSettings, settings);
        timeLimit = options.get("--time-limit", parsePositiveDouble, timeLimit);

        if (options.has("--seeds"))
            seeds = options.list<unsigned int>("--seeds", parseSCIPSeed);

        if (options.has("--model"))
            modelPath = options.text("--model");
    }
    catch (const CommandLineError& error)
    {
        return reportCommandLineError(argv[0], error);
    }

    for (const std::string& rule : rules)
    {
        if (!isMISBenchmarkRuleAvailable(rule))
        {
            std::cerr << "Unknown branching rule: " << rule << '\n';
            return 2;
        }
    }

    const bool usesModel =
        std::find(rules.begin(), rules.end(), "mislearned") != rules.end();

    if (usesModel != !modelPath.empty())
    {
        std::cerr
            << (usesModel
                ? "The mislearned rule needs --model.\n"
                : "--model is only used by the mislearned rule.\n");
        return 2;
    }

    if (usesModel)
    {
        LoadedMISBranchingModel model;
        std::string modelError;

        if (!loadMISBranchingModel(modelPath, model, modelError))
        {
            std::cerr << modelError << '\n';
            return 1;
        }
    }

    // Instances, in a fixed order.
    std::vector<std::filesystem::path> instancePaths;
    std::error_code directoryError;

    for (const auto& entry : std::filesystem::directory_iterator(
            instanceDirectory, directoryError))
    {
        if (entry.path().extension() == ".mis")
        {
            instancePaths.push_back(entry.path());
        }
    }

    if (directoryError)
    {
        std::cerr << "Cannot read instance directory: "
                  << instanceDirectory << '\n';
        return 1;
    }

    if (instancePaths.empty())
    {
        std::cerr << "No .mis instance found in: "
                  << instanceDirectory << '\n';
        return 1;
    }

    std::sort(instancePaths.begin(), instancePaths.end());

    if (std::filesystem::exists(outputPath))
    {
        std::cerr << "Output file already exists: " << outputPath << '\n';
        return 1;
    }

    std::ofstream output(outputPath);

    if (!output)
    {
        std::cerr << "Cannot open output file: " << outputPath << '\n';
        return 1;
    }

    output
        << "instance,n,m,settings,rule,seed,status,solved,nodes,"
        << "lp_iterations,time_seconds,primal_bound,dual_bound\n";

    const std::size_t totalRuns =
        instancePaths.size() * seeds.size() * rules.size();

    std::map<std::string, RuleSummary> summaries;
    std::size_t run = 0;

    // Rules are interleaved on each instance, so that a drift of the
    // machine load over time affects every rule alike.
    for (const auto& instancePath : instancePaths)
    {
        std::string errorMessage;
        LoadedMISInstance loaded;

        if (!loadMISInstance(instancePath, loaded, errorMessage))
        {
            std::cerr << errorMessage << '\n';
            return 1;
        }

        const std::string name = instancePath.filename().string();

        for (const unsigned int seed : seeds)
        {
            for (const std::string& rule : rules)
            {
                MISBenchmarkConfig config;
                config.branchingRule = rule;
                config.settings = settings;
                config.timeLimitSeconds = timeLimit;
                config.scipSeed = seed;

                if (rule == "mislearned")
                {
                    config.learnedModelPath = modelPath;
                }

                MISBenchmarkResult result;

                if (!benchmarkMIS(
                        loaded.instance, config, result, errorMessage))
                {
                    std::cerr << "Run failed (" << name << ", " << rule
                              << ", seed " << seed << "): "
                              << errorMessage << '\n'
                              << "The result table is incomplete: "
                              << outputPath << '\n';
                    return 1;
                }

                output
                    << name << ','
                    << loaded.instance.n << ','
                    << loaded.instance.edges.size() << ','
                    << misBenchmarkSettingsName(settings) << ','
                    << rule << ','
                    << seed << ','
                    << result.status << ','
                    << (result.solved ? 1 : 0) << ','
                    << result.exploredNodes << ','
                    << result.lpIterations << ','
                    << result.solvingTimeSeconds << ',';

                if (result.hasSolution)
                {
                    output << result.primalBound;
                }

                output << ',' << result.dualBound << '\n';
                output.flush();

                if (!output)
                {
                    std::cerr << "Failed to write: " << outputPath << '\n';
                    return 1;
                }

                RuleSummary& summary = summaries[rule];
                ++summary.runs;
                summary.solved += result.solved ? 1 : 0;
                summary.nodes.push_back(
                    static_cast<double>(result.exploredNodes));
                summary.times.push_back(result.solvingTimeSeconds);

                ++run;

                std::cout
                    << '[' << run << '/' << totalRuns << "] "
                    << name << ' ' << rule << " seed " << seed << ": "
                    << result.status << ", "
                    << result.exploredNodes << " nodes, "
                    << result.solvingTimeSeconds << " s\n";
            }
        }
    }

    // Summary, in the order the rules were given.
    std::cout
        << "\nSettings: " << misBenchmarkSettingsName(settings)
        << ", " << instancePaths.size() << " instances, "
        << seeds.size() << " seed(s), time limit " << timeLimit << " s\n"
        << "Shifted geometric means: nodes (shift 100),"
        << " time in seconds (shift 1)\n\n"
        << std::left << std::setw(22) << "rule"
        << std::right << std::setw(10) << "solved"
        << std::setw(12) << "nodes"
        << std::setw(12) << "time" << '\n';

    for (const std::string& rule : rules)
    {
        const RuleSummary& summary = summaries[rule];

        std::ostringstream solved;
        solved << summary.solved << '/' << summary.runs;

        std::cout
            << std::left << std::setw(22) << rule
            << std::right << std::setw(10) << solved.str()
            << std::setw(12) << std::fixed << std::setprecision(1)
            << misShiftedGeometricMean(summary.nodes, MIS_NODES_SHIFT)
            << std::setw(12) << std::setprecision(2)
            << misShiftedGeometricMean(summary.times, MIS_TIME_SHIFT_SECONDS) << '\n';
    }

    std::cout << "\nResults: " << outputPath << '\n';

    return 0;
}
