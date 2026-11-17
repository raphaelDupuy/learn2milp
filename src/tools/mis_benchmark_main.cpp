#include "io/MISInstanceIO.hpp"
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
        << "      [--parameters FILE] [--cut-selector NAME] [--model PATH]\n"
        << "      [--strong-branching K]\n\n"
        << "Solves every .mis instance of a directory with each branching\n"
        << "rule and each SCIP seed, one run at a time, and writes one CSV\n"
        << "row per run.\n\n"
        << "Options:\n"
        << "  --instances   Directory containing the .mis instances\n"
        << "  --rules       Comma-separated branching rules: SCIP rules such\n"
        << "                as relpscost, pscost, vanillafullstrong, or\n"
        << "                misfullstrong (the data collection expert),\n"
        << "                misfullstrong_nocutoff (the expert without its\n"
        << "                strong branching pruning),\n"
        << "                or mislearned (the rule driven by --model)\n"
        << "  --model       Model file of the mislearned rule\n"
        << "  --strong-branching\n"
        << "                Candidates of the mislearned rule evaluated by\n"
        << "                strong branching, those the model ranks first\n"
        << "                (default 0: branch on the model's best)\n"
        << "  --output      Path of the result table (never overwritten)\n"
        << "  --seeds       Comma-separated SCIP seed shifts (default 0)\n"
        << "  --settings    study: settings used for data collection"
        << " (default)\n"
        << "                default: SCIP with no parameter changed\n"
        << "  --parameters  SCIP settings file (name = value per line)\n"
        << "                applied after --settings, e.g. to choose the\n"
        << "                separators called at the nodes\n"
        << "  --cut-selector\n"
        << "                lookahead: at the root, cuts ranked by the LP\n"
        << "                bound increase of each cut added alone\n"
        << "                (default: SCIP's own selector)\n"
        << "  --time-limit  Time limit per run in seconds (default "
        << MIS_DEFAULT_TIME_LIMIT_SECONDS << ")\n"
        << "  -h, --help    Show this help and exit\n";
}

} // namespace

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
    double timeLimit = MIS_DEFAULT_TIME_LIMIT_SECONDS;
    std::string modelPath;
    std::string parameterFile;
    std::string cutSelector;
    int strongBranching = 0;

    try
    {
        const CommandLineOptions options(argc, argv, 1, {
            "--instances", "--rules", "--output", "--seeds",
            "--settings", "--parameters", "--cut-selector", "--model",
            "--time-limit", "--strong-branching"});

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

        if (options.has("--parameters"))
            parameterFile = options.text("--parameters");

        if (options.has("--cut-selector"))
            cutSelector = options.text("--cut-selector");

        strongBranching = options.get(
            "--strong-branching", parseNonNegativeInt, strongBranching);
    }
    catch (const CommandLineError& error)
    {
        return reportCommandLineError(argv[0], error);
    }

    // Every rule is checked, and the model loaded, before any run.
    const bool usesModel =
        std::find(rules.begin(), rules.end(), "mislearned") != rules.end();

    if (!modelPath.empty() && !usesModel)
    {
        std::cerr << "--model is only used by the mislearned rule.\n";
        return 2;
    }

    if (strongBranching > 0 && !usesModel)
    {
        std::cerr << "--strong-branching is only used by the mislearned rule.\n";
        return 2;
    }

    for (const std::string& rule : rules)
    {
        std::string ruleError;

        if (!validateMISBenchmarkRule(
                rule, rule == "mislearned" ? modelPath : "", ruleError))
        {
            std::cerr << ruleError << '\n';
            return 2;
        }
    }

    if (!parameterFile.empty())
    {
        std::string parameterError;

        if (!validateMISBenchmarkParameters(parameterFile, parameterError))
        {
            std::cerr << parameterError << '\n';
            return 2;
        }
    }

    if (!isMISBenchmarkCutSelectorAvailable(cutSelector))
    {
        std::cerr << "Unknown cut selector: " << cutSelector << '\n';
        return 2;
    }

    const std::string settingsLabel =
        misBenchmarkSettingsLabel(settings, parameterFile, cutSelector);

    // Instances, in a fixed order.
    std::vector<std::filesystem::path> instancePaths;

    try
    {
        instancePaths = listMISInstances(instanceDirectory);
    }
    catch (const std::filesystem::filesystem_error&)
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
        << "lp_iterations,time_seconds,primal_bound,dual_bound,"
        << "root_first_lp_bound,root_dual_bound,cuts_applied,"
        << "lookahead_scored,lookahead_improving\n";

    const std::size_t totalRuns =
        instancePaths.size() * seeds.size() * rules.size();

    std::map<std::string, MISRuleSummary> summaries;
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
                config.parameterFile = parameterFile;
                config.cutSelector = cutSelector;
                config.timeLimitSeconds = timeLimit;
                config.scipSeed = seed;

                if (rule == "mislearned")
                {
                    config.learnedModelPath = modelPath;
                    config.learnedStrongBranchingCandidates = strongBranching;
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
                    << settingsLabel << ','
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

                output
                    << ',' << result.dualBound
                    << ',' << result.rootFirstLPBound
                    << ',' << result.rootDualBound
                    << ',' << result.cutsApplied
                    << ',' << result.lookaheadScored
                    << ',' << result.lookaheadImproving << '\n';
                output.flush();

                if (!output)
                {
                    std::cerr << "Failed to write: " << outputPath << '\n';
                    return 1;
                }

                summaries[rule].add(result);

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
        << "\nSettings: " << settingsLabel
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
        const MISRuleSummary& summary = summaries[rule];

        std::ostringstream solved;
        solved << summary.solved << '/' << summary.runs;

        std::cout
            << std::left << std::setw(22) << rule
            << std::right << std::setw(10) << solved.str()
            << std::setw(12) << std::fixed << std::setprecision(1)
            << summary.meanNodes()
            << std::setw(12) << std::setprecision(2)
            << summary.meanTimeSeconds() << '\n';
    }

    std::cout << "\nResults: " << outputPath << '\n';

    return 0;
}
