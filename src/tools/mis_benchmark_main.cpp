#include "services/MISBenchmarkService.hpp"
#include "services/MISInstanceLoadingService.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace
{
void printUsage(const char* program)
{
    std::cerr
        << "Usage:\n"
        << "  " << program
        << " --instances DIR --rules R1,R2 --output PATH\n"
        << "      [--seeds S1,S2] [--settings study|default]"
        << " [--time-limit SECONDS]\n\n"
        << "Solves every .mis instance of a directory with each branching\n"
        << "rule and each SCIP seed, one run at a time, and writes one CSV\n"
        << "row per run.\n\n"
        << "Options:\n"
        << "  --instances   Directory containing the .mis instances\n"
        << "  --rules       Comma-separated branching rules: SCIP rules such\n"
        << "                as relpscost, pscost, vanillafullstrong, or\n"
        << "                misfullstrong (the data collection expert)\n"
        << "  --output      Path of the result table (never overwritten)\n"
        << "  --seeds       Comma-separated SCIP seed shifts (default 0)\n"
        << "  --settings    study: settings used for data collection"
        << " (default)\n"
        << "                default: SCIP with no parameter changed\n"
        << "  --time-limit  Time limit per run in seconds (default 3600)\n";
}

bool splitList(const std::string& text, std::vector<std::string>& items)
{
    items.clear();

    std::string item;
    std::istringstream stream(text);

    while (std::getline(stream, item, ','))
    {
        if (item.empty())
        {
            return false;
        }

        items.push_back(item);
    }

    // Reject an empty list, a trailing comma and duplicates.
    return !items.empty()
        && text.back() != ','
        && std::set<std::string>(items.begin(), items.end()).size()
            == items.size();
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

// Shifted geometric mean: exp(mean(log(x + shift))) - shift. It limits
// the weight of both the very easy and the very hard instances.
double shiftedGeometricMean(
    const std::vector<double>& values,
    double shift
)
{
    if (values.empty())
    {
        return 0.0;
    }

    double logSum = 0.0;

    for (const double value : values)
    {
        logSum += std::log(value + shift);
    }

    return std::exp(logSum / static_cast<double>(values.size())) - shift;
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
    std::string instanceDirectory;
    std::string outputPath;
    std::vector<std::string> rules;
    std::vector<unsigned int> seeds{0};
    MISBenchmarkSettings settings = MISBenchmarkSettings::Study;
    double timeLimit = 3600.0;

    bool hasInstances = false;
    bool hasRules = false;
    bool hasOutput = false;
    bool hasSeeds = false;
    bool hasSettings = false;
    bool hasTimeLimit = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string option = argv[i];

        if (i + 1 >= argc)
        {
            std::cerr << "Missing value for option: " << option << '\n';
            return 2;
        }

        const std::string value = argv[++i];

        if (option == "--instances")
        {
            if (hasInstances || value.empty())
            {
                std::cerr << "Invalid or duplicate --instances value.\n";
                return 2;
            }

            instanceDirectory = value;
            hasInstances = true;
        }
        else if (option == "--rules")
        {
            if (hasRules || !splitList(value, rules))
            {
                std::cerr << "Invalid or duplicate --rules value.\n";
                return 2;
            }

            hasRules = true;
        }
        else if (option == "--output")
        {
            if (hasOutput || value.empty())
            {
                std::cerr << "Invalid or duplicate --output value.\n";
                return 2;
            }

            outputPath = value;
            hasOutput = true;
        }
        else if (option == "--seeds")
        {
            std::vector<std::string> items;
            bool valid = !hasSeeds && splitList(value, items);

            seeds.clear();

            for (std::size_t s = 0; valid && s < items.size(); ++s)
            {
                unsigned int seed = 0;
                valid = parseSeed(items[s], seed);
                seeds.push_back(seed);
            }

            if (!valid)
            {
                std::cerr << "Invalid or duplicate --seeds value.\n";
                return 2;
            }

            hasSeeds = true;
        }
        else if (option == "--settings")
        {
            if (hasSettings || !parseMISBenchmarkSettings(value, settings))
            {
                std::cerr << "Invalid or duplicate --settings value.\n";
                return 2;
            }

            hasSettings = true;
        }
        else if (option == "--time-limit")
        {
            if (hasTimeLimit || !parsePositiveDouble(value, timeLimit))
            {
                std::cerr << "Invalid or duplicate --time-limit value.\n";
                return 2;
            }

            hasTimeLimit = true;
        }
        else
        {
            std::cerr << "Unknown option: " << option << '\n';
            return 2;
        }
    }

    if (!hasInstances || !hasRules || !hasOutput)
    {
        printUsage(argv[0]);
        return 2;
    }

    for (const std::string& rule : rules)
    {
        if (!isMISBenchmarkRuleAvailable(rule))
        {
            std::cerr << "Unknown branching rule: " << rule << '\n';
            return 2;
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
            << shiftedGeometricMean(summary.nodes, 100.0)
            << std::setw(12) << std::setprecision(2)
            << shiftedGeometricMean(summary.times, 1.0) << '\n';
    }

    std::cout << "\nResults: " << outputPath << '\n';

    return 0;
}
