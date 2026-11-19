#include "ui/TerminalUI.hpp"

#include "generators/MISGenerator.hpp"
#include "io/MISInstanceIO.hpp"
#include "io/MISStrongBranchingSampleIO.hpp"
#include "services/MISBenchmarkService.hpp"
#include "services/MISDatasetInspectionService.hpp"
#include "services/MISGenerationService.hpp"
#include "services/MISInstanceLoadingService.hpp"
#include "services/MISSolverLimits.hpp"
#include "tools/CommandLine.hpp"
#include "ui/FileBrowser.hpp"
#include "ui/TerminalInput.hpp"
#include "util/Text.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using namespace terminal;

const std::filesystem::path SINGLE_DIRECTORY = "data/instances/single";
const std::filesystem::path BATCH_DIRECTORY = "data/instances/batch";
const std::filesystem::path GENERATE_AND_SOLVE_DIRECTORY =
    "data/instances/generated_and_solved";

// A solution is printed in full up to this many vertices; beyond, only
// its first vertices are.
const std::size_t MAX_PRINTED_VERTICES = 20;

// What the shared parsers accept, for the messages.
const char* const NON_NEGATIVE = "a non-negative integer";
const char* const POSITIVE = "a positive integer";
const char* const PROBABILITY = "a number between 0 and 1";
const char* const SEED = "a non-negative integer";
const std::string SCIP_SEED =
    "an integer from 0 to " + std::to_string(std::numeric_limits<int>::max());

// Shortest usual notation: 3600, 0.5.
std::string formatNumber(double value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

std::string formatFixed(double value, int precision)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}

/*
 * Instance generation.
 */

std::string describeFamily(MISGraphFamily family)
{
    switch (family)
    {
        case MISGraphFamily::ErdosRenyi:
            return "Erdos-Renyi G(n, p): every edge with probability p";

        case MISGraphFamily::BarabasiAlbert:
            return "Barabasi-Albert: preferential attachment, every new"
                   " vertex joined to k existing ones";
    }

    return misGraphFamilyName(family);
}

bool readFamily(MISGraphFamily& family)
{
    const std::vector<MISGraphFamily>& families = misGraphFamilies();
    std::vector<std::string> options;

    for (const MISGraphFamily candidate : families)
        options.push_back(describeFamily(candidate));

    std::size_t choice = 0;

    if (!readChoice("Graph family:", options, choice))
        return false;

    family = families[choice];
    return true;
}

// Parameters of one instance, checked by the generator of its family.
bool readGeneratorConfig(MISGeneratorConfig& config)
{
    MISGraphFamily family = MISGraphFamily::ErdosRenyi;

    if (!readFamily(family))
        return false;

    while (true)
    {
        int n = 0;
        double p = 0.0;
        int k = 0;
        unsigned int seed = 0;

        if (!readValue("Number of vertices (n): ", n,
                parseNonNegativeInt, NON_NEGATIVE))
        {
            return false;
        }

        const bool erdosRenyi = family == MISGraphFamily::ErdosRenyi;

        if (erdosRenyi
            ? !readValue("Edge probability (p): ", p,
                  parseProbability, PROBABILITY)
            : !readValue("Edges of every new vertex (k): ", k,
                  parsePositiveInt, POSITIVE))
        {
            return false;
        }

        if (!readValue("Random seed: ", seed, parseSeed, SEED))
            return false;

        config = erdosRenyi
            ? MISGeneratorConfig{ErdosRenyiConfig{n, p, seed}}
            : MISGeneratorConfig{BarabasiAlbertConfig{n, k, seed}};

        try
        {
            validateMISGeneratorConfig(config);
            return true;
        }
        catch (const std::invalid_argument& exception)
        {
            std::cout << "Invalid parameters: " << exception.what()
                      << "\nPlease enter them again.\n";
        }
    }
}

/*
 * Asks for the parameters of an instance and writes it to directory,
 * named after its parameters. outputPath receives the instance file.
 */
bool generateInstance(
    const std::filesystem::path& directory,
    std::filesystem::path& outputPath
)
{
    MISGeneratorConfig config;

    if (!readGeneratorConfig(config))
        return false;

    outputPath = directory / (misInstanceId(config) + MIS_INSTANCE_EXTENSION);

    try
    {
        std::filesystem::create_directories(directory);
    }
    catch (const std::filesystem::filesystem_error& exception)
    {
        std::cout << "\nCannot create the output directory: "
                  << exception.what() << '\n';
        return false;
    }

    std::cout << "\nGenerating instance...\n";

    MISSingleGenerationResult generated;
    std::string errorMessage;

    if (!generateSingleMIS(config, outputPath, generated, errorMessage))
    {
        std::cout << "Generation failed: " << errorMessage << '\n';
        return false;
    }

    std::cout << "Instance generated: " << generated.graphPath.string()
              << " (" << generated.numberOfVertices << " vertices, "
              << generated.numberOfEdges << " edges)\n"
              << "Metadata: " << generated.metadataPath.string() << '\n';

    return true;
}

/*
 * Solving.
 */

/*
 * A generator parameter as stored in the metadata, where real numbers have
 * every digit needed to read them back exactly (0.10000000000000001):
 * numbers are shown with 10 significant digits (0.1), other values as is.
 */
std::string displayParameter(const std::string& value)
{
    try
    {
        std::size_t used = 0;
        const double number = std::stod(value, &used);

        if (used == value.size())
        {
            std::ostringstream stream;
            stream << std::setprecision(10) << number;
            return stream.str();
        }
    }
    catch (const std::exception&)
    {
    }

    return value;
}

void printInstanceSummary(const LoadedMISInstance& loaded)
{
    const InstanceMetadata& metadata = loaded.metadata;

    std::cout << "\nInstance loaded.\n";
    std::cout << "  Generator:  " << metadata.generatorType
              << " (version " << metadata.generatorVersion << ")\n";

    if (!metadata.generatorParameters.empty())
    {
        std::cout << "  Parameters: ";

        bool first = true;

        for (const auto& [name, value] : metadata.generatorParameters)
        {
            std::cout << (first ? "" : ", ") << name << " = "
                      << displayParameter(value);
            first = false;
        }

        std::cout << '\n';
    }

    if (metadata.generatorSeed.has_value())
        std::cout << "  Seed:       " << *metadata.generatorSeed << '\n';

    std::cout << "  Vertices:   " << loaded.instance.n << '\n';
    std::cout << "  Edges:      " << loaded.instance.edges.size() << '\n';
}

bool readModelPath(std::filesystem::path& modelPath)
{
    while (true)
    {
        std::filesystem::path path;

        if (!readPath("Model file (.model)", MODEL_KIND, path))
            return false;

        // Checked here so that a wrong file is reported before solving.
        std::string errorMessage;

        if (validateMISBenchmarkRule("mislearned", path, errorMessage))
        {
            modelPath = path;
            return true;
        }

        std::cout << errorMessage << '\n';
    }
}

// Candidates of the learned rule evaluated by strong branching (0: none).
bool readStrongBranchingCandidates(int& candidates)
{
    return readValue(
        "Strong branching on the model's first K candidates"
            " (0: branch on its best) [0]: ",
        candidates,
        parseNonNegativeInt,
        NON_NEGATIVE,
        std::optional<int>(0)
    );
}

// Branching rule and, for the learned rule, its model and candidates.
bool readBranchingRule(MISBenchmarkConfig& config)
{
    std::size_t choice = 0;

    if (!readChoice(
            "Branching rule:",
            {
                "SCIP's default (relpscost)",
                "Learned model (mislearned)",
                "Full strong branching expert (misfullstrong)",
                "The expert without its strong branching pruning"
                " (misfullstrong_nocutoff)",
                "Another SCIP rule, by name"
            },
            choice))
    {
        return false;
    }

    switch (choice)
    {
        case 0:
            config.branchingRule.clear();
            return true;

        case 1:
            config.branchingRule = "mislearned";
            return readModelPath(config.learnedModelPath)
                && readStrongBranchingCandidates(
                    config.learnedStrongBranchingCandidates);

        case 2:
            config.branchingRule = "misfullstrong";
            return true;

        case 3:
            config.branchingRule = "misfullstrong_nocutoff";
            return true;

        default:
            break;
    }

    while (true)
    {
        std::string name;

        if (!readNonEmptyLine("Rule name (e.g. pscost): ", name))
            return false;

        config.branchingRule = name;

        if (name == "mislearned")
        {
            return readModelPath(config.learnedModelPath)
                && readStrongBranchingCandidates(
                    config.learnedStrongBranchingCandidates);
        }

        std::string errorMessage;

        if (validateMISBenchmarkRule(name, {}, errorMessage))
            return true;

        std::cout << errorMessage << '\n';
    }
}

bool readSettings(MISBenchmarkSettings& settings)
{
    std::size_t choice = 0;

    if (!readChoice(
            "Solver settings:",
            {
                "study: those of the data collection (no restart,"
                " cuts at the root only)",
                "default: SCIP as shipped"
            },
            choice))
    {
        return false;
    }

    settings = choice == 0
        ? MISBenchmarkSettings::Study
        : MISBenchmarkSettings::Default;
    return true;
}

/*
 * A SCIP settings file, applied after the settings (for instance one of
 * experiments/cuts/settings/, to choose the separators of the nodes), or
 * none. Checked here, as by mis-benchmark --parameters.
 */
bool readParameterFile(std::filesystem::path& parameterFile)
{
    std::cout << "A SCIP settings file can change any parameter after the"
                 " settings above\n(e.g. experiments/cuts/settings/"
                 "no_node_gomory.set).\n";

    while (true)
    {
        std::filesystem::path path;

        if (!readPath("SCIP settings file (.set)", SETTINGS_KIND, path,
                PathRequirement::Optional))
        {
            return false;
        }

        std::string errorMessage;

        if (path.empty() || validateMISBenchmarkParameters(path, errorMessage))
        {
            parameterFile = path;
            return true;
        }

        std::cout << errorMessage << '\n';
    }
}

bool readTimeLimit(double& timeLimitSeconds)
{
    return readValue(
        "Time limit in seconds ["
            + formatNumber(MIS_DEFAULT_TIME_LIMIT_SECONDS) + "]: ",
        timeLimitSeconds,
        parsePositiveDouble,
        "a positive number",
        std::optional<double>(MIS_DEFAULT_TIME_LIMIT_SECONDS)
    );
}

bool readSolveOptions(MISBenchmarkConfig& config)
{
    config = MISBenchmarkConfig{};

    std::cout << '\n';

    return readBranchingRule(config)
        && readSettings(config.settings)
        && readParameterFile(config.parameterFile)
        && readTimeLimit(config.timeLimitSeconds)
        && readValue("SCIP random seed [0]: ", config.scipSeed,
            parseSCIPSeed, SCIP_SEED, std::optional<unsigned int>(0));
}

std::string describeRule(const MISBenchmarkConfig& config)
{
    if (config.branchingRule.empty())
        return "SCIP's default";

    std::string description = config.branchingRule;

    if (!config.learnedModelPath.empty())
        description += " (" + config.learnedModelPath.string() + ")";

    if (config.learnedStrongBranchingCandidates > 0)
    {
        description += ", strong branching on its first "
            + std::to_string(config.learnedStrongBranchingCandidates)
            + " candidate(s)";
    }

    return description;
}

std::string describeSettings(const MISBenchmarkConfig& config)
{
    std::string description = misBenchmarkSettingsName(config.settings);

    if (!config.parameterFile.empty())
        description += " + " + config.parameterFile.string();

    return description;
}

void printSelectedVertices(const std::vector<int>& vertices)
{
    const std::size_t printed =
        std::min(vertices.size(), MAX_PRINTED_VERTICES);

    std::cout << "Selected vertices: ";

    for (std::size_t i = 0; i < printed; ++i)
        std::cout << (i > 0 ? ", " : "") << vertices[i];

    if (printed < vertices.size())
    {
        std::cout << ", ... (first " << printed << " of "
                  << vertices.size() << ")";
    }

    std::cout << '\n';
}

void printSolveResult(
    const MISBenchmarkConfig& config,
    const MISBenchmarkResult& result
)
{
    std::cout << "\n========================================\n";
    std::cout << "             SOLVING RESULTS\n";
    std::cout << "========================================\n";

    std::cout << "Branching rule:    " << describeRule(config) << '\n';
    std::cout << "Settings:          " << describeSettings(config)
              << ", time limit " << config.timeLimitSeconds
              << " s, SCIP seed " << config.scipSeed << '\n';
    std::cout << "Status:            " << result.status << '\n';

    if (!result.hasSolution)
    {
        std::cout << "No feasible solution was found.\n";
    }
    else if (result.solved)
    {
        std::cout << "Maximum independent set: "
                  << formatFixed(result.primalBound, 0)
                  << " vertices (optimal)\n";
        printSelectedVertices(result.selectedVertices);
    }
    else
    {
        // Not proven optimal: the dual bound is an upper bound.
        std::cout << "Best independent set: "
                  << formatFixed(result.primalBound, 0)
                  << " vertices, upper bound "
                  << formatFixed(result.dualBound, 2);

        if (result.primalBound > 0.0)
        {
            std::cout << " (gap "
                      << formatFixed(100.0
                             * (result.dualBound - result.primalBound)
                             / result.primalBound, 2)
                      << " %)";
        }

        std::cout << '\n';
        printSelectedVertices(result.selectedVertices);
    }

    std::cout << "Explored nodes:    " << result.exploredNodes << '\n';
    std::cout << "LP iterations:     " << result.lpIterations << '\n';
    std::cout << "Solving time:      "
              << formatFixed(result.solvingTimeSeconds, 3) << " s\n";
    std::cout << "========================================\n";
}

// Loads an instance and prints its summary.
bool loadInstance(
    const std::filesystem::path& instancePath,
    LoadedMISInstance& loaded
)
{
    std::string errorMessage;

    if (!loadMISInstance(instancePath, loaded, errorMessage))
    {
        std::cout << "\nUnable to load the MIS instance: "
                  << errorMessage << '\n';
        return false;
    }

    printInstanceSummary(loaded);
    return true;
}

// Loads an instance, asks for the solving options and solves it.
void loadAndSolve(const std::filesystem::path& instancePath)
{
    LoadedMISInstance loaded;

    if (!loadInstance(instancePath, loaded))
        return;

    std::string errorMessage;
    MISBenchmarkConfig config;

    if (!readSolveOptions(config))
        return;

    std::cout << "\nSolving with SCIP...\n";

    MISBenchmarkResult result;

    if (!benchmarkMIS(loaded.instance, config, result, errorMessage))
    {
        std::cout << "Solving failed: " << errorMessage << '\n';
        return;
    }

    printSolveResult(config, result);
}

/*
 * Comparison of branching rules and settings files.
 */

// One of the compared rules: its name in the tables, and the branching
// rule, model and strong branching candidates it stands for.
struct ComparedRule
{
    std::string label;
    std::string branchingRule;
    std::filesystem::path modelPath;
    int strongBranchingCandidates = 0;
};

/*
 * A rule of the comparison, as typed: a SCIP rule name, misfullstrong,
 * misfullstrong_nocutoff, or a model file for the learned rule (labelled
 * after its file name). Fails, with an explanation, for anything else.
 */
bool parseComparedRule(
    const std::string& text,
    ComparedRule& rule,
    std::string& error
)
{
    const std::filesystem::path path(text);

    rule = path.extension() == MODEL_KIND
        ? ComparedRule{path.stem().string(), "mislearned", path}
        : ComparedRule{text, text, {}};

    return validateMISBenchmarkRule(rule.branchingRule, rule.modelPath, error);
}

bool samePath(const std::filesystem::path& a, const std::filesystem::path& b)
{
    std::error_code error;
    return a == b || std::filesystem::equivalent(a, b, error);
}

/*
 * Reads a comma-separated list, where "?" picks a file of the given kind
 * in the browser. An empty answer stands for defaultText.
 */
bool readItems(
    const std::string& prompt,
    const std::string& defaultText,
    const BrowsedKind& kind,
    std::vector<std::string>& items
)
{
    while (true)
    {
        std::string line;

        if (!readLine(prompt, line))
            return false;

        try
        {
            items = splitCommaSeparated(
                withoutSpacesAroundItems(line.empty() ? defaultText : line),
                "the list");
        }
        catch (const CommandLineError& error)
        {
            std::cout << error.what() << ".\n";
            continue;
        }

        bool picked = true;

        for (std::string& item : items)
        {
            if (item != "?")
                continue;

            std::filesystem::path file;

            if (!browse(kind, browserStart(kind), file))
                return false;

            if (file.empty())
            {
                std::cout << "No " << kind << " file was picked.\n";
                picked = false;
                break;
            }

            item = displayPath(file);
            std::cout << "Selected: " << item << '\n';
        }

        if (picked)
            return true;
    }
}

bool readComparedRules(std::vector<ComparedRule>& rules)
{
    std::cout << "\nRules to compare: SCIP rules (relpscost, pscost, ...),"
                 " misfullstrong,\nmisfullstrong_nocutoff, or model files"
                 " (.model) for the learned rule;\n? picks a model file in"
                 " the browser.\n";

    while (true)
    {
        std::vector<std::string> items;

        if (!readItems("Rules (comma-separated) [relpscost]: ", "relpscost",
                MODEL_KIND, items))
        {
            return false;
        }

        std::vector<ComparedRule> parsed;
        std::string error;

        for (const std::string& item : items)
        {
            ComparedRule rule;

            if (!parseComparedRule(item, rule, error))
                break;

            for (const ComparedRule& other : parsed)
            {
                if (other.branchingRule == rule.branchingRule
                    && samePath(other.modelPath, rule.modelPath))
                {
                    error = "rule given twice: " + item;
                    break;
                }

                // Two models of the same file name, in two directories.
                if (other.label == rule.label)
                    rule.label = displayPath(rule.modelPath);
            }

            if (!error.empty())
                break;

            parsed.push_back(rule);
        }

        if (error.empty())
        {
            rules = std::move(parsed);
            return true;
        }

        std::cout << "Invalid list: " << error
                  << (error.back() == '.' ? "\n" : ".\n");
    }
}

/*
 * Settings files to compare, each run with every rule: "-" stands for
 * none, so that "-, oddcycle.set" compares the settings alone with the
 * settings and that file.
 */
bool readParameterFiles(std::vector<std::filesystem::path>& files)
{
    std::cout << "\nSCIP settings files to compare, applied after the"
                 " settings (e.g.\nexperiments/cuts/settings/"
                 "no_node_gomory.set): - for none, ? picks one in the"
                 " browser.\n";

    while (true)
    {
        std::vector<std::string> items;

        if (!readItems("Settings files (comma-separated) [-]: ", "-",
                SETTINGS_KIND, items))
        {
            return false;
        }

        std::vector<std::filesystem::path> parsed;
        std::string error;

        for (const std::string& item : items)
        {
            const std::filesystem::path file =
                item == "-" ? std::filesystem::path() : std::filesystem::path(item);

            if (!file.empty() && !validateMISBenchmarkParameters(file, error))
                break;

            // Files are told apart by their name in the tables.
            const bool repeated = std::any_of(
                parsed.begin(), parsed.end(),
                [&](const std::filesystem::path& other) {
                    return other.stem() == file.stem();
                });

            if (repeated)
            {
                error = "given twice, or two files of the same name: " + item;
                break;
            }

            parsed.push_back(file);
        }

        if (error.empty())
        {
            files = std::move(parsed);
            return true;
        }

        std::cout << "Invalid list: " << error
                  << (error.back() == '.' ? "\n" : ".\n");
    }
}

void printComparisonRow(
    std::size_t labelWidth,
    const std::string& label,
    const std::string& seed,
    const std::string& status,
    const std::string& nodes,
    const std::string& lpIterations,
    const std::string& time,
    const std::string& size
)
{
    std::cout << std::left << std::setw(static_cast<int>(labelWidth))
              << label << std::right
              << std::setw(6) << seed
              << std::setw(11) << status
              << std::setw(11) << nodes
              << std::setw(12) << lpIterations
              << std::setw(11) << time
              << std::setw(6) << size << '\n';
}

// One compared configuration: a rule with a settings file.
struct ComparedConfiguration
{
    std::string label;
    MISBenchmarkConfig config;
};

void compareRules(const std::filesystem::path& instancePath)
{
    LoadedMISInstance loaded;

    if (!loadInstance(instancePath, loaded))
        return;

    std::vector<ComparedRule> rules;
    MISBenchmarkConfig common;
    std::vector<std::filesystem::path> parameterFiles;
    std::vector<unsigned int> seeds;

    if (!readComparedRules(rules))
        return;

    const bool learned = std::any_of(rules.begin(), rules.end(),
        [](const ComparedRule& rule) {
            return rule.branchingRule == "mislearned";
        });
    int strongBranching = 0;

    if (learned)
    {
        std::cout << "For the learned rule(s):\n";

        if (!readStrongBranchingCandidates(strongBranching))
            return;
    }

    if (!readSettings(common.settings)
        || !readParameterFiles(parameterFiles)
        || !readTimeLimit(common.timeLimitSeconds)
        || !readList<unsigned int>(
            "SCIP random seeds (comma-separated) [0]: ",
            seeds, parseSCIPSeed, "0"))
    {
        return;
    }

    // Rules vary fastest, so that they are side by side for each file.
    std::vector<ComparedConfiguration> configurations;

    for (const std::filesystem::path& file : parameterFiles)
    {
        for (const ComparedRule& rule : rules)
        {
            ComparedConfiguration configuration{rule.label, common};
            MISBenchmarkConfig& config = configuration.config;

            config.branchingRule = rule.branchingRule;
            config.learnedModelPath = rule.modelPath;
            config.parameterFile = file;

            if (rule.branchingRule == "mislearned" && strongBranching > 0)
            {
                config.learnedStrongBranchingCandidates = strongBranching;
                configuration.label += "+sb" + std::to_string(strongBranching);
            }

            if (!file.empty())
                configuration.label += "+" + file.stem().string();

            configurations.push_back(configuration);
        }
    }

    std::size_t labelWidth = 6;

    for (const ComparedConfiguration& configuration : configurations)
        labelWidth = std::max(labelWidth, configuration.label.size() + 2);

    std::cout << "\nSettings: " << misBenchmarkSettingsName(common.settings)
              << ", time limit " << common.timeLimitSeconds << " s, "
              << configurations.size() * seeds.size()
              << " run(s), one at a time.\n\n";

    printComparisonRow(labelWidth, "rule", "seed", "status", "nodes",
        "LP iter.", "time (s)", "size");

    std::vector<MISRuleSummary> summaries(configurations.size());
    std::vector<double> optimalSizes;

    for (std::size_t c = 0; c < configurations.size(); ++c)
    {
        for (const unsigned int seed : seeds)
        {
            MISBenchmarkConfig config = configurations[c].config;
            config.scipSeed = seed;

            MISBenchmarkResult result;
            std::string errorMessage;

            if (!benchmarkMIS(loaded.instance, config, result, errorMessage))
            {
                std::cout << configurations[c].label << ", seed " << seed
                          << ": solving failed: " << errorMessage << '\n';
                return;
            }

            summaries[c].add(result);

            if (result.solved)
                optimalSizes.push_back(result.primalBound);

            printComparisonRow(
                labelWidth, configurations[c].label, std::to_string(seed),
                result.status, std::to_string(result.exploredNodes),
                std::to_string(result.lpIterations),
                formatFixed(result.solvingTimeSeconds, 3),
                result.hasSolution ? formatFixed(result.primalBound, 0) : "-");
        }
    }

    if (seeds.size() > 1)
    {
        std::cout << "\nShifted geometric means over the seeds:"
                     " nodes (shift 100), time (shift 1 s)\n";

        for (std::size_t c = 0; c < configurations.size(); ++c)
        {
            std::cout << std::left
                      << std::setw(static_cast<int>(labelWidth))
                      << configurations[c].label << std::right
                      << "solved " << summaries[c].solved << '/'
                      << summaries[c].runs
                      << std::setw(11)
                      << formatFixed(summaries[c].meanNodes(), 0)
                      << " nodes"
                      << std::setw(10)
                      << formatFixed(summaries[c].meanTimeSeconds(), 3)
                      << " s\n";
        }
    }

    // Every configuration is exact: two different optima reveal a bug.
    const auto [smallest, largest] =
        std::minmax_element(optimalSizes.begin(), optimalSizes.end());

    if (!optimalSizes.empty() && *largest - *smallest > 0.5)
    {
        std::cout << "\nWARNING: the runs proved different optimal sizes ("
                  << formatFixed(*smallest, 0) << " and "
                  << formatFixed(*largest, 0) << "): one of the"
                     " configurations is not exact.\n";
    }
}

/*
 * Inspection of a data directory.
 */

std::string formatRange(const MISValueRange& range, int precision)
{
    return formatFixed(range.minimum, precision) + " / "
        + formatFixed(range.median, precision) + " / "
        + formatFixed(range.maximum, precision);
}

std::string formatCounts(const std::map<std::string, std::size_t>& counts)
{
    std::string text;

    for (const auto& [name, count] : counts)
        text += (text.empty() ? "" : ", ") + name + ": " + std::to_string(count);

    return text;
}

std::string formatDuration(double seconds)
{
    if (seconds < 120.0)
        return formatFixed(seconds, 1) + " s";

    if (seconds < 7200.0)
        return formatFixed(seconds / 60.0, 1) + " min";

    return formatFixed(seconds / 3600.0, 1) + " h";
}

void printInstanceSetSummary(const MISInstanceSetSummary& instances)
{
    std::cout << "Instances (.mis):  " << instances.numberOfInstances
              << " (" << formatCounts(instances.instancesPerGenerator)
              << ")\n"
              << "  vertices         " << formatRange(instances.vertices, 0)
              << "   (min / median / max)\n"
              << "  edges            " << formatRange(instances.edges, 0)
              << '\n';
}

void printCollectionSetSummary(const MISCollectionSetSummary& collections)
{
    std::cout << "Strong branching collections (.sb.meta): "
              << collections.numberOfCollections << " ("
              << formatCounts(collections.collectionsPerPolicy) << ")\n";

    if (collections.truncatedCollections > 0)
    {
        std::cout << "  truncated by a limit: "
                  << collections.truncatedCollections << '\n';
    }

    std::cout << "  samples          " << collections.samples
              << " (per collection "
              << formatRange(collections.samplesPerCollection, 0) << ")\n";

    std::cout << "  usable samples   " << collections.usableSamples;

    if (collections.collectionsWithoutUsableCount > 0)
    {
        std::cout << " (not counted: "
                  << collections.collectionsWithoutUsableCount
                  << " collection(s) in an older format)";
    }

    std::cout << '\n';

    std::cout << "  candidates       " << collections.candidates;

    if (collections.samples > 0)
    {
        std::cout << " ("
                  << formatFixed(static_cast<double>(collections.candidates)
                         / static_cast<double>(collections.samples), 1)
                  << " per sample)";
    }

    std::cout << '\n'
              << "  explored nodes   " << collections.exploredNodes << '\n';

    if (collections.policyDecisions > 0)
    {
        std::cout << "  model decisions  " << collections.policyDecisions
                  << ", "
                  << formatFixed(100.0
                         * static_cast<double>(collections.policyAgreements)
                         / static_cast<double>(collections.policyDecisions),
                         1)
                  << " % agreeing with the expert\n";
    }

    std::cout << "  solving time     "
              << formatDuration(collections.totalSolvingTimeSeconds)
              << " in total (per collection "
              << formatRange(collections.solvingTimeSeconds, 1) << " s)\n";
}

void inspectDirectory(const std::filesystem::path& directory)
{
    std::cout << "\nReading " << directory.string() << "...\n\n";

    MISDirectorySummary summary;
    std::string errorMessage;

    if (!inspectMISDirectory(directory, summary, errorMessage))
    {
        std::cout << "Inspection failed: " << errorMessage << '\n';
        return;
    }

    const bool hasInstances = summary.instances.numberOfInstances > 0;
    const bool hasCollections = summary.collections.numberOfCollections > 0;

    if (hasInstances)
        printInstanceSetSummary(summary.instances);

    if (hasCollections)
        printCollectionSetSummary(summary.collections);

    if (!hasInstances && !hasCollections && summary.skippedFiles.empty())
    {
        std::cout << "No instance (.mis) or strong branching collection"
                     " (.sb.meta) here.\n";
    }

    if (!summary.skippedFiles.empty())
    {
        const std::size_t shown =
            std::min<std::size_t>(summary.skippedFiles.size(), 5);

        std::cout << "Skipped, cannot be read: "
                  << summary.skippedFiles.size() << " file(s)\n";

        for (std::size_t i = 0; i < shown; ++i)
            std::cout << "  " << summary.skippedFiles[i] << '\n';

        if (shown < summary.skippedFiles.size())
            std::cout << "  ...\n";
    }

    // Datasets are split into subdirectories (train, valid, ...).
    std::vector<std::string> subdirectories;
    std::error_code error;

    for (std::filesystem::directory_iterator entry(directory, error), end;
            !error && entry != end; entry.increment(error))
    {
        std::error_code typeError;

        if (entry->is_directory(typeError))
            subdirectories.push_back(entry->path().filename().string());
    }

    std::sort(subdirectories.begin(), subdirectories.end());

    if (!subdirectories.empty())
    {
        std::cout << "Subdirectories:    ";

        for (std::size_t i = 0; i < subdirectories.size(); ++i)
            std::cout << (i > 0 ? ", " : "") << subdirectories[i];

        std::cout << '\n';
    }
}

void printTitle(const std::string& title)
{
    std::cout << "\n=== " << title << " ===\n\n";
}

} // namespace

void TerminalUI::displayMainMenu()
{
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "               LEARN2MILP\n";
    std::cout << "       Learning-Augmented MILP Solver\n";
    std::cout << "========================================\n\n";

    std::cout << "  1. Generate a single MIS instance\n";
    std::cout << "  2. Generate a batch of MIS instances\n";
    std::cout << "  3. Solve an existing MIS instance\n";
    std::cout << "  4. Generate and solve an MIS instance\n";
    std::cout << "  5. Compare branching rules and settings on an instance\n";
    std::cout << "  6. Inspect a data directory\n";
    std::cout << "  7. Exit\n\n";
}

void TerminalUI::run()
{
    while (true)
    {
        displayMainMenu();

        int choice = 0;

        if (!readValue("Select an option: ", choice,
                [](const std::string& text, int& value) {
                    return parsePositiveInt(text, value) && value <= 7;
                },
                "choose a number from 1 to 7"))
        {
            return;
        }

        switch (choice)
        {
            case 1:
                handleSingleGeneration();
                break;

            case 2:
                handleBatchGeneration();
                break;

            case 3:
                handleSolveExisting();
                break;

            case 4:
                handleGenerateAndSolve();
                break;

            case 5:
                handleCompareRules();
                break;

            case 6:
                handleInspectDirectory();
                break;

            default:
                std::cout << "\nExiting LEARN2MILP. Goodbye!\n";
                return;
        }
    }
}

void TerminalUI::handleSingleGeneration()
{
    printTitle("Generate a single MIS instance");

    std::filesystem::path instancePath;
    generateInstance(SINGLE_DIRECTORY, instancePath);
}

void TerminalUI::handleBatchGeneration()
{
    printTitle("Generate a batch of MIS instances");

    MISBatchConfig config;

    if (!readFamily(config.family))
        return;

    const bool erdosRenyi = config.family == MISGraphFamily::ErdosRenyi;

    if (!readList("Numbers of vertices (n, comma-separated): ",
            config.nValues, parseNonNegativeInt)
        || (erdosRenyi
            ? !readList("Edge probabilities (p, comma-separated): ",
                  config.pValues, parseProbability)
            : !readList("Edges of every new vertex (k, comma-separated): ",
                  config.kValues, parsePositiveInt))
        || !readList("Random seeds (comma-separated): ",
            config.seeds, parseSeed))
    {
        return;
    }

    std::string batchName;

    while (true)
    {
        if (!readNonEmptyLine(
                "Batch name (letters, digits, _ or -): ", batchName))
        {
            return;
        }

        const bool valid = std::all_of(
            batchName.begin(), batchName.end(),
            [](unsigned char c) {
                return std::isalnum(c) || c == '_' || c == '-';
            }
        );

        if (valid)
            break;

        std::cout << "Use only letters, digits, _ or -.\n";
    }

    config.outputDirectory = BATCH_DIRECTORY / batchName;

    const std::size_t total = config.nValues.size()
        * (config.pValues.size() + config.kValues.size())
        * config.seeds.size();

    std::cout << "\nGenerating " << total << " instances in "
              << config.outputDirectory.string() << "...\n";

    MISBatchResult generated;
    std::string errorMessage;

    const bool succeeded = generateBatchMIS(
        config, generated, errorMessage,
        [](const MISBatchProgress& progress) {
            std::cout << "\r  " << progress.index << " / "
                      << progress.total << std::flush;
        }
    );

    std::cout << '\n';

    if (!succeeded)
    {
        std::cout << "Batch generation failed: " << errorMessage << '\n';
        return;
    }

    std::cout << "Batch generated: " << generated.numberOfInstances
              << " instances.\n"
              << "Manifest: " << generated.manifestPath.string() << '\n';
}

void TerminalUI::handleSolveExisting()
{
    printTitle("Solve an existing MIS instance");

    std::filesystem::path instancePath;

    if (!readPath("MIS instance (.mis)", INSTANCE_KIND, instancePath))
        return;

    loadAndSolve(instancePath);
}

void TerminalUI::handleGenerateAndSolve()
{
    printTitle("Generate and solve an MIS instance");

    std::filesystem::path instancePath;

    if (generateInstance(GENERATE_AND_SOLVE_DIRECTORY, instancePath))
        loadAndSolve(instancePath);
}

void TerminalUI::handleCompareRules()
{
    printTitle("Compare branching rules and settings on an instance");

    std::filesystem::path instancePath;

    if (!readPath("MIS instance (.mis)", INSTANCE_KIND, instancePath))
        return;

    compareRules(instancePath);
}

void TerminalUI::handleInspectDirectory()
{
    printTitle("Inspect a data directory");

    std::cout << "Instances (.mis) and strong branching collections"
                 " (.sb.meta) of one directory,\ne.g."
                 " data/raw/strong_branching/er125_p01/train.\n";

    std::filesystem::path directory;

    if (!readPath("Directory", DIRECTORY_KIND, directory))
        return;

    inspectDirectory(directory);
}
