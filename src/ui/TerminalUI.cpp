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

#include <algorithm>
#include <cctype>
#include <cstddef>
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

const std::filesystem::path SINGLE_DIRECTORY = "data/instances/single";
const std::filesystem::path BATCH_DIRECTORY = "data/instances/batch";
const std::filesystem::path GENERATE_AND_SOLVE_DIRECTORY =
    "data/instances/generated_and_solved";

// A solution is printed in full up to this many vertices; beyond, only
// its first vertices are.
const std::size_t MAX_PRINTED_VERTICES = 20;

/*
 * Input. Every reader asks again until the answer is valid, and returns
 * false only when the input ends, the caller then going back to the menu.
 */

std::string trim(const std::string& text)
{
    const auto isSpace = [](unsigned char c) { return std::isspace(c); };

    const auto first = std::find_if_not(text.begin(), text.end(), isSpace);
    const auto last = std::find_if_not(text.rbegin(), text.rend(), isSpace);

    return first < last.base() ? std::string(first, last.base()) : "";
}

bool readLine(const std::string& prompt, std::string& value)
{
    std::cout << prompt;

    std::string line;

    if (!std::getline(std::cin, line))
    {
        std::cout << "\nInput interrupted.\n";
        return false;
    }

    value = trim(line);
    return true;
}

bool readNonEmptyLine(const std::string& prompt, std::string& value)
{
    while (true)
    {
        if (!readLine(prompt, value))
            return false;

        if (!value.empty())
            return true;

        std::cout << "The answer must not be empty.\n";
    }
}

/*
 * Reads one value with parser(text, value), one of the value parsers
 * shared with the command line tools (tools/CommandLine.hpp), so that the
 * menu and the tools accept the same values. requirement tells what a
 * valid answer is. An empty answer gives defaultValue when there is one.
 */
template <typename T, typename Parser>
bool readValue(
    const std::string& prompt,
    T& value,
    Parser parser,
    const std::string& requirement,
    std::optional<T> defaultValue = std::nullopt
)
{
    while (true)
    {
        std::string line;

        if (!readLine(prompt, line))
            return false;

        if (line.empty() && defaultValue.has_value())
        {
            value = *defaultValue;
            return true;
        }

        T parsed{};

        if (parser(line, parsed))
        {
            value = parsed;
            return true;
        }

        std::cout << "Invalid value: " << requirement << ".\n";
    }
}

/*
 * Reads a comma-separated list of distinct values, each read by parser,
 * as the list options of the tools. An empty answer stands for
 * defaultText when it is not empty.
 */
template <typename T, typename Parser>
bool readList(
    const std::string& prompt,
    std::vector<T>& values,
    Parser parser,
    const std::string& defaultText = ""
)
{
    while (true)
    {
        std::string line;

        if (!readLine(prompt, line))
            return false;

        if (line.empty())
            line = defaultText;

        try
        {
            values = parseCommaSeparated<T>(line, parser, "the list");
            return true;
        }
        catch (const CommandLineError& error)
        {
            std::cout << error.what() << ".\n";
        }
    }
}

// Shows numbered options and reads one of them (0-based index).
bool readChoice(
    const std::string& title,
    const std::vector<std::string>& options,
    std::size_t& choice,
    std::size_t defaultChoice = 0
)
{
    std::cout << title << '\n';

    for (std::size_t i = 0; i < options.size(); ++i)
        std::cout << "  " << i + 1 << ". " << options[i] << '\n';

    int number = 0;

    if (!readValue<int>(
            "Choice [" + std::to_string(defaultChoice + 1) + "]: ",
            number,
            [&](const std::string& text, int& value) {
                return parsePositiveInt(text, value)
                    && static_cast<std::size_t>(value) <= options.size();
            },
            "choose a number from 1 to " + std::to_string(options.size()),
            static_cast<int>(defaultChoice + 1)))
    {
        return false;
    }

    choice = static_cast<std::size_t>(number - 1);
    return true;
}

/*
 * File browser. A file is picked among those of one extension; a
 * directory is picked by moving into it. Each kind of file remembers
 * where it was last browsed.
 */

const std::size_t BROWSER_PAGE_SIZE = 20;

// Extension of the files to pick, or empty to pick a directory.
using BrowsedKind = std::string;

const BrowsedKind DIRECTORY_KIND;
const BrowsedKind INSTANCE_KIND = MIS_INSTANCE_EXTENSION;
const BrowsedKind MODEL_KIND = ".model";

std::map<BrowsedKind, std::filesystem::path>& lastBrowsedDirectories()
{
    static std::map<BrowsedKind, std::filesystem::path> directories;
    return directories;
}

std::filesystem::path browserStart(const BrowsedKind& kind)
{
    const auto& last = lastBrowsedDirectories();
    const auto found = last.find(kind);

    if (found != last.end() && std::filesystem::is_directory(found->second))
        return found->second;

    const std::filesystem::path preferred =
        kind == INSTANCE_KIND ? "data/instances"
        : kind == MODEL_KIND ? "experiments"
        : "data";

    std::error_code error;

    return std::filesystem::is_directory(preferred, error)
        ? preferred : std::filesystem::path(".");
}

// Absolute path without "." or "..", so that ".." always goes up.
std::filesystem::path normalized(const std::filesystem::path& path)
{
    std::error_code error;
    const std::filesystem::path canonical =
        std::filesystem::weakly_canonical(path, error);

    return error ? std::filesystem::absolute(path).lexically_normal()
                 : canonical;
}

// Relative to the working directory when inside it, absolute otherwise.
std::string displayPath(const std::filesystem::path& path)
{
    std::error_code error;
    const std::filesystem::path relative =
        path.lexically_relative(std::filesystem::current_path(error));

    if (error || relative.empty() || *relative.begin() == "..")
        return path.string();

    return relative.string();
}

bool hasExtension(const std::filesystem::path& path, const BrowsedKind& kind)
{
    return path.extension() == kind;
}

struct DirectoryListing
{
    std::vector<std::filesystem::path> directories;
    std::vector<std::filesystem::path> files;

    // Directory mode: instances and collections, to recognize a dataset.
    std::size_t instances = 0;
    std::size_t collections = 0;
};

DirectoryListing listDirectory(
    const std::filesystem::path& directory,
    const BrowsedKind& kind
)
{
    DirectoryListing listing;

    for (const auto& entry : std::filesystem::directory_iterator(
            directory,
            std::filesystem::directory_options::skip_permission_denied))
    {
        const std::filesystem::path& path = entry.path();
        const std::string name = path.filename().string();

        if (name.empty() || name[0] == '.')
            continue;

        std::error_code error;

        if (entry.is_directory(error))
        {
            listing.directories.push_back(path);
        }
        else if (entry.is_regular_file(error))
        {
            if (kind == DIRECTORY_KIND)
            {
                listing.instances +=
                    hasExtension(path, INSTANCE_KIND) ? 1 : 0;
                listing.collections +=
                    isMISCollectionMetadataFile(path) ? 1 : 0;
            }
            else if (hasExtension(path, kind))
            {
                listing.files.push_back(path);
            }
        }
    }

    std::sort(listing.directories.begin(), listing.directories.end());
    std::sort(listing.files.begin(), listing.files.end());
    return listing;
}

/*
 * Lets the user move through directories and pick a file of the given
 * kind, or a directory. selected stays empty if the user cancels.
 * Returns false only when the input ends.
 */
bool browse(
    const BrowsedKind& kind,
    std::filesystem::path start,
    std::filesystem::path& selected
)
{
    const bool pickDirectory = kind == DIRECTORY_KIND;
    std::filesystem::path directory = normalized(start);
    std::size_t page = 0;

    // The listing is printed again only after a move, not after a
    // mistyped answer.
    bool showListing = true;

    selected.clear();

    while (true)
    {
        DirectoryListing listing;

        try
        {
            listing = listDirectory(directory, kind);
        }
        catch (const std::filesystem::filesystem_error& exception)
        {
            std::cout << "Cannot read " << displayPath(directory) << ": "
                      << exception.code().message() << '\n';

            if (directory == directory.parent_path())
                return true;

            directory = directory.parent_path();
            continue;
        }

        // Entries: subdirectories first, then the files to pick.
        std::vector<std::filesystem::path> entries = listing.directories;
        entries.insert(
            entries.end(), listing.files.begin(), listing.files.end());

        const std::size_t pages = std::max<std::size_t>(
            1, (entries.size() + BROWSER_PAGE_SIZE - 1) / BROWSER_PAGE_SIZE);
        page = std::min(page, pages - 1);

        if (showListing)
        {
            std::cout << "\n[" << displayPath(directory) << "]";

            if (pickDirectory && (listing.instances > 0 || listing.collections > 0))
            {
                std::cout << "  " << listing.instances << " .mis, "
                          << listing.collections << ' '
                          << MIS_SB_METADATA_SUFFIX;
            }
            else if (!pickDirectory)
            {
                std::cout << "  " << listing.files.size() << ' ' << kind
                          << " file(s)";
            }

            std::cout << '\n';

            const std::size_t first = page * BROWSER_PAGE_SIZE;
            const std::size_t last =
                std::min(entries.size(), first + BROWSER_PAGE_SIZE);

            for (std::size_t i = first; i < last; ++i)
            {
                const bool isDirectory = i < listing.directories.size();

                std::cout << std::setw(5) << i + 1 << ". "
                          << entries[i].filename().string()
                          << (isDirectory ? "/" : "") << '\n';
            }

            if (entries.empty())
            {
                std::cout << "      (no subdirectory"
                          << (pickDirectory ? "" : " or " + kind + " file")
                          << ")\n";
            }

            std::cout << "Number to open" << (pickDirectory ? "" : " or pick")
                      << ", .. to go up"
                      << (pickDirectory ? ", . to pick this directory" : "");

            if (pages > 1)
            {
                std::cout << ", n/p: next/previous page (" << page + 1 << '/'
                          << pages << ")";
            }

            std::cout << ", a path, q to cancel.\n";
        }

        showListing = true;

        std::string answer;

        if (!readLine("> ", answer))
            return false;

        if (answer == "q")
            return true;

        if (answer == "n" || answer == "p")
        {
            page = answer == "n"
                ? std::min(page + 1, pages - 1)
                : (page > 0 ? page - 1 : 0);
            continue;
        }

        if (answer == "." && pickDirectory)
        {
            selected = directory;
            lastBrowsedDirectories()[kind] = directory;
            return true;
        }

        std::filesystem::path target;
        int number = 0;

        if (answer.empty())
        {
            showListing = false;
            continue;
        }

        if (parseNonNegativeInt(answer, number))
        {
            if (entries.empty())
            {
                std::cout << "Nothing to open here: "
                          << (pickDirectory ? ". picks this directory, " : "")
                          << ".. goes up, q cancels.\n";
                showListing = false;
                continue;
            }

            if (number < 1 || static_cast<std::size_t>(number) > entries.size())
            {
                std::cout << "Choose a number from 1 to " << entries.size()
                          << ".\n";
                showListing = false;
                continue;
            }

            target = entries[static_cast<std::size_t>(number - 1)];
        }
        else
        {
            const std::filesystem::path typed(answer);
            target = normalized(
                typed.is_absolute() ? typed : directory / typed);
        }

        std::error_code error;

        if (std::filesystem::is_directory(target, error))
        {
            directory = normalized(target);
            page = 0;
        }
        else if (!pickDirectory
                 && std::filesystem::is_regular_file(target, error)
                 && hasExtension(target, kind))
        {
            selected = target;
            lastBrowsedDirectories()[kind] = directory;
            return true;
        }
        else
        {
            std::cout << "Not " << (pickDirectory ? "a directory" :
                          "a directory or a " + kind + " file")
                      << ": " << answer << '\n';
            showListing = false;
        }
    }
}

/*
 * Asks for a file of the given kind, or a directory: typed, or picked in
 * the browser on an empty answer or "?". A typed directory, where a file
 * is expected, opens the browser there. Returns false when the user
 * cancels with "q" or the input ends.
 */
bool readPath(
    const std::string& what,
    const BrowsedKind& kind,
    std::filesystem::path& path
)
{
    const bool pickDirectory = kind == DIRECTORY_KIND;

    while (true)
    {
        std::string answer;

        if (!readLine(what + " (empty or ? to browse, q to cancel): ", answer))
            return false;

        if (answer == "q")
            return false;

        std::error_code error;
        std::filesystem::path start = browserStart(kind);

        if (!answer.empty() && answer != "?")
        {
            const std::filesystem::path typed(answer);

            if (pickDirectory
                ? std::filesystem::is_directory(typed, error)
                : std::filesystem::is_regular_file(typed, error))
            {
                path = typed;
                return true;
            }

            if (pickDirectory || !std::filesystem::is_directory(typed, error))
            {
                std::cout << "No such " << (pickDirectory ? "directory" : "file")
                          << ": " << answer << '\n';
                continue;
            }

            start = typed;
        }

        std::filesystem::path selected;

        if (!browse(kind, start, selected))
            return false;

        if (!selected.empty())
        {
            path = displayPath(selected);
            std::cout << "Selected: " << path.string() << '\n';
            return true;
        }
    }
}

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
            std::cout << (first ? "" : ", ") << name << " = " << value;
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

// Branching rule and, for the learned rule, its model.
bool readBranchingRule(MISBenchmarkConfig& config)
{
    std::size_t choice = 0;

    if (!readChoice(
            "Branching rule:",
            {
                "SCIP's default (relpscost)",
                "Learned model (mislearned)",
                "Full strong branching expert (misfullstrong)",
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
            return readModelPath(config.learnedModelPath);

        case 2:
            config.branchingRule = "misfullstrong";
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
            return readModelPath(config.learnedModelPath);

        std::string errorMessage;

        if (validateMISBenchmarkRule(name, {}, errorMessage))
            return true;

        std::cout << errorMessage << '\n';
    }
}

// Solver settings and time limit.
bool readSettingsAndTimeLimit(MISBenchmarkConfig& config)
{
    std::size_t settings = 0;

    if (!readChoice(
            "Solver settings:",
            {
                "study: those of the data collection (no restart,"
                " cuts at the root only)",
                "default: SCIP as shipped"
            },
            settings))
    {
        return false;
    }

    config.settings = settings == 0
        ? MISBenchmarkSettings::Study
        : MISBenchmarkSettings::Default;

    return readValue(
        "Time limit in seconds ["
            + formatNumber(MIS_DEFAULT_TIME_LIMIT_SECONDS) + "]: ",
        config.timeLimitSeconds,
        parsePositiveDouble,
        "a positive number",
        std::optional<double>(MIS_DEFAULT_TIME_LIMIT_SECONDS)
    );
}

bool readSolveOptions(MISBenchmarkConfig& config)
{
    config = MISBenchmarkConfig{};

    std::cout << '\n';

    if (!readBranchingRule(config) || !readSettingsAndTimeLimit(config))
        return false;

    return readValue("SCIP random seed [0]: ", config.scipSeed,
        parseSCIPSeed, SCIP_SEED, std::optional<unsigned int>(0));
}

std::string describeRule(const MISBenchmarkConfig& config)
{
    if (config.branchingRule.empty())
        return "SCIP's default";

    if (config.learnedModelPath.empty())
        return config.branchingRule;

    return config.branchingRule + " ("
        + config.learnedModelPath.string() + ")";
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
    std::cout << "Settings:          "
              << misBenchmarkSettingsName(config.settings)
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
 * Comparison of branching rules.
 */

// One of the compared rules: its name in the tables, and the branching
// rule and model it stands for.
struct ComparedRule
{
    std::string label;
    std::string branchingRule;
    std::filesystem::path modelPath;
};

/*
 * A rule of the comparison, as typed: a SCIP rule name, misfullstrong,
 * or a model file for the learned rule (labelled after its file name).
 * Fails, with an explanation, for anything else.
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

bool readComparedRules(std::vector<ComparedRule>& rules)
{
    std::cout << "\nRules to compare: SCIP rules (relpscost, pscost, ...),"
                 " misfullstrong,\nor model files (.model) for the learned"
                 " rule; ? picks a model file in the browser.\n";

    while (true)
    {
        std::string line;

        if (!readLine("Rules (comma-separated) [relpscost]: ", line))
            return false;

        if (line.empty())
            line = "relpscost";

        std::istringstream stream(line);
        std::string item;
        std::vector<ComparedRule> parsed;
        std::string error;

        while (error.empty() && std::getline(stream, item, ','))
        {
            item = trim(item);

            if (item == "?")
            {
                std::filesystem::path model;

                if (!browse(MODEL_KIND, browserStart(MODEL_KIND), model))
                    return false;

                if (model.empty())
                {
                    error = "no model file was picked";
                    break;
                }

                item = displayPath(model);
                std::cout << "Selected: " << item << '\n';
            }

            ComparedRule rule;

            if (!parseComparedRule(item, rule, error))
                break;

            const bool repeated = std::any_of(
                parsed.begin(), parsed.end(),
                [&](const ComparedRule& other) {
                    return other.label == rule.label;
                });

            if (repeated)
                error = "rule given twice: " + rule.label;
            else
                parsed.push_back(rule);
        }

        if (error.empty() && parsed.empty())
            error = "the list must not be empty";

        if (error.empty())
        {
            rules = std::move(parsed);
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

void compareRules(const std::filesystem::path& instancePath)
{
    LoadedMISInstance loaded;

    if (!loadInstance(instancePath, loaded))
        return;

    std::vector<ComparedRule> rules;
    MISBenchmarkConfig common;
    std::vector<unsigned int> seeds;

    if (!readComparedRules(rules) || !readSettingsAndTimeLimit(common)
        || !readList<unsigned int>(
            "SCIP random seeds (comma-separated) [0]: ",
            seeds, parseSCIPSeed, "0"))
    {
        return;
    }

    std::size_t labelWidth = 6;

    for (const ComparedRule& rule : rules)
        labelWidth = std::max(labelWidth, rule.label.size() + 2);

    std::cout << "\nSettings: " << misBenchmarkSettingsName(common.settings)
              << ", time limit " << common.timeLimitSeconds << " s, "
              << rules.size() * seeds.size() << " run(s), one at a time.\n\n";

    printComparisonRow(labelWidth, "rule", "seed", "status", "nodes",
        "LP iter.", "time (s)", "size");

    std::vector<MISRuleSummary> summaries(rules.size());
    std::vector<double> optimalSizes;

    for (std::size_t r = 0; r < rules.size(); ++r)
    {
        for (const unsigned int seed : seeds)
        {
            MISBenchmarkConfig config = common;
            config.branchingRule = rules[r].branchingRule;
            config.learnedModelPath = rules[r].modelPath;
            config.scipSeed = seed;

            MISBenchmarkResult result;
            std::string errorMessage;

            if (!benchmarkMIS(loaded.instance, config, result, errorMessage))
            {
                std::cout << rules[r].label << ", seed " << seed
                          << ": solving failed: " << errorMessage << '\n';
                return;
            }

            summaries[r].add(result);

            if (result.solved)
                optimalSizes.push_back(result.primalBound);

            printComparisonRow(
                labelWidth, rules[r].label, std::to_string(seed),
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

        for (std::size_t r = 0; r < rules.size(); ++r)
        {
            std::cout << std::left
                      << std::setw(static_cast<int>(labelWidth))
                      << rules[r].label << std::right
                      << "solved " << summaries[r].solved << '/'
                      << summaries[r].runs
                      << std::setw(11)
                      << formatFixed(summaries[r].meanNodes(), 0)
                      << " nodes"
                      << std::setw(10)
                      << formatFixed(summaries[r].meanTimeSeconds(), 3)
                      << " s\n";
        }
    }

    // Every rule is exact: two different optima reveal a bug.
    const auto [smallest, largest] =
        std::minmax_element(optimalSizes.begin(), optimalSizes.end());

    if (!optimalSizes.empty() && *largest - *smallest > 0.5)
    {
        std::cout << "\nWARNING: the runs proved different optimal sizes ("
                  << formatFixed(*smallest, 0) << " and "
                  << formatFixed(*largest, 0) << "): one of the rules is"
                     " not exact.\n";
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

    for (const auto& entry :
            std::filesystem::directory_iterator(directory, error))
    {
        if (entry.is_directory())
            subdirectories.push_back(entry.path().filename().string());
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
    std::cout << "  5. Compare branching rules on an instance\n";
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
    printTitle("Compare branching rules on an instance");

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
