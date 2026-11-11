#include "ui/TerminalUI.hpp"

#include "services/MISGenerationService.hpp"
#include "services/MISInstanceLoadingService.hpp"
#include "services/MISSolvingService.hpp"

#include <fstream>
#include <iomanip>
#include <cctype>
#include <charconv>
#include <system_error>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>

namespace
{

bool readLine(const std::string& prompt, std::string& value)
{
    std::cout << prompt;

    if (!std::getline(std::cin, value))
    {
        std::cout << "\nInput interrupted.\n";
        return false;
    }

    return true;
}

bool readInt(const std::string& prompt, int& value)
{
    std::string line;

    if (!readLine(prompt, line))
        return false;

    std::istringstream stream(line);
    int parsed = 0;
    char extra = '\0';

    if (!(stream >> parsed) || (stream >> extra))
    {
        std::cout << "Invalid integer. Please try again.\n";
        return readInt(prompt, value);
    }

    value = parsed;
    return true;
}

bool readDouble(const std::string& prompt, double& value)
{
    std::string line;

    if (!readLine(prompt, line))
        return false;

    std::istringstream stream(line);
    double parsed = 0.0;
    char extra = '\0';

    if (!(stream >> parsed) || (stream >> extra)
        || !std::isfinite(parsed))
    {
        std::cout << "Invalid number. Please try again.\n";
        return readDouble(prompt, value);
    }

    value = parsed;
    return true;
}

bool readSeed(const std::string& prompt, unsigned int& value)
{
    std::string line;

    if (!readLine(prompt, line))
        return false;

    std::istringstream stream(line);
    long long parsed = 0;
    char extra = '\0';

    if (!(stream >> parsed) || (stream >> extra)
        || parsed < 0
        || parsed > std::numeric_limits<unsigned int>::max())
    {
        std::cout << "Invalid seed. Please enter a non-negative integer.\n";
        return readSeed(prompt, value);
    }

    value = static_cast<unsigned int>(parsed);
    return true;
}
template <typename T, typename Parser>
bool readList(
    const std::string& prompt,
    std::vector<T>& values,
    Parser parser
)
{
    std::string line;

    if (!readLine(prompt, line))
        return false;

    std::istringstream stream(line);
    std::string token;
    std::vector<T> parsedValues;

    while (std::getline(stream, token, ','))
    {
        std::istringstream tokenStream(token);
        T value{};
        char extra = '\0';

        if (!(tokenStream >> value) || (tokenStream >> extra))
        {
            std::cout << "Invalid list. Separate values with commas.\n";
            return readList(prompt, values, parser);
        }

        if (!parser(value))
        {
            std::cout << "One or more values are outside the valid range.\n";
            return readList(prompt, values, parser);
        }

        if (std::find(
                parsedValues.begin(),
                parsedValues.end(),
                value) != parsedValues.end())
        {
            std::cout << "Duplicate values are not allowed.\n";
            return readList(prompt, values, parser);
        }

        parsedValues.push_back(value);
    }

    if (parsedValues.empty())
    {
        std::cout << "The list must not be empty.\n";
        return readList(prompt, values, parser);
    }

    values = std::move(parsedValues);
    return true;
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
    std::cout << "  5. Exit\n\n";
}

void TerminalUI::run()
{
    bool running = true;

    while (running)
    {
        displayMainMenu();

        int choice = 0;

        if (!readInt("Select an option: ", choice))
            break;

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
                running = false;
                std::cout << "\nExiting LEARN2MILP. Goodbye!\n";
                break;

            default:
                std::cout << "\nInvalid option. Choose a number from 1 to 5.\n";
                break;
        }
    }
}

void TerminalUI::handleSingleGeneration()
{
    std::cout << "\n";
    std::cout << "=== Generate a single MIS instance ===\n\n";

    int n = 0;
    double p = 0.0;
    unsigned int seed = 0;

    if (!readInt("Number of vertices (n): ", n))
        return;

    if (n < 0)
    {
        std::cout << "The number of vertices must be non-negative.\n";
        return;
    }

    if (!readDouble("Edge probability (p, between 0 and 1): ", p))
        return;

    if (p < 0.0 || p > 1.0)
    {
        std::cout << "Probability must be between 0 and 1.\n";
        return;
    }

    if (!readSeed("Random seed: ", seed))
        return;

    char probabilityBuffer[64];

    const auto conversion = std::to_chars(
        probabilityBuffer,
        probabilityBuffer + sizeof(probabilityBuffer),
        p,
        std::chars_format::general
    );

    if (conversion.ec != std::errc{})
    {
        std::cout << "\nError while formatting edge probability.\n";
        return;
    }

    std::string probabilityToken(
        probabilityBuffer,
        conversion.ptr
    );

    std::replace(
        probabilityToken.begin(),
        probabilityToken.end(),
        '.',
        '_'
    );

    const std::string filename =
        "mis_n" + std::to_string(n)
        + "_p" + probabilityToken
        + "_seed" + std::to_string(seed)
        + ".mis";

    const std::filesystem::path outputDirectory =
        "data/instances/single";

    const std::filesystem::path outputPath =
        outputDirectory / filename;

    try
    {
        std::filesystem::create_directories(outputDirectory);
    }
    catch (const std::filesystem::filesystem_error& exception)
    {
        std::cout << "\nCannot create output directory: "
                  << exception.what() << '\n';
        return;
    }

    std::string errorMessage;

    const ErdosRenyiConfig config{n, p, seed};

    std::cout << "\nGenerating instance...\n";

    if (!generateSingleMIS(config, outputPath, errorMessage))
    {
        std::cout << "\nGeneration failed: "
                  << errorMessage << '\n';
        return;
    }

    std::cout << "\nInstance generated successfully!\n";
    std::cout << "File: " << outputPath << '\n';
    std::cout << "Metadata: " << outputPath.string() << ".meta\n";
}
void TerminalUI::handleBatchGeneration()
{
    std::cout << "\n";
    std::cout << "=== Generate a batch of MIS instances ===\n\n";

    std::vector<int> nValues;
    std::vector<double> pValues;
    std::vector<long long> seedsAsIntegers;

    // Read vertex counts
    if (!readList<int>(
            "Number of vertices (comma-separated): ",
            nValues,
            [](int n) {
                return n >= 0;
            }))
        return;

    // Read edge probabilities
    if (!readList<double>(
            "Edge probabilities (comma-separated): ",
            pValues,
            [](double p) {
                return std::isfinite(p)
                    && p >= 0.0
                    && p <= 1.0;
            }))
        return;

    // Read seeds as signed integers to reject negative values
    if (!readList<long long>(
            "Random seeds (comma-separated): ",
            seedsAsIntegers,
            [](long long seed) {
                return seed >= 0
                    && seed <= std::numeric_limits<unsigned int>::max();
            }))
        return;

    // Convert validated seeds to unsigned int
    std::vector<unsigned int> seeds;

    for (long long seed : seedsAsIntegers)
    {
        seeds.push_back(static_cast<unsigned int>(seed));
    }

    // Read and validate batch name
    std::string batchName;

    if (!readLine("Batch name (letters, numbers, _ or -): ", batchName))
        return;

    if (batchName.empty())
    {
        std::cout << "Batch name cannot be empty.\n";
        return;
    }

    const bool validBatchName = std::all_of(
        batchName.begin(),
        batchName.end(),
        [](unsigned char c) {
            return std::isalnum(c) || c == '_' || c == '-';
        }
    );

    if (!validBatchName)
    {
        std::cout << "Invalid batch name. Use only letters, numbers, _ or -.\n";
        return;
    }

    // Configure batch generation
    MISBatchConfig config;
    config.nValues = nValues;
    config.pValues = pValues;
    config.seeds = seeds;
    config.outputDirectory =
        std::filesystem::path("data/instances/batch") / batchName;

    const std::size_t total =
        nValues.size()
        * pValues.size()
        * seeds.size();

    std::cout << "\nThis will generate "
              << total << " instances.\n";

    // Generate instances
    std::string errorMessage;

    std::cout << "Generating batch...\n";

    if (!generateBatchMIS(config, errorMessage))
    {
        std::cout << "\nBatch generation failed: "
                  << errorMessage << '\n';
        return;
    }

    // Display results
    std::cout << "\nBatch generated successfully!\n";
    std::cout << "Output directory: "
              << config.outputDirectory << '\n';
    std::cout << "Number of instances: "
              << total << '\n';
    std::cout << "Manifest: "
              << config.outputDirectory / "manifest.csv"
              << '\n';
}


void TerminalUI::handleSolveExisting()
{
    std::cout << "\n";
    std::cout << "=== Solve an existing MIS instance ===\n\n";

    // Read input file path.
    std::string inputPath;

    if (!readLine("Path to MIS instance (.mis): ", inputPath))
        return;

    if (inputPath.empty())
    {
        std::cout << "Input path cannot be empty.\n";
        return;
    }

    // Load MIS instance and its metadata.
    LoadedMISInstance loadedInstance;
    std::string errorMessage;

    if (!loadMISInstance(inputPath, loadedInstance, errorMessage))
    {
        std::cout << "\nUnable to load MIS instance: "
                  << errorMessage << '\n';
        return;
    }

    const MISInstance& instance = loadedInstance.instance;
    const InstanceMetadata& metadata = loadedInstance.metadata;

    std::cout << "\nInstance loaded successfully.\n";
    std::cout << "Problem type: "
              << metadata.problemType << '\n';
    std::cout << "Generator: "
              << metadata.generatorType << '\n';
    std::cout << "Generator version: "
              << metadata.generatorVersion << '\n';

    if (metadata.generatorSeed.has_value())
    {
        std::cout << "Generator seed: "
                  << *metadata.generatorSeed << '\n';
    }
    std::cout << "Number of vertices: "
              << instance.n << '\n';
    std::cout << "Number of edges: "
              << instance.edges.size() << '\n';

    // Solve the MIS instance.
    MISResult result;

    std::cout << "\nSolving MIS instance with SCIP...\n";

    if (!solveMIS(instance, result))
    {
        std::cout << "\nSolving failed: "
                  << result.errorMessage << '\n';
        return;
    }

    // Display solver status.
    std::cout << "\n========================================\n";
    std::cout << "             SOLVING RESULTS\n";
    std::cout << "========================================\n";

    std::cout << "Status: "
              << result.status << '\n';

    if (!result.hasSolution)
    {
        std::cout << "No feasible solution was found.\n";
    }
    else
    {
        std::cout << std::fixed << std::setprecision(0);
        std::cout << "Maximum independent set size: "
                  << result.objectiveValue << '\n';

        std::cout << "Selected vertices: ";

        for (std::size_t i = 0;
             i < result.selectedVertices.size();
             ++i)
        {
            if (i > 0)
                std::cout << ", ";

            std::cout << result.selectedVertices[i];
        }

        std::cout << '\n';
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Solving time: "
              << result.solvingTimeSeconds << " seconds\n";

    std::cout << "Explored nodes: "
              << result.exploredNodes << '\n';

    std::cout << "========================================\n";
}


void TerminalUI::handleGenerateAndSolve()
{
    std::cout << "\n";
    std::cout << "=== Generate and solve an MIS instance ===\n\n";

    int n = 0;
    double p = 0.0;
    unsigned int seed = 0;

    // Read and validate the instance parameters.
    if (!readInt("Number of vertices (n): ", n))
        return;

    if (n < 0)
    {
        std::cout << "The number of vertices must be non-negative.\n";
        return;
    }

    if (!readDouble("Edge probability (p, between 0 and 1): ", p))
        return;

    if (p < 0.0 || p > 1.0)
    {
        std::cout << "Probability must be between 0 and 1.\n";
        return;
    }

    if (!readSeed("Random seed: ", seed))
        return;

    // Build a deterministic filename.
    char probabilityBuffer[64];

    const auto conversion = std::to_chars(
        probabilityBuffer,
        probabilityBuffer + sizeof(probabilityBuffer),
        p,
        std::chars_format::general
    );

    if (conversion.ec != std::errc{})
    {
        std::cout << "\nError while formatting edge probability.\n";
        return;
    }

    std::string probabilityToken(
        probabilityBuffer,
        conversion.ptr
    );

    std::replace(
        probabilityToken.begin(),
        probabilityToken.end(),
        '.',
        '_'
    );

    const std::string filename =
        "mis_n" + std::to_string(n)
        + "_p" + probabilityToken
        + "_seed" + std::to_string(seed)
        + ".mis";

    const std::filesystem::path outputDirectory =
        "data/instances/generated_and_solved";

    const std::filesystem::path outputPath =
        outputDirectory / filename;

    // Create the output directory.
    try
    {
        std::filesystem::create_directories(outputDirectory);
    }
    catch (const std::filesystem::filesystem_error& exception)
    {
        std::cout << "\nCannot create output directory: "
                  << exception.what() << '\n';
        return;
    }

    // Generate and save the instance.
    const ErdosRenyiConfig config{n, p, seed};

    std::string errorMessage;

    std::cout << "\nGenerating instance...\n";

    if (!generateSingleMIS(config, outputPath, errorMessage))
    {
        std::cout << "\nGeneration failed: "
                  << errorMessage << '\n';
        return;
    }

    std::cout << "Instance generated successfully.\n";
    std::cout << "File: " << outputPath << '\n';
    std::cout << "Metadata: " << outputPath.string() << ".meta\n";

    // Reload the generated instance and its metadata.
    LoadedMISInstance loadedInstance;

    if (!loadMISInstance(outputPath, loadedInstance, errorMessage))
    {
        std::cout << "\nCannot load generated instance: "
                  << errorMessage << '\n';
        return;
    }

    const MISInstance& instance = loadedInstance.instance;
    const InstanceMetadata& metadata = loadedInstance.metadata;

    std::cout << "\nInstance loaded successfully.\n";
    std::cout << "Problem type: "
              << metadata.problemType << '\n';
    std::cout << "Generator: "
              << metadata.generatorType << '\n';
    std::cout << "Generator version: "
              << metadata.generatorVersion << '\n';

    if (metadata.generatorSeed.has_value())
    {
        std::cout << "Generator seed: "
                  << *metadata.generatorSeed << '\n';
    }
    std::cout << "Number of vertices: "
              << instance.n << '\n';
    std::cout << "Number of edges: "
              << instance.edges.size() << '\n';

    // Solve the generated instance.
    MISResult result;

    std::cout << "\nSolving MIS instance with SCIP...\n";

    if (!solveMIS(instance, result))
    {
        std::cout << "\nSolving failed: "
                  << result.errorMessage << '\n';
        return;
    }

    // Display solver results.
    std::cout << "\n========================================\n";
    std::cout << "             SOLVING RESULTS\n";
    std::cout << "========================================\n";

    std::cout << "Status: "
              << result.status << '\n';

    if (!result.hasSolution)
    {
        std::cout << "No feasible solution was found.\n";
    }
    else
    {
        std::cout << std::fixed << std::setprecision(0);
        std::cout << "Maximum independent set size: "
                  << result.objectiveValue << '\n';

        std::cout << "Selected vertices: ";

        for (std::size_t i = 0;
             i < result.selectedVertices.size();
             ++i)
        {
            if (i > 0)
                std::cout << ", ";

            std::cout << result.selectedVertices[i];
        }

        std::cout << '\n';
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Solving time: "
              << result.solvingTimeSeconds << " seconds\n";

    std::cout << "Explored nodes: "
              << result.exploredNodes << '\n';

    std::cout << "========================================\n";
}
