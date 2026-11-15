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

#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void printUsage(const char* program)
{
    std::cerr
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
        << "Existing files are never overwritten.\n";
}

bool parseNonNegativeInt(const std::string& text, int& value)
{
    try
    {
        std::size_t position = 0;
        const long long parsed = std::stoll(text, &position);

        if (position != text.size()
            || parsed < 0
            || parsed > std::numeric_limits<int>::max())
        {
            return false;
        }

        value = static_cast<int>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool parsePositiveInt(const std::string& text, int& value)
{
    return parseNonNegativeInt(text, value) && value > 0;
}

bool parseSeed(const std::string& text, unsigned int& value)
{
    try
    {
        if (text.empty() || text.front() == '-')
        {
            return false;
        }

        std::size_t position = 0;
        const unsigned long long parsed = std::stoull(text, &position);

        if (position != text.size()
            || parsed > std::numeric_limits<unsigned int>::max())
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

bool parseProbability(const std::string& text, double& value)
{
    try
    {
        std::size_t position = 0;
        value = std::stod(text, &position);

        return position == text.size()
            && std::isfinite(value)
            && value >= 0.0
            && value <= 1.0;
    }
    catch (...)
    {
        return false;
    }
}

std::vector<std::string> splitCommaSeparated(const std::string& text)
{
    if (text.empty())
    {
        throw std::invalid_argument("Empty parameter list.");
    }

    std::vector<std::string> values;
    std::size_t start = 0;

    while (start < text.size())
    {
        const std::size_t comma = text.find(',', start);
        const std::size_t end =
            comma == std::string::npos ? text.size() : comma;

        if (end == start)
        {
            throw std::invalid_argument(
                "Empty value in comma-separated list."
            );
        }

        values.push_back(text.substr(start, end - start));

        if (comma == std::string::npos)
        {
            return values;
        }

        start = comma + 1;

        if (start == text.size())
        {
            throw std::invalid_argument(
                "Trailing comma in comma-separated list."
            );
        }
    }

    return values;
}

template <typename T, typename Parser>
std::vector<T> parseList(
    const std::string& text,
    Parser parser,
    const std::string& parameterName
)
{
    std::vector<T> values;
    std::set<T> uniqueValues;

    for (const auto& token : splitCommaSeparated(text))
    {
        T value{};

        if (!parser(token, value))
        {
            throw std::invalid_argument(
                "Invalid value '" + token + "' for " + parameterName + "."
            );
        }

        if (!uniqueValues.insert(value).second)
        {
            throw std::invalid_argument(
                "Duplicate value '" + token + "' in " + parameterName + "."
            );
        }

        values.push_back(value);
    }

    return values;
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

int runSingle(int argc, char* argv[])
{
    MISGraphFamily family = MISGraphFamily::ErdosRenyi;
    int n = 0;
    double p = 0.0;
    int k = 0;
    unsigned int seed = 0;
    std::string outputPath;

    bool hasFamily = false;
    bool hasN = false;
    bool hasP = false;
    bool hasK = false;
    bool hasSeed = false;
    bool hasOutput = false;

    for (int i = 2; i < argc; ++i)
    {
        const std::string option = argv[i];

        if (i + 1 >= argc)
        {
            std::cerr << "Missing value for option: " << option << '\n';
            return 2;
        }

        const std::string value = argv[++i];
        bool valid = true;

        if (option == "--family")
        {
            valid = !hasFamily && parseMISGraphFamily(value, family);
            hasFamily = true;
        }
        else if (option == "--n")
        {
            valid = !hasN && parseNonNegativeInt(value, n);
            hasN = true;
        }
        else if (option == "--p")
        {
            valid = !hasP && parseProbability(value, p);
            hasP = true;
        }
        else if (option == "--k")
        {
            valid = !hasK && parsePositiveInt(value, k);
            hasK = true;
        }
        else if (option == "--seed")
        {
            valid = !hasSeed && parseSeed(value, seed);
            hasSeed = true;
        }
        else if (option == "--output")
        {
            valid = !hasOutput && !value.empty();
            outputPath = value;
            hasOutput = true;
        }
        else
        {
            std::cerr << "Unknown option: " << option << '\n';
            return 2;
        }

        if (!valid)
        {
            std::cerr << "Invalid or duplicate " << option << " value.\n";
            return 2;
        }
    }

    const bool erdosRenyi = family == MISGraphFamily::ErdosRenyi;

    if (erdosRenyi ? hasK : hasP)
    {
        std::cerr << "Option " << otherFamilyOption(family, false)
                  << " is not a parameter of "
                  << misGraphFamilyName(family) << " graphs.\n";
        return 2;
    }

    if (!hasN || !(erdosRenyi ? hasP : hasK) || !hasSeed || !hasOutput)
    {
        printUsage(argv[0]);
        return 2;
    }

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
    std::string familyText;
    std::string nText;
    std::string pText;
    std::string kText;
    std::string seedsText;
    std::string outputText;

    bool hasFamily = false;
    bool hasN = false;
    bool hasP = false;
    bool hasK = false;
    bool hasSeeds = false;
    bool hasOutput = false;

    for (int i = 2; i < argc; ++i)
    {
        const std::string option = argv[i];

        if (i + 1 >= argc)
        {
            throw std::invalid_argument(
                "Missing value for option " + option + "."
            );
        }

        const std::string value = argv[++i];

        if (option == "--family" && !hasFamily)
        {
            familyText = value;
            hasFamily = true;
        }
        else if (option == "--n-values" && !hasN)
        {
            nText = value;
            hasN = true;
        }
        else if (option == "--p-values" && !hasP)
        {
            pText = value;
            hasP = true;
        }
        else if (option == "--k-values" && !hasK)
        {
            kText = value;
            hasK = true;
        }
        else if (option == "--seeds" && !hasSeeds)
        {
            seedsText = value;
            hasSeeds = true;
        }
        else if (option == "--output-dir" && !hasOutput)
        {
            outputText = value;
            hasOutput = true;
        }
        else
        {
            throw std::invalid_argument(
                "Unknown or duplicate option: " + option
            );
        }
    }

    MISBatchConfig config;

    if (hasFamily && !parseMISGraphFamily(familyText, config.family))
    {
        throw std::invalid_argument("Unknown graph family: " + familyText);
    }

    const bool erdosRenyi = config.family == MISGraphFamily::ErdosRenyi;

    if (erdosRenyi ? hasK : hasP)
    {
        throw std::invalid_argument(
            std::string("Option ") + otherFamilyOption(config.family, true)
            + " is not a parameter of "
            + misGraphFamilyName(config.family) + " graphs."
        );
    }

    if (!hasN || !(erdosRenyi ? hasP : hasK) || !hasSeeds || !hasOutput)
    {
        throw std::invalid_argument(
            std::string("Required options: --n-values, ")
            + familyOption(config.family, true)
            + ", --seeds, --output-dir."
        );
    }

    config.nValues = parseList<int>(nText, parseNonNegativeInt, "--n-values");

    if (erdosRenyi)
    {
        config.pValues =
            parseList<double>(pText, parseProbability, "--p-values");
    }
    else
    {
        config.kValues =
            parseList<int>(kText, parsePositiveInt, "--k-values");
    }

    config.seeds = parseList<unsigned int>(seedsText, parseSeed, "--seeds");
    config.outputDirectory = outputText;

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
