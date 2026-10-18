
#include "generators/ErdosRenyiGenerator.hpp"
#include "io/MISInstanceIO.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace
{
constexpr std::size_t MAX_BATCH_SIZE = 100000;

class BatchFileCleanup
{
public:
    void track(const std::filesystem::path& path)
    {
        paths_.push_back(path);
    }

    void commit()
    {
        committed_ = true;
    }

    ~BatchFileCleanup()
    {
        if (committed_)
        {
            return;
        }

        for (auto it = paths_.rbegin(); it != paths_.rend(); ++it)
        {
            std::error_code error;
            std::filesystem::remove(*it, error);
        }
    }

private:
    std::vector<std::filesystem::path> paths_;
    bool committed_ = false;
};

bool parseNonNegativeInt(const std::string& text, int& value)
{
    try
    {
        std::size_t pos = 0;
        const long long parsed = std::stoll(text, &pos);

        if (pos != text.size()
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

bool parseSeed(const std::string& text, unsigned int& value)
{
    try
    {
        if (text.empty() || text.front() == '-')
        {
            return false;
        }

        std::size_t pos = 0;
        const unsigned long long parsed = std::stoull(text, &pos);

        if (pos != text.size()
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
        std::size_t pos = 0;
        value = std::stod(text, &pos);

        return pos == text.size()
            && std::isfinite(value)
            && value >= 0.0
            && value <= 1.0;
    }
    catch (...)
    {
        return false;
    }
}

std::vector<std::string> splitCommaSeparated(
    const std::string& text
)
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
    const auto tokens = splitCommaSeparated(text);
    std::vector<T> values;
    std::set<T> uniqueValues;

    for (const auto& token : tokens)
    {
        T value{};

        if (!parser(token, value))
        {
            throw std::invalid_argument(
                "Invalid value '" + token
                + "' for " + parameterName + "."
            );
        }

        if (!uniqueValues.insert(value).second)
        {
            throw std::invalid_argument(
                "Duplicate value '" + token
                + "' in " + parameterName + "."
            );
        }

        values.push_back(value);
    }

    return values;
}

std::string probabilityToken(double p)
{
    std::ostringstream stream;
    stream << std::setprecision(
        std::numeric_limits<double>::max_digits10
    ) << p;

    std::string token = stream.str();

    // Produce a filename-safe representation.
    std::replace(token.begin(), token.end(), '.', '_');
    std::replace(token.begin(), token.end(), '-', 'm');
    std::replace(token.begin(), token.end(), '+', 'p');

    return token;
}

std::string probabilityText(double p)
{
    std::ostringstream stream;
    stream << std::setprecision(
        std::numeric_limits<double>::max_digits10
    ) << p;

    return stream.str();
}

struct BatchInstance
{
    int n;
    double p;
    unsigned int seed;
    std::string id;
    std::string filename;
};

std::string makeInstanceId(
    int n,
    double p,
    unsigned int seed
)
{
    return "erdos_renyi_n" + std::to_string(n)
        + "_p" + probabilityToken(p)
        + "_seed" + std::to_string(seed);
}

int runBatch(int argc, char* argv[])
{
    std::string nText;
    std::string pText;
    std::string seedsText;
    std::string outputText;

    bool hasN = false;
    bool hasP = false;
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

        if (option == "--n-values" && !hasN)
        {
            nText = value;
            hasN = true;
        }
        else if (option == "--p-values" && !hasP)
        {
            pText = value;
            hasP = true;
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

    if (!hasN || !hasP || !hasSeeds || !hasOutput)
    {
        throw std::invalid_argument(
            "Required options: --n-values, --p-values, "
            "--seeds, --output-dir."
        );
    }

    const auto nValues = parseList<int>(
        nText,
        parseNonNegativeInt,
        "--n-values"
    );

    const auto pValues = parseList<double>(
        pText,
        parseProbability,
        "--p-values"
    );

    const auto seeds = parseList<unsigned int>(
        seedsText,
        parseSeed,
        "--seeds"
    );

    if (nValues.size() > MAX_BATCH_SIZE / pValues.size()
        || nValues.size() * pValues.size()
            > MAX_BATCH_SIZE / seeds.size())
    {
        throw std::invalid_argument(
            "Batch exceeds the maximum of 100000 instances."
        );
    }

    const std::size_t total =
        nValues.size() * pValues.size() * seeds.size();

    const std::filesystem::path outputDir(outputText);

    if (outputDir.empty())
    {
        throw std::invalid_argument(
            "Output directory must not be empty."
        );
    }

    std::filesystem::create_directories(outputDir);

    const auto manifestPath = outputDir / "manifest.csv";

    // Do not overwrite a previous batch.
    if (std::filesystem::exists(manifestPath))
    {
        throw std::runtime_error(
            "Manifest already exists: "
            + manifestPath.string()
            + ". Choose a new output directory."
        );
    }

    std::vector<BatchInstance> instances;
    instances.reserve(total);

    for (const int n : nValues)
    {
        for (const double p : pValues)
        {
            for (const unsigned int seed : seeds)
            {
                const std::string id = makeInstanceId(n, p, seed);
                const std::string filename = id + ".mis";

                const auto graphPath = outputDir / filename;
                const auto metadataPath =
                    std::filesystem::path(graphPath.string() + ".meta");

                if (std::filesystem::exists(graphPath)
                    || std::filesystem::exists(metadataPath))
                {
                    throw std::runtime_error(
                        "Output file already exists for instance "
                        + id + ". Choose a clean output directory."
                    );
                }

                instances.push_back(
                    BatchInstance{n, p, seed, id, filename}
                );
            }
        }
    }

    ErdosRenyiGenerator generator;
    std::ostringstream manifest;

    manifest
        << "instance_id,file,family,generator_version,"
        << "seed,n,p,m\n";

    for (std::size_t i = 0; i < instances.size(); ++i)
    {
        const auto& item = instances[i];

        const GeneratedMISInstance generated =
            generator.generate(
                ErdosRenyiConfig{item.n, item.p, item.seed}
            );

        const auto graphPath = outputDir / item.filename;
        const auto metadataPath =
            std::filesystem::path(graphPath.string() + ".meta");

        std::string errorMessage;

        {
            std::ofstream graphFile(graphPath);

            if (!graphFile
                || !writeMISInstance(
                    graphFile,
                    generated.graph,
                    errorMessage))
            {
                throw std::runtime_error(
                    "Could not write " + graphPath.string()
                    + ": " + errorMessage
                );
            }

            graphFile.close();

            if (!graphFile)
            {
                throw std::runtime_error(
                    "Failed to finish writing "
                    + graphPath.string()
                );
            }
        }

        {
            std::ofstream metadataFile(metadataPath);

            if (!metadataFile)
            {
                throw std::runtime_error(
                    "Could not write "
                    + metadataPath.string()
                );
            }

            const auto& metadata = generated.metadata;

            metadataFile
                << "MIS_GENERATION_METADATA 1\n"
                << "family " << metadata.family << '\n'
                << "seed " << metadata.seed << '\n'
                << "n " << metadata.n << '\n'
                << "p " << probabilityText(metadata.p) << '\n'
                << "generator_version "
                << metadata.generatorVersion << '\n';

            metadataFile.close();

            if (!metadataFile)
            {
                throw std::runtime_error(
                    "Failed to finish writing "
                    + metadataPath.string()
                );
            }
        }

        manifest
            << item.id << ','
            << item.filename << ','
            << generated.metadata.family << ','
            << generated.metadata.generatorVersion << ','
            << item.seed << ','
            << item.n << ','
            << probabilityText(item.p) << ','
            << generated.graph.edges.size() << '\n';

        std::cout
            << '[' << (i + 1) << '/' << total << "] "
            << item.filename
            << " (m=" << generated.graph.edges.size() << ")\n";
    }

    const auto temporaryManifest =
        outputDir / "manifest.csv.tmp";

    {
        std::ofstream manifestFile(temporaryManifest);

        if (!manifestFile)
        {
            throw std::runtime_error(
                "Could not create temporary manifest."
            );
        }

        manifestFile << manifest.str();
        manifestFile.close();

        if (!manifestFile)
        {
            throw std::runtime_error(
                "Failed to finish writing manifest."
            );
        }
    }

    std::filesystem::rename(temporaryManifest, manifestPath);

    std::cout
        << "\nBatch completed: " << total << " instances\n"
        << "Output directory: " << outputDir.string() << '\n'
        << "Manifest: " << manifestPath.string() << '\n';

    return 0;
}
}

int runMISGeneratorBatch(int argc, char* argv[])
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
