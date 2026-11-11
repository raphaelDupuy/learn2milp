#include "services/MISGenerationService.hpp"
#include "io/InstanceMetadataIO.hpp"
#include "io/MISInstanceIO.hpp"

#include <charconv>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <system_error>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <sstream>
#include <vector>

bool generateSingleMIS(
    const ErdosRenyiConfig& config,
    const std::filesystem::path& outputPath,
    std::string& errorMessage
)
{
    try
    {
        if (outputPath.empty())
        {
            throw std::invalid_argument(
                "Output path must not be empty."
            );
        }

        const std::filesystem::path metadataPath =
            outputPath.string() + ".meta";

        if (std::filesystem::exists(outputPath)
            || std::filesystem::exists(metadataPath))
        {
            throw std::runtime_error(
                "Output file already exists. "
                "Choose another output path."
            );
        }

        ErdosRenyiGenerator generator;
        const GeneratedMISInstance generated =
            generator.generate(config);

        {
            std::ofstream graphFile(outputPath);

            if (!graphFile)
            {
                throw std::runtime_error(
                    "Cannot open output file: "
                    + outputPath.string()
                );
            }

            if (!writeMISInstance(
                    graphFile,
                    generated.graph,
                    errorMessage))
            {
                throw std::runtime_error(
                    "Failed to write instance: " + errorMessage
                );
            }

            graphFile.close();

            if (!graphFile)
            {
                throw std::runtime_error(
                    "Failed to finish writing: "
                    + outputPath.string()
                );
            }
        }

        {
            std::ofstream metadataFile(metadataPath);

            if (!metadataFile)
            {
                throw std::runtime_error(
                    "Cannot open metadata file: "
                    + metadataPath.string()
                );
            }

            const auto& metadata = generated.metadata;
            std::string metadataError;

            if (!writeInstanceMetadata(
                    metadataFile,
                    metadata,
                    metadataError))
            {
                throw std::runtime_error(
                    "Failed to write metadata: " + metadataError
                );
            }

            metadataFile.close();

            if (!metadataFile)
            {
                throw std::runtime_error(
                    "Failed to finish writing metadata: "
                    + metadataPath.string()
                );
            }
        }

        errorMessage.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        // Avoid leaving an incomplete instance behind.
        std::error_code ignored;
        std::filesystem::remove(outputPath, ignored);
        std::filesystem::remove(
            outputPath.string() + ".meta",
            ignored
        );

        errorMessage = exception.what();
        return false;
    }
}

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
            return;

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

std::string probabilityToken(double p)
{
    char buffer[64];

    const auto conversion = std::to_chars(
        buffer,
        buffer + sizeof(buffer),
        p,
        std::chars_format::general
    );

    if (conversion.ec != std::errc{})
    {
        throw std::runtime_error(
            "Failed to format probability value."
        );
    }

    std::string token(buffer, conversion.ptr);

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
}

bool generateBatchMIS(
    const MISBatchConfig& config,
    std::string& errorMessage
)
{
    try
    {
        if (config.nValues.empty()
            || config.pValues.empty()
            || config.seeds.empty())
        {
            throw std::invalid_argument(
                "Batch parameter lists must not be empty."
            );
        }

        std::set<int> uniqueN;
        for (int n : config.nValues)
        {
            if (n < 0 || !uniqueN.insert(n).second)
                throw std::invalid_argument(
                    "Invalid or duplicate n value."
                );
        }

        std::set<double> uniqueP;
        for (double p : config.pValues)
        {
            if (!std::isfinite(p) || p < 0.0 || p > 1.0
                || !uniqueP.insert(p).second)
                throw std::invalid_argument(
                    "Invalid or duplicate probability value."
                );
        }

        std::set<unsigned int> uniqueSeeds;
        for (unsigned int seed : config.seeds)
        {
            if (!uniqueSeeds.insert(seed).second)
                throw std::invalid_argument(
                    "Duplicate seed value."
                );
        }

        if (config.nValues.size() > MAX_BATCH_SIZE / config.pValues.size()
            || config.nValues.size() * config.pValues.size()
                > MAX_BATCH_SIZE / config.seeds.size())
        {
            throw std::invalid_argument(
                "Batch exceeds the maximum of 100000 instances."
            );
        }

        if (config.outputDirectory.empty())
            throw std::invalid_argument(
                "Output directory must not be empty."
            );

        const auto& outputDir = config.outputDirectory;
        std::filesystem::create_directories(outputDir);

        const auto manifestPath = outputDir / "manifest.csv";
        const auto temporaryManifest = outputDir / "manifest.csv.tmp";

        if (std::filesystem::exists(manifestPath)
            || std::filesystem::exists(temporaryManifest))
        {
            throw std::runtime_error(
                "Manifest already exists. Choose another output directory."
            );
        }

        struct BatchInstance
        {
            int n;
            double p;
            unsigned int seed;
            std::string id;
            std::string filename;
        };

        std::vector<BatchInstance> instances;

        const std::size_t total =
            config.nValues.size()
            * config.pValues.size()
            * config.seeds.size();

        instances.reserve(total);

        for (int n : config.nValues)
        {
            for (double p : config.pValues)
            {
                for (unsigned int seed : config.seeds)
                {
                    const std::string id =
                        "erdos_renyi_n" + std::to_string(n)
                        + "_p" + probabilityToken(p)
                        + "_seed" + std::to_string(seed);

                    const std::string filename = id + ".mis";
                    const auto graphPath = outputDir / filename;
                    const auto metadataPath =
                        std::filesystem::path(
                            graphPath.string() + ".meta"
                        );

                    if (std::filesystem::exists(graphPath)
                        || std::filesystem::exists(metadataPath))
                    {
                        throw std::runtime_error(
                            "Output file already exists: " + filename
                        );
                    }

                    instances.push_back(
                        {n, p, seed, id, filename}
                    );
                }
            }
        }

        BatchFileCleanup cleanup;
        ErdosRenyiGenerator generator;

        std::ostringstream manifest;
        manifest
            << "instance_id,file,generator_type,generator_version,"
            << "seed,n,p,m\n";

        for (const auto& item : instances)
        {
            const auto generated = generator.generate(
                ErdosRenyiConfig{
                    item.n, item.p, item.seed
                }
            );

            const auto graphPath = outputDir / item.filename;
            const auto metadataPath =
                std::filesystem::path(
                    graphPath.string() + ".meta"
                );

            cleanup.track(graphPath);

            {
                std::ofstream graphFile(graphPath);
                std::string ioError;

                if (!graphFile
                    || !writeMISInstance(
                        graphFile, generated.graph, ioError))
                {
                    throw std::runtime_error(
                        "Failed to write " + graphPath.string()
                        + ": " + ioError
                    );
                }

                graphFile.close();

                if (!graphFile)
                    throw std::runtime_error(
                        "Failed to finish writing graph file."
                    );
            }

            cleanup.track(metadataPath);

            {
                std::ofstream metadataFile(metadataPath);

                if (!metadataFile)
                    throw std::runtime_error(
                        "Cannot create metadata file: "
                        + metadataPath.string()
                    );

                const auto& metadata = generated.metadata;
                std::string metadataError;

                if (!writeInstanceMetadata(
                        metadataFile,
                        metadata,
                        metadataError))
                {
                    throw std::runtime_error(
                        "Failed to write metadata: " + metadataError
                    );
                }

                metadataFile.close();

                if (!metadataFile)
                {
                    throw std::runtime_error(
                        "Failed to finish writing metadata."
                    );
                }
            }

            manifest
                << item.id << ','
                << item.filename << ','
                << generated.metadata.generatorType << ','
                << generated.metadata.generatorVersion << ','
                << item.seed << ','
                << item.n << ','
                << probabilityText(item.p) << ','
                << generated.graph.edges.size() << '\n';
        }

        cleanup.track(temporaryManifest);

        {
            std::ofstream manifestFile(temporaryManifest);

            if (!manifestFile)
                throw std::runtime_error(
                    "Cannot create temporary manifest."
                );

            manifestFile << manifest.str();
            manifestFile.close();

            if (!manifestFile)
                throw std::runtime_error(
                    "Failed to finish writing manifest."
                );
        }

        std::filesystem::rename(temporaryManifest, manifestPath);
        cleanup.commit();

        errorMessage.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        errorMessage = exception.what();
        return false;
    }
}
