#include "services/MISGenerationService.hpp"

#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "io/InstanceMetadataIO.hpp"
#include "io/MISInstanceIO.hpp"

namespace
{

// Removes every tracked file, newest first, unless committed.
class FileCleanup
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

    ~FileCleanup()
    {
        if (committed_)
            return;

        for (auto it = paths_.rbegin(); it != paths_.rend(); ++it)
        {
            std::error_code ignored;
            std::filesystem::remove(*it, ignored);
        }
    }

private:
    std::vector<std::filesystem::path> paths_;
    bool committed_ = false;
};

void closeOutput(std::ofstream& file, const std::filesystem::path& path)
{
    file.close();

    if (!file)
    {
        throw std::runtime_error(
            "Failed to finish writing: " + path.string()
        );
    }
}

// Writes the graph and metadata files of one instance; both are tracked
// by cleanup before they are created.
void writeInstanceFiles(
    const GeneratedMISInstance& generated,
    const std::filesystem::path& graphPath,
    FileCleanup& cleanup
)
{
    const auto metadataPath = misInstanceMetadataPath(graphPath);
    std::string ioError;

    cleanup.track(graphPath);

    {
        std::ofstream graphFile(graphPath);

        if (!graphFile)
        {
            throw std::runtime_error(
                "Cannot open output file: " + graphPath.string()
            );
        }

        if (!writeMISInstance(graphFile, generated.graph, ioError))
        {
            throw std::runtime_error(
                "Failed to write instance " + graphPath.string()
                + ": " + ioError
            );
        }

        closeOutput(graphFile, graphPath);
    }

    cleanup.track(metadataPath);

    {
        std::ofstream metadataFile(metadataPath);

        if (!metadataFile)
        {
            throw std::runtime_error(
                "Cannot open metadata file: " + metadataPath.string()
            );
        }

        if (!writeInstanceMetadata(
                metadataFile, generated.metadata, ioError))
        {
            throw std::runtime_error(
                "Failed to write metadata: " + ioError
            );
        }

        closeOutput(metadataFile, metadataPath);
    }
}

void requireAbsent(const std::filesystem::path& path)
{
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error(
            "Output file already exists: " + path.string()
        );
    }
}

template <typename T>
void requireDistinct(const std::vector<T>& values, const char* listName)
{
    if (std::set<T>(values.begin(), values.end()).size() != values.size())
    {
        throw std::invalid_argument(
            std::string("Duplicate value in the ") + listName + " list."
        );
    }
}

// Every instance of a batch, in generation order.
std::vector<MISGeneratorConfig> batchInstances(const MISBatchConfig& config)
{
    const bool erdosRenyi = config.family == MISGraphFamily::ErdosRenyi;
    const std::size_t familyValues = erdosRenyi
        ? config.pValues.size() : config.kValues.size();
    const std::size_t otherValues = erdosRenyi
        ? config.kValues.size() : config.pValues.size();

    if (config.nValues.empty() || familyValues == 0 || config.seeds.empty())
    {
        throw std::invalid_argument(
            std::string("Batch parameter lists must not be empty: n, ")
            + (erdosRenyi ? "p" : "k") + " and seeds."
        );
    }

    if (otherValues != 0)
    {
        throw std::invalid_argument(
            std::string(erdosRenyi ? "k" : "p")
            + " values are not parameters of "
            + misGraphFamilyName(config.family) + " graphs."
        );
    }

    for (const double p : config.pValues)
    {
        if (!std::isfinite(p))
            throw std::invalid_argument("Invalid probability value.");
    }

    requireDistinct(config.nValues, "n");
    requireDistinct(config.pValues, "p");
    requireDistinct(config.kValues, "k");
    requireDistinct(config.seeds, "seed");

    if (config.nValues.size() > MIS_MAX_BATCH_SIZE / familyValues
        || config.nValues.size() * familyValues
            > MIS_MAX_BATCH_SIZE / config.seeds.size())
    {
        throw std::invalid_argument(
            "Batch exceeds the maximum of "
            + std::to_string(MIS_MAX_BATCH_SIZE) + " instances."
        );
    }

    std::vector<MISGeneratorConfig> instances;

    for (const int n : config.nValues)
    {
        for (std::size_t f = 0; f < familyValues; ++f)
        {
            for (const unsigned int seed : config.seeds)
            {
                if (erdosRenyi)
                {
                    instances.push_back(
                        ErdosRenyiConfig{n, config.pValues[f], seed});
                }
                else
                {
                    instances.push_back(
                        BarabasiAlbertConfig{n, config.kValues[f], seed});
                }

                // Before anything is written.
                validateMISGeneratorConfig(instances.back());
            }
        }
    }

    return instances;
}

} // namespace

bool generateSingleMIS(
    const MISGeneratorConfig& config,
    const std::filesystem::path& outputPath,
    MISSingleGenerationResult& result,
    std::string& errorMessage
)
{
    try
    {
        if (outputPath.empty())
        {
            throw std::invalid_argument("Output path must not be empty.");
        }

        requireAbsent(outputPath);
        requireAbsent(misInstanceMetadataPath(outputPath));

        const GeneratedMISInstance generated = generateMISInstance(config);

        // From here on, the files are removed on failure.
        FileCleanup cleanup;
        writeInstanceFiles(generated, outputPath, cleanup);
        cleanup.commit();

        MISSingleGenerationResult written;
        written.graphPath = outputPath;
        written.metadataPath = misInstanceMetadataPath(outputPath);
        written.numberOfVertices = generated.graph.n;
        written.numberOfEdges = generated.graph.edges.size();
        written.seed = misGeneratorSeed(config);

        result = std::move(written);
        errorMessage.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        errorMessage = exception.what();
        return false;
    }
}

bool generateBatchMIS(
    const MISBatchConfig& config,
    MISBatchResult& result,
    std::string& errorMessage,
    const MISBatchProgressCallback& progress
)
{
    try
    {
        // Validate the whole request before touching the file system.
        const std::vector<MISGeneratorConfig> instances =
            batchInstances(config);

        if (config.outputDirectory.empty())
        {
            throw std::invalid_argument(
                "Output directory must not be empty."
            );
        }

        const auto& outputDir = config.outputDirectory;
        const auto manifestPath = outputDir / "manifest.csv";
        const auto temporaryManifest = outputDir / "manifest.csv.tmp";

        std::filesystem::create_directories(outputDir);

        if (std::filesystem::exists(manifestPath)
            || std::filesystem::exists(temporaryManifest))
        {
            throw std::runtime_error(
                "A manifest already exists in " + outputDir.string()
                + ". Choose another output directory."
            );
        }

        std::vector<std::string> ids;
        ids.reserve(instances.size());

        for (const auto& instance : instances)
        {
            ids.push_back(misInstanceId(instance));

            const auto graphPath =
                outputDir / (ids.back() + MIS_INSTANCE_EXTENSION);
            requireAbsent(graphPath);
            requireAbsent(misInstanceMetadataPath(graphPath));
        }

        // From here on, every created file is removed on failure.
        FileCleanup cleanup;
        std::ostringstream manifest;

        manifest << "instance_id,file,generator_type,generator_version,seed";

        for (const auto& name : misManifestParameterNames(config.family))
            manifest << ',' << name;

        manifest << ",m\n";

        for (std::size_t i = 0; i < instances.size(); ++i)
        {
            const GeneratedMISInstance generated =
                generateMISInstance(instances[i]);

            const std::string fileName = ids[i] + MIS_INSTANCE_EXTENSION;
            writeInstanceFiles(generated, outputDir / fileName, cleanup);

            manifest
                << ids[i] << ','
                << fileName << ','
                << generated.metadata.generatorType << ','
                << generated.metadata.generatorVersion << ','
                << misGeneratorSeed(instances[i]);

            for (const auto& value : misManifestParameterValues(instances[i]))
                manifest << ',' << value;

            manifest << ',' << generated.graph.edges.size() << '\n';

            if (progress)
            {
                progress(MISBatchProgress{
                    i + 1,
                    instances.size(),
                    fileName,
                    generated.graph.edges.size()});
            }
        }

        cleanup.track(temporaryManifest);

        {
            std::ofstream manifestFile(temporaryManifest);

            if (!manifestFile)
            {
                throw std::runtime_error(
                    "Cannot create temporary manifest: "
                    + temporaryManifest.string()
                );
            }

            manifestFile << manifest.str();
            closeOutput(manifestFile, temporaryManifest);
        }

        std::filesystem::rename(temporaryManifest, manifestPath);
        cleanup.commit();

        result.numberOfInstances = instances.size();
        result.manifestPath = manifestPath;
        errorMessage.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        errorMessage = exception.what();
        return false;
    }
}
