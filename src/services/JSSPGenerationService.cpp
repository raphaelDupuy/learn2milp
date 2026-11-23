#include "services/JSSPGenerationService.hpp"

#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "io/InstanceMetadataIO.hpp"
#include "io/JSSPInstanceIO.hpp"
#include "util/FileOutput.hpp"

namespace
{

// Writes the instance and metadata files; both are tracked by cleanup
// before they are created.
void writeInstanceFiles(
    const GeneratedJSSPInstance& generated,
    const std::filesystem::path& instancePath,
    FileCleanup& cleanup
)
{
    const auto metadataPath = instanceMetadataPath(instancePath);
    std::string ioError;

    cleanup.track(instancePath);

    {
        std::ofstream file(instancePath);

        if (!file)
        {
            throw std::runtime_error(
                "Cannot open output file: " + instancePath.string());
        }

        if (!writeJSSPInstance(file, generated.instance, ioError))
        {
            throw std::runtime_error(
                "Failed to write instance " + instancePath.string()
                + ": " + ioError);
        }

        closeOutput(file, instancePath);
    }

    cleanup.track(metadataPath);

    {
        std::ofstream file(metadataPath);

        if (!file)
        {
            throw std::runtime_error(
                "Cannot open metadata file: " + metadataPath.string());
        }

        if (!writeInstanceMetadata(file, generated.metadata, ioError))
            throw std::runtime_error("Failed to write metadata: " + ioError);

        closeOutput(file, metadataPath);
    }
}

template <typename T>
void requireDistinct(const std::vector<T>& values, const char* listName)
{
    if (std::set<T>(values.begin(), values.end()).size() != values.size())
    {
        throw std::invalid_argument(
            std::string("Duplicate value in the ") + listName + " list.");
    }
}

// Every instance of a batch, in generation order.
std::vector<TaillardConfig> batchInstances(const JSSPBatchConfig& config)
{
    if (config.jobsValues.empty() || config.machinesValues.empty()
        || config.seeds.empty())
    {
        throw std::invalid_argument(
            "Batch parameter lists must not be empty: jobs, machines and"
            " seeds.");
    }

    requireDistinct(config.jobsValues, "jobs");
    requireDistinct(config.machinesValues, "machines");
    requireDistinct(config.seeds, "seed");

    if (config.jobsValues.size() > JSSP_MAX_BATCH_SIZE / config.machinesValues.size()
        || config.jobsValues.size() * config.machinesValues.size()
            > JSSP_MAX_BATCH_SIZE / config.seeds.size())
    {
        throw std::invalid_argument(
            "Batch exceeds the maximum of "
            + std::to_string(JSSP_MAX_BATCH_SIZE) + " instances.");
    }

    std::vector<TaillardConfig> instances;

    for (const int jobs : config.jobsValues)
    {
        for (const int machines : config.machinesValues)
        {
            for (const unsigned int seed : config.seeds)
            {
                instances.push_back(taillardConfigFromSeed(jobs, machines, seed));

                // Before anything is written.
                TaillardGenerator::validate(instances.back());
            }
        }
    }

    return instances;
}

} // namespace

bool generateSingleJSSP(
    const TaillardConfig& config,
    const std::filesystem::path& outputPath,
    JSSPSingleGenerationResult& result,
    std::string& errorMessage
)
{
    try
    {
        if (outputPath.empty())
            throw std::invalid_argument("Output path must not be empty.");

        requireAbsent(outputPath);
        requireAbsent(instanceMetadataPath(outputPath));

        const GeneratedJSSPInstance generated = TaillardGenerator{}.generate(config);

        // From here on, the files are removed on failure.
        FileCleanup cleanup;
        writeInstanceFiles(generated, outputPath, cleanup);
        cleanup.commit();

        JSSPSingleGenerationResult written;
        written.instancePath = outputPath;
        written.metadataPath = instanceMetadataPath(outputPath);
        written.jobs = config.jobs;
        written.machines = config.machines;
        written.lowerBound = jsspOneMachineBound(generated.instance);
        written.dispatchMakespan = jsspDispatchMakespan(generated.instance);

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

bool generateBatchJSSP(
    const JSSPBatchConfig& config,
    JSSPBatchResult& result,
    std::string& errorMessage,
    const JSSPBatchProgressCallback& progress
)
{
    try
    {
        // Validate the whole request before touching the file system.
        const std::vector<TaillardConfig> instances = batchInstances(config);

        if (config.outputDirectory.empty())
            throw std::invalid_argument("Output directory must not be empty.");

        const auto& outputDir = config.outputDirectory;
        const auto manifestPath = outputDir / "manifest.csv";
        const auto temporaryManifest = outputDir / "manifest.csv.tmp";

        std::filesystem::create_directories(outputDir);

        if (std::filesystem::exists(manifestPath)
            || std::filesystem::exists(temporaryManifest))
        {
            throw std::runtime_error(
                "A manifest already exists in " + outputDir.string()
                + ". Choose another output directory.");
        }

        std::vector<std::string> ids;
        ids.reserve(instances.size());

        for (const auto& instance : instances)
        {
            ids.push_back(taillardInstanceId(instance));

            const auto path = outputDir / (ids.back() + JSSP_INSTANCE_EXTENSION);
            requireAbsent(path);
            requireAbsent(instanceMetadataPath(path));
        }

        // From here on, every created file is removed on failure.
        FileCleanup cleanup;
        std::ostringstream manifest;

        manifest << "instance_id,file,generator_type,generator_version,seed,"
                    "jobs,machines,time_seed,machine_seed,lower_bound,"
                    "dispatch_makespan\n";

        for (std::size_t i = 0; i < instances.size(); ++i)
        {
            const TaillardConfig& instance = instances[i];
            const GeneratedJSSPInstance generated =
                TaillardGenerator{}.generate(instance);

            const std::string fileName = ids[i] + JSSP_INSTANCE_EXTENSION;
            writeInstanceFiles(generated, outputDir / fileName, cleanup);

            const int lowerBound = jsspOneMachineBound(generated.instance);
            const int dispatch = jsspDispatchMakespan(generated.instance);

            manifest
                << ids[i] << ','
                << fileName << ','
                << generated.metadata.generatorType << ','
                << generated.metadata.generatorVersion << ','
                << *instance.seed << ','
                << instance.jobs << ','
                << instance.machines << ','
                << instance.timeSeed << ','
                << instance.machineSeed << ','
                << lowerBound << ','
                << dispatch << '\n';

            if (progress)
            {
                progress(JSSPBatchProgress{
                    i + 1, instances.size(), fileName, lowerBound, dispatch});
            }
        }

        cleanup.track(temporaryManifest);

        {
            std::ofstream manifestFile(temporaryManifest);

            if (!manifestFile)
            {
                throw std::runtime_error(
                    "Cannot create temporary manifest: "
                    + temporaryManifest.string());
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
