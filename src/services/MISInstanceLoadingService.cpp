
#include "services/MISInstanceLoadingService.hpp"

#include <fstream>
#include <utility>

#include "io/InstanceMetadataIO.hpp"
#include "io/MISInstanceIO.hpp"

bool loadMISInstance(
    const std::filesystem::path& instancePath,
    LoadedMISInstance& loadedInstance,
    std::string& errorMessage
)
{
    if (instancePath.empty())
    {
        errorMessage = "The MIS instance path is empty.";
        return false;
    }

    MISInstance parsedInstance;

    std::ifstream instanceFile(instancePath);
    if (!instanceFile.is_open())
    {
        errorMessage = "Unable to open MIS instance file: "
            + instancePath.string();
        return false;
    }

    if (!readMISInstance(instanceFile, parsedInstance, errorMessage))
    {
        errorMessage = "Invalid MIS instance file '"
            + instancePath.string() + "': " + errorMessage;
        return false;
    }

    instanceFile.close();

    const std::filesystem::path metadataPath(
        instancePath.string() + ".meta"
    );

    InstanceMetadata parsedMetadata;

    std::ifstream metadataFile(metadataPath);
    if (!metadataFile.is_open())
    {
        errorMessage = "Unable to open metadata file: "
            + metadataPath.string();
        return false;
    }

    if (!readInstanceMetadata(
        metadataFile,
        parsedMetadata,
        errorMessage
    ))
    {
        errorMessage = "Invalid metadata file '"
            + metadataPath.string() + "': " + errorMessage;
        return false;
    }

    if (parsedMetadata.problemType != "MIS")
    {
        errorMessage = "Metadata problem type mismatch: expected 'MIS', got '"
            + parsedMetadata.problemType + "'.";
        return false;
    }

    LoadedMISInstance parsedLoadedInstance;
    parsedLoadedInstance.instance = std::move(parsedInstance);
    parsedLoadedInstance.metadata = std::move(parsedMetadata);

    loadedInstance = std::move(parsedLoadedInstance);
    errorMessage.clear();

    return true;
}
