
#include "services/MISInstanceLoadingService.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace
{
    const std::filesystem::path testDirectory =
        "test_MISInstanceLoadingService";


    bool writeTestFiles(
        const std::string& instanceContent,
        const std::string& metadataContent,
        const bool writeMetadata = true
    )
    {
        std::error_code error;
        std::filesystem::create_directories(testDirectory, error);

        if (error)
            return false;

        const std::filesystem::path instancePath =
            testDirectory / "test.mis";

        const std::filesystem::path metadataPath(
            instancePath.string() + ".meta"
        );

        // Remove files left by previous tests.
        std::filesystem::remove(metadataPath, error);

        if (error)
            return false;

        std::ofstream instanceFile(instancePath);
        if (!instanceFile)
            return false;

        instanceFile << instanceContent;
        instanceFile.close();

        if (!instanceFile)
            return false;

        if (!writeMetadata)
            return true;

        std::ofstream metadataFile(metadataPath);

        if (!metadataFile)
            return false;

        metadataFile << metadataContent;
        metadataFile.close();

        return static_cast<bool>(metadataFile);
    }


    bool testSuccessfulLoading()
    {
        const std::string instanceContent =
            "MIS_INSTANCE 1\n"
            "n 3\n"
            "m 2\n"
            "0 1\n"
            "1 2\n";

        const std::string metadataContent =
            "INSTANCE_METADATA 1\n"
            "problem_type \"MIS\"\n"
            "generator_type \"erdos_renyi\"\n"
            "generator_version \"1.0\"\n"
            "seed 42\n"
            "parameter.n \"3\"\n"
            "parameter.p \"0.5\"\n";

        if (!writeTestFiles(instanceContent, metadataContent))
            return false;

        LoadedMISInstance loaded;
        std::string errorMessage;

        if (!loadMISInstance(
            testDirectory / "test.mis",
            loaded,
            errorMessage
        ))
        {
            std::cerr << "Loading failed: "
                      << errorMessage << '\n';
            return false;
        }

        if (loaded.instance.n != 3
            || loaded.instance.edges.size() != 2
            || loaded.metadata.problemType != "MIS"
            || loaded.metadata.generatorType != "erdos_renyi"
            || loaded.metadata.generatorVersion != "1.0"
            || loaded.metadata.generatorSeed != 42)
        {
            std::cerr << "Loaded instance or metadata is incorrect.\n";
            return false;
        }

        return true;
    }

    bool testRejectsMissingMetadata()
    {
        const std::string instanceContent =
            "MIS_INSTANCE 1\n"
            "n 2\n"
            "m 1\n"
            "0 1\n";

        if (!writeTestFiles(instanceContent, "", false))
            return false;

        LoadedMISInstance loaded;
        std::string errorMessage;

        if (loadMISInstance(
            testDirectory / "test.mis",
            loaded,
            errorMessage
        ))
        {
            std::cerr << "Instance without metadata was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }

    bool testRejectsInvalidMetadata()
    {
        const std::string instanceContent =
            "MIS_INSTANCE 1\n"
            "n 2\n"
            "m 1\n"
            "0 1\n";

        const std::string metadataContent =
            "INVALID_HEADER\n";

        if (!writeTestFiles(instanceContent, metadataContent))
            return false;

        LoadedMISInstance loaded;
        std::string errorMessage;

        if (loadMISInstance(
            testDirectory / "test.mis",
            loaded,
            errorMessage
        ))
        {
            std::cerr << "Invalid metadata was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }

    bool testRejectsWrongProblemType()
    {
        const std::string instanceContent =
            "MIS_INSTANCE 1\n"
            "n 2\n"
            "m 1\n"
            "0 1\n";

        const std::string metadataContent =
            "INSTANCE_METADATA 1\n"
            "problem_type \"JSSP\"\n"
            "generator_type \"test\"\n"
            "generator_version \"1.0\"\n";

        if (!writeTestFiles(instanceContent, metadataContent))
            return false;

        LoadedMISInstance loaded;
        std::string errorMessage;

        if (loadMISInstance(
            testDirectory / "test.mis",
            loaded,
            errorMessage
        ))
        {
            std::cerr << "Metadata with wrong problem type was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }

    bool testPreservesDestinationOnFailure()
    {
        const std::string instanceContent =
            "MIS_INSTANCE 1\n"
            "n 2\n"
            "m 1\n"
            "0 1\n";

        const std::string metadataContent =
            "INSTANCE_METADATA 1\n"
            "problem_type \"JSSP\"\n"
            "generator_type \"test\"\n"
            "generator_version \"1.0\"\n";

        if (!writeTestFiles(instanceContent, metadataContent))
            return false;

        LoadedMISInstance loaded;
        loaded.instance.n = 99;
        loaded.metadata.problemType = "UNCHANGED";

        std::string errorMessage;

        if (loadMISInstance(
            testDirectory / "test.mis",
            loaded,
            errorMessage
        ))
        {
            std::cerr << "Loading unexpectedly succeeded.\n";
            return false;
        }

        if (loaded.instance.n != 99
            || loaded.metadata.problemType != "UNCHANGED")
        {
            std::cerr << "Destination was modified after failure.\n";
            return false;
        }

        return true;
    }
}

int main()
{
    std::error_code error;
    std::filesystem::remove_all(testDirectory, error);

    const bool success =
        testSuccessfulLoading()
        && testRejectsMissingMetadata()
        && testRejectsInvalidMetadata()
        && testRejectsWrongProblemType()
        && testPreservesDestinationOnFailure();

    std::filesystem::remove_all(testDirectory, error);

    if (!success)
        return 1;

    std::cout << "All MISInstanceLoadingService tests passed.\n";
    return 0;
}
