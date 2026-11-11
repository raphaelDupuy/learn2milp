
#include "io/InstanceMetadataIO.hpp"

#include <iostream>
#include <optional>
#include <sstream>
#include <string>

namespace
{
    bool testSerialization()
    {
        InstanceMetadata metadata{
            "MIS",
            "erdos_renyi",
            "1.0",
            42,
            {
                {"n", "30"},
                {"p", "0.2"}
            }
        };

        std::ostringstream output;
        std::string errorMessage;

        if (!writeInstanceMetadata(output, metadata, errorMessage))
        {
            std::cerr << "Serialization failed: "
                      << errorMessage << '\n';
            return false;
        }

        const std::string expected =
            "INSTANCE_METADATA 1\n"
            "problem_type \"MIS\"\n"
            "generator_type \"erdos_renyi\"\n"
            "generator_version \"1.0\"\n"
            "seed 42\n"
            "parameter.n \"30\"\n"
            "parameter.p \"0.2\"\n";

        if (output.str() != expected)
        {
            std::cerr << "Unexpected serialized metadata:\n"
                      << output.str();
            return false;
        }

        return true;
    }

    bool testMetadataWithoutSeed()
    {
        InstanceMetadata metadata{
            "MIS",
            "erdos_renyi",
            "1.0",
            std::nullopt,
            {}
        };

        std::ostringstream output;
        std::string errorMessage;

        if (!writeInstanceMetadata(output, metadata, errorMessage))
        {
            std::cerr << "Serialization without seed failed: "
                      << errorMessage << '\n';
            return false;
        }

        if (output.str().find("seed ") != std::string::npos)
        {
            std::cerr << "Metadata without seed unexpectedly contains a seed.\n";
            return false;
        }

        return true;
    }

    bool testRejectsIncompleteMetadata()
    {
        InstanceMetadata metadata{
            "",
            "erdos_renyi",
            "1.0",
            42,
            {}
        };

        std::ostringstream output;
        std::string errorMessage;

        if (writeInstanceMetadata(output, metadata, errorMessage))
        {
            std::cerr << "Incomplete metadata was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }

    bool testRejectsInvalidParameterKey()
    {
        InstanceMetadata metadata{
            "MIS",
            "erdos_renyi",
            "1.0",
            42,
            {
                {"invalid key", "value"}
            }
        };

        std::ostringstream output;
        std::string errorMessage;

        if (writeInstanceMetadata(output, metadata, errorMessage))
        {
            std::cerr << "Invalid parameter key was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }
}

int main()
{
    if (!testSerialization()
        || !testMetadataWithoutSeed()
        || !testRejectsIncompleteMetadata()
        || !testRejectsInvalidParameterKey())
    {
        return 1;
    }

    std::cout << "All InstanceMetadataIO tests passed.\n";
    return 0;
}
