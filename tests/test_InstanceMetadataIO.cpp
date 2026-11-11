
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

    bool testRejectsMissingRequiredField()
    {
        const std::string serialized =
            "INSTANCE_METADATA 1\n"
            "problem_type \"MIS\"\n"
            "generator_type \"erdos_renyi\"\n";

        std::istringstream input(serialized);
        InstanceMetadata metadata;
        std::string errorMessage;

        if (readInstanceMetadata(input, metadata, errorMessage))
        {
            std::cerr << "Metadata with missing required field was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }

    bool testRejectsDuplicateField()
    {
        const std::string serialized =
            "INSTANCE_METADATA 1\n"
            "problem_type \"MIS\"\n"
            "problem_type \"JSSP\"\n"
            "generator_type \"erdos_renyi\"\n"
            "generator_version \"1.0\"\n";

        std::istringstream input(serialized);
        InstanceMetadata metadata;
        std::string errorMessage;

        if (readInstanceMetadata(input, metadata, errorMessage))
        {
            std::cerr << "Duplicate metadata field was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }

    bool testRejectsDuplicateParameter()
    {
        const std::string serialized =
            "INSTANCE_METADATA 1\n"
            "problem_type \"MIS\"\n"
            "generator_type \"erdos_renyi\"\n"
            "generator_version \"1.0\"\n"
            "parameter.n \"30\"\n"
            "parameter.n \"40\"\n";

        std::istringstream input(serialized);
        InstanceMetadata metadata;
        std::string errorMessage;

        if (readInstanceMetadata(input, metadata, errorMessage))
        {
            std::cerr << "Duplicate parameter was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }

    bool testDeserialization()
    {
        const std::string serialized =
            "INSTANCE_METADATA 1\n"
            "problem_type \"MIS\"\n"
            "generator_type \"erdos_renyi\"\n"
            "generator_version \"1.0\"\n"
            "seed 42\n"
            "parameter.n \"30\"\n"
            "parameter.p \"0.2\"\n";

        std::istringstream input(serialized);
        InstanceMetadata metadata;
        std::string errorMessage;

        if (!readInstanceMetadata(input, metadata, errorMessage))
        {
            std::cerr << "Deserialization failed: "
                      << errorMessage << '\n';
            return false;
        }

        if (metadata.problemType != "MIS"
            || metadata.generatorType != "erdos_renyi"
            || metadata.generatorVersion != "1.0"
            || metadata.generatorSeed != 42
            || metadata.generatorParameters.size() != 2
            || metadata.generatorParameters.at("n") != "30"
            || metadata.generatorParameters.at("p") != "0.2")
        {
            std::cerr << "Deserialized metadata does not match expected values.\n";
            return false;
        }

        return true;
    }

    bool testDeserializationWithoutSeed()
    {
        const std::string serialized =
            "INSTANCE_METADATA 1\n"
            "problem_type \"MIS\"\n"
            "generator_type \"erdos_renyi\"\n"
            "generator_version \"1.0\"\n"
            "parameter.n \"30\"\n";

        std::istringstream input(serialized);
        InstanceMetadata metadata;
        std::string errorMessage;

        if (!readInstanceMetadata(input, metadata, errorMessage))
        {
            std::cerr << "Deserialization without seed failed: "
                      << errorMessage << '\n';
            return false;
        }

        if (metadata.generatorSeed.has_value())
        {
            std::cerr << "Metadata unexpectedly contains a seed.\n";
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

    bool testRejectsInvalidSeed()
    {
        const std::string invalidSeeds[] = {
            "-1",
            "abc",
            "429496729999999999999"
        };

        for (const std::string& seed : invalidSeeds)
        {
            const std::string serialized =
                "INSTANCE_METADATA 1\n"
                "problem_type \"MIS\"\n"
                "generator_type \"erdos_renyi\"\n"
                "generator_version \"1.0\"\n"
                "seed " + seed + "\n";

            std::istringstream input(serialized);
            InstanceMetadata metadata;
            std::string errorMessage;

            if (readInstanceMetadata(input, metadata, errorMessage))
            {
                std::cerr << "Invalid seed was accepted: "
                          << seed << '\n';
                return false;
            }

            if (errorMessage.empty())
            {
                std::cerr << "No error message for invalid seed: "
                          << seed << '\n';
                return false;
            }
        }

        return true;
    }

    bool testRejectsUnknownField()
    {
        const std::string serialized =
            "INSTANCE_METADATA 1\n"
            "problem_type \"MIS\"\n"
            "generator_type \"erdos_renyi\"\n"
            "generator_version \"1.0\"\n"
            "unknown_field \"value\"\n";

        std::istringstream input(serialized);
        InstanceMetadata metadata;
        std::string errorMessage;

        if (readInstanceMetadata(input, metadata, errorMessage))
        {
            std::cerr << "Unknown metadata field was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }

    bool testPreservesMetadataOnFailure()
    {
        InstanceMetadata metadata{
            "JSSP",
            "custom_generator",
            "2.0",
            123,
            {
                {"machines", "5"}
            }
        };

        const std::string serialized =
            "INSTANCE_METADATA 1\n"
            "problem_type \"MIS\"\n"
            "generator_type \"erdos_renyi\"\n";

        std::istringstream input(serialized);
        std::string errorMessage;

        if (readInstanceMetadata(input, metadata, errorMessage))
        {
            std::cerr << "Invalid metadata was unexpectedly accepted.\n";
            return false;
        }

        if (metadata.problemType != "JSSP"
            || metadata.generatorType != "custom_generator"
            || metadata.generatorVersion != "2.0"
            || metadata.generatorSeed != 123
            || metadata.generatorParameters.size() != 1
            || metadata.generatorParameters.at("machines") != "5")
        {
            std::cerr << "Metadata was modified after failed deserialization.\n";
            return false;
        }

        return true;
    }

    bool testRejectsUnquotedValue()
    {
        const std::string serialized =
            "INSTANCE_METADATA 1\n"
            "problem_type MIS\n"
            "generator_type \"erdos_renyi\"\n"
            "generator_version \"1.0\"\n";

        std::istringstream input(serialized);
        InstanceMetadata metadata;
        std::string errorMessage;

        if (readInstanceMetadata(input, metadata, errorMessage))
        {
            std::cerr << "Unquoted metadata value was accepted.\n";
            return false;
        }

        if (errorMessage.empty())
        {
            std::cerr << "No error message was provided for an unquoted value.\n";
            return false;
        }

        return true;
    }

    bool testDeserializationWithWindowsLineEndings()
    {
        const std::string serialized =
            "INSTANCE_METADATA 1\r\n"
            "problem_type \"MIS\"\r\n"
            "generator_type \"erdos_renyi\"\r\n"
            "generator_version \"1.0\"\r\n"
            "seed 42\r\n"
            "parameter.n \"30\"\r\n";

        std::istringstream input(serialized);
        InstanceMetadata metadata;
        std::string errorMessage;

        if (!readInstanceMetadata(input, metadata, errorMessage))
        {
            std::cerr << "Deserialization with Windows line endings failed: "
                      << errorMessage << '\n';
            return false;
        }

        if (metadata.problemType != "MIS"
            || metadata.generatorType != "erdos_renyi"
            || metadata.generatorVersion != "1.0"
            || metadata.generatorSeed != 42
            || metadata.generatorParameters.size() != 1
            || metadata.generatorParameters.at("n") != "30")
        {
            std::cerr << "Deserialized metadata with Windows line endings "
                      << "does not match expected values.\n";
            return false;
        }

        return true;
    }

    bool testRejectsLineBreakInMetadataValue()
    {
        InstanceMetadata metadata{
            "MIS\nJSSP",
            "erdos_renyi",
            "1.0",
            42,
            {}
        };

        std::ostringstream output;
        std::string errorMessage;

        if (writeInstanceMetadata(output, metadata, errorMessage))
        {
            std::cerr << "Metadata containing a line break was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }

    bool testRejectsLineBreakInGeneratorParameter()
    {
        InstanceMetadata metadata{
            "MIS",
            "erdos_renyi",
            "1.0",
            42,
            {
                {"description", "first line\nsecond line"}
            }
        };

        std::ostringstream output;
        std::string errorMessage;

        if (writeInstanceMetadata(output, metadata, errorMessage))
        {
            std::cerr << "Generator parameter containing a line break "
                      << "was accepted.\n";
            return false;
        }

        return !errorMessage.empty();
    }
}

int main()
{
    if (!testSerialization()
        || !testDeserialization()
        || !testDeserializationWithoutSeed()
        || !testMetadataWithoutSeed()
        || !testRejectsIncompleteMetadata()
        || !testRejectsInvalidParameterKey()
        || !testRejectsMissingRequiredField()
        || !testRejectsDuplicateField()
        || !testRejectsDuplicateParameter()
        || !testRejectsInvalidSeed()
        || !testRejectsUnknownField()
        || !testPreservesMetadataOnFailure()
        || !testRejectsUnquotedValue()
        || !testDeserializationWithWindowsLineEndings()
        || !testRejectsLineBreakInMetadataValue()
        || !testRejectsLineBreakInGeneratorParameter())
    {
        return 1;
    }

    std::cout << "All InstanceMetadataIO tests passed.\n";
    return 0;
}
