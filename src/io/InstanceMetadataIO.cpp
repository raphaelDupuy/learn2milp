
#include "io/InstanceMetadataIO.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <iomanip>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>
#include <utility>

namespace
{
    bool isValidParameterKey(const std::string& key)
    {
        if (key.empty())
        {
            return false;
        }

        return std::all_of(
            key.begin(),
            key.end(),
            [](unsigned char c)
            {
                return std::isalnum(c)
                    || c == '_'
                    || c == '-'
                    || c == '.';
            }
        );
    }

    bool containsLineBreak(const std::string& value)
    {
        return value.find('\n') != std::string::npos
            || value.find('\r') != std::string::npos;
    }

    bool readQuotedValue(
        std::istringstream& lineStream,
        std::string& value
    )
    {
        lineStream >> std::ws;

        if (lineStream.peek() != '"')
        {
            return false;
        }

        if (!(lineStream >> std::quoted(value)))
        {
            return false;
        }

        lineStream >> std::ws;

        return lineStream.eof();
    }
}

bool writeInstanceMetadata(
    std::ostream& output,
    const InstanceMetadata& metadata,
    std::string& errorMessage
)
{
    if (metadata.problemType.empty()
        || metadata.generatorType.empty()
        || metadata.generatorVersion.empty())
    {
        errorMessage = "Incomplete instance metadata.";
        return false;
    }

    if (containsLineBreak(metadata.problemType)
        || containsLineBreak(metadata.generatorType)
        || containsLineBreak(metadata.generatorVersion))
    {
        errorMessage = "Metadata fields must not contain line breaks.";
        return false;
    }

    for (const auto& [key, value] : metadata.generatorParameters)
    {
        if (!isValidParameterKey(key))
        {
            errorMessage = "Invalid generator parameter key: " + key;
            return false;
        }

        if (containsLineBreak(value))
        {
            errorMessage =
                "Generator parameter values must not contain line breaks.";
            return false;
        }
    }

    output
        << "INSTANCE_METADATA 1\n"
        << "problem_type " << std::quoted(metadata.problemType) << '\n'
        << "generator_type " << std::quoted(metadata.generatorType) << '\n'
        << "generator_version "
        << std::quoted(metadata.generatorVersion) << '\n';

    if (metadata.generatorSeed.has_value())
    {
        output << "seed " << *metadata.generatorSeed << '\n';
    }

    for (const auto& [key, value] : metadata.generatorParameters)
    {
        output
            << "parameter."
            << key << ' '
            << std::quoted(value) << '\n';
    }

    if (!output)
    {
        errorMessage = "Failed to write instance metadata.";
        return false;
    }

    errorMessage.clear();
    return true;
}

bool readInstanceMetadata(
    std::istream& input,
    InstanceMetadata& metadata,
    std::string& errorMessage
)
{
    InstanceMetadata parsedMetadata;

    std::string line;
    std::size_t lineNumber = 0;

    bool hasProblemType = false;
    bool hasGeneratorType = false;
    bool hasGeneratorVersion = false;
    bool hasSeed = false;

    // Read and validate the file header.
    if (!std::getline(input, line))
    {
        errorMessage = "Empty metadata file.";
        return false;
    }

    if (!line.empty() && line.back() == '\r')
    {
        line.pop_back();
    }

    ++lineNumber;

    if (line != "INSTANCE_METADATA 1")
    {
        errorMessage = "Invalid metadata header.";
        return false;
    }

    // Read metadata fields.
    while (std::getline(input, line))
    {
        ++lineNumber;

        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        if (line.empty())
        {
            continue;
        }

        std::istringstream lineStream(line);

        std::string field;
        if (!(lineStream >> field))
        {
            continue;
        }

        if (field == "problem_type"
            || field == "generator_type"
            || field == "generator_version")
        {
            std::string value;

            if (!readQuotedValue(lineStream, value))
            {
                errorMessage =
                    "Invalid value at line "
                    + std::to_string(lineNumber) + ".";
                return false;
            }

            if (value.empty())
            {
                errorMessage =
                    "Empty required field at line "
                    + std::to_string(lineNumber) + ".";
                return false;
            }

            if (field == "problem_type")
            {
                if (hasProblemType)
                {
                    errorMessage = "Duplicate problem_type field.";
                    return false;
                }

                parsedMetadata.problemType = value;
                hasProblemType = true;
            }
            else if (field == "generator_type")
            {
                if (hasGeneratorType)
                {
                    errorMessage = "Duplicate generator_type field.";
                    return false;
                }

                parsedMetadata.generatorType = value;
                hasGeneratorType = true;
            }
            else
            {
                if (hasGeneratorVersion)
                {
                    errorMessage = "Duplicate generator_version field.";
                    return false;
                }

                parsedMetadata.generatorVersion = value;
                hasGeneratorVersion = true;
            }
        }
        else if (field == "seed")
        {
            if (hasSeed)
            {
                errorMessage = "Duplicate seed field.";
                return false;
            }

            std::string seedText;
            if (!(lineStream >> seedText))
            {
                errorMessage = "Missing seed value.";
                return false;
            }

            lineStream >> std::ws;
            if (!lineStream.eof())
            {
                errorMessage = "Unexpected content after seed.";
                return false;
            }

            unsigned int seed = 0;

            const auto result = std::from_chars(
                seedText.data(),
                seedText.data() + seedText.size(),
                seed
            );

            if (result.ec != std::errc{}
                || result.ptr != seedText.data() + seedText.size())
            {
                errorMessage = "Invalid seed value.";
                return false;
            }

            parsedMetadata.generatorSeed = seed;
            hasSeed = true;
        }
        else if (field.rfind("parameter.", 0) == 0)
        {
            const std::string key = field.substr(10);

            if (!isValidParameterKey(key))
            {
                errorMessage = "Invalid generator parameter key: " + key;
                return false;
            }

            std::string value;
            if (!readQuotedValue(lineStream, value))
            {
                errorMessage = "Invalid parameter value.";
                return false;
            }

            if (!parsedMetadata.generatorParameters.emplace(key, value).second)
            {
                errorMessage = "Duplicate generator parameter: " + key;
                return false;
            }
        }
        else
        {
            errorMessage = "Unknown metadata field: " + field;
            return false;
        }
    }

    if (input.bad())
    {
        errorMessage = "Failed to read metadata.";
        return false;
    }

    if (!hasProblemType || !hasGeneratorType || !hasGeneratorVersion)
    {
        errorMessage = "Missing required metadata fields.";
        return false;
    }

    metadata = std::move(parsedMetadata);
    errorMessage.clear();

    return true;
}
