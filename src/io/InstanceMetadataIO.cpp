
#include "io/InstanceMetadataIO.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <ostream>

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

    for (const auto& [key, value] : metadata.generatorParameters)
    {
        if (!isValidParameterKey(key))
        {
            errorMessage = "Invalid generator parameter key: " + key;
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
