
#pragma once

#include <map>
#include <optional>
#include <string>

struct InstanceMetadata
{
    std::string problemType;

    std::string generatorType;
    std::string generatorVersion;

    std::optional<unsigned int> generatorSeed;

    std::map<std::string, std::string> generatorParameters;
};
