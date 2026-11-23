#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>

#include "instances/InstanceMetadata.hpp"

// Metadata file of an instance file: PATH.meta, for every problem.
inline std::filesystem::path instanceMetadataPath(
    const std::filesystem::path& instancePath
)
{
    return std::filesystem::path(instancePath.string() + ".meta");
}

bool writeInstanceMetadata(
    std::ostream& output,
    const InstanceMetadata& metadata,
    std::string& errorMessage
);

bool readInstanceMetadata(
    std::istream& input,
    InstanceMetadata& metadata,
    std::string& errorMessage
);
