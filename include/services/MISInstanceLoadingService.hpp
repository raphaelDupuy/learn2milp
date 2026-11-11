#pragma once

#include <filesystem>
#include <string>

#include "instances/InstanceMetadata.hpp"
#include "problems/MIS.hpp"

struct LoadedMISInstance
{
    MISInstance instance;
    InstanceMetadata metadata;
};

bool loadMISInstance(
    const std::filesystem::path& instancePath,
    LoadedMISInstance& loadedInstance,
    std::string& errorMessage
);