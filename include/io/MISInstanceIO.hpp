
#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "problems/MIS.hpp"

// Instance files: NAME.mis (the graph) and NAME.mis.meta (its metadata,
// io/InstanceMetadataIO.hpp).
extern const char* const MIS_INSTANCE_EXTENSION;

std::filesystem::path misInstanceMetadataPath(
    const std::filesystem::path& instancePath
);

// The .mis files of a directory (not of its subdirectories), in name
// order. Throws std::filesystem::filesystem_error if it cannot be read.
std::vector<std::filesystem::path> listMISInstances(
    const std::filesystem::path& directory
);

// Checks that the instance is a simple graph in canonical form: endpoints
// in [0, n), u < v for every edge, no duplicate edge. On failure,
// errorMessage describes the first problem found; it is left untouched
// on success.
bool validateMISInstance(
    const MISInstance& instance,
    std::string& errorMessage
);

bool writeMISInstance(
    std::ostream& output,
    const MISInstance& instance,
    std::string& errorMessage
);

bool readMISInstance(
    std::istream& input,
    MISInstance& instance,
    std::string& errorMessage
);
