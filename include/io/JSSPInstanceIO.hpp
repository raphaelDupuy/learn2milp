#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "problems/JSSP.hpp"

/*
 * Instance files: NAME.jsp (the instance) and NAME.jsp.meta (its
 * metadata, io/InstanceMetadataIO.hpp, absent for instances that were not
 * generated here).
 *
 * The format is the one of the OR-Library and of the published benchmarks
 * (ft, la, ta, ...): the number of jobs and of machines, then one line
 * per job with, for each operation in processing order, its machine
 * (numbered from 0) and its duration:
 *
 *   3 2
 *   0 5 1 3
 *   1 2 0 4
 *   0 1 1 1
 *
 * When reading, a '#' starts a comment that runs to the end of the line,
 * as in the files of JSPLIB.
 */
extern const char* const JSSP_INSTANCE_EXTENSION;

// The .jsp files of a directory (not of its subdirectories), in name
// order. Throws std::filesystem::filesystem_error if it cannot be read.
std::vector<std::filesystem::path> listJSSPInstances(
    const std::filesystem::path& directory
);

// Fails, without writing anything, if the instance is not valid
// (validateJSSPInstance()).
bool writeJSSPInstance(
    std::ostream& output,
    const JSSPInstance& instance,
    std::string& errorMessage
);

// Fails if the file is malformed or the instance is not valid; instance
// is only changed on success.
bool readJSSPInstance(
    std::istream& input,
    JSSPInstance& instance,
    std::string& errorMessage
);
