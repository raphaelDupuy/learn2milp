
#pragma once

#include <iosfwd>
#include <string>

#include "problems/MIS.hpp"

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
