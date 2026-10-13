
#pragma once

#include <iosfwd>
#include <string>

#include "problems/MIS.hpp"

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
