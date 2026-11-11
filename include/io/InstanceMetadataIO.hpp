#pragma once

#include <iosfwd>
#include <string>

#include "instances/InstanceMetadata.hpp"

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
