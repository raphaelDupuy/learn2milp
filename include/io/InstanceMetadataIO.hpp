
#pragma once

#include <iosfwd>
#include <string>

#include "instances/InstanceMetadata.hpp"

bool writeInstanceMetadata(
    std::ostream& output,
    const InstanceMetadata& metadata,
    std::string& errorMessage
);
