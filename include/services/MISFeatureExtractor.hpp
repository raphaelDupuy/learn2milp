#pragma once

#include <string>

#include "features/MISFeatures.hpp"
#include "problems/MIS.hpp"

bool extractMISFeatures(
    const MISInstance& instance,
    MISFeatures& features,
    std::string& errorMessage
);