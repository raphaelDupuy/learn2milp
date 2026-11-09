#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "generators/ErdosRenyiGenerator.hpp"

struct MISBatchConfig
{
    std::vector<int> nValues;
    std::vector<double> pValues;
    std::vector<unsigned int> seeds;
    std::filesystem::path outputDirectory;
};

bool generateSingleMIS(
    const ErdosRenyiConfig& config,
    const std::filesystem::path& outputPath,
    std::string& errorMessage
);

bool generateBatchMIS(
    const MISBatchConfig& config,
    std::string& errorMessage
);
