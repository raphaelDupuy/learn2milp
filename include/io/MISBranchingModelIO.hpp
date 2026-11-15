#pragma once

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "learning/MISBranchingModel.h"

/*
 * Text format of a branching model (written by
 * ml/branching_model_file.py):
 *
 *   MIS_BRANCHING_MODEL 1
 *   features <count>
 *   feature <name> <mean> <standard deviation>     one line per feature
 *   layers <count>
 *   layer <inputs> <outputs> <relu|linear>         then, for each layer:
 *   <inputs values>                                one line per output
 *   <outputs values>                               one line of biases
 *
 * A model reads all the compiled features or a subset of them, in any
 * order. It is refused if it names a feature that this program does not
 * compute: it was trained on another feature set.
 */

// A model with the storage its C view points to.
class LoadedMISBranchingModel
{
public:
    LoadedMISBranchingModel() = default;

    // Movable but not copyable: a copy would keep pointing to the
    // storage of the original.
    LoadedMISBranchingModel(LoadedMISBranchingModel&&) = default;
    LoadedMISBranchingModel& operator=(LoadedMISBranchingModel&&) = default;
    LoadedMISBranchingModel(const LoadedMISBranchingModel&) = delete;
    LoadedMISBranchingModel& operator=(
        const LoadedMISBranchingModel&) = delete;

    // View for the C functions. It points into this object and must not
    // outlive it or be used after the object is modified.
    const MISBranchingModel& view() const;

    int numberOfLayers() const;

    // Total number of weights and biases.
    std::size_t numberOfParameters() const;

private:
    friend bool readMISBranchingModel(
        std::istream&, LoadedMISBranchingModel&, std::string&);

    std::vector<int> indices_;
    std::vector<double> means_;
    std::vector<double> stds_;
    std::vector<std::vector<double>> weights_;
    std::vector<std::vector<double>> biases_;
    std::vector<MISBranchingModelLayer> layers_;
    MISBranchingModel model_{};
};

bool readMISBranchingModel(
    std::istream& input,
    LoadedMISBranchingModel& model,
    std::string& errorMessage
);

bool loadMISBranchingModel(
    const std::filesystem::path& path,
    LoadedMISBranchingModel& model,
    std::string& errorMessage
);
