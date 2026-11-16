#pragma once

#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#include "learning/MISBranchingModel.h"

/*
 * Text format of a branching model (written by
 * ml/branching_model_file.py):
 *
 *   MIS_BRANCHING_MODEL <1|2|3>
 *   features <count>
 *   feature <name> <mean> <standard deviation>     one line per feature
 *   [bipartite encoder, version 3 only, see below]
 *   layers <count>
 *   then, for each layer, one of:
 *   layer <inputs> <outputs> <relu|linear>         dense layer:
 *   <inputs values>                                  one line per output (W)
 *   <outputs values>                                 one line of biases
 *   graph_layer <inputs> <outputs> <relu|linear>   graph layer (version 2):
 *   <inputs values>                                  one line per output (W),
 *                                                    then W_neighbours and
 *                                                    W_node the same way
 *   <outputs values>                                 one line of biases
 *
 * The encoder of a bipartite model (version 3) is
 *
 *   vertex_inputs <count>, vertex_input <name> <mean> <std> lines
 *   constraint_inputs <count>, constraint_input <name> <mean> <std> lines
 *   vertex_layers <count>, dense layers
 *   constraint_layers <count>, dense layers
 *   message_layers <count>, then for each:
 *   message_layer <to_constraints|to_vertices> <targets> <sources> <hidden>
 *                 <outputs> <relu|linear>
 *   L (hidden lines of targets values), R (hidden lines of sources values),
 *   one line of hidden biases, P (outputs lines of hidden values),
 *   Q (outputs lines of targets values), one line of output biases
 *
 * with the vertex and constraint inputs of MISbranchingVertexInputName()
 * and MISbranchingConstraintInputName(), in that order.
 *
 * The layers are described in learning/MISBranchingModel.h. A model reads
 * all the compiled features or a subset of them, in any order. It is
 * refused if it names a feature that this program does not compute: it
 * was trained on another feature set.
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

    // Layers of the encoder included.
    int numberOfLayers() const;

    // Total number of weights and biases.
    std::size_t numberOfParameters() const;

private:
    friend bool readMISBranchingModel(
        std::istream&, LoadedMISBranchingModel&, std::string&);

    std::vector<int> indices_;
    std::vector<double> means_;
    std::vector<double> stds_;

    // Every weight matrix and bias vector. Moving the outer vector keeps
    // the inner buffers, which the layers point to.
    std::vector<std::vector<double>> storage_;

    std::vector<MISBranchingModelLayer> layers_;
    std::vector<MISBranchingModelLayer> vertexLayers_;
    std::vector<MISBranchingModelLayer> constraintLayers_;
    std::vector<MISBranchingMessageLayer> messageLayers_;

    // On the heap, so that moving the object keeps model_.encoder valid.
    std::unique_ptr<MISBranchingBipartiteEncoder> encoder_;

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
