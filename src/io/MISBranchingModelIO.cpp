#include "io/MISBranchingModelIO.hpp"

#include <cmath>
#include <fstream>
#include <istream>
#include <sstream>
#include <utility>

#include "learning/MISBranchingFeatures.h"

namespace
{

// Largest accepted layer width: far above any model evaluated at every
// node, and small enough to reject a corrupted size before allocating.
const int MAX_LAYER_WIDTH = 4096;
const int MAX_LAYERS = 64;

// Reads one line of exactly count finite numbers.
bool readNumbers(
    std::istream& input,
    int count,
    std::vector<double>& values
)
{
    std::string line;

    if (!std::getline(input, line))
    {
        return false;
    }

    std::istringstream stream(line);
    double value = 0.0;

    while (stream >> value)
    {
        if (!std::isfinite(value))
        {
            return false;
        }

        values.push_back(value);
    }

    // Stopped on something that is not a number, or wrong count.
    return stream.eof()
        && values.size() % static_cast<std::size_t>(count) == 0
        && !values.empty();
}

// Reads a line and splits it into whitespace-separated tokens.
bool readTokens(std::istream& input, std::vector<std::string>& tokens)
{
    std::string line;
    tokens.clear();

    if (!std::getline(input, line))
    {
        return false;
    }

    std::istringstream stream(line);
    std::string token;

    while (stream >> token)
    {
        tokens.push_back(token);
    }

    return true;
}

bool parseCount(const std::string& text, int maximum, int& value)
{
    try
    {
        std::size_t position = 0;
        const long long parsed = std::stoll(text, &position);

        if (position != text.size() || parsed < 1 || parsed > maximum)
        {
            return false;
        }

        value = static_cast<int>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool parseReal(const std::string& text, double& value)
{
    try
    {
        std::size_t position = 0;
        value = std::stod(text, &position);

        return position == text.size() && std::isfinite(value);
    }
    catch (...)
    {
        return false;
    }
}

// Count of a section header "<keyword> <count>", from minimum to maximum.
bool readCount(
    std::istream& input,
    const std::string& keyword,
    int minimum,
    int maximum,
    int& value,
    std::string& errorMessage
)
{
    std::vector<std::string> tokens;

    if (readTokens(input, tokens) && tokens.size() == 2 && tokens[0] == keyword)
    {
        if (minimum == 0 && tokens[1] == "0")
        {
            value = 0;
            return true;
        }

        if (parseCount(tokens[1], maximum, value) && value >= minimum)
            return true;
    }

    errorMessage = "Expected " + keyword + " <integer>.";
    return false;
}

// Reads rows lines of columns finite numbers as one stored matrix;
// nullptr if a line is malformed.
const double* readMatrix(
    std::istream& input,
    int rows,
    int columns,
    std::vector<std::vector<double>>& storage
)
{
    std::vector<double> values;

    for (int r = 0; r < rows; ++r)
    {
        std::vector<double> row;

        if (!readNumbers(input, columns, row)
            || row.size() != static_cast<std::size_t>(columns))
        {
            return nullptr;
        }

        values.insert(values.end(), row.begin(), row.end());
    }

    storage.push_back(std::move(values));
    return storage.back().data();
}

// Reads count dense (or, if allowed, graph) layers chaining from width.
bool readLayers(
    std::istream& input,
    int count,
    bool graphAllowed,
    const std::string& stage,
    int& width,
    std::vector<MISBranchingModelLayer>& layers,
    std::vector<std::vector<double>>& storage,
    std::string& errorMessage
)
{
    std::vector<std::string> tokens;

    for (int l = 0; l < count; ++l)
    {
        const std::string where = stage + " " + std::to_string(l) + ": ";
        int inputs = 0;
        int outputs = 0;

        if (!readTokens(input, tokens)
            || tokens.size() != 4
            || (tokens[0] != "layer"
                && !(graphAllowed && tokens[0] == "graph_layer"))
            || !parseCount(tokens[1], MAX_LAYER_WIDTH, inputs)
            || !parseCount(tokens[2], MAX_LAYER_WIDTH, outputs)
            || (tokens[3] != "relu" && tokens[3] != "linear"))
        {
            errorMessage = where
                + (graphAllowed
                    ? "expected: layer|graph_layer <inputs> <outputs> "
                      "<relu|linear>."
                    : "expected: layer <inputs> <outputs> <relu|linear>.");
            return false;
        }

        if (inputs != width)
        {
            errorMessage =
                where + "takes " + std::to_string(inputs)
                + " inputs, but the previous stage produces "
                + std::to_string(width) + ".";
            return false;
        }

        // A graph layer has three matrices, W, W_neighbours and W_node.
        MISBranchingModelLayer layer{};
        layer.number_of_inputs = inputs;
        layer.number_of_outputs = outputs;
        layer.relu = tokens[3] == "relu" ? 1 : 0;
        layer.graph = tokens[0] == "graph_layer" ? 1 : 0;
        layer.weights = readMatrix(input, outputs, inputs, storage);

        if (layer.graph && layer.weights != nullptr)
        {
            layer.neighbour_weights =
                readMatrix(input, outputs, inputs, storage);

            if (layer.neighbour_weights != nullptr)
            {
                layer.node_weights =
                    readMatrix(input, outputs, inputs, storage);
            }
        }

        if (layer.weights == nullptr
            || (layer.graph
                && (layer.neighbour_weights == nullptr
                    || layer.node_weights == nullptr)))
        {
            errorMessage = where + "every weight row must have "
                + std::to_string(inputs) + " finite numbers.";
            return false;
        }

        layer.biases = readMatrix(input, 1, outputs, storage);

        if (layer.biases == nullptr)
        {
            errorMessage = where + "bias row must have "
                + std::to_string(outputs) + " finite numbers.";
            return false;
        }

        layers.push_back(layer);
        width = outputs;
    }

    return true;
}

// Reads the named inputs of one side of a bipartite model: exactly the
// expected names, in order, with positive deviations.
bool readInputs(
    std::istream& input,
    const std::string& side,
    int expectedCount,
    const char* (*name)(int),
    const double*& means,
    const double*& stds,
    std::vector<std::vector<double>>& storage,
    std::string& errorMessage
)
{
    int count = 0;

    if (!readCount(input, side + "_inputs", 1, MAX_LAYER_WIDTH, count,
            errorMessage))
    {
        return false;
    }

    if (count != expectedCount)
    {
        errorMessage = "The model has " + std::to_string(count) + " " + side
            + " inputs, but this program computes "
            + std::to_string(expectedCount) + ".";
        return false;
    }

    std::vector<double> meanValues;
    std::vector<double> stdValues;
    std::vector<std::string> tokens;

    for (int i = 0; i < count; ++i)
    {
        double mean = 0.0;
        double std = 0.0;

        if (!readTokens(input, tokens)
            || tokens.size() != 4
            || tokens[0] != side + "_input"
            || tokens[1] != name(i)
            || !parseReal(tokens[2], mean)
            || !parseReal(tokens[3], std)
            || !(std > 0.0))
        {
            errorMessage = "Expected " + side + "_input " + name(i)
                + " <mean> <positive std>.";
            return false;
        }

        meanValues.push_back(mean);
        stdValues.push_back(std);
    }

    storage.push_back(std::move(meanValues));
    means = storage.back().data();
    storage.push_back(std::move(stdValues));
    stds = storage.back().data();
    return true;
}

// Reads the encoder of a bipartite model; vertexWidth receives the width
// of the final vertex states.
bool readEncoder(
    std::istream& input,
    int featureCount,
    MISBranchingBipartiteEncoder& encoder,
    std::vector<MISBranchingModelLayer>& vertexLayers,
    std::vector<MISBranchingModelLayer>& constraintLayers,
    std::vector<MISBranchingMessageLayer>& messageLayers,
    std::vector<std::vector<double>>& storage,
    int& vertexWidth,
    std::string& errorMessage
)
{
    int constraintWidth = MIS_BRANCHING_CONSTRAINT_INPUT_COUNT;
    int count = 0;

    vertexWidth = featureCount + MIS_BRANCHING_VERTEX_INPUT_COUNT;

    if (!readInputs(input, "vertex", MIS_BRANCHING_VERTEX_INPUT_COUNT,
            MISbranchingVertexInputName, encoder.vertex_input_means,
            encoder.vertex_input_stds, storage, errorMessage)
        || !readInputs(input, "constraint",
            MIS_BRANCHING_CONSTRAINT_INPUT_COUNT,
            MISbranchingConstraintInputName, encoder.constraint_input_means,
            encoder.constraint_input_stds, storage, errorMessage)
        || !readCount(input, "vertex_layers", 0, MAX_LAYERS, count,
            errorMessage)
        || !readLayers(input, count, false, "Vertex layer", vertexWidth,
            vertexLayers, storage, errorMessage)
        || !readCount(input, "constraint_layers", 0, MAX_LAYERS, count,
            errorMessage)
        || !readLayers(input, count, false, "Constraint layer",
            constraintWidth, constraintLayers, storage, errorMessage)
        || !readCount(input, "message_layers", 0, MAX_LAYERS, count,
            errorMessage))
    {
        return false;
    }

    std::vector<std::string> tokens;

    for (int l = 0; l < count; ++l)
    {
        const std::string where = "Message layer " + std::to_string(l) + ": ";
        MISBranchingMessageLayer layer{};

        if (!readTokens(input, tokens)
            || tokens.size() != 7
            || tokens[0] != "message_layer"
            || (tokens[1] != "to_constraints" && tokens[1] != "to_vertices")
            || !parseCount(tokens[2], MAX_LAYER_WIDTH, layer.number_of_target_inputs)
            || !parseCount(tokens[3], MAX_LAYER_WIDTH, layer.number_of_source_inputs)
            || !parseCount(tokens[4], MAX_LAYER_WIDTH, layer.number_of_hidden)
            || !parseCount(tokens[5], MAX_LAYER_WIDTH, layer.number_of_outputs)
            || (tokens[6] != "relu" && tokens[6] != "linear"))
        {
            errorMessage = where + "expected: message_layer "
                "<to_constraints|to_vertices> <targets> <sources> <hidden> "
                "<outputs> <relu|linear>.";
            return false;
        }

        layer.to_vertices = tokens[1] == "to_vertices" ? 1 : 0;
        layer.relu = tokens[6] == "relu" ? 1 : 0;

        int& targetWidth = layer.to_vertices ? vertexWidth : constraintWidth;
        const int sourceWidth =
            layer.to_vertices ? constraintWidth : vertexWidth;

        if (layer.number_of_target_inputs != targetWidth
            || layer.number_of_source_inputs != sourceWidth)
        {
            errorMessage = where + "takes " + tokens[2] + " target and "
                + tokens[3] + " source inputs, but the previous stages "
                "produce " + std::to_string(targetWidth) + " and "
                + std::to_string(sourceWidth) + ".";
            return false;
        }

        const int targets = layer.number_of_target_inputs;
        const int hidden = layer.number_of_hidden;
        const int outputs = layer.number_of_outputs;

        layer.target_weights = readMatrix(input, hidden, targets, storage);
        layer.source_weights = layer.target_weights == nullptr ? nullptr
            : readMatrix(input, hidden, layer.number_of_source_inputs, storage);
        layer.hidden_biases = layer.source_weights == nullptr ? nullptr
            : readMatrix(input, 1, hidden, storage);
        layer.message_weights = layer.hidden_biases == nullptr ? nullptr
            : readMatrix(input, outputs, hidden, storage);
        layer.own_weights = layer.message_weights == nullptr ? nullptr
            : readMatrix(input, outputs, targets, storage);
        layer.biases = layer.own_weights == nullptr ? nullptr
            : readMatrix(input, 1, outputs, storage);

        if (layer.biases == nullptr)
        {
            errorMessage = where + "a matrix or bias line has the wrong "
                "number of finite numbers.";
            return false;
        }

        messageLayers.push_back(layer);
        targetWidth = outputs;
    }

    return true;
}

} // namespace

const MISBranchingModel& LoadedMISBranchingModel::view() const
{
    return model_;
}

int LoadedMISBranchingModel::numberOfLayers() const
{
    return static_cast<int>(
        layers_.size() + vertexLayers_.size() + constraintLayers_.size()
        + messageLayers_.size());
}

std::size_t LoadedMISBranchingModel::numberOfParameters() const
{
    std::size_t count = 0;

    for (const std::vector<double>& values : storage_)
        count += values.size();

    // The means and deviations of the inputs are not parameters.
    if (encoder_ != nullptr)
    {
        count -= 2 * (MIS_BRANCHING_VERTEX_INPUT_COUNT
            + MIS_BRANCHING_CONSTRAINT_INPUT_COUNT);
    }

    return count;
}

bool readMISBranchingModel(
    std::istream& input,
    LoadedMISBranchingModel& model,
    std::string& errorMessage
)
{
    LoadedMISBranchingModel parsed;
    std::vector<std::string> tokens;

    if (!readTokens(input, tokens)
        || tokens.size() != 2
        || tokens[0] != "MIS_BRANCHING_MODEL"
        || (tokens[1] != "1" && tokens[1] != "2" && tokens[1] != "3"))
    {
        errorMessage = "Invalid header or unsupported model format version.";
        return false;
    }

    // Graph layers exist in version 2, bipartite encoders in version 3.
    const int version = tokens[1][0] - '0';
    int featureCount = 0;

    if (!readTokens(input, tokens)
        || tokens.size() != 2
        || tokens[0] != "features"
        || !parseCount(tokens[1], MAX_LAYER_WIDTH, featureCount))
    {
        errorMessage = "Expected feature-count field: features <integer>.";
        return false;
    }

    if (featureCount > MIS_BRANCHING_FEATURE_COUNT)
    {
        errorMessage =
            "The model expects " + std::to_string(featureCount)
            + " features, but this program computes "
            + std::to_string(MIS_BRANCHING_FEATURE_COUNT) + ".";
        return false;
    }

    for (int f = 0; f < featureCount; ++f)
    {
        double mean = 0.0;
        double std = 0.0;

        if (!readTokens(input, tokens)
            || tokens.size() != 4
            || tokens[0] != "feature"
            || !parseReal(tokens[2], mean)
            || !parseReal(tokens[3], std))
        {
            errorMessage =
                "Expected feature field: feature <name> <mean> <std>.";
            return false;
        }

        int index = -1;

        for (int k = 0; k < MIS_BRANCHING_FEATURE_COUNT; ++k)
        {
            if (tokens[1] == MISbranchingFeatureName(k))
                index = k;
        }

        if (index < 0)
        {
            errorMessage =
                "Feature '" + tokens[1] + "' of the model is not computed "
                "by this program: the model was trained on another "
                "feature set.";
            return false;
        }

        for (const int used : parsed.indices_)
        {
            if (used == index)
            {
                errorMessage =
                    "Feature '" + tokens[1] + "' appears twice.";
                return false;
            }
        }

        parsed.indices_.push_back(index);

        if (!(std > 0.0))
        {
            errorMessage =
                "Standard deviation of feature '" + tokens[1]
                + "' must be positive.";
            return false;
        }

        parsed.means_.push_back(mean);
        parsed.stds_.push_back(std);
    }

    int width = featureCount;

    if (version == 3)
    {
        parsed.encoder_ = std::make_unique<MISBranchingBipartiteEncoder>();

        if (!readEncoder(input, featureCount, *parsed.encoder_,
                parsed.vertexLayers_, parsed.constraintLayers_,
                parsed.messageLayers_, parsed.storage_, width, errorMessage))
        {
            return false;
        }
    }

    int layerCount = 0;

    if (!readCount(input, "layers", 1, MAX_LAYERS, layerCount, errorMessage)
        || !readLayers(input, layerCount, version == 2, "Layer", width,
            parsed.layers_, parsed.storage_, errorMessage))
    {
        return false;
    }

    if (width != 1 || parsed.layers_.back().relu != 0)
    {
        errorMessage =
            "The last layer must be linear and have a single output.";
        return false;
    }

    input >> std::ws;

    if (input.peek() != std::char_traits<char>::eof())
    {
        errorMessage = "Unexpected content after the last layer.";
        return false;
    }

    // Point the C view to the final storage.
    parsed.model_.number_of_features = featureCount;
    parsed.model_.feature_indices = parsed.indices_.data();
    parsed.model_.feature_means = parsed.means_.data();
    parsed.model_.feature_stds = parsed.stds_.data();
    parsed.model_.number_of_layers = layerCount;
    parsed.model_.layers = parsed.layers_.data();

    if (parsed.encoder_ != nullptr)
    {
        MISBranchingBipartiteEncoder& encoder = *parsed.encoder_;
        encoder.number_of_vertex_layers =
            static_cast<int>(parsed.vertexLayers_.size());
        encoder.vertex_layers = parsed.vertexLayers_.data();
        encoder.number_of_constraint_layers =
            static_cast<int>(parsed.constraintLayers_.size());
        encoder.constraint_layers = parsed.constraintLayers_.data();
        encoder.number_of_message_layers =
            static_cast<int>(parsed.messageLayers_.size());
        encoder.message_layers = parsed.messageLayers_.data();
        parsed.model_.encoder = parsed.encoder_.get();
    }

    if (!MISbranchingModelIsValid(&parsed.model_))
    {
        errorMessage = "The model is not consistent.";
        return false;
    }

    // Moving the vectors keeps their heap storage, so the view stays valid.
    model = std::move(parsed);
    errorMessage.clear();
    return true;
}

bool loadMISBranchingModel(
    const std::filesystem::path& path,
    LoadedMISBranchingModel& model,
    std::string& errorMessage
)
{
    std::ifstream file(path);

    if (!file.is_open())
    {
        errorMessage = "Unable to open model file: " + path.string();
        return false;
    }

    if (!readMISBranchingModel(file, model, errorMessage))
    {
        errorMessage = "Invalid model file '" + path.string() + "': "
            + errorMessage;
        return false;
    }

    return true;
}
