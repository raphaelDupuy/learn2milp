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

} // namespace

const MISBranchingModel& LoadedMISBranchingModel::view() const
{
    return model_;
}

int LoadedMISBranchingModel::numberOfLayers() const
{
    return static_cast<int>(layers_.size());
}

std::size_t LoadedMISBranchingModel::numberOfParameters() const
{
    std::size_t count = 0;

    for (std::size_t l = 0; l < layers_.size(); ++l)
    {
        count += weights_[l].size() + biases_[l].size();
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
        || tokens != std::vector<std::string>{"MIS_BRANCHING_MODEL", "1"})
    {
        errorMessage = "Invalid header or unsupported model format version.";
        return false;
    }

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

    int layerCount = 0;

    if (!readTokens(input, tokens)
        || tokens.size() != 2
        || tokens[0] != "layers"
        || !parseCount(tokens[1], MAX_LAYERS, layerCount))
    {
        errorMessage = "Expected layer-count field: layers <integer>.";
        return false;
    }

    int expectedInputs = featureCount;
    std::vector<int> relu;

    for (int l = 0; l < layerCount; ++l)
    {
        const std::string where = "Layer " + std::to_string(l) + ": ";
        int inputs = 0;
        int outputs = 0;

        if (!readTokens(input, tokens)
            || tokens.size() != 4
            || tokens[0] != "layer"
            || !parseCount(tokens[1], MAX_LAYER_WIDTH, inputs)
            || !parseCount(tokens[2], MAX_LAYER_WIDTH, outputs)
            || (tokens[3] != "relu" && tokens[3] != "linear"))
        {
            errorMessage =
                where + "expected: layer <inputs> <outputs> <relu|linear>.";
            return false;
        }

        if (inputs != expectedInputs)
        {
            errorMessage =
                where + "takes " + std::to_string(inputs)
                + " inputs, but the previous stage produces "
                + std::to_string(expectedInputs) + ".";
            return false;
        }

        std::vector<double> weights;

        for (int o = 0; o < outputs; ++o)
        {
            std::vector<double> row;

            if (!readNumbers(input, inputs, row)
                || row.size() != static_cast<std::size_t>(inputs))
            {
                errorMessage =
                    where + "weight row " + std::to_string(o) + " must have "
                    + std::to_string(inputs) + " finite numbers.";
                return false;
            }

            weights.insert(weights.end(), row.begin(), row.end());
        }

        std::vector<double> biases;

        if (!readNumbers(input, outputs, biases)
            || biases.size() != static_cast<std::size_t>(outputs))
        {
            errorMessage =
                where + "bias row must have " + std::to_string(outputs)
                + " finite numbers.";
            return false;
        }

        parsed.weights_.push_back(std::move(weights));
        parsed.biases_.push_back(std::move(biases));
        relu.push_back(tokens[3] == "relu" ? 1 : 0);

        MISBranchingModelLayer layer{};
        layer.number_of_inputs = inputs;
        layer.number_of_outputs = outputs;
        layer.relu = relu.back();
        parsed.layers_.push_back(layer);

        expectedInputs = outputs;
    }

    if (expectedInputs != 1 || relu.back() != 0)
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
    for (std::size_t l = 0; l < parsed.layers_.size(); ++l)
    {
        parsed.layers_[l].weights = parsed.weights_[l].data();
        parsed.layers_[l].biases = parsed.biases_[l].data();
    }

    parsed.model_.number_of_features = featureCount;
    parsed.model_.feature_indices = parsed.indices_.data();
    parsed.model_.feature_means = parsed.means_.data();
    parsed.model_.feature_stds = parsed.stds_.data();
    parsed.model_.number_of_layers = layerCount;
    parsed.model_.layers = parsed.layers_.data();

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
