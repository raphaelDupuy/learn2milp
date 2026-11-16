#include "io/MISBranchingModelIO.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "learning/MISAdjacency.hpp"
#include "learning/MISBranchingFeatures.h"

namespace
{

int failures = 0;

void check(bool condition, const std::string& testName)
{
    if (condition)
    {
        std::cout << "[PASS] " << testName << '\n';
    }
    else
    {
        std::cerr << "[FAIL] " << testName << '\n';
        ++failures;
    }
}

const int K = MIS_BRANCHING_FEATURE_COUNT;

// "feature" lines with the compiled names, mean 0 and std 1.
std::string featureLines()
{
    std::string text = "features " + std::to_string(K) + "\n";

    for (int f = 0; f < K; ++f)
    {
        text += std::string("feature ") + MISbranchingFeatureName(f)
            + " 0 1\n";
    }

    return text;
}

// One line of K numbers: value at the given positions, 0 elsewhere.
std::string row(const std::vector<std::pair<int, double>>& entries)
{
    std::vector<double> values(K, 0.0);

    for (const auto& [index, value] : entries)
        values[index] = value;

    std::ostringstream line;

    for (int f = 0; f < K; ++f)
        line << (f > 0 ? " " : "") << values[f];

    return line.str() + "\n";
}

const std::string HEADER = "MIS_BRANCHING_MODEL 1\n";

// Linear model: score = sum of (f + 1) * x_f + 0.5.
std::string linearModel()
{
    std::string weights;

    for (int f = 0; f < K; ++f)
        weights += (f > 0 ? " " : "") + std::to_string(f + 1);

    return HEADER + featureLines()
        + "layers 1\nlayer " + std::to_string(K) + " 1 linear\n"
        + weights + "\n0.5\n";
}

// Network: u0 = relu(x0 - x1), u1 = relu(x1), score = 2 u0 - 3 u1 + 1.
std::string networkModel()
{
    return HEADER + featureLines()
        + "layers 2\nlayer " + std::to_string(K) + " 2 relu\n"
        + row({{0, 1.0}, {1, -1.0}})
        + row({{1, 1.0}})
        + "0 0\n"
        + "layer 2 1 linear\n2 -3\n1\n";
}

bool parse(const std::string& text, LoadedMISBranchingModel& model)
{
    std::istringstream input(text);
    std::string error;

    return readMISBranchingModel(input, model, error);
}

double evaluate(
    const LoadedMISBranchingModel& model,
    const std::vector<std::pair<int, double>>& entries
)
{
    std::vector<double> features(K, 0.0);

    for (const auto& [index, value] : entries)
        features[index] = value;

    std::vector<double> work(MISbranchingModelWorkSize(&model.view()));

    return MISevaluateBranchingModel(
        &model.view(), features.data(), work.data());
}

// A small node for the graph layers: edges 0-1, 1-2, 0-3 and 4-5, with
// candidates 2, 0, 1 and 4 (in that order, LP values 0.2, 0.1, 0.5 and
// 0.4). Vertices 3 and 5 are not candidates: 0 has the candidate
// neighbour 1 only, and 4 has none.
const std::vector<int> NODE_VERTICES = {2, 0, 1, 4};
const std::vector<double> NODE_LP_VALUES = {0.2, 0.1, 0.5, 0.4};

MISAdjacency nodeGraph()
{
    return buildMISAdjacency(MISInstance{6, {{0, 1}, {1, 2}, {0, 3}, {4, 5}}});
}

// Scores of the small node, reading feature 0 as the LP value; false if
// the evaluation fails.
bool evaluateNode(
    const LoadedMISBranchingModel& model,
    const std::vector<int>& vertices,
    std::vector<double>& scores
)
{
    const MISAdjacency adjacency = nodeGraph();
    const MISFeatureGraph graph = adjacency.view();
    const int n = static_cast<int>(vertices.size());

    std::vector<double> features(static_cast<std::size_t>(n) * K, 0.0);

    for (int c = 0; c < n; ++c)
        features[static_cast<std::size_t>(c) * K] = NODE_LP_VALUES[c];

    std::vector<double> work(static_cast<std::size_t>(
        MISbranchingModelNodeWorkSize(&model.view(), n)));
    std::vector<int> positions(static_cast<std::size_t>(graph.number_of_vertices));
    scores.assign(static_cast<std::size_t>(n), 0.0);

    return MISevaluateBranchingModelOnNode(
        &model.view(), &graph, n, vertices.data(), features.data(),
        work.data(), positions.data(), scores.data()) == 0;
}

bool near(const std::vector<double>& values, const std::vector<double>& expected)
{
    if (values.size() != expected.size())
        return false;

    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (std::fabs(values[i] - expected[i]) > 1e-12)
            return false;
    }

    return true;
}

void expectRejection(const std::string& testName, const std::string& text)
{
    LoadedMISBranchingModel model;
    std::istringstream input(text);
    std::string error;

    check(
        !readMISBranchingModel(input, model, error) && !error.empty(),
        testName
    );
}

std::string replaced(
    std::string text,
    const std::string& from,
    const std::string& to
)
{
    const std::size_t position = text.find(from);

    if (position == std::string::npos)
    {
        std::cerr << "  test setup: '" << from << "' not found\n";
        ++failures;
        return text;
    }

    return text.replace(position, from.size(), to);
}

} // namespace

int main()
{
    // Linear model.
    {
        LoadedMISBranchingModel model;

        check(parse(linearModel(), model), "Linear model is read");

        std::vector<std::pair<int, double>> ones;

        for (int f = 0; f < K; ++f)
            ones.push_back({f, 1.0});

        check(
            model.numberOfLayers() == 1
            && model.numberOfParameters() == static_cast<std::size_t>(K + 1)
            && MISbranchingModelIsValid(&model.view())
            && std::fabs(evaluate(model, ones) - (K * (K + 1) / 2 + 0.5))
                < 1e-12
            && std::fabs(evaluate(model, {{2, -2.0}}) - (-6.0 + 0.5)) < 1e-12,
            "Linear model is a weighted sum plus the bias"
        );
    }

    // Standardization.
    {
        const std::string text = replaced(
            linearModel(),
            std::string("feature ") + MISbranchingFeatureName(0) + " 0 1",
            std::string("feature ") + MISbranchingFeatureName(0) + " 1 2");

        LoadedMISBranchingModel model;

        check(
            parse(text, model)
            // Feature 0 has weight 1: (5 - 1) / 2 = 2, plus the bias.
            && std::fabs(evaluate(model, {{0, 5.0}}) - 2.5) < 1e-12,
            "Features are standardized with their mean and deviation"
        );
    }

    // Network with a hidden ReLU layer.
    {
        LoadedMISBranchingModel model;

        check(parse(networkModel(), model), "Network model is read");

        check(
            model.numberOfLayers() == 2
            && model.numberOfParameters()
                == static_cast<std::size_t>(2 * K + 2 + 2 + 1)
            && std::fabs(evaluate(model, {{0, 3.0}, {1, 1.0}}) - 2.0) < 1e-12
            && std::fabs(evaluate(model, {{1, -2.0}}) - 5.0) < 1e-12
            && std::fabs(evaluate(model, {{1, 1.0}}) - (-2.0)) < 1e-12,
            "Hidden units are clipped at zero before the output layer"
        );

        // Moving a model keeps its view usable.
        LoadedMISBranchingModel moved = std::move(model);

        check(
            MISbranchingModelIsValid(&moved.view())
            && std::fabs(evaluate(moved, {{0, 3.0}, {1, 1.0}}) - 2.0) < 1e-12,
            "A moved model can still be evaluated"
        );
    }

    // Rejections.
    const std::string linear = linearModel();
    const std::string network = networkModel();
    const std::string name0 = MISbranchingFeatureName(0);
    const std::string name1 = MISbranchingFeatureName(1);

    expectRejection("Empty file is rejected", "");

    expectRejection(
        "Unsupported format version is rejected",
        replaced(linear, "MIS_BRANCHING_MODEL 1", "MIS_BRANCHING_MODEL 3"));

    // Graph layers.
    const std::string graphLinear =
        "MIS_BRANCHING_MODEL 2\nfeatures 1\nfeature " + name0 + " 0 1\n"
        "layers 1\ngraph_layer 1 1 linear\n1\n10\n100\n0.5\n";

    {
        LoadedMISBranchingModel model;
        std::vector<double> scores;

        // score = x + 10 mean(x of the candidate neighbours)
        //           + 100 mean(x of the candidates) + 0.5, mean x = 0.3
        check(
            parse(graphLinear, model)
            && MISbranchingModelIsValid(&model.view())
            && MISbranchingModelHasGraphLayers(&model.view())
            && model.numberOfParameters() == 4
            && evaluateNode(model, NODE_VERTICES, scores)
            && near(scores, {0.2 + 5.0 + 30.5, 0.1 + 5.0 + 30.5,
                             0.5 + 1.5 + 30.5, 0.4 + 0.0 + 30.5}),
            "Graph layer adds the means over the candidate neighbours and "
            "over the node"
        );
    }

    {
        // h = relu(0.3 - x) = 0.1, 0.2, 0, 0, then h + 10 mean(h of the
        // candidate neighbours).
        LoadedMISBranchingModel model;
        std::vector<double> scores;

        check(
            parse("MIS_BRANCHING_MODEL 2\nfeatures 1\nfeature " + name0
                    + " 0 1\nlayers 2\nlayer 1 1 relu\n-1\n0.3\n"
                    "graph_layer 1 1 linear\n1\n10\n0\n0\n",
                model)
            && model.numberOfLayers() == 2
            && evaluateNode(model, NODE_VERTICES, scores)
            && near(scores, {0.1, 0.2, 1.5, 0.0}),
            "Graph layer reads the states left by the previous layer"
        );

        // Alone, vertex 4 takes the first LP value, 0.2: h = 0.1.
        std::vector<double> alone;

        check(
            evaluateNode(model, {4}, alone) && near(alone, {0.1})
            && !evaluateNode(model, {2, 0, 2}, scores)
            && !evaluateNode(model, {2, 6}, scores),
            "Node evaluation refuses repeated or out-of-range vertices"
        );
    }

    {
        // Without graph layer, the node evaluation is the candidate one.
        LoadedMISBranchingModel model;
        parse(networkModel(), model);

        std::vector<double> scores;
        bool same = evaluateNode(model, NODE_VERTICES, scores);

        for (std::size_t c = 0; same && c < NODE_VERTICES.size(); ++c)
            same = scores[c] == evaluate(model, {{0, NODE_LP_VALUES[c]}});

        check(
            same && !MISbranchingModelHasGraphLayers(&model.view())
            && MISbranchingModelNodeWorkSize(&model.view(), 4)
                == MISbranchingModelWorkSize(&model.view()),
            "Node evaluation of a model without graph layer is the candidate "
            "evaluation, bit for bit"
        );
    }

    expectRejection(
        "Graph layer in a version 1 file is rejected",
        replaced(graphLinear, "MIS_BRANCHING_MODEL 2", "MIS_BRANCHING_MODEL 1"));

    expectRejection(
        "Graph layer without its three matrices is rejected",
        replaced(graphLinear, "1\n10\n100\n0.5\n", "1\n10\n0.5\n"));

    {
        LoadedMISBranchingModel model;
        parse(graphLinear, model);

        MISBranchingModelLayer incomplete = model.view().layers[0];
        incomplete.node_weights = nullptr;

        MISBranchingModel broken = model.view();
        broken.layers = &incomplete;

        check(
            !MISbranchingModelIsValid(&broken),
            "Graph layer without its neighbour and node matrices is not valid"
        );
    }

    expectRejection(
        "Wrong feature count is rejected",
        replaced(linear, "features " + std::to_string(K),
            "features " + std::to_string(K - 1)));

    expectRejection(
        "Unknown feature name is rejected",
        replaced(linear, "feature " + name0 + " ", "feature other_name "));

    expectRejection(
        "Feature named twice is rejected",
        replaced(linear, "feature " + name1 + " ", "feature " + name0 + " "));

    // A model may read a subset of the features, in its own order.
    {
        LoadedMISBranchingModel model;

        const bool parsed = parse(
            HEADER + "features 2\nfeature " + name1 + " 1 2\nfeature "
                + name0 + " 0 1\nlayers 1\nlayer 2 1 linear\n1 10\n0\n",
            model);

        // score = (x1 - 1) / 2 + 10 x0
        check(
            parsed
            && MISbranchingModelIsValid(&model.view())
            && model.view().number_of_features == 2
            && std::fabs(evaluate(model, {{0, 3.0}, {1, 5.0}, {2, 99.0}})
                - 32.0) < 1e-12,
            "Model on a subset of the features is evaluated on its columns"
        );
    }

    expectRejection(
        "Zero standard deviation is rejected",
        replaced(linear, "feature " + name0 + " 0 1",
            "feature " + name0 + " 0 0"));

    expectRejection(
        "Negative standard deviation is rejected",
        replaced(linear, "feature " + name0 + " 0 1",
            "feature " + name0 + " 0 -1"));

    expectRejection(
        "Non-numeric mean is rejected",
        replaced(linear, "feature " + name0 + " 0 1",
            "feature " + name0 + " x 1"));

    expectRejection(
        "Layer with the wrong input size is rejected",
        replaced(network, "layer 2 1 linear", "layer 3 1 linear"));

    expectRejection(
        "Final layer with an activation is rejected",
        replaced(network, "layer 2 1 linear", "layer 2 1 relu"));

    expectRejection(
        "Final layer with two outputs is rejected",
        replaced(network, "layer 2 1 linear\n2 -3\n1\n",
            "layer 2 2 linear\n2 -3\n1 1\n0 0\n"));

    expectRejection(
        "Unknown activation is rejected",
        replaced(network, "2 relu", "2 tanh"));

    expectRejection(
        "Weight row with too few numbers is rejected",
        replaced(network, "layer 2 1 linear\n2 -3\n", "layer 2 1 linear\n2\n"));

    expectRejection(
        "Weight row with too many numbers is rejected",
        replaced(network, "layer 2 1 linear\n2 -3\n",
            "layer 2 1 linear\n2 -3 4\n"));

    expectRejection(
        "Non-finite weight is rejected",
        replaced(network, "layer 2 1 linear\n2 -3\n",
            "layer 2 1 linear\n2 nan\n"));

    expectRejection(
        "Non-numeric weight is rejected",
        replaced(network, "layer 2 1 linear\n2 -3\n",
            "layer 2 1 linear\n2 abc\n"));

    expectRejection(
        "Missing bias row is rejected",
        replaced(network, "layer 2 1 linear\n2 -3\n1\n",
            "layer 2 1 linear\n2 -3\n"));

    expectRejection(
        "Missing layer is rejected",
        replaced(network, "layers 2", "layers 3"));

    expectRejection(
        "Content after the last layer is rejected",
        network + "layer 1 1 linear\n1\n0\n");

    {
        LoadedMISBranchingModel model;
        std::string error;

        check(
            !loadMISBranchingModel("/nonexistent/model", model, error)
            && error.find("Unable to open model file") == 0,
            "Missing model file is reported"
        );
    }

    // The C validity check on hand-built models.
    {
        LoadedMISBranchingModel model;
        parse(linear, model);

        MISBranchingModel copy = model.view();
        copy.number_of_features = K - 1;

        // Without indices, a model must read every feature.
        MISBranchingModel partial = model.view();
        partial.feature_indices = nullptr;

        MISBranchingModel unindexed = partial;
        unindexed.number_of_features = K - 1;

        const int outOfRange[1] = {K};
        MISBranchingModel misplaced = model.view();
        misplaced.feature_indices = outOfRange;
        misplaced.number_of_features = 1;

        MISBranchingModelLayer activated = model.view().layers[0];
        activated.relu = 1;

        MISBranchingModel withActivation = model.view();
        withActivation.layers = &activated;

        check(
            !MISbranchingModelIsValid(nullptr)
            && !MISbranchingModelIsValid(&copy)
            && MISbranchingModelIsValid(&partial)
            && !MISbranchingModelIsValid(&unindexed)
            && !MISbranchingModelIsValid(&misplaced)
            && !MISbranchingModelIsValid(&withActivation)
            && MISbranchingModelWorkSize(&model.view()) == 2 * K,
            "Inconsistent hand-built models are not valid"
        );
    }

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS branching model IO tests passed.\n";
    return 0;
}
