/*
 * Tests of the learned branching rule.
 *
 * The rule is driven here by a hand-written model whose behaviour is
 * known: a linear model with weight 1 on the "fractionality" feature,
 * i.e. "branch on the most fractional vertex".
 *
 * The key check ties inference to training: at the root node, the rule
 * must see exactly what the data collection records (vertex states,
 * candidates and LP values), so that the features it computes are the
 * ones the model was trained on.
 */

#include "branching/MISLearnedBranchrule.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <scip/cons_linear.h>

#include "branching/MISFullstrongBranchrule.h"
#include "generators/ErdosRenyiGenerator.hpp"
#include "io/MISBranchingModelIO.hpp"
#include "learning/MISAdjacency.hpp"
#include "problems/MIS.hpp"
#include "services/MISSolverSettings.hpp"

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
const int FRACTIONALITY = 1;

// Linear model: score = value of one feature.
bool singleFeatureModel(int feature, LoadedMISBranchingModel& model)
{
    std::ostringstream text;

    text << "MIS_BRANCHING_MODEL 1\nfeatures " << K << '\n';

    for (int f = 0; f < K; ++f)
        text << "feature " << MISbranchingFeatureName(f) << " 0 1\n";

    text << "layers 1\nlayer " << K << " 1 linear\n";

    for (int f = 0; f < K; ++f)
        text << (f > 0 ? " " : "") << (f == feature ? 1 : 0);

    text << "\n0\n";

    std::istringstream input(text.str());
    std::string error;

    return readMISBranchingModel(input, model, error);
}

// Graph model: two hidden units from the fractionality, the LP value and
// the degree, a graph layer, then a linear output.
bool graphModel(LoadedMISBranchingModel& model)
{
    std::ostringstream text;

    text << "MIS_BRANCHING_MODEL 2\nfeatures " << K << '\n';

    for (int f = 0; f < K; ++f)
        text << "feature " << MISbranchingFeatureName(f) << " 0 1\n";

    const auto row = [](std::ostringstream& out,
                        std::vector<std::pair<int, double>> entries)
    {
        for (int f = 0; f < K; ++f)
        {
            double value = 0.0;

            for (const auto& [index, weight] : entries)
                if (index == f)
                    value = weight;

            out << (f > 0 ? " " : "") << value;
        }

        out << '\n';
    };

    text << "layers 3\nlayer " << K << " 2 relu\n";
    row(text, {{FRACTIONALITY, 4.0}, {2, 0.05}});
    row(text, {{0, -2.0}, {2, 0.1}});
    text << "0.1 1\n"
         << "graph_layer 2 2 relu\n"
         << "1 0.5\n-0.5 1\n"      // W
         << "2 -1\n0.25 0.5\n"     // W_neighbours
         << "-1 0.5\n0.5 -0.25\n"  // W_node
         << "0 0.2\n"
         << "layer 2 1 linear\n1 -0.75\n0\n";

    std::istringstream input(text.str());
    std::string error;

    return readMISBranchingModel(input, model, error);
}

// Scores of a model, computed here without the library's evaluation:
// the reference the rule must agree with.
std::vector<double> referenceScores(
    const MISBranchingModel& model,
    const MISFeatureGraph& graph,
    int candidates,
    const int* vertices,
    const double* features)
{
    std::vector<std::vector<double>> states(static_cast<std::size_t>(candidates));

    for (int c = 0; c < candidates; ++c)
    {
        for (int i = 0; i < model.number_of_features; ++i)
        {
            const double value = features[c * K + model.feature_indices[i]];
            states[c].push_back(
                (value - model.feature_means[i]) / model.feature_stds[i]);
        }
    }

    std::vector<std::vector<int>> neighbours(static_cast<std::size_t>(candidates));

    for (int c = 0; c < candidates; ++c)
    {
        for (int d = 0; d < candidates; ++d)
        {
            const int* first = graph.adjacency + graph.adjacency_offsets[vertices[c]];
            const int* last = graph.adjacency + graph.adjacency_offsets[vertices[c] + 1];

            if (std::find(first, last, vertices[d]) != last)
                neighbours[c].push_back(d);
        }
    }

    for (int l = 0; l < model.number_of_layers; ++l)
    {
        const MISBranchingModelLayer& layer = model.layers[l];
        const int in = layer.number_of_inputs;
        std::vector<double> node(static_cast<std::size_t>(in), 0.0);

        for (int c = 0; c < candidates; ++c)
            for (int i = 0; i < in; ++i)
                node[i] += states[c][i] / candidates;

        std::vector<std::vector<double>> next(static_cast<std::size_t>(candidates));

        for (int c = 0; c < candidates; ++c)
        {
            std::vector<double> mean(static_cast<std::size_t>(in), 0.0);

            for (const int d : neighbours[c])
                for (int i = 0; i < in; ++i)
                    mean[i] += states[d][i] / neighbours[c].size();

            for (int o = 0; o < layer.number_of_outputs; ++o)
            {
                double sum = layer.biases[o];

                for (int i = 0; i < in; ++i)
                {
                    sum += layer.weights[o * in + i] * states[c][i];

                    if (layer.graph)
                    {
                        sum += layer.neighbour_weights[o * in + i] * mean[i]
                            + layer.node_weights[o * in + i] * node[i];
                    }
                }

                next[c].push_back(layer.relu ? std::max(sum, 0.0) : sum);
            }
        }

        states = std::move(next);
    }

    std::vector<double> scores;

    for (int c = 0; c < candidates; ++c)
        scores.push_back(states[c][0]);

    return scores;
}

struct GraphDecisionRecorder
{
    const MISFeatureGraph* graph = nullptr;
    const MISBranchingModel* model = nullptr;
    long long decisions = 0;
    long long withNeighbours = 0;
    std::vector<std::string> problems;
};

void checkGraphDecision(
    void* userdata,
    const MISLearnedBranchingDecision* decision)
{
    auto& recorder = *static_cast<GraphDecisionRecorder*>(userdata);
    const int k = decision->number_of_candidates;
    const std::vector<double> expected = referenceScores(
        *recorder.model, *recorder.graph, k, decision->candidate_vertices,
        decision->features);

    ++recorder.decisions;

    int best = 0;

    for (int c = 0; c < k; ++c)
    {
        if (std::fabs(decision->scores[c] - expected[c])
                > 1e-9 * std::max(1.0, std::fabs(expected[c]))
            && recorder.problems.size() < 10)
        {
            recorder.problems.push_back(
                "node " + std::to_string(decision->node_id)
                + ": score differs from the reference");
        }

        if (decision->scores[c] > decision->scores[best])
            best = c;
    }

    if (decision->selected_candidate != best && recorder.problems.size() < 10)
        recorder.problems.push_back("selection is not the first best score");

    // Count the nodes where the graph layer has something to mix.
    for (int c = 0; c < k; ++c)
    {
        const int v = decision->candidate_vertices[c];
        bool linked = false;

        for (int d = 0; d < k && !linked; ++d)
        {
            const int* first = recorder.graph->adjacency
                + recorder.graph->adjacency_offsets[v];
            const int* last = recorder.graph->adjacency
                + recorder.graph->adjacency_offsets[v + 1];
            linked = std::find(first, last, decision->candidate_vertices[d]) != last;
        }

        if (linked)
        {
            ++recorder.withNeighbours;
            break;
        }
    }
}

// Deterministic pseudo-random weights in [-0.5, 0.5].
std::string weightLines(int rows, int columns, int seed)
{
    std::ostringstream text;

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < columns; ++c)
        {
            const double x = std::sin(12.9898 * seed + 78.233 * (r * columns + c + 1)) * 43758.5453;
            text << (c > 0 ? " " : "") << (x - std::floor(x) - 0.5);
        }

        text << '\n';
    }

    return text.str();
}

// Bipartite model reading three features: vertex and constraint layers of
// width 4, one round of messages, then a linear output.
bool bipartiteModel(LoadedMISBranchingModel& model)
{
    std::ostringstream text;

    text << "MIS_BRANCHING_MODEL 3\nfeatures 3\n"
         << "feature " << MISbranchingFeatureName(0) << " 0.4 0.2\n"
         << "feature " << MISbranchingFeatureName(1) << " 0.35 0.12\n"
         << "feature " << MISbranchingFeatureName(2) << " 10 3\n"
         << "vertex_inputs " << MIS_BRANCHING_VERTEX_INPUT_COUNT << '\n';

    for (int i = 0; i < MIS_BRANCHING_VERTEX_INPUT_COUNT; ++i)
        text << "vertex_input " << MISbranchingVertexInputName(i) << ' ' << 0.1 * i << ' ' << 1.0 + 0.5 * i << '\n';

    text << "constraint_inputs " << MIS_BRANCHING_CONSTRAINT_INPUT_COUNT << '\n';

    for (int i = 0; i < MIS_BRANCHING_CONSTRAINT_INPUT_COUNT; ++i)
        text << "constraint_input " << MISbranchingConstraintInputName(i) << " 0.2 0.7\n";

    text << "vertex_layers 1\nlayer 9 4 relu\n" << weightLines(4, 9, 1) << weightLines(1, 4, 2)
         << "constraint_layers 1\nlayer 3 4 relu\n" << weightLines(4, 3, 3) << weightLines(1, 4, 4)
         << "message_layers 2\nmessage_layer to_constraints 4 4 3 4 relu\n"
         << weightLines(3, 4, 5) << weightLines(3, 4, 6) << weightLines(1, 3, 7)
         << weightLines(4, 3, 8) << weightLines(4, 4, 9) << weightLines(1, 4, 10)
         << "message_layer to_vertices 4 4 3 4 relu\n"
         << weightLines(3, 4, 11) << weightLines(3, 4, 12) << weightLines(1, 3, 13)
         << weightLines(4, 3, 14) << weightLines(4, 4, 15) << weightLines(1, 4, 16)
         << "layers 1\nlayer 4 1 linear\n" << weightLines(1, 4, 17) << "0\n";

    std::istringstream input(text.str());
    std::string error;
    const bool ok = readMISBranchingModel(input, model, error);

    if (!ok)
        std::cerr << "  bipartite model: " << error << '\n';

    return ok;
}

// Scores of a bipartite model, computed here without the library.
std::vector<double> referenceBipartiteScores(
    const MISBranchingModel& model,
    const MISFeatureGraph& graph,
    const MISVertexState* states,
    int candidates,
    const int* vertices,
    const double* features)
{
    const MISBranchingBipartiteEncoder& encoder = *model.encoder;
    using Vector = std::vector<double>;

    const auto dense = [](const MISBranchingModelLayer& layer, const Vector& x)
    {
        Vector y;

        for (int o = 0; o < layer.number_of_outputs; ++o)
        {
            double sum = layer.biases[o];

            for (int i = 0; i < layer.number_of_inputs; ++i)
                sum += layer.weights[o * layer.number_of_inputs + i] * x[i];

            y.push_back(layer.relu ? std::max(sum, 0.0) : sum);
        }

        return y;
    };

    std::map<int, int> candidateOf;

    for (int c = 0; c < candidates; ++c)
        candidateOf[vertices[c]] = c;

    std::vector<int> freeVertices;
    std::map<int, int> rowOf;
    std::vector<double> lp(static_cast<std::size_t>(graph.number_of_vertices), 0.0);

    for (int v = 0; v < graph.number_of_vertices; ++v)
    {
        if (states[v] == MIS_VERTEX_FREE_LP_ZERO || states[v] == MIS_VERTEX_FREE_LP_ONE
            || states[v] == MIS_VERTEX_FRACTIONAL)
        {
            rowOf[v] = static_cast<int>(freeVertices.size());
            freeVertices.push_back(v);
        }

        if (states[v] == MIS_VERTEX_FREE_LP_ONE)
            lp[v] = 1.0;

        if (candidateOf.count(v) > 0)
            lp[v] = features[candidateOf[v] * K];
    }

    std::vector<std::pair<int, int>> constraints;

    for (const int v : freeVertices)
    {
        for (int k = graph.adjacency_offsets[v]; k < graph.adjacency_offsets[v + 1]; ++k)
        {
            const int u = graph.adjacency[k];

            if (u > v && rowOf.count(u) > 0)
                constraints.push_back({v, u});
        }
    }

    std::vector<Vector> vertexStates;

    for (const int v : freeVertices)
    {
        Vector x;
        int freeDegree = 0;

        for (const auto& [a, b] : constraints)
            freeDegree += (a == v) + (b == v);

        for (int f = 0; f < model.number_of_features; ++f)
        {
            x.push_back(candidateOf.count(v) == 0 ? 0.0
                : (features[candidateOf[v] * K + model.feature_indices[f]]
                    - model.feature_means[f]) / model.feature_stds[f]);
        }

        const double inputs[] = {
            static_cast<double>(states[v] == MIS_VERTEX_FREE_LP_ZERO),
            static_cast<double>(states[v] == MIS_VERTEX_FREE_LP_ONE),
            static_cast<double>(states[v] == MIS_VERTEX_FRACTIONAL),
            lp[v],
            static_cast<double>(graph.adjacency_offsets[v + 1] - graph.adjacency_offsets[v]),
            static_cast<double>(freeDegree)};

        for (int i = 0; i < MIS_BRANCHING_VERTEX_INPUT_COUNT; ++i)
            x.push_back((inputs[i] - encoder.vertex_input_means[i]) / encoder.vertex_input_stds[i]);

        vertexStates.push_back(x);
    }

    std::vector<Vector> constraintStates;

    for (const auto& [a, b] : constraints)
    {
        const double slack = 1.0 - lp[a] - lp[b];
        const double inputs[] = {
            slack,
            slack <= 1e-6 ? 1.0 : 0.0,
            static_cast<double>((states[a] == MIS_VERTEX_FRACTIONAL) + (states[b] == MIS_VERTEX_FRACTIONAL))};
        Vector x;

        for (int i = 0; i < MIS_BRANCHING_CONSTRAINT_INPUT_COUNT; ++i)
            x.push_back((inputs[i] - encoder.constraint_input_means[i]) / encoder.constraint_input_stds[i]);

        constraintStates.push_back(x);
    }

    for (int l = 0; l < encoder.number_of_vertex_layers; ++l)
        for (Vector& x : vertexStates)
            x = dense(encoder.vertex_layers[l], x);

    for (int l = 0; l < encoder.number_of_constraint_layers; ++l)
        for (Vector& x : constraintStates)
            x = dense(encoder.constraint_layers[l], x);

    for (int l = 0; l < encoder.number_of_message_layers; ++l)
    {
        const MISBranchingMessageLayer& layer = encoder.message_layers[l];
        std::vector<Vector>& targets = layer.to_vertices ? vertexStates : constraintStates;
        const std::vector<Vector>& sources = layer.to_vertices ? constraintStates : vertexStates;
        std::vector<Vector> updated;

        for (std::size_t t = 0; t < targets.size(); ++t)
        {
            std::vector<std::size_t> from;

            if (layer.to_vertices)
            {
                for (std::size_t e = 0; e < constraints.size(); ++e)
                    if (constraints[e].first == freeVertices[t] || constraints[e].second == freeVertices[t])
                        from.push_back(e);
            }
            else
            {
                from.push_back(static_cast<std::size_t>(rowOf[constraints[t].first]));
                from.push_back(static_cast<std::size_t>(rowOf[constraints[t].second]));
            }

            Vector mean(static_cast<std::size_t>(layer.number_of_hidden), 0.0);

            for (const std::size_t s : from)
            {
                for (int h = 0; h < layer.number_of_hidden; ++h)
                {
                    double message = layer.hidden_biases[h];

                    for (int i = 0; i < layer.number_of_target_inputs; ++i)
                        message += layer.target_weights[h * layer.number_of_target_inputs + i] * targets[t][i];

                    for (int i = 0; i < layer.number_of_source_inputs; ++i)
                        message += layer.source_weights[h * layer.number_of_source_inputs + i] * sources[s][i];

                    mean[h] += std::max(message, 0.0) / from.size();
                }
            }

            Vector y;

            for (int o = 0; o < layer.number_of_outputs; ++o)
            {
                double sum = layer.biases[o];

                for (int h = 0; h < layer.number_of_hidden; ++h)
                    sum += layer.message_weights[o * layer.number_of_hidden + h] * mean[h];

                for (int i = 0; i < layer.number_of_target_inputs; ++i)
                    sum += layer.own_weights[o * layer.number_of_target_inputs + i] * targets[t][i];

                y.push_back(layer.relu ? std::max(sum, 0.0) : sum);
            }

            updated.push_back(y);
        }

        targets = updated;
    }

    std::vector<double> scores;

    for (int c = 0; c < candidates; ++c)
    {
        Vector x = vertexStates[static_cast<std::size_t>(rowOf[vertices[c]])];

        for (int l = 0; l < model.number_of_layers; ++l)
            x = dense(model.layers[l], x);

        scores.push_back(x[0]);
    }

    return scores;
}

struct BipartiteDecisionRecorder
{
    const MISFeatureGraph* graph = nullptr;
    const MISBranchingModel* model = nullptr;
    long long decisions = 0;
    std::vector<std::string> problems;
};

void checkBipartiteDecision(
    void* userdata,
    const MISLearnedBranchingDecision* decision)
{
    auto& recorder = *static_cast<BipartiteDecisionRecorder*>(userdata);
    const int k = decision->number_of_candidates;
    const std::vector<double> expected = referenceBipartiteScores(
        *recorder.model, *recorder.graph, decision->vertex_states, k,
        decision->candidate_vertices, decision->features);

    ++recorder.decisions;

    int best = 0;

    for (int c = 0; c < k; ++c)
    {
        if (std::fabs(decision->scores[c] - expected[c])
                > 1e-9 * std::max(1.0, std::fabs(expected[c]))
            && recorder.problems.size() < 10)
        {
            recorder.problems.push_back(
                "node " + std::to_string(decision->node_id)
                + ": bipartite score differs from the reference");
        }

        if (decision->scores[c] > decision->scores[best])
            best = c;
    }

    if (decision->selected_candidate != best && recorder.problems.size() < 10)
        recorder.problems.push_back("selection is not the first best score");
}

// The root node as one rule saw it.
struct RootView
{
    bool seen = false;
    std::vector<MISVertexState> states;
    std::vector<int> vertices;
    std::vector<double> lpValues;
};

struct DecisionRecorder
{
    const MISFeatureGraph* graph = nullptr;
    long long decisions = 0;
    std::vector<std::string> problems;
    RootView root;

    void fail(const std::string& message)
    {
        if (problems.size() < 10)
            problems.push_back(message);
    }
};

void checkDecision(
    void* userdata,
    const MISLearnedBranchingDecision* decision)
{
    auto& recorder = *static_cast<DecisionRecorder*>(userdata);
    const int k = decision->number_of_candidates;
    const std::string where =
        "node " + std::to_string(decision->node_id) + ": ";

    ++recorder.decisions;

    if (k <= 0 || decision->number_of_vertices
            != recorder.graph->number_of_vertices)
    {
        recorder.fail(where + "unexpected sizes");
        return;
    }

    // The features are those of the shared feature function.
    std::vector<double> work(
        MISbranchingFeatureWorkSize(decision->number_of_vertices));
    std::vector<double> expected(static_cast<std::size_t>(k) * K);

    MISBranchingNode node;
    node.vertex_states = decision->vertex_states;
    node.node_depth = decision->node_depth;
    node.number_of_candidates = k;
    node.candidate_vertices = decision->candidate_vertices;
    node.candidate_lp_values = decision->candidate_lp_values;
    node.candidate_histories = decision->candidate_histories;
    node.objective_gap = decision->objective_gap;

    if (MIScomputeBranchingFeatures(
            recorder.graph, &node, work.data(), expected.data()) != 0)
    {
        recorder.fail(where + "inputs rejected by the feature function");
        return;
    }

    int best = 0;

    for (int c = 0; c < k; ++c)
    {
        for (int f = 0; f < K; ++f)
        {
            if (decision->features[c * K + f] != expected[c * K + f])
                recorder.fail(where + "feature differs from recomputation");
        }

        // With this model, the score is the fractionality itself.
        const double value = decision->candidate_lp_values[c];
        const double fractionality = std::min(value, 1.0 - value);

        if (decision->scores[c] != decision->features[c * K + FRACTIONALITY]
            || std::fabs(decision->scores[c] - fractionality) > 1e-12)
            recorder.fail(where + "score is not the fractionality");

        if (decision->scores[c] > decision->scores[best])
            best = c;
    }

    if (decision->selected_candidate != best)
        recorder.fail(where + "selection is not the first best score");

    if (!recorder.root.seen)
    {
        recorder.root.seen = true;

        if (decision->node_depth != 0)
            recorder.fail(where + "first decision is not at the root");

        recorder.root.states.assign(
            decision->vertex_states,
            decision->vertex_states + decision->number_of_vertices);
        recorder.root.vertices.assign(
            decision->candidate_vertices, decision->candidate_vertices + k);
        recorder.root.lpValues.assign(
            decision->candidate_lp_values, decision->candidate_lp_values + k);
    }
}

// Root node as the data collection records it.
void recordExpertRoot(
    void* userdata,
    const MISStrongBranchingSample* sample,
    const MISStrongBranchingCandidate* candidates,
    size_t numberOfCandidates)
{
    auto& root = *static_cast<RootView*>(userdata);

    if (root.seen)
        return;

    root.seen = true;
    root.states.assign(
        sample->vertex_states,
        sample->vertex_states + sample->number_of_vertices);

    for (size_t c = 0; c < numberOfCandidates; ++c)
    {
        root.vertices.push_back(candidates[c].vertex_id);
        root.lpValues.push_back(candidates[c].lp_value);
    }
}

struct Solver
{
    SCIP* scip = nullptr;
    std::vector<SCIP_VAR*> vars;

    ~Solver()
    {
        if (scip != nullptr)
            SCIPfree(&scip);
    }

    bool create(const MISInstance& instance, bool studySettings)
    {
        return SCIPcreate(&scip) == SCIP_OKAY
            && SCIPincludeDefaultPlugins(scip) == SCIP_OKAY
            && buildMISModel(scip, instance, vars) == SCIP_OKAY
            && SCIPsetIntParam(scip, "display/verblevel", 0) == SCIP_OKAY
            && (!studySettings
                || applyBranchingStudySettings(scip) == SCIP_OKAY);
    }
};

} // namespace

int main()
{
    const MISInstance instance =
        ErdosRenyiGenerator{}.generate({100, 0.10, 1}).graph;
    const MISAdjacency adjacency = buildMISAdjacency(instance);
    const MISFeatureGraph graph = adjacency.view();

    LoadedMISBranchingModel model;

    check(
        singleFeatureModel(FRACTIONALITY, model),
        "Hand-written model is accepted"
    );

    // Reference optimum, with SCIP's own rule.
    double optimum = 0.0;
    {
        Solver reference;

        check(
            reference.create(instance, true)
            && SCIPsolve(reference.scip) == SCIP_OKAY
            && SCIPgetStatus(reference.scip) == SCIP_STATUS_OPTIMAL,
            "Reference solve succeeds"
        );

        optimum = SCIPgetPrimalbound(reference.scip);
    }

    // Solve with the learned rule, twice on the same SCIP.
    DecisionRecorder recorder;
    recorder.graph = &graph;
    {
        Solver solver;

        const bool included =
            solver.create(instance, true)
            && SCIPincludeBranchruleMISLearned(
                solver.scip, solver.vars.data(),
                static_cast<int>(solver.vars.size()),
                &graph, &model.view(), checkDecision, &recorder)
                == SCIP_OKAY;

        check(included, "Learned rule is included");

        bool solved = included;
        long long nodes = 0;
        long long firstCycleDecisions = 0;

        for (int cycle = 0; solved && cycle < 2; ++cycle)
        {
            solved =
                SCIPsolve(solver.scip) == SCIP_OKAY
                && SCIPgetStatus(solver.scip) == SCIP_STATUS_OPTIMAL
                && std::fabs(SCIPgetPrimalbound(solver.scip) - optimum)
                    < 1e-6;

            nodes = SCIPgetNNodes(solver.scip);

            if (cycle == 0)
                firstCycleDecisions = recorder.decisions;

            solved = solved
                && SCIPfreeTransform(solver.scip) == SCIP_OKAY;
        }

        check(
            solved,
            "Learned rule reaches the reference optimum, twice in a row"
        );

        // The second solve starts from the solution of the first one,
        // so its tree is different: only require that the rule ran.
        check(
            firstCycleDecisions > 1 && nodes > 1
            && recorder.decisions > firstCycleDecisions,
            "The rule branches in both solves ("
                + std::to_string(firstCycleDecisions) + " decisions in the "
                + "first, " + std::to_string(
                    recorder.decisions - firstCycleDecisions)
                + " in the second)"
        );
    }

    for (const std::string& problem : recorder.problems)
        std::cerr << "  " << problem << '\n';

    check(
        recorder.problems.empty(),
        "Features, scores and selections are consistent at every node"
    );

    // A graph model: the rule's scores match a reference computation at
    // every node.
    {
        LoadedMISBranchingModel graphNetwork;

        check(graphModel(graphNetwork), "Hand-written graph model is accepted");

        GraphDecisionRecorder graphRecorder;
        graphRecorder.graph = &graph;
        graphRecorder.model = &graphNetwork.view();

        Solver solver;

        const bool solved =
            solver.create(instance, true)
            && SCIPincludeBranchruleMISLearned(
                solver.scip, solver.vars.data(),
                static_cast<int>(solver.vars.size()),
                &graph, &graphNetwork.view(), checkGraphDecision,
                &graphRecorder) == SCIP_OKAY
            && SCIPsolve(solver.scip) == SCIP_OKAY
            && SCIPgetStatus(solver.scip) == SCIP_STATUS_OPTIMAL
            && std::fabs(SCIPgetPrimalbound(solver.scip) - optimum) < 1e-6;

        for (const std::string& problem : graphRecorder.problems)
            std::cerr << "  " << problem << '\n';

        check(
            solved && graphRecorder.decisions > 1
            && graphRecorder.withNeighbours > 0
            && graphRecorder.problems.empty(),
            "Graph model scores match the reference at every node ("
                + std::to_string(graphRecorder.decisions) + " decisions)"
        );
    }

    // A bipartite model: the rule's scores match a reference computation
    // at every node.
    {
        LoadedMISBranchingModel bipartite;

        check(bipartiteModel(bipartite), "Hand-written bipartite model is accepted");

        BipartiteDecisionRecorder bipartiteRecorder;
        bipartiteRecorder.graph = &graph;
        bipartiteRecorder.model = &bipartite.view();

        Solver solver;

        const bool solved =
            solver.create(instance, true)
            && SCIPincludeBranchruleMISLearned(
                solver.scip, solver.vars.data(),
                static_cast<int>(solver.vars.size()),
                &graph, &bipartite.view(), checkBipartiteDecision,
                &bipartiteRecorder) == SCIP_OKAY
            && SCIPsolve(solver.scip) == SCIP_OKAY
            && SCIPgetStatus(solver.scip) == SCIP_STATUS_OPTIMAL
            && std::fabs(SCIPgetPrimalbound(solver.scip) - optimum) < 1e-6;

        for (const std::string& problem : bipartiteRecorder.problems)
            std::cerr << "  " << problem << '\n';

        check(
            solved && bipartiteRecorder.decisions > 1
            && bipartiteRecorder.problems.empty(),
            "Bipartite model scores match the reference at every node ("
                + std::to_string(bipartiteRecorder.decisions) + " decisions)"
        );
    }

    // Root node seen by the data collection expert.
    RootView expertRoot;
    {
        Solver solver;

        const bool ok =
            solver.create(instance, true)
            && SCIPincludeBranchruleMISFullstrong(
                solver.scip, solver.vars.data(),
                static_cast<int>(solver.vars.size()),
                recordExpertRoot, &expertRoot) == SCIP_OKAY
            && SCIPsetLongintParam(solver.scip, "limits/nodes", 1)
                == SCIP_OKAY
            && SCIPsolve(solver.scip) == SCIP_OKAY;

        check(ok && expertRoot.seen, "Expert root sample is recorded");
    }

    check(
        recorder.root.seen
        && recorder.root.states == expertRoot.states
        && recorder.root.vertices == expertRoot.vertices
        && recorder.root.lpValues == expertRoot.lpValues
        && !expertRoot.vertices.empty(),
        "At the root, the learned rule sees what the collection records ("
            + std::to_string(expertRoot.vertices.size()) + " candidates)"
    );

    // Invalid configurations are refused at inclusion.
    {
        Solver solver;
        solver.create(instance, true);

        const MISAdjacency smaller =
            buildMISAdjacency(MISInstance{5, {{0, 1}}});
        const MISFeatureGraph smallerGraph = smaller.view();

        MISBranchingModel invalid = model.view();
        invalid.number_of_layers = 0;

        const int n = static_cast<int>(solver.vars.size());

        check(
            SCIPincludeBranchruleMISLearned(
                solver.scip, solver.vars.data(), n, &smallerGraph,
                &model.view(), nullptr, nullptr) == SCIP_INVALIDDATA
            && SCIPincludeBranchruleMISLearned(
                solver.scip, solver.vars.data(), n, &graph,
                &invalid, nullptr, nullptr) == SCIP_INVALIDDATA
            && SCIPincludeBranchruleMISLearned(
                solver.scip, solver.vars.data(), n, &graph,
                nullptr, nullptr, nullptr) == SCIP_INVALIDDATA
            && SCIPincludeBranchruleMISLearned(
                solver.scip, solver.vars.data(), n, nullptr,
                &model.view(), nullptr, nullptr) == SCIP_INVALIDDATA
            && SCIPfindBranchrule(solver.scip, "mislearned") == nullptr,
            "Mismatched graph and invalid model are refused"
        );
    }

    // Two vertices merged into one LP column: the solve must fail loudly.
    {
        const MISInstance small =
            ErdosRenyiGenerator{}.generate({40, 0.15, 7}).graph;
        const MISAdjacency smallAdjacency = buildMISAdjacency(small);
        const MISFeatureGraph smallGraph = smallAdjacency.view();

        Solver solver;
        bool ok = solver.create(small, false)
            && SCIPsetIntParam(solver.scip, "separating/maxrounds", 0)
                == SCIP_OKAY
            && SCIPsetIntParam(solver.scip, "separating/maxroundsroot", 0)
                == SCIP_OKAY
            && SCIPsetIntParam(solver.scip, "presolving/maxrestarts", 0)
                == SCIP_OKAY;

        SCIP_CONS* cons = nullptr;
        SCIP_VAR* pair[2] = {solver.vars[0], solver.vars[1]};
        SCIP_Real ones[2] = {1.0, 1.0};

        ok = ok
            && SCIPcreateConsBasicLinear(
                solver.scip, &cons, "force_aggregation", 2, pair, ones,
                1.0, 1.0) == SCIP_OKAY
            && SCIPaddCons(solver.scip, cons) == SCIP_OKAY
            && SCIPreleaseCons(solver.scip, &cons) == SCIP_OKAY
            && SCIPincludeBranchruleMISLearned(
                solver.scip, solver.vars.data(),
                static_cast<int>(solver.vars.size()),
                &smallGraph, &model.view(), nullptr, nullptr) == SCIP_OKAY;

        SCIPmessagehdlrSetQuiet(SCIPgetMessagehdlr(solver.scip), TRUE);

        check(
            ok && SCIPsolve(solver.scip) == SCIP_INVALIDDATA,
            "Merged vertices make the solve fail instead of guessing"
        );
    }

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS learned branching rule tests passed.\n";
    return 0;
}
