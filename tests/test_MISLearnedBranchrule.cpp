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

#include <cmath>
#include <iostream>
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
