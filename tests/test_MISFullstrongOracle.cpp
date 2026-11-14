/*
 * Oracle test: compares MISFullstrongBranchrule against SCIP's own
 * vanillafullstrong rule on the same nodes.
 *
 * vanillafullstrong runs first (higher priority) in "donotbranch" mode:
 * it scores the candidates without side effects (idempotent) and returns
 * DIDNOTRUN, so our rule then runs on the very same LP. In our callback,
 * the scores and the best candidate must match SCIPgetVanillafullstrongData().
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "branching/MISFullstrongBranchrule.h"
#include "generators/ErdosRenyiGenerator.hpp"
#include "problems/MIS.hpp"

namespace
{

enum class Settings
{
    // Presolve and cuts off: LP columns are exactly the vertex variables.
    PureBranchAndBound,
    // Data collection settings: presolve on, no restarts, root-only cuts.
    Collection
};

struct OracleContext
{
    SCIP* scip = nullptr;
    const std::vector<SCIP_VAR*>* originalVars = nullptr;

    long long samples = 0;
    long long comparedCandidates = 0;
    long long complementedCandidates = 0;
    long long fixedVertexStates = 0;
    std::vector<std::string> failures;

    void fail(long long nodeId, const std::string& message)
    {
        if (failures.size() < 20)
            failures.push_back("node " + std::to_string(nodeId) + ": " + message);
        else if (failures.size() == 20)
            failures.push_back("...");
    }
};

bool sameScore(double a, double b)
{
    return std::fabs(a - b) <= 1e-6 * std::max(1.0, std::fabs(b));
}

void compareWithVanilla(
    void* userdata,
    const MISStrongBranchingSample* sample,
    const MISStrongBranchingCandidate* candidates,
    size_t numberOfCandidates)
{
    auto& context = *static_cast<OracleContext*>(userdata);
    const long long node = sample->node_id;

    ++context.samples;

    SCIP_VAR** vanillaCands = nullptr;
    SCIP_Real* vanillaScores = nullptr;
    int vanillaNCands = 0;
    int vanillaNPrio = 0;
    int vanillaBest = -1;

    if (SCIPgetVanillafullstrongData(context.scip, &vanillaCands, &vanillaScores,
            &vanillaNCands, &vanillaNPrio, &vanillaBest) != SCIP_OKAY)
    {
        context.fail(node, "SCIPgetVanillafullstrongData failed");
        return;
    }

    if (static_cast<size_t>(vanillaNCands) != numberOfCandidates
        || sample->number_of_candidates != vanillaNCands)
    {
        context.fail(node, "candidate count differs: ours "
            + std::to_string(numberOfCandidates) + ", vanilla "
            + std::to_string(vanillaNCands));
        return;
    }

    if (sample->number_of_priority_candidates != vanillaNPrio)
        context.fail(node, "priority candidate count differs");

    // Objective bounds share the LP's internal (minimization) space, and
    // map back to the original bounds reported by SCIP.
    if (!sample->lp_objective_valid
        || std::fabs(sample->lp_objective - SCIPgetLPObjval(context.scip)) > 1e-9)
        context.fail(node, "lp_objective differs from the node LP");

    if (sample->dual_bound_valid
        && (sample->dual_bound > sample->lp_objective + 1e-6
            || std::fabs(SCIPretransformObj(context.scip, sample->dual_bound)
                    - SCIPgetDualbound(context.scip)) > 1e-6))
        context.fail(node, "inconsistent dual bound");

    if (sample->primal_bound_valid
        && (sample->primal_bound < sample->lp_objective - 1e-6
            || std::fabs(SCIPretransformObj(context.scip, sample->primal_bound)
                    - SCIPgetPrimalbound(context.scip)) > 1e-6))
        context.fail(node, "inconsistent primal bound");

    // Vertex states, recomputed through an independent path: the active
    // column of each vertex (x_v = scalar * column + constant) instead of
    // the bounds SCIP maintains on the transformed vertex variable.
    const int numberOfVertices = static_cast<int>(context.originalVars->size());

    if (sample->number_of_vertices != numberOfVertices
        || sample->vertex_states == nullptr)
    {
        context.fail(node, "missing vertex states");
        return;
    }

    int fractionalVertices = 0;

    for (int v = 0; v < numberOfVertices; ++v)
    {
        SCIP_VAR* column = nullptr;
        SCIP_Real scalar = 1.0;
        SCIP_Real constant = 0.0;

        if (SCIPgetTransformedVar(context.scip, (*context.originalVars)[v],
                &column) != SCIP_OKAY
            || SCIPgetProbvarSum(context.scip, &column, &scalar, &constant)
                != SCIP_OKAY)
        {
            context.fail(node, "vertex cannot be resolved");
            continue;
        }

        MISVertexState expected;

        if (column == nullptr || scalar == 0.0)
        {
            // Fixed by presolving: x_v = constant.
            expected = constant < 0.5
                ? MIS_VERTEX_FIXED_ZERO : MIS_VERTEX_FIXED_ONE;
        }
        else
        {
            const double a = scalar * SCIPvarGetLbLocal(column) + constant;
            const double b = scalar * SCIPvarGetUbLocal(column) + constant;
            const double lower = std::min(a, b);
            const double upper = std::max(a, b);
            const double value = scalar * SCIPvarGetLPSol(column) + constant;

            if (upper - lower < 0.5)
                expected = upper < 0.5
                    ? MIS_VERTEX_FIXED_ZERO : MIS_VERTEX_FIXED_ONE;
            else if (value > 1e-6 && value < 1.0 - 1e-6)
                expected = MIS_VERTEX_FRACTIONAL;
            else
                expected = value < 0.5
                    ? MIS_VERTEX_FREE_LP_ZERO : MIS_VERTEX_FREE_LP_ONE;
        }

        if (sample->vertex_states[v] != expected)
            context.fail(node, "vertex " + std::to_string(v)
                + ": state " + std::to_string(sample->vertex_states[v])
                + " vs expected " + std::to_string(expected));

        if (sample->vertex_states[v] == MIS_VERTEX_FRACTIONAL)
            ++fractionalVertices;

        if (sample->vertex_states[v] == MIS_VERTEX_FIXED_ZERO
            || sample->vertex_states[v] == MIS_VERTEX_FIXED_ONE)
            ++context.fixedVertexStates;
    }

    if (fractionalVertices != static_cast<int>(numberOfCandidates))
        context.fail(node, "fractional vertices are not exactly the candidates");

    for (size_t c = 0; c < numberOfCandidates; ++c)
    {
        const int v = candidates[c].vertex_id;

        if (v >= 0 && v < numberOfVertices
            && sample->vertex_states[v] != MIS_VERTEX_FRACTIONAL)
            context.fail(node, "candidate vertex is not marked fractional");
    }

    int selectedCount = 0;
    int selectedPosition = -1;

    for (size_t c = 0; c < numberOfCandidates; ++c)
    {
        const MISStrongBranchingCandidate& candidate = candidates[c];
        const std::string where = "candidate " + std::to_string(c) + ": ";

        if (candidate.candidate_position != static_cast<int>(c))
            context.fail(node, where + "wrong candidate_position");

        // Same variable, in the same order: resolve the vertex variable
        // through presolve (x_v = scalar * column + constant).
        const int v = candidate.vertex_id;
        SCIP_VAR* column = nullptr;
        SCIP_Real scalar = 1.0;
        SCIP_Real constant = 0.0;

        if (v < 0 || v >= static_cast<int>(context.originalVars->size())
            || SCIPgetTransformedVar(context.scip, (*context.originalVars)[v],
                   &column) != SCIP_OKAY
            || SCIPgetProbvarSum(context.scip, &column, &scalar, &constant) != SCIP_OKAY
            || column != vanillaCands[c])
        {
            context.fail(node, where + "vertex does not match vanilla candidate");
            continue;
        }

        if (scalar < 0.0)
            ++context.complementedCandidates;

        // The record is in vertex space: lp_value must be x_v, not the column.
        const double expectedValue =
            scalar * SCIPvarGetLPSol(column) + constant;

        if (std::fabs(candidate.lp_value - expectedValue) > 1e-6)
            context.fail(node, where + "lp_value is not expressed on x_v");

        // Same score (vanilla uses -infinity for unscored candidates).
        const bool vanillaScored = !SCIPisInfinity(context.scip, -vanillaScores[c]);

        if (candidate.score_valid != vanillaScored)
            context.fail(node, where + "scored status differs");
        else if (vanillaScored && !sameScore(candidate.score, vanillaScores[c]))
            context.fail(node, where + "score " + std::to_string(candidate.score)
                + " vs vanilla " + std::to_string(vanillaScores[c]));

        // Internal consistency of the record.
        if (candidate.evaluation_status == MIS_SB_EVALUATED)
        {
            if (candidate.down.skipped || candidate.up.skipped)
                context.fail(node, where + "evaluated but a direction is skipped");
            if (candidate.expert_rank < 0)
                context.fail(node, where + "evaluated but no expert rank");
        }

        if (candidate.expert_selected)
        {
            ++selectedCount;
            selectedPosition = static_cast<int>(c);

            if (candidate.expert_rank != 0)
                context.fail(node, where + "selected but rank != 0");
        }

        ++context.comparedCandidates;
    }

    if (selectedCount != 1)
        context.fail(node, "expected exactly one selected candidate, got "
            + std::to_string(selectedCount));
    else if (selectedPosition != vanillaBest)
        context.fail(node, "best candidate " + std::to_string(selectedPosition)
            + " vs vanilla " + std::to_string(vanillaBest));
    else if (!sample->expert_selection_valid
        || sample->expert_selected_vertex_id != candidates[selectedPosition].vertex_id)
        context.fail(node, "sample-level expert selection inconsistent");
}

#define CHECK_SCIP(call)                                                    \
    do                                                                      \
    {                                                                       \
        const SCIP_RETCODE checkRetcode = (call);                           \
        if (checkRetcode != SCIP_OKAY)                                      \
        {                                                                   \
            std::cerr << #call << " failed: " << checkRetcode << '\n';      \
            return false;                                                   \
        }                                                                   \
    } while (false)

/* Reference optimum with default SCIP settings. */
bool solveReference(const MISInstance& instance, double& objective)
{
    SCIP* scip = nullptr;
    CHECK_SCIP(SCIPcreate(&scip));
    CHECK_SCIP(SCIPincludeDefaultPlugins(scip));
    CHECK_SCIP(SCIPsetIntParam(scip, "display/verblevel", 0));

    std::vector<SCIP_VAR*> vars;
    CHECK_SCIP(buildMISModel(scip, instance, vars));
    CHECK_SCIP(SCIPsolve(scip));

    const bool optimal = SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL;
    objective = SCIPgetPrimalbound(scip);

    CHECK_SCIP(SCIPfree(&scip));
    return optimal;
}

/*
 * Guard: when presolve merges two vertices into one LP column (here forced
 * by x_0 + x_1 = 1, aggregated as x_0 = 1 - x_1), the column no longer
 * identifies a single vertex. The rule must fail loudly (SCIP_INVALIDDATA)
 * rather than emit a sample attributed to the wrong vertex.
 */
bool sharedColumnIsRejected()
{
    // Same graph as a pure B&B case, known to need branching.
    MISInstance instance = ErdosRenyiGenerator{}.generate({40, 0.15, 7}).graph;

    SCIP* scip = nullptr;
    CHECK_SCIP(SCIPcreate(&scip));
    CHECK_SCIP(SCIPincludeDefaultPlugins(scip));
    CHECK_SCIP(SCIPsetIntParam(scip, "display/verblevel", 0));
    CHECK_SCIP(SCIPsetIntParam(scip, "presolving/maxrestarts", 0));
    CHECK_SCIP(SCIPsetIntParam(scip, "separating/maxrounds", 0));
    CHECK_SCIP(SCIPsetIntParam(scip, "separating/maxroundsroot", 0));

    std::vector<SCIP_VAR*> vars;
    CHECK_SCIP(buildMISModel(scip, instance, vars));

    SCIP_CONS* cons = nullptr;
    SCIP_VAR* pair[2] = {vars[0], vars[1]};
    SCIP_Real ones[2] = {1.0, 1.0};
    CHECK_SCIP(SCIPcreateConsBasicLinear(scip, &cons, "force_aggregation",
        2, pair, ones, 1.0, 1.0));
    CHECK_SCIP(SCIPaddCons(scip, cons));
    CHECK_SCIP(SCIPreleaseCons(scip, &cons));

    OracleContext context;
    context.scip = scip;
    context.originalVars = &vars;

    CHECK_SCIP(SCIPincludeBranchruleMISFullstrong(
        scip, vars.data(), static_cast<int>(vars.size()),
        compareWithVanilla, &context));

    // Check that presolve did merge the two vertices.
    CHECK_SCIP(SCIPpresolve(scip));

    SCIP_VAR* columns[2] = {nullptr, nullptr};
    for (int v = 0; v < 2; ++v)
    {
        SCIP_Real scalar = 1.0;
        SCIP_Real constant = 0.0;
        CHECK_SCIP(SCIPgetTransformedVar(scip, vars[v], &columns[v]));
        CHECK_SCIP(SCIPgetProbvarSum(scip, &columns[v], &scalar, &constant));
    }

    if (columns[0] == nullptr || columns[0] != columns[1])
    {
        std::cerr << "  guard setup: presolve did not merge x_0 and x_1\n";
        SCIPfree(&scip);
        return false;
    }

    SCIPmessagehdlrSetQuiet(SCIPgetMessagehdlr(scip), TRUE);
    const SCIP_RETCODE solveRetcode = SCIPsolve(scip);
    SCIPfree(&scip);

    if (solveRetcode != SCIP_INVALIDDATA)
    {
        std::cerr << "  shared column was not rejected (retcode "
                  << solveRetcode << ", " << context.samples << " samples)\n";
        return false;
    }

    return true;
}

bool runOracle(const MISInstance& instance, Settings settings,
    long long nodeLimit, OracleContext& context,
    double& objective, long long& nodes, bool& optimal)
{
    SCIP* scip = nullptr;
    CHECK_SCIP(SCIPcreate(&scip));
    CHECK_SCIP(SCIPincludeDefaultPlugins(scip));
    CHECK_SCIP(SCIPsetIntParam(scip, "display/verblevel", 0));

    std::vector<SCIP_VAR*> vars;
    CHECK_SCIP(buildMISModel(scip, instance, vars));

    context.scip = scip;
    context.originalVars = &vars;

    CHECK_SCIP(SCIPincludeBranchruleMISFullstrong(
        scip, vars.data(), static_cast<int>(vars.size()),
        compareWithVanilla, &context));

    CHECK_SCIP(SCIPsetIntParam(scip, "presolving/maxrestarts", 0));
    CHECK_SCIP(SCIPsetIntParam(scip, "separating/maxrounds", 0));
    CHECK_SCIP(SCIPsetLongintParam(scip, "limits/nodes", nodeLimit));

    if (settings == Settings::PureBranchAndBound)
    {
        // Keep LP columns identical to the original variables, and keep
        // the tree large enough to compare many nodes.
        CHECK_SCIP(SCIPsetIntParam(scip, "presolving/maxrounds", 0));
        CHECK_SCIP(SCIPsetIntParam(scip, "separating/maxroundsroot", 0));
    }

    // Vanilla oracle: runs first, scores without side effects, never branches.
    CHECK_SCIP(SCIPsetIntParam(scip, "branching/vanillafullstrong/priority", 2000000));
    CHECK_SCIP(SCIPsetBoolParam(scip, "branching/vanillafullstrong/idempotent", TRUE));
    CHECK_SCIP(SCIPsetBoolParam(scip, "branching/vanillafullstrong/donotbranch", TRUE));
    CHECK_SCIP(SCIPsetBoolParam(scip, "branching/vanillafullstrong/collectscores", TRUE));
    CHECK_SCIP(SCIPsetBoolParam(scip, "branching/vanillafullstrong/scoreall", TRUE));
    CHECK_SCIP(SCIPsetBoolParam(scip, "branching/vanillafullstrong/integralcands", FALSE));

    CHECK_SCIP(SCIPsolve(scip));

    optimal = SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL;
    objective = SCIPgetPrimalbound(scip);
    nodes = SCIPgetNNodes(scip);

    CHECK_SCIP(SCIPfree(&scip));
    return true;
}

} // namespace

int main()
{
    struct Case
    {
        Settings settings;
        int n;
        double p;
        unsigned int seed;
        long long nodeLimit;   // -1: no limit, the optimum is checked
    };

    const Case cases[] = {
        {Settings::PureBranchAndBound, 30, 0.20, 1, -1},
        {Settings::PureBranchAndBound, 40, 0.15, 7, -1},
        {Settings::PureBranchAndBound, 50, 0.10, 42, -1},

        // Data collection settings (presolve, root cuts) on instances that
        // still need branching. Node limits keep the test fast.
        {Settings::Collection, 100, 0.10, 1, -1},
        {Settings::Collection, 150, 0.05, 3, 15},
        {Settings::Collection, 125, 0.10, 1, 15},
    };

    bool ok = true;
    long long totalComplemented = 0;

    for (const Case& testCase : cases)
    {
        const std::string label =
            std::string(testCase.settings == Settings::Collection
                ? "[collection] " : "[pure B&B]   ")
            + "G(" + std::to_string(testCase.n) + ", "
            + std::to_string(testCase.p) + ", seed "
            + std::to_string(testCase.seed) + ")";

        const MISInstance instance = ErdosRenyiGenerator{}
            .generate({testCase.n, testCase.p, testCase.seed}).graph;

        OracleContext context;
        double objective = 0.0;
        long long nodes = 0;
        bool optimal = false;

        if (!runOracle(instance, testCase.settings, testCase.nodeLimit,
                context, objective, nodes, optimal))
        {
            std::cerr << label << ": solve failed\n";
            ok = false;
            continue;
        }

        std::cout << label << ": " << nodes << " nodes, " << context.samples
                  << " samples, " << context.comparedCandidates
                  << " candidates compared ("
                  << context.complementedCandidates << " complemented), "
                  << (optimal ? "optimum " : "stopped, primal ") << objective
                  << '\n';

        totalComplemented += context.complementedCandidates;

        for (const std::string& failure : context.failures)
            std::cerr << "  " << failure << '\n';

        if (!context.failures.empty())
            ok = false;

        if (testCase.nodeLimit < 0)
        {
            double reference = 0.0;

            if (!solveReference(instance, reference) || !optimal
                || std::fabs(objective - reference) > 1e-6)
            {
                std::cerr << "  wrong optimum: " << objective
                          << " vs reference " << reference << '\n';
                ok = false;
            }
        }

        if (context.fixedVertexStates == 0)
        {
            std::cerr << "  no fixed vertex was ever recorded\n";
            ok = false;
        }

        if (context.samples == 0 || nodes <= 1)
        {
            std::cerr << "  the rule never branched: the instance is too easy\n";
            ok = false;
        }
    }

    if (sharedColumnIsRejected())
        std::cout << "[guard]      shared column rejected with SCIP_INVALIDDATA\n";
    else
        ok = false;

    if (totalComplemented == 0)
        std::cout << "note: no complemented candidate occurred; "
                     "the vertex-space conversion was not exercised.\n";

    if (!ok)
        return 1;

    std::cout << "MISFullstrongOracle test passed.\n";
    return 0;
}
