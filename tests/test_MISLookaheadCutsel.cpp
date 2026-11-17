/*
 * Tests of the lookahead cut selector: it scores the root cuts by the LP
 * bound increase of each cut alone, selects them at the root, leaves the
 * other nodes to SCIP's selectors, and keeps the solve exact.
 */

#include "cuts/MISLookaheadCutsel.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "generators/ErdosRenyiGenerator.hpp"
#include "problems/MIS.hpp"

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

/*
 * A probe between the lookahead selector and SCIP's: it counts the
 * selections that reach it and always leaves them to the next selector.
 * (SCIP 10.1 does not export the call counts of a cut selector.)
 */
struct ProbeCounts
{
    long long rootCalls = 0;
    long long otherCalls = 0;
};

SCIP_DECL_CUTSELSELECT(probeSelect)
{
    (void) scip; (void) cuts; (void) ncuts; (void) forcedcuts;
    (void) nforcedcuts; (void) maxnselectedcuts;

    auto* counts = reinterpret_cast<ProbeCounts*>(SCIPcutselGetData(cutsel));
    ++(root ? counts->rootCalls : counts->otherCalls);

    *nselectedcuts = 0;
    *result = SCIP_DIDNOTFIND;
    return SCIP_OKAY;
}

struct Solve
{
    bool ok = false;
    double optimum = 0.0;
    long long nodes = 0;
    double rootDualBound = 0.0;
    ProbeCounts probe;
    MISLookaheadStatistics statistics{};
};

enum class Cuts
{
    RootOnly,       // the study settings
    EveryNode,      // zero-half cuts and the cut pool at every node
    RootNoPresolve  // root only, and no presolving (tiny instances)
};

Solve solve(const MISInstance& instance, bool lookahead, Cuts cuts)
{
    Solve result;
    SCIP* scip = nullptr;

    if (SCIPcreate(&scip) != SCIP_OKAY)
        return result;

    std::vector<SCIP_VAR*> vars;
    SCIP_CUTSEL* probe = nullptr;

    // Between the lookahead selector and SCIP's hybrid selector (8000).
    bool ok = SCIPincludeDefaultPlugins(scip) == SCIP_OKAY
        && (!lookahead || SCIPincludeCutselMISLookahead(scip) == SCIP_OKAY)
        && SCIPincludeCutselBasic(scip, &probe, "probe", "counts selections",
            9000, probeSelect,
            reinterpret_cast<SCIP_CUTSELDATA*>(&result.probe)) == SCIP_OKAY
        && buildMISModel(scip, instance, vars) == SCIP_OKAY
        && SCIPsetIntParam(scip, "display/verblevel", 0) == SCIP_OKAY
        && SCIPsetIntParam(scip, "presolving/maxrestarts", 0) == SCIP_OKAY;

    if (ok && cuts != Cuts::EveryNode)
        ok = SCIPsetIntParam(scip, "separating/maxrounds", 0) == SCIP_OKAY;

    if (ok && cuts == Cuts::EveryNode)
    {
        ok = SCIPsetIntParam(scip, "separating/poolfreq", 1) == SCIP_OKAY
            && SCIPsetIntParam(scip, "separating/zerohalf/freq", 1) == SCIP_OKAY
            && SCIPsetIntParam(scip, "separating/zerohalf/expbackoff", 1) == SCIP_OKAY;
    }

    if (ok && cuts == Cuts::RootNoPresolve)
        ok = SCIPsetIntParam(scip, "presolving/maxrounds", 0) == SCIP_OKAY;

    ok = ok
        && SCIPsolve(scip) == SCIP_OKAY
        && SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL;

    if (ok)
    {
        result.optimum = SCIPgetPrimalbound(scip);
        result.nodes = SCIPgetNNodes(scip);
        result.rootDualBound = SCIPgetDualboundRoot(scip);

        if (lookahead)
        {
            ok = SCIPgetMISLookaheadStatistics(scip, &result.statistics)
                == SCIP_OKAY;
        }
    }

    SCIPfree(&scip);
    result.ok = ok;
    return result;
}

} // namespace

int main()
{
    // Needs branching with cuts at the root only.
    const MISInstance instance =
        ErdosRenyiGenerator{}.generate({100, 0.10, 1}).graph;

    const Solve reference = solve(instance, false, Cuts::RootOnly);
    const Solve root = solve(instance, true, Cuts::RootOnly);

    check(
        reference.ok && root.ok
        && std::fabs(root.optimum - reference.optimum) < 1e-6,
        "The lookahead selector keeps the optimum ("
            + std::to_string(root.nodes) + " nodes, SCIP's selector "
            + std::to_string(reference.nodes) + ")"
    );

    const MISLookaheadStatistics& s = root.statistics;

    check(
        s.rounds > 0 && s.selected > 0 && s.selected <= s.scored
        && s.improving <= s.scored && s.dive_iterations > 0
        && s.seconds > 0.0,
        "Every candidate cut of the root is scored by a dive ("
            + std::to_string(s.scored) + " scored, "
            + std::to_string(s.improving) + " improving, "
            + std::to_string(s.selected) + " selected in "
            + std::to_string(s.rounds) + " rounds)"
    );

    check(
        reference.probe.rootCalls > 0 && root.probe.rootCalls == 0,
        "The lookahead selector makes every root selection ("
            + std::to_string(reference.probe.rootCalls)
            + " root selections reach the next selector without it, 0 with it)"
    );

    // With cuts at the nodes, SCIP's own selector decides there.
    {
        const Solve nodes = solve(instance, true, Cuts::EveryNode);

        check(
            nodes.ok
            && std::fabs(nodes.optimum - reference.optimum) < 1e-6
            && nodes.nodes > 1
            && nodes.probe.rootCalls == 0
            && nodes.probe.otherCalls > 0,
            "Outside the root, the selection goes to the next selectors ("
                + std::to_string(nodes.probe.otherCalls) + " selections in "
                + std::to_string(nodes.nodes) + " nodes)"
        );
    }

    // Ten disjoint 5-cycles: the LP bound is 25 (one half everywhere), the
    // optimum 20, and the odd cycle inequality of each cycle raises the
    // bound by 0.5 on its own. The ten are orthogonal, all selected, and
    // close the gap at the root.
    {
        MISInstance cycles{50, {}};

        for (int c = 0; c < 10; ++c)
        {
            for (int j = 0; j < 5; ++j)
            {
                const int u = 5 * c + j;
                const int v = 5 * c + (j + 1) % 5;
                cycles.edges.push_back({std::min(u, v), std::max(u, v)});
            }
        }

        const Solve odd = solve(cycles, true, Cuts::RootNoPresolve);

        check(
            odd.ok
            && std::fabs(odd.optimum - 20.0) < 1e-6
            && odd.statistics.improving >= 1
            && odd.nodes == 1,
            "A cut that raises the bound on its own is scored as improving ("
                + std::to_string(odd.statistics.improving) + " of "
                + std::to_string(odd.statistics.scored) + ", solved at the root)"
        );
    }

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All lookahead cut selector tests passed.\n";
    return 0;
}
