#include "generators/ErdosRenyiGenerator.hpp"
#include "learning/MISBranchingFeatures.h"
#include "problems/GraphProblem.hpp"
#include "services/MISBenchmarkService.hpp"
#include "services/MISCollectionService.hpp"

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{
int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

// Closed neighbourhoods as bit masks (n <= 20, for the enumeration).
std::vector<unsigned> closedMasks(const MISInstance& graph)
{
    std::vector<unsigned> masks(graph.n);

    for (int v = 0; v < graph.n; ++v)
        masks[v] = 1u << v;

    for (const auto& [u, v] : graph.edges)
    {
        masks[u] |= 1u << v;
        masks[v] |= 1u << u;
    }

    return masks;
}

bool dominates(const MISInstance& graph, const std::vector<int>& vertices)
{
    std::vector<bool> covered(graph.n, false);

    for (const int v : vertices)
        covered[v] = true;

    std::vector<bool> chosen = covered;

    for (const auto& [u, v] : graph.edges)
    {
        if (chosen[u])
            covered[v] = true;
        if (chosen[v])
            covered[u] = true;
    }

    for (const bool c : covered)
    {
        if (!c)
            return false;
    }

    return true;
}

// Size of a minimum dominating set, by enumeration of all subsets.
int bruteForceDomination(const MISInstance& graph)
{
    const std::vector<unsigned> masks = closedMasks(graph);
    const unsigned all = (1u << graph.n) - 1;
    int best = graph.n;

    for (unsigned subset = 0; subset <= all; ++subset)
    {
        unsigned covered = 0;
        int size = 0;

        for (int v = 0; v < graph.n; ++v)
        {
            if (subset & (1u << v))
            {
                covered |= masks[v];
                ++size;
            }
        }

        if (covered == all && size < best)
            best = size;
    }

    return best;
}

// Optimum found by SCIP on the model, or -1 on failure.
double solveModel(GraphProblem problem, const MISInstance& graph)
{
    SCIP* scip = nullptr;

    if (SCIPcreate(&scip) != SCIP_OKAY)
        return -1.0;

    double value = -1.0;
    std::vector<SCIP_VAR*> vars;

    if (SCIPincludeDefaultPlugins(scip) == SCIP_OKAY
        && SCIPsetIntParam(scip, "display/verblevel", 0) == SCIP_OKAY
        && buildGraphProblemModel(scip, problem, graph, vars) == SCIP_OKAY
        && SCIPsolve(scip) == SCIP_OKAY
        && SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL)
    {
        value = SCIPgetPrimalbound(scip);
    }

    SCIPfree(&scip);
    return value;
}

void testNames()
{
    GraphProblem problem = GraphProblem::MaximumIndependentSet;

    check(parseGraphProblem("dominating", problem)
              && problem == GraphProblem::MinimumDominatingSet,
          "dominating is parsed");
    check(parseGraphProblem("mis", problem)
              && problem == GraphProblem::MaximumIndependentSet,
          "mis is parsed");
    check(!parseGraphProblem("vertex_cover", problem)
              && problem == GraphProblem::MaximumIndependentSet,
          "an unknown problem is refused");
    check(std::string(graphProblemName(GraphProblem::MinimumDominatingSet))
              == "dominating",
          "name of the dominating set problem");
}

void testAgainstEnumeration()
{
    for (const double p : {0.1, 0.2, 0.4})
    {
        for (unsigned int seed = 1; seed <= 4; ++seed)
        {
            MISInstance graph = ErdosRenyiGenerator{}.generate({12, p, seed}).graph;
            // A duplicate edge must not change anything.
            if (!graph.edges.empty())
                graph.edges.push_back(graph.edges.front());

            const int optimum = bruteForceDomination(graph);
            const double solved =
                solveModel(GraphProblem::MinimumDominatingSet, graph);

            check(std::fabs(solved - optimum) < 1e-6,
                  "G(12, " + std::to_string(p) + ") seed " + std::to_string(seed)
                      + ": SCIP finds the optimum of the enumeration ("
                      + std::to_string(solved) + " vs "
                      + std::to_string(optimum) + ")");
        }
    }

    // Without edges, every vertex dominates only itself.
    check(std::fabs(solveModel(GraphProblem::MinimumDominatingSet,
                               MISInstance{5, {}}) - 5.0) < 1e-6,
          "a graph without edges needs every vertex");
}

void testInvalidGraph()
{
    SCIP* scip = nullptr;
    SCIPcreate(&scip);
    SCIPincludeDefaultPlugins(scip);

    std::vector<SCIP_VAR*> vars;
    check(buildGraphProblemModel(scip, GraphProblem::MinimumDominatingSet,
                                 MISInstance{3, {{0, 3}}}, vars)
              == SCIP_INVALIDDATA
              && vars.empty(),
          "an edge outside the graph is refused");

    SCIPfree(&scip);
}

// The rules of the study and the learned rule on the dominating set.
void testBenchmark(const std::filesystem::path& workDir)
{
    const MISInstance graph = ErdosRenyiGenerator{}.generate({90, 0.08, 2}).graph;

    const auto solve = [&](const std::string& rule, MISBenchmarkResult& result,
                           const std::vector<int>& start = {},
                           const std::filesystem::path& model = {})
    {
        MISBenchmarkConfig config;
        config.problem = GraphProblem::MinimumDominatingSet;
        config.branchingRule = rule;
        config.startSolution = start;
        config.learnedModelPath = model;

        std::string error;
        const bool ok = benchmarkMIS(graph, config, result, error);

        if (!ok)
            std::cerr << "  " << rule << ": " << error << '\n';

        return ok && result.solved;
    };

    MISBenchmarkResult reference;
    const bool solved = solve("relpscost", reference);
    check(solved && reference.exploredNodes > 1,
          "relpscost solves the instance by branching ("
              + std::to_string(reference.exploredNodes) + " nodes)");
    check(dominates(graph, reference.selectedVertices)
              && std::fabs(reference.primalBound
                           - static_cast<double>(reference.selectedVertices.size()))
                  < 1e-6,
          "its solution is a dominating set of the size it reports ("
              + std::to_string(reference.selectedVertices.size()) + " vertices, "
              + std::to_string(reference.primalBound) + ")");

    for (const std::string rule : {"misfullstrong", "misfullstrong_nocutoff"})
    {
        MISBenchmarkResult result;
        check(solve(rule, result)
                  && std::fabs(result.primalBound - reference.primalBound) < 1e-6,
              rule + " proves the same optimum");
    }

    MISBenchmarkResult started;
    check(solve("relpscost", started, reference.selectedVertices)
              && std::fabs(started.primalBound - reference.primalBound) < 1e-6,
          "the optimum given first is accepted");

    // A model that branches on the most fractional candidate.
    const auto modelPath = workDir / "most_fractional.model";
    {
        std::ofstream file(modelPath);
        file << "MIS_BRANCHING_MODEL 1\nfeatures " << MIS_BRANCHING_FEATURE_COUNT
             << '\n';

        for (int f = 0; f < MIS_BRANCHING_FEATURE_COUNT; ++f)
            file << "feature " << MISbranchingFeatureName(f) << " 0 1\n";

        file << "layers 1\nlayer " << MIS_BRANCHING_FEATURE_COUNT << " 1 linear\n";

        for (int f = 0; f < MIS_BRANCHING_FEATURE_COUNT; ++f)
            file << (f > 0 ? " " : "") << (f == 1 ? 1 : 0);

        file << "\n0\n";
    }

    MISBenchmarkResult learned;
    check(solve("mislearned", learned, {}, modelPath)
              && std::fabs(learned.primalBound - reference.primalBound) < 1e-6
              && learned.learnedFeatureSeconds > 0.0,
          "the learned rule proves the same optimum");
}

void testCollection(const std::filesystem::path& workDir)
{
    const MISInstance graph = ErdosRenyiGenerator{}.generate({90, 0.08, 2}).graph;

    MISCollectionConfig config;
    config.instanceName = "dominating.mis";
    config.problem = GraphProblem::MinimumDominatingSet;

    MISCollectionResult result;
    std::string error;
    const bool ok = collectMISStrongBranchingSamples(
        graph, config, workDir / "dominating", result, error);

    std::ifstream metadata(result.metadataPath);
    std::string line;
    bool named = false;

    while (std::getline(metadata, line))
        named = named || line == "problem \"dominating\"";

    check(ok && result.complete && result.numberOfSamples > 0 && named,
          "samples are collected on the dominating set, and the metadata "
          "name the problem " + error);
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " WORK_DIR\n";
        return 2;
    }

    const std::filesystem::path workDir(argv[1]);
    std::filesystem::remove_all(workDir);
    std::filesystem::create_directories(workDir);

    testNames();
    testAgainstEnumeration();
    testInvalidGraph();
    testBenchmark(workDir);
    testCollection(workDir);

    if (failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }

    std::cout << "All dominating set tests passed\n";
    return 0;
}
