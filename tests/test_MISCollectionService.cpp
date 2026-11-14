/*
 * Integration test of the strong branching data collection.
 *
 * The dataset files are read back and checked for internal consistency
 * (tables, candidates, vertex states), for consistency with the graph
 * (LP feasibility of the recorded states) and for consistency with the
 * search tree (parents, depths, fixings inherited by the children).
 */

#include "services/MISCollectionService.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "generators/ErdosRenyiGenerator.hpp"
#include "io/MISStrongBranchingSampleIO.hpp"
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

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}

struct Table
{
    std::string headerLine;
    std::map<std::string, std::size_t> columns;
    std::vector<std::vector<std::string>> rows;

    const std::string& text(std::size_t row, const std::string& column) const
    {
        return rows[row][columns.at(column)];
    }

    long long integer(std::size_t row, const std::string& column) const
    {
        return std::stoll(text(row, column));
    }

    double real(std::size_t row, const std::string& column) const
    {
        return std::stod(text(row, column));
    }
};

std::vector<std::string> split(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    std::istringstream stream(line);

    while (std::getline(stream, field, ','))
        fields.push_back(field);

    if (!line.empty() && line.back() == ',')
        fields.emplace_back();

    return fields;
}

bool readTable(const std::filesystem::path& path, Table& table)
{
    std::ifstream file(path);

    if (!file || !std::getline(file, table.headerLine))
        return false;

    const auto names = split(table.headerLine);

    for (std::size_t i = 0; i < names.size(); ++i)
        table.columns[names[i]] = i;

    std::string line;

    while (std::getline(file, line))
    {
        auto fields = split(line);

        if (fields.size() != names.size())
            return false;

        table.rows.push_back(std::move(fields));
    }

    return true;
}

// Reads "key value" lines; quoted values are unquoted.
std::map<std::string, std::string> readMetadata(
    const std::filesystem::path& path,
    std::string& firstLine
)
{
    std::map<std::string, std::string> metadata;
    std::ifstream file(path);
    std::string line;

    std::getline(file, firstLine);

    while (std::getline(file, line))
    {
        const auto space = line.find(' ');

        if (space == std::string::npos)
            continue;

        std::string value = line.substr(space + 1);

        if (value.size() >= 2 && value.front() == '"' && value.back() == '"')
            value = value.substr(1, value.size() - 2);

        metadata[line.substr(0, space)] = value;
    }

    return metadata;
}

bool solveReference(const MISInstance& instance, double& objective)
{
    SCIP* scip = nullptr;

    if (SCIPcreate(&scip) != SCIP_OKAY)
        return false;

    std::vector<SCIP_VAR*> vars;

    const bool solved =
        SCIPincludeDefaultPlugins(scip) == SCIP_OKAY
        && SCIPsetIntParam(scip, "display/verblevel", 0) == SCIP_OKAY
        && buildMISModel(scip, instance, vars) == SCIP_OKAY
        && SCIPsolve(scip) == SCIP_OKAY
        && SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL;

    if (solved)
        objective = SCIPgetPrimalbound(scip);

    SCIPfree(&scip);
    return solved;
}

/*
 * Checks every invariant of a collected dataset. Returns the list of
 * violations (empty when the dataset is consistent).
 */
std::vector<std::string> validateDataset(
    const MISInstance& instance,
    const MISCollectionResult& result,
    const double* optimum
)
{
    std::vector<std::string> problems;

    const auto fail = [&problems](const std::string& message)
    {
        if (problems.size() < 15)
            problems.push_back(message);
    };

    Table nodes;
    Table candidates;

    if (!readTable(result.nodesPath, nodes)
        || !readTable(result.candidatesPath, candidates))
    {
        fail("tables cannot be read or have ragged rows");
        return problems;
    }

    if (nodes.headerLine != MIS_SB_NODE_CSV_HEADER
        || candidates.headerLine != MIS_SB_CANDIDATE_CSV_HEADER)
        fail("unexpected header row");

    if (nodes.rows.size() != result.numberOfSamples
        || candidates.rows.size() != result.numberOfCandidates)
        fail("row counts differ from the collection result");

    // Metadata.
    std::string firstLine;
    const auto metadata = readMetadata(result.metadataPath, firstLine);

    const auto meta = [&metadata](const std::string& key)
    {
        const auto it = metadata.find(key);
        return it == metadata.end() ? std::string() : it->second;
    };

    if (firstLine != "MIS_SB_DATASET 1")
        fail("unexpected metadata header");

    if (meta("number_of_vertices") != std::to_string(instance.n)
        || meta("number_of_edges") != std::to_string(instance.edges.size())
        || meta("number_of_samples") != std::to_string(nodes.rows.size())
        || meta("number_of_candidates")
            != std::to_string(candidates.rows.size())
        || meta("explored_nodes") != std::to_string(result.exploredNodes)
        || meta("status") != result.status
        || meta("complete") != (result.complete ? "1" : "0")
        || meta("parameter.presolving/maxrestarts") != "0"
        || meta("parameter.separating/maxrounds") != "0"
        || meta("branching_rule") != "misfullstrong"
        || meta("scip_version").empty()
        || meta("original_objective_slope").empty())
        fail("metadata does not match the collection result");

    const double intercept =
        std::atof(meta("original_objective_intercept").c_str());
    const double slope =
        std::atof(meta("original_objective_slope").c_str());

    // Adjacency lists.
    std::vector<std::vector<int>> adjacency(instance.n);

    for (const auto& [u, v] : instance.edges)
    {
        adjacency[u].push_back(v);
        adjacency[v].push_back(u);
    }

    // Candidates grouped by sample.
    std::map<long long, std::vector<std::size_t>> candidatesOfSample;

    for (std::size_t r = 0; r < candidates.rows.size(); ++r)
    {
        candidatesOfSample[candidates.integer(r, "sample_index")]
            .push_back(r);
    }

    struct NodeInfo
    {
        long long depth;
        std::string states;
        long long selectedVertex;
        bool cutoff;
    };

    std::map<long long, NodeInfo> nodeInfo;
    long long cutoffSamples = 0;

    for (std::size_t r = 0; r < nodes.rows.size(); ++r)
    {
        const std::string where = "sample " + std::to_string(r) + ": ";
        const long long nodeId = nodes.integer(r, "node_id");
        const std::string& states = nodes.text(r, "vertex_states");

        if (nodes.integer(r, "sample_index") != static_cast<long long>(r))
            fail(where + "sample_index is not sequential");

        // Vertex states.
        if (nodes.integer(r, "number_of_vertices") != instance.n
            || states.size() != static_cast<std::size_t>(instance.n)
            || states.find_first_not_of("ZOzof") != std::string::npos)
        {
            fail(where + "malformed vertex_states");
            continue;
        }

        long long fractional = 0;

        for (const char state : states)
        {
            if (state == 'f')
                ++fractional;
        }

        // The recorded LP solution must be feasible for the graph: a
        // vertex at 1 forces all its neighbours to 0.
        for (int v = 0; v < instance.n; ++v)
        {
            if (states[v] != 'O' && states[v] != 'o')
                continue;

            for (const int neighbour : adjacency[v])
            {
                if (states[neighbour] != 'Z' && states[neighbour] != 'z')
                    fail(where + "vertex at 1 has a neighbour not at 0");
            }
        }

        // Objective bounds, all in the same (minimization) space.
        const double lpObjective = nodes.real(r, "lp_objective");

        if (nodes.integer(r, "lp_objective_valid") != 1)
            fail(where + "invalid LP objective");

        if (nodes.integer(r, "dual_bound_valid") == 1
            && nodes.real(r, "dual_bound") > lpObjective + 1e-6)
            fail(where + "dual bound above the LP objective");

        if (nodes.integer(r, "primal_bound_valid") == 1)
        {
            if (nodes.real(r, "primal_bound") < lpObjective - 1e-6)
                fail(where + "primal bound below the LP objective");

            // Back in the original space, a primal bound is the size of
            // a known independent set and the LP bound is optimistic.
            if (optimum != nullptr)
            {
                const double primal =
                    intercept + slope * nodes.real(r, "primal_bound");
                const double dual =
                    intercept + slope * nodes.real(r, "dual_bound");

                if (primal > *optimum + 1e-6 || dual < *optimum - 1e-6
                    || std::fabs(primal - std::round(primal)) > 1e-6)
                    fail(where + "bounds inconsistent with the optimum");
            }
        }

        // Candidates of this sample.
        const auto& rowsOfSample = candidatesOfSample[r];
        const long long numberOfCandidates =
            nodes.integer(r, "number_of_candidates");

        if (static_cast<long long>(rowsOfSample.size()) != numberOfCandidates
            || fractional != numberOfCandidates
            || numberOfCandidates == 0)
        {
            fail(where + "candidate count mismatch");
            continue;
        }

        long long selected = 0;
        long long selectedVertex = -1;
        bool selectedIsDeadEnd = false;
        std::set<long long> vertices;
        bool allEvaluated = true;

        for (std::size_t k = 0; k < rowsOfSample.size(); ++k)
        {
            const std::size_t c = rowsOfSample[k];
            const long long vertex = candidates.integer(c, "vertex_id");
            const double lpValue = candidates.real(c, "lp_value");
            const double fractionality =
                candidates.real(c, "fractionality");

            if (candidates.integer(c, "node_id") != nodeId
                || candidates.integer(c, "candidate_position")
                    != static_cast<long long>(k))
                fail(where + "candidate rows out of order");

            if (vertex < 0 || vertex >= instance.n
                || !vertices.insert(vertex).second
                || states[vertex] != 'f')
            {
                fail(where + "candidate is not a fractional vertex");
                continue;
            }

            if (lpValue <= 0.0 || lpValue >= 1.0
                || std::fabs(fractionality
                    - std::min(lpValue, 1.0 - lpValue)) > 1e-9
                || candidates.real(c, "local_lower_bound") != 0.0
                || candidates.real(c, "local_upper_bound") != 1.0)
                fail(where + "inconsistent candidate LP data");

            if (std::fabs(candidates.real(c, "gain_reference_lp_objective")
                    - lpObjective) > 1e-9)
                fail(where + "gain reference differs from LP objective");

            if (candidates.text(c, "evaluation_status") == "evaluated")
            {
                if (candidates.integer(c, "score_valid") != 1
                    || candidates.integer(c, "expert_rank") < 0
                    || candidates.integer(c, "down_skipped") != 0
                    || candidates.integer(c, "up_skipped") != 0
                    || candidates.real(c, "down_gain") < 0.0
                    || candidates.real(c, "up_gain") < 0.0
                    || candidates.real(c, "score") <= 0.0)
                    fail(where + "inconsistent evaluated candidate");
            }
            else
            {
                allEvaluated = false;
            }

            if (candidates.integer(c, "expert_selected") == 1)
            {
                ++selected;
                selectedVertex = vertex;
                selectedIsDeadEnd =
                    candidates.integer(c, "down_infeasible") == 1
                    && candidates.integer(c, "up_infeasible") == 1;

                if (candidates.integer(c, "expert_rank") != 0)
                    fail(where + "selected candidate is not ranked first");
            }
        }

        // Nodes reached while the solver is stopping are not recorded,
        // so every recorded sample is a complete expert decision.
        if (!allEvaluated)
            fail(where + "candidate without strong branching evaluation");

        if (nodes.integer(r, "expert_selection_valid") != 1
            || selected != 1
            || selectedVertex
                != nodes.integer(r, "expert_selected_vertex_id"))
            fail(where + "missing or inconsistent expert selection");

        // The node is pruned exactly when both branches of the selected
        // candidate are infeasible or cut off.
        const bool cutoff = nodes.integer(r, "node_cutoff") == 1;

        if (cutoff != selectedIsDeadEnd)
            fail(where + "node_cutoff disagrees with the selected candidate");

        if (cutoff)
            ++cutoffSamples;

        if (!nodeInfo.emplace(
                nodeId,
                NodeInfo{
                    nodes.integer(r, "node_depth"),
                    states,
                    selectedVertex,
                    cutoff}).second)
            fail(where + "node recorded twice");
    }

    // Search tree: this rule branches at every recorded node, so the
    // parent of a recorded node is recorded too. A child inherits its
    // parent's fixings and has the parent's branching vertex fixed.
    long long roots = 0;

    for (std::size_t r = 0; r < nodes.rows.size(); ++r)
    {
        const std::string where = "sample " + std::to_string(r) + ": ";
        const auto self = nodeInfo.find(nodes.integer(r, "node_id"));

        if (self == nodeInfo.end())
            continue;

        if (nodes.integer(r, "parent_node_valid") == 0)
        {
            ++roots;

            if (self->second.depth != 0)
                fail(where + "root node with a non-zero depth");

            continue;
        }

        const auto parent =
            nodeInfo.find(nodes.integer(r, "parent_node_id"));

        if (parent == nodeInfo.end())
        {
            fail(where + "parent node was never recorded");
            continue;
        }

        if (self->second.depth != parent->second.depth + 1)
            fail(where + "depth is not the parent's depth plus one");

        if (parent->second.cutoff)
            fail(where + "parent was pruned but has a recorded child");

        for (int v = 0; v < instance.n; ++v)
        {
            const char inParent = parent->second.states[v];

            if ((inParent == 'Z' || inParent == 'O')
                && self->second.states[v] != inParent)
                fail(where + "child lost a fixing of its parent");
        }

        const long long branched = parent->second.selectedVertex;

        if (branched >= 0
            && self->second.states[branched] != 'Z'
            && self->second.states[branched] != 'O')
            fail(where + "parent's branching vertex is not fixed");
    }

    if (!nodes.rows.empty() && roots != 1)
        fail("expected exactly one root sample");

    // A complete search ends by pruning nodes: some samples are cut off.
    if (result.complete && nodes.rows.size() > 1 && cutoffSamples == 0)
        fail("no pruned node in a complete run");

    return problems;
}

void report(
    const std::string& testName,
    const std::vector<std::string>& problems
)
{
    for (const auto& problem : problems)
        std::cerr << "  " << problem << '\n';

    check(problems.empty(), testName);
}

bool noFileWithPrefix(const std::filesystem::path& prefix)
{
    const std::string name = prefix.filename().string();

    if (!std::filesystem::exists(prefix.parent_path()))
        return true;

    for (const auto& entry :
            std::filesystem::directory_iterator(prefix.parent_path()))
    {
        if (entry.path().filename().string().rfind(name, 0) == 0)
            return false;
    }

    return true;
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

    // Large enough to need branching with presolving and root cuts.
    const MISInstance instance =
        ErdosRenyiGenerator{}.generate({100, 0.10, 1}).graph;

    MISCollectionConfig config;
    config.instanceName = "erdos_renyi_n100_p0_1_seed1.mis";

    std::string error;

    // Complete run.
    MISCollectionResult first;
    const bool collected = collectMISStrongBranchingSamples(
        instance, config, workDir / "first", first, error);

    check(collected, "Collection succeeds: " + error);

    if (!collected)
        return 1;

    std::cout << "  " << first.exploredNodes << " nodes, "
              << first.numberOfSamples << " samples, "
              << first.numberOfCandidates << " candidates, status "
              << first.status << '\n';

    double optimum = 0.0;
    const bool hasReference = solveReference(instance, optimum);

    check(hasReference, "Reference optimum is available");

    check(
        first.status == "OPTIMAL" && first.complete && first.hasSolution
        && std::fabs(first.primalBound - optimum) < 1e-6
        && std::fabs(first.dualBound - optimum) < 1e-6,
        "Complete run reaches the reference optimum"
    );

    check(
        first.numberOfSamples > 1 && first.exploredNodes > 1,
        "Instance needs branching"
    );

    check(
        first.nodesPath == workDir / "first.sb.nodes.csv"
        && first.candidatesPath == workDir / "first.sb.candidates.csv"
        && first.metadataPath == workDir / "first.sb.meta"
        && !std::filesystem::exists(first.nodesPath.string() + ".tmp")
        && !std::filesystem::exists(first.candidatesPath.string() + ".tmp")
        && !std::filesystem::exists(first.metadataPath.string() + ".tmp"),
        "Output files are named after the prefix, without leftovers"
    );

    report(
        "Complete dataset is consistent",
        validateDataset(instance, first, &optimum)
    );

    // Reproducibility.
    MISCollectionResult second;

    check(
        collectMISStrongBranchingSamples(
            instance, config, workDir / "second", second, error)
        && readFile(first.nodesPath) == readFile(second.nodesPath)
        && readFile(first.candidatesPath)
            == readFile(second.candidatesPath),
        "Two runs produce identical tables"
    );

    // Existing outputs are never overwritten.
    const std::string nodesBefore = readFile(first.nodesPath);
    MISCollectionResult overwrite;

    check(
        !collectMISStrongBranchingSamples(
            instance, config, workDir / "first", overwrite, error)
        && !error.empty()
        && readFile(first.nodesPath) == nodesBefore,
        "Existing outputs are not overwritten"
    );

    // Truncated run.
    MISCollectionConfig limited = config;
    limited.nodeLimit = 5;

    MISCollectionResult truncated;

    check(
        collectMISStrongBranchingSamples(
            instance, limited, workDir / "truncated", truncated, error)
        && truncated.status == "NODELIMIT"
        && !truncated.complete
        && truncated.numberOfSamples > 0
        && truncated.numberOfSamples < first.numberOfSamples,
        "Node limit produces a truncated dataset"
    );

    report(
        "Truncated dataset is consistent",
        validateDataset(instance, truncated, &optimum)
    );

    // The truncated run follows the same tree as the complete one.
    {
        Table complete;
        Table partial;

        bool prefix =
            readTable(first.nodesPath, complete)
            && readTable(truncated.nodesPath, partial)
            && partial.rows.size() <= complete.rows.size();

        for (std::size_t r = 0; prefix && r < partial.rows.size(); ++r)
        {
            prefix = partial.text(r, "node_id")
                    == complete.text(r, "node_id")
                && partial.text(r, "vertex_states")
                    == complete.text(r, "vertex_states")
                && partial.text(r, "expert_selected_vertex_id")
                    == complete.text(r, "expert_selected_vertex_id");
        }

        check(prefix, "Truncated run is a prefix of the complete run");
    }

    // Time limit: wherever the run is interrupted, only complete expert
    // decisions are recorded. The cut-off point depends on the machine,
    // so only the invariants are checked.
    {
        MISCollectionConfig hurried = config;
        hurried.timeLimitSeconds =
            std::max(0.01, first.solvingTimeSeconds / 3.0);

        MISCollectionResult result;

        const bool ok = collectMISStrongBranchingSamples(
            instance, hurried, workDir / "hurried", result, error);

        check(
            ok
            && (result.status == "TIMELIMIT" || result.status == "OPTIMAL")
            && result.numberOfSamples <= first.numberOfSamples,
            "Time limit is honoured (" + result.status + ", "
                + std::to_string(result.numberOfSamples) + " samples)"
        );

        if (ok)
        {
            report(
                "Time-limited dataset is consistent",
                validateDataset(instance, result, &optimum)
            );
        }
    }

    // Instances without any branching still produce valid, empty tables.
    {
        const MISInstance edgeless{5, {}};
        MISCollectionResult result;

        const bool ok = collectMISStrongBranchingSamples(
            edgeless, config, workDir / "edgeless", result, error);

        check(
            ok && result.numberOfSamples == 0 && result.complete
            && std::fabs(result.primalBound - 5.0) < 1e-6
            && readFile(result.nodesPath)
                == std::string(MIS_SB_NODE_CSV_HEADER) + "\n"
            && readFile(result.candidatesPath)
                == std::string(MIS_SB_CANDIDATE_CSV_HEADER) + "\n",
            "Edgeless graph produces header-only tables"
        );

        const MISInstance empty{0, {}};

        check(
            collectMISStrongBranchingSamples(
                empty, config, workDir / "empty", result, error)
            && result.numberOfSamples == 0,
            "Empty graph is accepted"
        );
    }

    // Invalid requests are rejected and leave no file behind.
    {
        MISCollectionResult result;

        check(
            !collectMISStrongBranchingSamples(
                instance, config, "", result, error),
            "Empty output prefix is rejected"
        );

        MISCollectionConfig invalid = config;
        invalid.instanceName.clear();

        check(
            !collectMISStrongBranchingSamples(
                instance, invalid, workDir / "invalid", result, error),
            "Empty instance name is rejected"
        );

        invalid = config;
        invalid.timeLimitSeconds = 0.0;

        check(
            !collectMISStrongBranchingSamples(
                instance, invalid, workDir / "invalid", result, error),
            "Non-positive time limit is rejected"
        );

        invalid = config;
        invalid.nodeLimit = 0;

        check(
            !collectMISStrongBranchingSamples(
                instance, invalid, workDir / "invalid", result, error),
            "Zero node limit is rejected"
        );

        const MISInstance outOfRange{3, {{0, 3}}};

        check(
            !collectMISStrongBranchingSamples(
                outOfRange, config, workDir / "invalid", result, error),
            "Invalid instance is rejected"
        );

        check(
            !collectMISStrongBranchingSamples(
                instance,
                config,
                workDir / "missing_directory" / "invalid",
                result,
                error),
            "Missing output directory is reported"
        );

        check(
            noFileWithPrefix(workDir / "invalid")
            && !std::filesystem::exists(workDir / "missing_directory"),
            "Rejected requests leave no file behind"
        );
    }

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS collection service tests passed.\n";
    return 0;
}
