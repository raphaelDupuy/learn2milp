/*
 * Tests of the branching rule benchmark.
 *
 * The key check ties evaluation to data collection: benchmarking the
 * "misfullstrong" expert must explore exactly the tree that the data
 * collection recorded for the same instance and seed.
 */

#include "services/MISBenchmarkService.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "generators/ErdosRenyiGenerator.hpp"
#include "learning/MISBranchingFeatures.h"
#include "services/MISCollectionService.hpp"

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

bool run(
    const MISInstance& instance,
    const std::string& rule,
    MISBenchmarkSettings settings,
    unsigned int seed,
    MISBenchmarkResult& result
)
{
    MISBenchmarkConfig config;
    config.branchingRule = rule;
    config.settings = settings;
    config.scipSeed = seed;

    std::string error;

    if (!benchmarkMIS(instance, config, result, error))
    {
        std::cerr << "  " << rule << ": " << error << '\n';
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

    // Needs branching with the study settings.
    const MISInstance instance =
        ErdosRenyiGenerator{}.generate({100, 0.10, 1}).graph;

    // Every rule proves the same optimum.
    MISBenchmarkResult reference;

    check(
        run(instance, "relpscost", MISBenchmarkSettings::Default, 0,
            reference)
        && reference.solved && reference.status == "OPTIMAL"
        && reference.hasSolution
        && std::fabs(reference.primalBound - reference.dualBound) < 1e-6,
        "SCIP as shipped solves the instance"
    );

    for (const char* rule : {
            "relpscost", "pscost", "vanillafullstrong", "misfullstrong"})
    {
        MISBenchmarkResult result;

        const bool ok =
            run(instance, rule, MISBenchmarkSettings::Study, 0, result);

        check(
            ok && result.solved
            && std::fabs(result.primalBound - reference.primalBound) < 1e-6
            && result.exploredNodes > 1
            && result.lpIterations > 0
            && result.solvingTimeSeconds > 0.0,
            std::string(rule) + " reaches the same optimum ("
                + std::to_string(result.exploredNodes) + " nodes)"
        );
    }

    // The rule really changes the search.
    {
        MISBenchmarkResult pscost;
        MISBenchmarkResult expert;

        check(
            run(instance, "pscost", MISBenchmarkSettings::Study, 0, pscost)
            && run(instance, "misfullstrong", MISBenchmarkSettings::Study,
                0, expert)
            && expert.exploredNodes < pscost.exploredNodes,
            "Full strong branching explores fewer nodes than pscost"
        );
    }

    // Without its strong branching pruning, the expert branches where it
    // would prune: same optimum, never fewer nodes.
    {
        MISBenchmarkResult pruned;
        MISBenchmarkResult unpruned;

        const bool ran =
            isMISBenchmarkRuleAvailable("misfullstrong_nocutoff")
            && run(instance, "misfullstrong", MISBenchmarkSettings::Study, 0, pruned)
            && run(instance, "misfullstrong_nocutoff", MISBenchmarkSettings::Study,
                0, unpruned);

        check(
            ran
            && unpruned.solved
            && std::fabs(unpruned.primalBound - pruned.primalBound) < 1e-6
            && unpruned.exploredNodes >= pruned.exploredNodes,
            "The expert without pruning reaches the same optimum ("
                + std::to_string(pruned.exploredNodes) + " nodes pruned, "
                + std::to_string(unpruned.exploredNodes) + " unpruned)"
        );
    }

    // Node counts are reproducible for a given seed.
    {
        MISBenchmarkResult first;
        MISBenchmarkResult second;

        check(
            run(instance, "relpscost", MISBenchmarkSettings::Study, 3, first)
            && run(instance, "relpscost", MISBenchmarkSettings::Study, 3,
                second)
            && first.exploredNodes == second.exploredNodes
            && first.lpIterations == second.lpIterations,
            "Same seed gives the same search"
        );
    }

    // Evaluation and data collection run in the same conditions: the
    // expert explores exactly the tree that the collection recorded.
    {
        MISCollectionConfig collection;
        collection.instanceName = "instance";

        MISCollectionResult collected;
        MISBenchmarkResult expert;
        std::string error;

        const bool same =
            collectMISStrongBranchingSamples(
                instance, collection, workDir / "expert", collected, error)
            && run(instance, "misfullstrong", MISBenchmarkSettings::Study,
                0, expert)
            && collected.exploredNodes > 1
            && expert.exploredNodes == collected.exploredNodes
            && std::fabs(expert.primalBound - collected.primalBound) < 1e-6;

        check(
            same,
            "Benchmarked expert matches the collected search tree ("
                + std::to_string(collected.exploredNodes) + " nodes)"
        );
    }

    // Learned rule, driven by a hand-written model file ("branch on the
    // most fractional vertex").
    {
        const auto modelPath = workDir / "most_fractional.model";
        {
            std::ofstream file(modelPath);

            file << "MIS_BRANCHING_MODEL 1\nfeatures "
                 << MIS_BRANCHING_FEATURE_COUNT << '\n';

            for (int f = 0; f < MIS_BRANCHING_FEATURE_COUNT; ++f)
                file << "feature " << MISbranchingFeatureName(f) << " 0 1\n";

            file << "layers 1\nlayer " << MIS_BRANCHING_FEATURE_COUNT
                 << " 1 linear\n";

            for (int f = 0; f < MIS_BRANCHING_FEATURE_COUNT; ++f)
                file << (f > 0 ? " " : "") << (f == 1 ? 1 : 0);

            file << "\n0\n";
        }

        MISBenchmarkConfig config;
        config.branchingRule = "mislearned";
        config.learnedModelPath = modelPath;

        MISBenchmarkResult learned;
        MISBenchmarkResult again;
        std::string error;

        const bool ok =
            benchmarkMIS(instance, config, learned, error)
            && benchmarkMIS(instance, config, again, error);

        check(
            ok && learned.solved
            && std::fabs(learned.primalBound - reference.primalBound) < 1e-6
            && learned.exploredNodes > 1
            && learned.exploredNodes == again.exploredNodes,
            "Learned rule reaches the same optimum ("
                + std::to_string(learned.exploredNodes) + " nodes) " + error
        );

        // Strong branching on the candidates the model ranks first: on all
        // of them, the rule makes the expert's decisions.
        {
            MISBenchmarkConfig expertConfig;
            expertConfig.branchingRule = "misfullstrong";

            MISBenchmarkResult expert;
            MISBenchmarkResult allCandidates;
            MISBenchmarkResult threeCandidates;

            MISBenchmarkConfig strong = config;
            strong.learnedStrongBranchingCandidates = 100000;

            const bool ran =
                benchmarkMIS(instance, expertConfig, expert, error)
                && benchmarkMIS(instance, strong, allCandidates, error);

            strong.learnedStrongBranchingCandidates = 3;

            const bool ranThree =
                benchmarkMIS(instance, strong, threeCandidates, error);

            check(
                ran && allCandidates.solved
                && allCandidates.exploredNodes == expert.exploredNodes
                && std::fabs(allCandidates.primalBound - expert.primalBound) < 1e-6,
                "Strong branching on every candidate repeats the expert ("
                    + std::to_string(allCandidates.exploredNodes) + " and "
                    + std::to_string(expert.exploredNodes) + " nodes)"
            );

            check(
                ranThree && threeCandidates.solved
                && std::fabs(threeCandidates.primalBound - reference.primalBound) < 1e-6,
                "Strong branching on three candidates reaches the optimum ("
                    + std::to_string(threeCandidates.exploredNodes) + " nodes)"
            );

            MISBenchmarkConfig wrong;
            wrong.branchingRule = "relpscost";
            wrong.learnedStrongBranchingCandidates = 3;

            check(
                !benchmarkMIS(instance, wrong, threeCandidates, error),
                "Strong branching candidates given to another rule are rejected"
            );
        }

        MISBenchmarkResult result;

        config.learnedModelPath = workDir / "missing.model";

        check(
            !benchmarkMIS(instance, config, result, error)
            && error.find("Unable to open model file") == 0,
            "Missing model file is reported"
        );

        config.learnedModelPath.clear();

        check(
            !benchmarkMIS(instance, config, result, error),
            "Learned rule without a model is rejected"
        );

        config.branchingRule = "relpscost";
        config.learnedModelPath = modelPath;

        check(
            !benchmarkMIS(instance, config, result, error),
            "Model given to another rule is rejected"
        );

        config.branchingRule = "mislearned";

        check(
            benchmarkMIS(MISInstance{0, {}}, config, result, error)
            && isMISBenchmarkRuleAvailable("mislearned"),
            "Learned rule accepts an empty graph"
        );
    }

    // Time limit: reported, not an error.
    {
        MISBenchmarkConfig config;
        config.branchingRule = "pscost";
        config.timeLimitSeconds = 0.001;

        MISBenchmarkResult result;
        std::string error;

        check(
            benchmarkMIS(instance, config, result, error)
            && !result.solved
            && result.status == "TIMELIMIT",
            "Time limit is reported as an unsolved run"
        );
    }

    // Degenerate graphs.
    {
        MISBenchmarkResult result;

        check(
            run(MISInstance{5, {}}, "misfullstrong",
                MISBenchmarkSettings::Study, 0, result)
            && result.solved
            && std::fabs(result.primalBound - 5.0) < 1e-6,
            "Edgeless graph is solved with the expert"
        );

        check(
            run(MISInstance{0, {}}, "misfullstrong",
                MISBenchmarkSettings::Study, 0, result),
            "Empty graph is accepted with the expert"
        );
    }

    // Rejections.
    {
        MISBenchmarkConfig config;
        MISBenchmarkResult result;
        std::string error;

        config.branchingRule = "no_such_rule";

        check(
            !benchmarkMIS(instance, config, result, error)
            && error == "Unknown branching rule: no_such_rule",
            "Unknown branching rule is rejected"
        );

        config.branchingRule = "relpscost";
        config.timeLimitSeconds = 0.0;

        check(
            !benchmarkMIS(instance, config, result, error),
            "Non-positive time limit is rejected"
        );

        config.timeLimitSeconds = 10.0;

        check(
            !benchmarkMIS(MISInstance{3, {{0, 3}}}, config, result, error)
            && error.rfind("Invalid MIS instance: ", 0) == 0,
            "Invalid instance is rejected"
        );

        check(
            isMISBenchmarkRuleAvailable("relpscost")
            && isMISBenchmarkRuleAvailable("misfullstrong")
            && !isMISBenchmarkRuleAvailable("no_such_rule")
            && !isMISBenchmarkRuleAvailable(""),
            "Rule availability is reported"
        );
    }

    // Settings names.
    {
        MISBenchmarkSettings parsed = MISBenchmarkSettings::Study;

        check(
            parseMISBenchmarkSettings("default", parsed)
            && parsed == MISBenchmarkSettings::Default
            && parseMISBenchmarkSettings("study", parsed)
            && parsed == MISBenchmarkSettings::Study
            && !parseMISBenchmarkSettings("other", parsed)
            && std::string(misBenchmarkSettingsName(
                MISBenchmarkSettings::Study)) == "study",
            "Settings names round-trip"
        );
    }

    // SCIP parameter files, applied after the settings.
    {
        const auto write = [&](const std::string& name, const std::string& text)
        {
            const auto path = workDir / name;
            std::ofstream(path) << text;
            return path;
        };

        check(
            misBenchmarkSettingsLabel(
                MISBenchmarkSettings::Default, "dir/no_node_gomory.set")
                == "default+no_node_gomory"
            && misBenchmarkSettingsLabel(MISBenchmarkSettings::Study, {})
                == "study",
            "Settings labels name the parameter file"
        );

        // The study settings written as a file on top of SCIP's defaults
        // give exactly the search of the study settings.
        {
            MISBenchmarkConfig study;
            study.branchingRule = "relpscost";

            MISBenchmarkConfig asFile = study;
            asFile.settings = MISBenchmarkSettings::Default;
            asFile.parameterFile = write("as_study.set",
                "# the study settings, as a parameter file\n"
                "presolving/maxrestarts = 0\n"
                "\n"
                "separating/maxrounds = 0   # cuts at the root only\n");

            MISBenchmarkResult fromSettings;
            MISBenchmarkResult fromFile;
            std::string error;

            // Solved before the message is built (argument order).
            const bool ran =
                benchmarkMIS(instance, study, fromSettings, error)
                && benchmarkMIS(instance, asFile, fromFile, error);

            check(
                ran
                && fromFile.solved
                && fromFile.exploredNodes == fromSettings.exploredNodes
                && fromFile.lpIterations == fromSettings.lpIterations,
                "A parameter file reproduces the study settings ("
                    + std::to_string(fromFile.exploredNodes) + " and "
                    + std::to_string(fromSettings.exploredNodes) + " nodes) "
                    + error
            );
        }

        {
            MISBenchmarkConfig config;
            config.branchingRule = "relpscost";
            config.parameterFile = write("one_node.set", "limits/nodes = 1\n");

            MISBenchmarkResult result;
            std::string error;

            check(
                benchmarkMIS(instance, config, result, error)
                && !result.solved && result.status == "NODELIMIT",
                "A parameter file changes the search (node limit)"
            );

            // The run's own time limit takes precedence over the file.
            config.parameterFile = write("short_time.set", "limits/time = 0.001\n");

            check(
                benchmarkMIS(instance, config, result, error) && result.solved,
                "The time limit of the run takes precedence over the file"
            );
        }

        std::string error;
        MISBenchmarkResult result;
        MISBenchmarkConfig config;
        config.branchingRule = "relpscost";

        config.parameterFile = write("unknown.set", "separating/gomori/freq = 0\n");

        check(
            !validateMISBenchmarkParameters(config.parameterFile, error)
            && error.find("separating/gomori/freq") != std::string::npos
            && !benchmarkMIS(instance, config, result, error),
            "A misspelled parameter is rejected"
        );

        config.parameterFile = write("out_of_range.set", "separating/gomory/freq = -5\n");

        check(
            !validateMISBenchmarkParameters(config.parameterFile, error)
            && !benchmarkMIS(instance, config, result, error),
            "A value out of range is rejected"
        );

        config.parameterFile = write("no_value.set", "separating/gomory/freq\n");

        check(
            !validateMISBenchmarkParameters(config.parameterFile, error)
            && error.find("line 1") != std::string::npos,
            "A line without a value is rejected"
        );

        config.parameterFile = workDir / "missing.set";

        check(
            !validateMISBenchmarkParameters(config.parameterFile, error)
            && error.find("Unable to open parameter file") == 0
            && validateMISBenchmarkParameters(
                write("valid.set", "separating/gomory/freq = 0\n"), error),
            "A missing file is reported, a valid one accepted"
        );
    }

    // Root bounds, and the lookahead cut selector.
    {
        MISBenchmarkConfig config;
        config.branchingRule = "relpscost";

        MISBenchmarkResult scipSelector;
        MISBenchmarkResult lookahead;
        std::string error;

        const bool ranScip = benchmarkMIS(instance, config, scipSelector, error);

        config.cutSelector = "lookahead";
        const bool ranLookahead = benchmarkMIS(instance, config, lookahead, error);

        // Maximization: first LP bound >= bound after the root cuts >= optimum.
        check(
            ranScip && scipSelector.solved
            && scipSelector.rootFirstLPBound >= scipSelector.rootDualBound - 1e-6
            && scipSelector.rootDualBound >= scipSelector.primalBound - 1e-6
            && scipSelector.rootFirstLPBound > scipSelector.primalBound + 0.5
            && scipSelector.cutsApplied > 0
            && scipSelector.lookaheadScored == 0,
            "Root bounds are reported (first LP "
                + std::to_string(scipSelector.rootFirstLPBound) + ", after cuts "
                + std::to_string(scipSelector.rootDualBound) + ", optimum "
                + std::to_string(scipSelector.primalBound) + ")"
        );

        check(
            ranLookahead && lookahead.solved
            && std::fabs(lookahead.primalBound - scipSelector.primalBound) < 1e-6
            && lookahead.lookaheadScored > 0
            && lookahead.lookaheadImproving <= lookahead.lookaheadScored,
            "The lookahead cut selector reaches the same optimum ("
                + std::to_string(lookahead.lookaheadScored) + " cuts scored, "
                + std::to_string(lookahead.lookaheadImproving) + " improving)"
        );

        config.cutSelector = "no_such_selector";

        check(
            !benchmarkMIS(instance, config, lookahead, error)
            && error == "Unknown cut selector: no_such_selector"
            && isMISBenchmarkCutSelectorAvailable("lookahead")
            && isMISBenchmarkCutSelectorAvailable("")
            && misBenchmarkSettingsLabel(
                MISBenchmarkSettings::Study, {}, "lookahead") == "study+lookahead",
            "Cut selectors are checked and named in the settings label"
        );
    }

    // Node log: one line per node solved, without changing the search.
    {
        // Rows of a node log, values as text, the header first.
        const auto readLog = [](const std::filesystem::path& path)
        {
            std::vector<std::vector<std::string>> rows;
            std::ifstream in(path);
            std::string line;

            while (std::getline(in, line))
            {
                std::vector<std::string> fields;
                std::size_t start = 0;

                for (std::size_t comma; (comma = line.find(',', start)) != std::string::npos;
                     start = comma + 1)
                {
                    fields.push_back(line.substr(start, comma - start));
                }

                fields.push_back(line.substr(start));
                rows.push_back(fields);
            }

            return rows;
        };

        // Empty for a value that was not measured.
        const auto value = [](const std::string& text)
        {
            return text.empty() ? std::nan("") : std::stod(text);
        };

        const MISInstance larger =
            ErdosRenyiGenerator{}.generate({125, 0.10, 3}).graph;

        // Zero-half cuts at every node, on top of SCIP's defaults.
        MISBenchmarkConfig config;
        config.branchingRule = "relpscost";
        config.settings = MISBenchmarkSettings::Default;
        config.parameterFile = workDir / "zerohalf_every_node.set";
        std::ofstream(config.parameterFile)
            << "separating/gomory/freq = 0\n"
            << "separating/zerohalf/freq = 1\n"
            << "separating/zerohalf/expbackoff = 1\n";

        MISBenchmarkConfig logging = config;
        logging.nodeLogPath = workDir / "nodes.csv";

        MISBenchmarkResult plain;
        MISBenchmarkResult logged;
        std::string error;

        const bool ran = benchmarkMIS(larger, config, plain, error)
            && benchmarkMIS(larger, logging, logged, error);

        check(
            ran && logged.solved
            && logged.exploredNodes == plain.exploredNodes
            && logged.lpIterations == plain.lpIterations,
            "The node log does not change the search ("
                + std::to_string(logged.exploredNodes) + " and "
                + std::to_string(plain.exploredNodes) + " nodes) " + error
        );

        // Columns used below.
        const std::size_t depth = 3, status = 4, lpFirst = 8, lpCutLoop = 9,
            lpFinal = 10, rounds = 14, cuts = 15, columns = 31;

        const auto rows = readLog(logging.nodeLogPath);
        bool wellFormed = !rows.empty() && rows[0].size() == columns
            && rows[0][0] == "run" && rows[0][lpFirst] == "lp_first"
            && rows[0][lpCutLoop] == "lp_cut_loop"
            && rows[0][cuts] == "cuts" && rows[0][16] == "pool_cuts"
            && rows[0][30] == "lp_resolves";
        bool boundsRise = true;
        long long nodeCuts = 0;
        long long nodesWithCuts = 0;
        long long pruned = 0;

        for (std::size_t i = 1; i < rows.size(); ++i)
        {
            const auto& row = rows[i];

            if (row.size() != columns)
            {
                wellFormed = false;
                break;
            }

            // Minimization: the cut loop, then the later solves of the node,
            // can only raise its LP bound.
            const double first = value(row[lpFirst]);
            const double afterLoop = value(row[lpCutLoop]);
            const double last = value(row[lpFinal]);

            if ((std::isfinite(first) && std::isfinite(afterLoop) && afterLoop < first - 1e-6)
                || (std::isfinite(afterLoop) && std::isfinite(last) && last < afterLoop - 1e-6))
            {
                boundsRise = false;
            }

            if (row[depth] != "0")
            {
                nodeCuts += std::stoll(row[cuts]);
                nodesWithCuts += std::stoll(row[cuts]) > 0 ? 1 : 0;
                pruned += row[status] == "infeasible" ? 1 : 0;
            }
        }

        const bool rootFirst = rows.size() > 1 && rows[1][1] == "1"
            && rows[1][2] == "0" && rows[1][depth] == "0"
            && value(rows[1][lpCutLoop]) > value(rows[1][lpFirst]) + 0.5
            && std::stoll(rows[1][cuts]) > 0;

        check(
            wellFormed && rootFirst
            && static_cast<long long>(rows.size()) - 1 == logged.exploredNodes,
            "The node log has one line per node, the root first ("
                + std::to_string(rows.size() - 1) + " lines)"
        );

        check(
            boundsRise && nodesWithCuts > 0,
            "The node log shows the cuts at the nodes ("
                + std::to_string(nodeCuts) + " cuts at "
                + std::to_string(nodesWithCuts) + " nodes, "
                + std::to_string(pruned) + " nodes pruned)"
        );

        // The study settings separate at the root only.
        MISBenchmarkConfig study;
        study.branchingRule = "relpscost";
        study.nodeLogPath = workDir / "study_nodes.csv";

        MISBenchmarkResult studied;
        bool noNodeCuts = benchmarkMIS(larger, study, studied, error);
        const auto studyRows = readLog(study.nodeLogPath);
        long long resolved = 0;

        for (std::size_t i = 1; i < studyRows.size(); ++i)
        {
            const auto& row = studyRows[i];

            if (row[depth] == "0")
                continue;

            // No cut loop at the nodes: its LP bound is the first one, and
            // only the later solves (after strong branching) can raise it.
            const double first = value(row[lpFirst]);

            if (row[rounds] != "0" || row[cuts] != "0"
                || (std::isfinite(first) && value(row[lpCutLoop]) != first))
            {
                noNodeCuts = false;
            }

            resolved += value(row[lpFinal]) > first + 1e-6 ? 1 : 0;
        }

        check(
            noNodeCuts && studyRows.size() > 2,
            "With the study settings, the node log shows no cut loop at the nodes ("
                + std::to_string(resolved) + " bounds raised by later solves)"
        );

        check(
            !benchmarkMIS(larger, logging, logged, error)
            && error.find("Node log already exists") == 0,
            "An existing node log is not overwritten"
        );
    }

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS benchmark service tests passed.\n";
    return 0;
}
