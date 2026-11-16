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

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS benchmark service tests passed.\n";
    return 0;
}
