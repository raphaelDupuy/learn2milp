#include "services/MISBenchmarkService.hpp"

#include <climits>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <vector>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "branching/MISFullstrongBranchrule.h"
#include "branching/MISLearnedBranchrule.h"
#include "io/MISBranchingModelIO.hpp"
#include "io/MISInstanceIO.hpp"
#include "learning/MISAdjacency.hpp"
#include "services/MISSolverSettings.hpp"
#include "services/SCIPStatusName.hpp"

namespace
{

const char* const MIS_FULLSTRONG_RULE = "misfullstrong";
const char* const MIS_LEARNED_RULE = "mislearned";

// Highest value accepted by SCIP for a branching rule priority.
const int MAX_BRANCHING_PRIORITY = INT_MAX / 4;

// Frees the SCIP instance on every exit path.
struct SCIPHandle
{
    SCIP* scip = nullptr;

    ~SCIPHandle()
    {
        if (scip != nullptr)
            SCIPfree(&scip);
    }
};

void require(SCIP_RETCODE retcode, const char* action)
{
    if (retcode != SCIP_OKAY)
    {
        throw std::runtime_error(
            std::string("SCIP failed to ") + action + "."
        );
    }
}

// The expert only branches here: its samples are not recorded.
void ignoreSample(
    void*,
    const MISStrongBranchingSample*,
    const MISStrongBranchingCandidate*,
    size_t
)
{
}

} // namespace

const char* misBenchmarkSettingsName(MISBenchmarkSettings settings)
{
    switch (settings)
    {
        case MISBenchmarkSettings::Study:
            return "study";

        case MISBenchmarkSettings::Default:
            return "default";
    }

    return "unknown";
}

bool parseMISBenchmarkSettings(
    const std::string& name,
    MISBenchmarkSettings& settings
)
{
    for (const MISBenchmarkSettings candidate : {
            MISBenchmarkSettings::Study,
            MISBenchmarkSettings::Default})
    {
        if (name == misBenchmarkSettingsName(candidate))
        {
            settings = candidate;
            return true;
        }
    }

    return false;
}

bool isMISBenchmarkRuleAvailable(const std::string& branchingRule)
{
    if (branchingRule == MIS_FULLSTRONG_RULE
        || branchingRule == MIS_LEARNED_RULE)
    {
        return true;
    }

    SCIPHandle handle;

    return SCIPcreate(&handle.scip) == SCIP_OKAY
        && SCIPincludeDefaultPlugins(handle.scip) == SCIP_OKAY
        && SCIPfindBranchrule(handle.scip, branchingRule.c_str())
            != nullptr;
}

bool benchmarkMIS(
    const MISInstance& instance,
    const MISBenchmarkConfig& config,
    MISBenchmarkResult& result,
    std::string& errorMessage
)
{
    try
    {
        if (!std::isfinite(config.timeLimitSeconds)
            || config.timeLimitSeconds <= 0.0)
        {
            throw std::invalid_argument(
                "Time limit must be a positive number of seconds."
            );
        }

        if (config.scipSeed > static_cast<unsigned int>(INT_MAX))
        {
            throw std::invalid_argument(
                "SCIP seed exceeds the supported range."
            );
        }

        std::string instanceError;

        if (!validateMISInstance(instance, instanceError))
        {
            throw std::invalid_argument(
                "Invalid MIS instance: " + instanceError
            );
        }

        const bool learned = config.branchingRule == MIS_LEARNED_RULE;

        if (learned == config.learnedModelPath.empty())
        {
            throw std::invalid_argument(
                learned
                ? "The mislearned rule needs a model file."
                : "A model file is only used by the mislearned rule."
            );
        }

        // Declared before the solver: the learned rule points to them,
        // so they must be destroyed after it.
        LoadedMISBranchingModel model;
        MISAdjacency adjacency;
        MISFeatureGraph graph{};

        if (learned)
        {
            std::string modelError;

            if (!loadMISBranchingModel(
                    config.learnedModelPath, model, modelError))
            {
                throw std::invalid_argument(modelError);
            }

            adjacency = buildMISAdjacency(instance);
            graph = adjacency.view();
        }

        SCIPHandle handle;
        require(SCIPcreate(&handle.scip), "create a solver instance");

        SCIP* const scip = handle.scip;
        require(
            SCIPincludeDefaultPlugins(scip),
            "include the default plugins"
        );

        std::vector<SCIP_VAR*> vars;
        require(
            buildMISModel(scip, instance, vars),
            "build the MIS model"
        );

        require(
            SCIPsetIntParam(scip, "display/verblevel", 0),
            "disable console output"
        );

        if (config.settings == MISBenchmarkSettings::Study)
        {
            require(
                applyBranchingStudySettings(scip),
                "apply the branching study settings"
            );
        }

        require(
            SCIPsetRealParam(
                scip, "limits/time", config.timeLimitSeconds),
            "set the time limit"
        );
        require(
            SCIPsetIntParam(
                scip,
                "randomization/randomseedshift",
                static_cast<int>(config.scipSeed)),
            "set the random seed"
        );

        // Select the branching rule by giving it the highest priority.
        if (!config.branchingRule.empty())
        {
            if (config.branchingRule == MIS_FULLSTRONG_RULE
                && !vars.empty())
            {
                require(
                    SCIPincludeBranchruleMISFullstrong(
                        scip,
                        vars.data(),
                        static_cast<int>(vars.size()),
                        ignoreSample,
                        nullptr),
                    "include the strong branching rule"
                );
            }
            else if (learned && !vars.empty())
            {
                require(
                    SCIPincludeBranchruleMISLearned(
                        scip,
                        vars.data(),
                        static_cast<int>(vars.size()),
                        &graph,
                        &model.view(),
                        nullptr,
                        nullptr),
                    "include the learned branching rule"
                );
            }
            else if (config.branchingRule != MIS_FULLSTRONG_RULE
                && !learned
                && SCIPfindBranchrule(
                    scip, config.branchingRule.c_str()) == nullptr)
            {
                throw std::invalid_argument(
                    "Unknown branching rule: " + config.branchingRule
                );
            }

            SCIP_BRANCHRULE* const rule =
                SCIPfindBranchrule(scip, config.branchingRule.c_str());

            // Null only for our own rules on a graph without vertices,
            // where there is nothing to branch on.
            if (rule != nullptr)
            {
                require(
                    SCIPsetBranchrulePriority(
                        scip, rule, MAX_BRANCHING_PRIORITY),
                    "select the branching rule"
                );
            }
        }

        require(SCIPsolve(scip), "solve the instance");

        const SCIP_STATUS status = SCIPgetStatus(scip);

        MISBenchmarkResult measured;

        measured.status = getSCIPStatusName(status);
        measured.solved = status == SCIP_STATUS_OPTIMAL;
        measured.exploredNodes =
            static_cast<long long>(SCIPgetNNodes(scip));
        measured.lpIterations =
            static_cast<long long>(SCIPgetNLPIterations(scip));
        measured.solvingTimeSeconds = SCIPgetSolvingTime(scip);

        measured.hasSolution = SCIPgetBestSol(scip) != nullptr;
        measured.primalBound =
            measured.hasSolution ? SCIPgetPrimalbound(scip) : 0.0;
        measured.dualBound = SCIPgetDualbound(scip);

        result = measured;
        errorMessage.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        errorMessage = exception.what();
        return false;
    }
}
