#include "services/MISBenchmarkService.hpp"

#include <climits>
#include <cmath>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <string>
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

// The expert without its strong branching pruning: the tree a perfect
// imitation of its choices would explore.
const char* const MIS_FULLSTRONG_NOCUTOFF_RULE = "misfullstrong_nocutoff";
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

std::string trim(const std::string& text)
{
    const auto first = text.find_first_not_of(" \t\r");

    if (first == std::string::npos)
        return "";

    return text.substr(first, text.find_last_not_of(" \t\r") - first + 1);
}

/*
 * Applies a SCIP settings file. SCIP itself only warns about a parameter
 * it does not know, which would silently ignore a misspelled name: every
 * name is checked first.
 */
void applyParameterFile(SCIP* scip, const std::filesystem::path& file)
{
    std::ifstream input(file);

    if (!input)
    {
        throw std::invalid_argument(
            "Unable to open parameter file: " + file.string()
        );
    }

    std::string line;
    int lineNumber = 0;

    while (std::getline(input, line))
    {
        ++lineNumber;
        line = trim(line.substr(0, line.find('#')));

        if (line.empty())
            continue;

        const auto equals = line.find('=');
        const std::string name =
            trim(equals == std::string::npos ? line : line.substr(0, equals));

        if (equals == std::string::npos || name.empty())
        {
            throw std::invalid_argument(
                "Invalid line " + std::to_string(lineNumber)
                + " in parameter file " + file.string()
                + " (expected: name = value)"
            );
        }

        if (SCIPgetParam(scip, name.c_str()) == nullptr)
        {
            throw std::invalid_argument(
                "Unknown SCIP parameter in " + file.string() + ": " + name
            );
        }
    }

    if (SCIPreadParams(scip, file.string().c_str()) != SCIP_OKAY)
    {
        throw std::invalid_argument(
            "SCIP rejected a value of the parameter file: " + file.string()
        );
    }
}

// The mislearned rule, and only it, takes a model file.
void requireModelOfRule(
    const std::string& branchingRule,
    const std::filesystem::path& learnedModelPath
)
{
    const bool learned = branchingRule == MIS_LEARNED_RULE;

    if (learned == learnedModelPath.empty())
    {
        throw std::invalid_argument(
            learned
            ? "The mislearned rule needs a model file."
            : "A model file is only used by the mislearned rule."
        );
    }
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

std::string misBenchmarkSettingsLabel(
    MISBenchmarkSettings settings,
    const std::filesystem::path& parameterFile
)
{
    std::string label = misBenchmarkSettingsName(settings);

    if (!parameterFile.empty())
        label += "+" + parameterFile.stem().string();

    for (char& c : label)
    {
        if (c == ',')
            c = '_';
    }

    return label;
}

bool validateMISBenchmarkParameters(
    const std::filesystem::path& parameterFile,
    std::string& errorMessage
)
{
    try
    {
        SCIPHandle handle;
        require(SCIPcreate(&handle.scip), "create a solver instance");
        require(
            SCIPincludeDefaultPlugins(handle.scip),
            "include the default plugins"
        );
        require(
            SCIPsetIntParam(handle.scip, "display/verblevel", 0),
            "disable console output"
        );

        applyParameterFile(handle.scip, parameterFile);

        errorMessage.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        errorMessage = exception.what();
        return false;
    }
}

double misShiftedGeometricMean(const std::vector<double>& values, double shift)
{
    if (values.empty())
        return 0.0;

    double logSum = 0.0;

    for (const double value : values)
        logSum += std::log(value + shift);

    return std::exp(logSum / static_cast<double>(values.size())) - shift;
}

bool isMISBenchmarkRuleAvailable(const std::string& branchingRule)
{
    if (branchingRule == MIS_FULLSTRONG_RULE
        || branchingRule == MIS_FULLSTRONG_NOCUTOFF_RULE
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

bool validateMISBenchmarkRule(
    const std::string& branchingRule,
    const std::filesystem::path& learnedModelPath,
    std::string& errorMessage
)
{
    try
    {
        requireModelOfRule(branchingRule, learnedModelPath);

        if (!isMISBenchmarkRuleAvailable(branchingRule))
        {
            throw std::invalid_argument(
                "Unknown branching rule: " + branchingRule
            );
        }

        if (!learnedModelPath.empty())
        {
            LoadedMISBranchingModel model;
            std::string modelError;

            if (!loadMISBranchingModel(learnedModelPath, model, modelError))
                throw std::invalid_argument(modelError);
        }

        errorMessage.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        errorMessage = exception.what();
        return false;
    }
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

        if (config.learnedStrongBranchingCandidates < 0
            || (config.learnedStrongBranchingCandidates > 0 && !learned))
        {
            throw std::invalid_argument(
                "Strong branching candidates are only used by the mislearned "
                "rule, and cannot be negative.");
        }
        const bool expert = config.branchingRule == MIS_FULLSTRONG_RULE
            || config.branchingRule == MIS_FULLSTRONG_NOCUTOFF_RULE;

        requireModelOfRule(config.branchingRule, config.learnedModelPath);

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

        if (!config.parameterFile.empty())
            applyParameterFile(scip, config.parameterFile);

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
            if (expert && !vars.empty())
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

                if (config.branchingRule == MIS_FULLSTRONG_NOCUTOFF_RULE)
                {
                    require(
                        SCIPsetMISFullstrongPruning(scip, FALSE),
                        "disable strong branching pruning"
                    );
                }
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

                if (config.learnedStrongBranchingCandidates > 0)
                {
                    require(
                        SCIPsetMISLearnedStrongBranching(
                            scip, config.learnedStrongBranchingCandidates),
                        "set the strong branching candidates"
                    );
                }
            }
            else if (!expert
                && !learned
                && SCIPfindBranchrule(
                    scip, config.branchingRule.c_str()) == nullptr)
            {
                throw std::invalid_argument(
                    "Unknown branching rule: " + config.branchingRule
                );
            }

            SCIP_BRANCHRULE* const rule = SCIPfindBranchrule(
                scip,
                expert ? MIS_FULLSTRONG_RULE : config.branchingRule.c_str());

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

        SCIP_SOL* const solution = SCIPgetBestSol(scip);

        measured.hasSolution = solution != nullptr;
        measured.primalBound =
            measured.hasSolution ? SCIPgetPrimalbound(scip) : 0.0;
        measured.dualBound = SCIPgetDualbound(scip);

        if (measured.hasSolution)
        {
            for (std::size_t v = 0; v < vars.size(); ++v)
            {
                if (SCIPgetSolVal(scip, solution, vars[v]) > 0.5)
                    measured.selectedVertices.push_back(static_cast<int>(v));
            }
        }

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

void MISRuleSummary::add(const MISBenchmarkResult& result)
{
    ++runs;
    solved += result.solved ? 1 : 0;
    nodes.push_back(static_cast<double>(result.exploredNodes));
    times.push_back(result.solvingTimeSeconds);
}

double MISRuleSummary::meanNodes() const
{
    return misShiftedGeometricMean(nodes, MIS_NODES_SHIFT);
}

double MISRuleSummary::meanTimeSeconds() const
{
    return misShiftedGeometricMean(times, MIS_TIME_SHIFT_SECONDS);
}
