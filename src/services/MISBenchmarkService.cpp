#include "services/MISBenchmarkService.hpp"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cmath>
#include <exception>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "branching/MISFullstrongBranchrule.h"
#include "branching/MISLearnedBranchrule.h"
#include "cuts/MISLookaheadCutsel.h"
#include "cuts/MISNodeLog.h"
#include "cuts/MISSeparationGate.h"
#include "io/MISBranchingModelIO.hpp"
#include "io/MISInstanceIO.hpp"
#include "learning/MISAdjacency.hpp"
#include "services/MISSolverSettings.hpp"
#include "services/SCIPStatusName.hpp"
#include "util/Text.hpp"

namespace
{

const char* const MIS_FULLSTRONG_RULE = "misfullstrong";

// The expert without its strong branching pruning: the tree a perfect
// imitation of its choices would explore.
const char* const MIS_FULLSTRONG_NOCUTOFF_RULE = "misfullstrong_nocutoff";
const char* const MIS_LEARNED_RULE = "mislearned";
const char* const MIS_LOOKAHEAD_CUT_SELECTOR = "lookahead";

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

/*
 * Creates a silent solver with SCIP's default plugins and the separation
 * gate (off unless a parameter file sets separating/misgate/freq). The
 * validation of a parameter file and the runs use it alike, so that the
 * file names the same parameters in both.
 */
void createSolver(SCIPHandle& handle)
{
    require(SCIPcreate(&handle.scip), "create a solver instance");
    require(
        SCIPincludeDefaultPlugins(handle.scip),
        "include the default plugins"
    );
    require(
        SCIPincludeSepaMISSeparationGate(handle.scip),
        "include the separation gate"
    );
    require(
        SCIPsetIntParam(handle.scip, "display/verblevel", 0),
        "disable console output"
    );
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

/*
 * Collects SCIP's error messages instead of printing them, until
 * destroyed, so that they can be reported with the error they explain.
 */
class SCIPErrorCapture
{
public:
    SCIPErrorCapture() { SCIPmessageSetErrorPrinting(collect, &text_); }
    ~SCIPErrorCapture() { SCIPmessageSetErrorPrintingDefault(); }

    SCIPErrorCapture(const SCIPErrorCapture&) = delete;
    SCIPErrorCapture& operator=(const SCIPErrorCapture&) = delete;

    // The first message, without SCIP's "[file:line] ERROR: " prefix;
    // empty if there was none.
    std::string first() const
    {
        std::string line = text_.substr(0, text_.find('\n'));
        const std::size_t prefix = line.find("ERROR: ");

        if (prefix != std::string::npos)
            line.erase(0, prefix + 7);

        return trimWhitespace(line);
    }

private:
    static void collect(void* data, FILE*, const char* message)
    {
        static_cast<std::string*>(data)->append(message);
    }

    std::string text_;
};

/*
 * True when the parameter holds the value written in a settings file (in
 * SCIP's format: TRUE or FALSE, a number, one character, or a string in
 * double quotes, possibly followed by "fix").
 */
bool holdsValue(SCIP_PARAM* param, std::string value)
{
    value = trimWhitespace(value);

    if (value.size() > 4 && value.compare(value.size() - 3, 3, "fix") == 0
        && (value[value.size() - 4] == ' ' || value[value.size() - 4] == '\t'))
    {
        value = trimWhitespace(value.substr(0, value.size() - 3));
    }

    try
    {
        std::size_t used = 0;

        switch (SCIPparamGetType(param))
        {
            case SCIP_PARAMTYPE_BOOL:
            {
                std::string upper;
                for (const char c : value)
                    upper += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));

                return (upper == "TRUE" || upper == "FALSE")
                    && (upper == "TRUE") == (SCIPparamGetBool(param) == TRUE);
            }
            case SCIP_PARAMTYPE_INT:
            {
                const long long parsed = std::stoll(value, &used);
                return used == value.size() && parsed == SCIPparamGetInt(param);
            }
            case SCIP_PARAMTYPE_LONGINT:
            {
                const long long parsed = std::stoll(value, &used);
                return used == value.size() && parsed == SCIPparamGetLongint(param);
            }
            case SCIP_PARAMTYPE_REAL:
            {
                const double parsed = std::stod(value, &used);
                const double held = SCIPparamGetReal(param);
                return used == value.size()
                    && std::fabs(parsed - held)
                        <= 1e-9 * std::max(1.0, std::fabs(parsed));
            }
            case SCIP_PARAMTYPE_CHAR:
                return value.size() == 1 && value[0] == SCIPparamGetChar(param);
            case SCIP_PARAMTYPE_STRING:
                return value.size() >= 2 && value.front() == '"'
                    && value.back() == '"'
                    && value.substr(1, value.size() - 2)
                        == SCIPparamGetString(param);
        }
    }
    catch (const std::exception&)
    {
        return false;
    }

    return false;
}

/*
 * Applies a SCIP settings file. SCIP itself only warns about a parameter
 * it does not know, which would silently ignore a misspelled name: every
 * name is checked first. And when a value is malformed (a string without
 * its double quotes), SCIP stops reading at that line, ignores the rest of
 * the file and still reports success: every value is checked after.
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

    struct Entry
    {
        int line;
        std::string name;
        std::string value;
    };

    std::vector<Entry> entries;
    std::string line;
    int lineNumber = 0;

    while (std::getline(input, line))
    {
        ++lineNumber;
        line = trimWhitespace(line.substr(0, line.find('#')));

        if (line.empty())
            continue;

        const auto equals = line.find('=');
        const std::string name =
            trimWhitespace(
                equals == std::string::npos ? line : line.substr(0, equals));

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

        entries.push_back({lineNumber, name, line.substr(equals + 1)});
    }

    SCIPErrorCapture errors;

    if (SCIPreadParams(scip, file.string().c_str()) != SCIP_OKAY)
    {
        throw std::invalid_argument(
            "SCIP rejected a value of the parameter file " + file.string()
            + ": " + errors.first()
        );
    }

    // A parameter set on several lines holds the value of the last one;
    // the first line not applied is reported, the malformed one if any.
    std::map<std::string, int> lastLine;

    for (const Entry& entry : entries)
        lastLine[entry.name] = entry.line;

    for (const Entry& entry : entries)
    {
        if (lastLine[entry.name] != entry.line)
            continue;

        if (!holdsValue(SCIPgetParam(scip, entry.name.c_str()), entry.value))
        {
            throw std::invalid_argument(
                "SCIP did not apply line " + std::to_string(entry.line)
                + " of the parameter file " + file.string() + " ("
                + entry.name + " = " + trimWhitespace(entry.value) + "): "
                + (errors.first().empty()
                    ? "check its value and format (strings in double quotes)"
                    : errors.first())
            );
        }
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

// SCIP's statistics of the solve, as JSON (SCIPprintStatisticsJson).
void writeStatistics(SCIP* scip, const std::filesystem::path& path)
{
    std::FILE* const file = std::fopen(path.string().c_str(), "w");

    if (file == nullptr)
    {
        throw std::runtime_error(
            "Cannot create the statistics file " + path.string() + ".");
    }

    const SCIP_RETCODE retcode = SCIPprintStatisticsJson(scip, file);
    const bool closed = std::fclose(file) == 0;

    require(retcode, "write the statistics");

    if (!closed)
    {
        throw std::runtime_error(
            "Cannot write the statistics file " + path.string() + ".");
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
    const std::filesystem::path& parameterFile,
    const std::string& cutSelector
)
{
    std::string label = misBenchmarkSettingsName(settings);

    if (!parameterFile.empty())
        label += "+" + parameterFile.stem().string();

    if (!cutSelector.empty())
        label += "+" + cutSelector;

    for (char& c : label)
    {
        if (c == ',')
            c = '_';
    }

    return label;
}

bool isMISBenchmarkCutSelectorAvailable(const std::string& cutSelector)
{
    return cutSelector.empty() || cutSelector == MIS_LOOKAHEAD_CUT_SELECTOR;
}

bool validateMISBenchmarkParameters(
    const std::filesystem::path& parameterFile,
    std::string& errorMessage
)
{
    try
    {
        SCIPHandle handle;
        createSolver(handle);
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

        if (!isMISBenchmarkCutSelectorAvailable(config.cutSelector))
        {
            throw std::invalid_argument(
                "Unknown cut selector: " + config.cutSelector
            );
        }

        const bool lookahead = config.cutSelector == MIS_LOOKAHEAD_CUT_SELECTOR;

        if (!config.nodeLogPath.empty()
            && std::filesystem::exists(config.nodeLogPath))
        {
            throw std::invalid_argument(
                "Node log already exists: " + config.nodeLogPath.string()
            );
        }

        if (!config.statisticsPath.empty()
            && std::filesystem::exists(config.statisticsPath))
        {
            throw std::invalid_argument(
                "Statistics file already exists: "
                + config.statisticsPath.string()
            );
        }

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
        createSolver(handle);

        SCIP* const scip = handle.scip;

        if (lookahead)
        {
            require(
                SCIPincludeCutselMISLookahead(scip),
                "include the lookahead cut selector"
            );
        }

        if (!config.nodeLogPath.empty())
        {
            const std::string action =
                "create the node log " + config.nodeLogPath.string();

            require(
                SCIPincludeEventHdlrMISNodeLog(
                    scip, config.nodeLogPath.string().c_str()),
                action.c_str()
            );
        }

        std::vector<SCIP_VAR*> vars;
        require(
            buildGraphProblemModel(scip, config.problem, instance, vars),
            "build the model"
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

        if (!config.startSolution.empty())
        {
            SCIP_SOL* start = nullptr;
            SCIP_Bool stored = FALSE;
            require(
                SCIPcreateOrigSol(scip, &start, nullptr),
                "create the start solution"
            );

            for (const int v : config.startSolution)
            {
                if (v < 0 || v >= static_cast<int>(vars.size()))
                {
                    SCIPfreeSol(scip, &start);
                    throw std::invalid_argument(
                        "Start solution: no vertex " + std::to_string(v));
                }

                require(
                    SCIPsetSolVal(scip, start, vars[v], 1.0),
                    "set a value of the start solution"
                );
            }

            require(
                SCIPaddSolFree(scip, &start, &stored),
                "add the start solution"
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
        measured.rootFirstLPBound = SCIPgetFirstLPDualboundRoot(scip);
        measured.rootDualBound = SCIPgetDualboundRoot(scip);

        // SCIP reports an infinite root bound when the root closes the gap
        // on its own: the bound at the end of the root is then the optimum.
        if (SCIPisInfinity(scip, std::fabs(measured.rootDualBound))
            && measured.solved)
        {
            measured.rootDualBound = measured.dualBound;
        }
        measured.cutsApplied =
            static_cast<long long>(SCIPgetNCutsApplied(scip));

        if (lookahead)
        {
            MISLookaheadStatistics statistics;
            require(
                SCIPgetMISLookaheadStatistics(scip, &statistics),
                "read the lookahead statistics"
            );

            measured.lookaheadScored = statistics.scored;
            measured.lookaheadImproving = statistics.improving;
        }

        MISSeparationGateStatistics gate;
        require(
            SCIPgetMISSeparationGateStatistics(scip, &gate),
            "read the separation gate statistics"
        );
        measured.gateSeparated = gate.separated;
        measured.gateSkipped = gate.skipped;

        // The rule is not included on a graph without vertices.
        if (learned && SCIPfindBranchrule(scip, MIS_LEARNED_RULE) != nullptr)
        {
            SCIP_Real featureSeconds = 0.0;
            SCIP_Real modelSeconds = 0.0;
            require(
                SCIPgetMISLearnedTimes(scip, &featureSeconds, &modelSeconds),
                "read the time of the learned rule"
            );
            measured.learnedFeatureSeconds = featureSeconds;
            measured.learnedModelSeconds = modelSeconds;
        }

        if (!config.statisticsPath.empty())
            writeStatistics(scip, config.statisticsPath);

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
