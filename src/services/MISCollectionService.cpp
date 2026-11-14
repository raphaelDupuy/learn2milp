#include "services/MISCollectionService.hpp"

#include <charconv>
#include <climits>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <system_error>
#include <vector>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "branching/MISFullstrongBranchrule.h"
#include "io/MISInstanceIO.hpp"
#include "io/MISStrongBranchingSampleIO.hpp"
#include "services/MISSolverSettings.hpp"
#include "services/SCIPStatusName.hpp"

const char* const MIS_SB_NODES_SUFFIX = ".sb.nodes.csv";
const char* const MIS_SB_CANDIDATES_SUFFIX = ".sb.candidates.csv";
const char* const MIS_SB_METADATA_SUFFIX = ".sb.meta";

namespace
{

// Removes every tracked file unless the collection is committed.
class OutputCleanup
{
public:
    void track(const std::filesystem::path& path)
    {
        paths_.push_back(path);
    }

    void commit()
    {
        committed_ = true;
    }

    ~OutputCleanup()
    {
        if (committed_)
            return;

        for (const auto& path : paths_)
        {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
    }

private:
    std::vector<std::filesystem::path> paths_;
    bool committed_ = false;
};

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

// Bridges the C sample callback to the CSV writer. Errors cannot cross
// the C boundary as exceptions: they are stored and the solve is stopped.
struct CollectionState
{
    SCIP* scip = nullptr;
    MISStrongBranchingSampleWriter* writer = nullptr;

    bool failed = false;
    std::string errorMessage;
};

void recordSample(
    void* userdata,
    const MISStrongBranchingSample* sample,
    const MISStrongBranchingCandidate* candidates,
    size_t numberOfCandidates
)
{
    auto& state = *static_cast<CollectionState*>(userdata);

    if (state.failed)
        return;

    try
    {
        if (!state.writer->writeSample(
                *sample,
                candidates,
                numberOfCandidates,
                state.errorMessage))
        {
            state.failed = true;
        }
    }
    catch (const std::exception& exception)
    {
        state.failed = true;
        state.errorMessage = exception.what();
    }
    catch (...)
    {
        state.failed = true;
        state.errorMessage = "Unknown error while writing a sample.";
    }

    if (state.failed)
        SCIPinterruptSolve(state.scip);
}

// Shortest decimal text that reads back to the same double.
std::string formatDouble(double value)
{
    char buffer[64];

    const auto conversion = std::to_chars(
        buffer,
        buffer + sizeof(buffer),
        value
    );

    if (conversion.ec != std::errc{})
        throw std::runtime_error("Failed to format a number.");

    return std::string(buffer, conversion.ptr);
}

void require(SCIP_RETCODE retcode, const char* action)
{
    if (retcode != SCIP_OKAY)
    {
        throw std::runtime_error(
            std::string("SCIP failed to ") + action + "."
        );
    }
}

void closeOutput(
    std::ofstream& file,
    const std::filesystem::path& path
)
{
    file.close();

    if (!file)
    {
        throw std::runtime_error(
            "Failed to finish writing: " + path.string()
        );
    }
}

std::ofstream openOutput(const std::filesystem::path& path)
{
    std::ofstream file(path);

    if (!file)
    {
        throw std::runtime_error(
            "Cannot open output file: " + path.string()
        );
    }

    return file;
}

std::filesystem::path withSuffix(
    const std::filesystem::path& prefix,
    const std::string& suffix
)
{
    return std::filesystem::path(prefix.string() + suffix);
}

} // namespace

bool collectMISStrongBranchingSamples(
    const MISInstance& instance,
    const MISCollectionConfig& config,
    const std::filesystem::path& outputPrefix,
    MISCollectionResult& result,
    std::string& errorMessage
)
{
    try
    {
        // Validate the request before touching the file system.
        if (outputPrefix.empty())
        {
            throw std::invalid_argument(
                "Output prefix must not be empty."
            );
        }

        if (config.instanceName.empty()
            || config.instanceName.find('\n') != std::string::npos
            || config.instanceName.find('\r') != std::string::npos)
        {
            throw std::invalid_argument(
                "Instance name must be a non-empty single line."
            );
        }

        if (!std::isfinite(config.timeLimitSeconds)
            || config.timeLimitSeconds <= 0.0)
        {
            throw std::invalid_argument(
                "Time limit must be a positive number of seconds."
            );
        }

        if (config.nodeLimit < -1 || config.nodeLimit == 0)
        {
            throw std::invalid_argument(
                "Node limit must be positive, or -1 for no limit."
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

        MISCollectionResult collected;
        collected.nodesPath =
            withSuffix(outputPrefix, MIS_SB_NODES_SUFFIX);
        collected.candidatesPath =
            withSuffix(outputPrefix, MIS_SB_CANDIDATES_SUFFIX);
        collected.metadataPath =
            withSuffix(outputPrefix, MIS_SB_METADATA_SUFFIX);

        const auto temporaryNodes =
            withSuffix(collected.nodesPath, ".tmp");
        const auto temporaryCandidates =
            withSuffix(collected.candidatesPath, ".tmp");
        const auto temporaryMetadata =
            withSuffix(collected.metadataPath, ".tmp");

        for (const auto& path : {
                collected.nodesPath,
                collected.candidatesPath,
                collected.metadataPath,
                temporaryNodes,
                temporaryCandidates,
                temporaryMetadata})
        {
            if (std::filesystem::exists(path))
            {
                throw std::runtime_error(
                    "Output file already exists: " + path.string()
                );
            }
        }

        // From here on, every created file is removed on failure.
        OutputCleanup cleanup;

        cleanup.track(temporaryNodes);
        std::ofstream nodesFile = openOutput(temporaryNodes);

        cleanup.track(temporaryCandidates);
        std::ofstream candidatesFile = openOutput(temporaryCandidates);

        MISStrongBranchingSampleWriter writer(nodesFile, candidatesFile);

        if (!writer.writeHeaders(errorMessage))
            throw std::runtime_error(errorMessage);

        // Build the model.
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

        // Collection settings (see the header).
        require(
            SCIPsetIntParam(scip, "display/verblevel", 0),
            "disable console output"
        );
        require(
            applyBranchingStudySettings(scip),
            "apply the branching study settings"
        );
        require(
            SCIPsetRealParam(
                scip, "limits/time", config.timeLimitSeconds),
            "set the time limit"
        );
        require(
            SCIPsetLongintParam(scip, "limits/nodes", config.nodeLimit),
            "set the node limit"
        );
        require(
            SCIPsetIntParam(
                scip,
                "randomization/randomseedshift",
                static_cast<int>(config.scipSeed)),
            "set the random seed"
        );

        CollectionState state;
        state.scip = scip;
        state.writer = &writer;

        // An instance without vertices has nothing to branch on.
        if (!vars.empty())
        {
            require(
                SCIPincludeBranchruleMISFullstrong(
                    scip,
                    vars.data(),
                    static_cast<int>(vars.size()),
                    recordSample,
                    &state),
                "include the strong branching rule"
            );
        }

        const SCIP_RETCODE solveRetcode = SCIPsolve(scip);

        if (state.failed)
        {
            throw std::runtime_error(
                "Failed to record a sample: " + state.errorMessage
            );
        }

        require(solveRetcode, "solve the instance");

        closeOutput(nodesFile, temporaryNodes);
        closeOutput(candidatesFile, temporaryCandidates);

        // Gather the outcome of the run.
        const SCIP_STATUS status = SCIPgetStatus(scip);

        collected.status = getSCIPStatusName(status);
        collected.complete =
            status == SCIP_STATUS_OPTIMAL
            || status == SCIP_STATUS_INFEASIBLE;

        collected.exploredNodes =
            static_cast<long long>(SCIPgetNNodes(scip));
        collected.numberOfSamples = writer.numberOfSamples();
        collected.numberOfCandidates = writer.numberOfCandidates();
        collected.solvingTimeSeconds = SCIPgetSolvingTime(scip);

        collected.hasSolution = SCIPgetBestSol(scip) != nullptr;
        collected.primalBound =
            collected.hasSolution ? SCIPgetPrimalbound(scip) : 0.0;
        collected.dualBound = SCIPgetDualbound(scip);

        // The samples store objective values in SCIP's internal space.
        // This affine map brings them back to the original objective:
        //     original = intercept + slope * internal.
        const double objectiveIntercept = SCIPretransformObj(scip, 0.0);
        const double objectiveSlope =
            SCIPretransformObj(scip, 1.0) - objectiveIntercept;

        cleanup.track(temporaryMetadata);

        {
            std::ofstream metadataFile = openOutput(temporaryMetadata);

            metadataFile
                << "MIS_SB_DATASET 1\n"
                << "instance_name "
                << std::quoted(config.instanceName) << '\n'
                << "number_of_vertices " << instance.n << '\n'
                << "number_of_edges " << instance.edges.size() << '\n'
                << "scip_version \""
                << SCIPmajorVersion() << '.'
                << SCIPminorVersion() << '.'
                << SCIPtechVersion() << "\"\n"
                << "branching_rule \"misfullstrong\"\n"
                << "parameter.presolving/maxrestarts 0\n"
                << "parameter.separating/maxrounds 0\n"
                << "parameter.limits/time "
                << formatDouble(config.timeLimitSeconds) << '\n'
                << "parameter.limits/nodes " << config.nodeLimit << '\n'
                << "parameter.randomization/randomseedshift "
                << config.scipSeed << '\n'
                << "status " << std::quoted(collected.status) << '\n'
                << "complete " << (collected.complete ? 1 : 0) << '\n'
                << "explored_nodes " << collected.exploredNodes << '\n'
                << "number_of_samples "
                << collected.numberOfSamples << '\n'
                << "number_of_candidates "
                << collected.numberOfCandidates << '\n'
                << "has_solution "
                << (collected.hasSolution ? 1 : 0) << '\n';

            if (collected.hasSolution)
            {
                metadataFile
                    << "primal_bound "
                    << formatDouble(collected.primalBound) << '\n';
            }

            metadataFile
                << "dual_bound "
                << formatDouble(collected.dualBound) << '\n'
                << "original_objective_intercept "
                << formatDouble(objectiveIntercept) << '\n'
                << "original_objective_slope "
                << formatDouble(objectiveSlope) << '\n'
                << "solving_time_seconds "
                << formatDouble(collected.solvingTimeSeconds) << '\n';

            closeOutput(metadataFile, temporaryMetadata);
        }

        // Publish the tables first and the metadata last: a metadata
        // file never exists without its two complete tables.
        cleanup.track(collected.nodesPath);
        std::filesystem::rename(temporaryNodes, collected.nodesPath);

        cleanup.track(collected.candidatesPath);
        std::filesystem::rename(
            temporaryCandidates, collected.candidatesPath);

        cleanup.track(collected.metadataPath);
        std::filesystem::rename(
            temporaryMetadata, collected.metadataPath);

        cleanup.commit();

        result = std::move(collected);
        errorMessage.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        errorMessage = exception.what();
        return false;
    }
}
