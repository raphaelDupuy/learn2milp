#include "services/MISFeaturizationService.hpp"

#include <charconv>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <vector>

#include "io/CsvReader.hpp"
#include "io/MISInstanceIO.hpp"
#include "io/MISStrongBranchingSampleIO.hpp"
#include "learning/MISAdjacency.hpp"
#include "learning/MISBranchingFeatures.h"

namespace
{

// Removes the output file unless the featurization is committed.
class OutputCleanup
{
public:
    explicit OutputCleanup(std::filesystem::path path)
        : path_(std::move(path))
    {
    }

    void commit()
    {
        committed_ = true;
    }

    ~OutputCleanup()
    {
        if (!committed_)
        {
            std::error_code ignored;
            std::filesystem::remove(path_, ignored);
        }
    }

private:
    std::filesystem::path path_;
    bool committed_ = false;
};

// Shortest decimal text that reads back to the same double.
void appendDouble(std::string& row, double value)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error("A feature value is not finite.");
    }

    if (value == 0.0)
    {
        value = 0.0;
    }

    char buffer[64];

    const auto conversion = std::to_chars(
        buffer,
        buffer + sizeof(buffer),
        value
    );

    if (conversion.ec != std::errc{})
    {
        throw std::runtime_error("Failed to format a number.");
    }

    row.append(buffer, conversion.ptr);
}

// Columns of a table, resolved once from its header.
class Columns
{
public:
    Columns(const CsvReader& reader, const std::string& tableName)
        : reader_(reader)
        , tableName_(tableName)
    {
    }

    int require(const std::string& name) const
    {
        const int index = reader_.column(name);

        if (index < 0)
        {
            throw std::runtime_error(
                "The " + tableName_ + " table has no column '" + name + "'."
            );
        }

        return index;
    }

private:
    const CsvReader& reader_;
    const std::string& tableName_;
};

long long integerField(
    const std::vector<std::string>& fields,
    int column,
    const char* name
)
{
    long long value = 0;

    if (!parseCsvInteger(fields[column], value))
    {
        throw std::runtime_error(
            std::string("Invalid integer in column '") + name + "': "
            + fields[column]
        );
    }

    return value;
}

double realField(
    const std::vector<std::string>& fields,
    int column,
    const char* name
)
{
    double value = 0.0;

    if (!parseCsvReal(fields[column], value))
    {
        throw std::runtime_error(
            std::string("Invalid number in column '") + name + "': "
            + fields[column]
        );
    }

    return value;
}

MISVertexState stateFromCode(char code)
{
    for (const MISVertexState state : {
            MIS_VERTEX_FIXED_ZERO,
            MIS_VERTEX_FIXED_ONE,
            MIS_VERTEX_FREE_LP_ZERO,
            MIS_VERTEX_FREE_LP_ONE,
            MIS_VERTEX_FRACTIONAL,
            MIS_VERTEX_UNKNOWN})
    {
        if (misVertexStateCode(state) == code)
        {
            return state;
        }
    }

    throw std::runtime_error(
        std::string("Unknown vertex state code: ") + code
    );
}

std::ifstream openInput(const std::filesystem::path& path)
{
    std::ifstream file(path);

    if (!file)
    {
        throw std::runtime_error(
            "Cannot open input file: " + path.string()
        );
    }

    return file;
}

} // namespace

std::string misFeatureTableHeader()
{
    std::string header =
        "sample_index,candidate_position,vertex_id,"
        "node_cutoff,expert_selection_valid,evaluated,"
        "score,expert_rank,expert_selected";

    for (int f = 0; f < MIS_BRANCHING_FEATURE_COUNT; ++f)
    {
        header += ',';
        header += MISbranchingFeatureName(f);
    }

    return header;
}

bool featurizeMISStrongBranchingSamples(
    const MISInstance& instance,
    const std::filesystem::path& nodesPath,
    const std::filesystem::path& candidatesPath,
    const std::filesystem::path& outputPath,
    MISFeaturizationResult& result,
    std::string& errorMessage
)
{
    try
    {
        if (outputPath.empty())
        {
            throw std::invalid_argument("Output path must not be empty.");
        }

        std::string instanceError;

        if (!validateMISInstance(instance, instanceError))
        {
            throw std::invalid_argument(
                "Invalid MIS instance: " + instanceError
            );
        }

        const std::filesystem::path temporaryPath(
            outputPath.string() + ".tmp"
        );

        if (std::filesystem::exists(outputPath)
            || std::filesystem::exists(temporaryPath))
        {
            throw std::runtime_error(
                "Output file already exists: " + outputPath.string()
            );
        }

        std::ifstream nodesFile = openInput(nodesPath);
        std::ifstream candidatesFile = openInput(candidatesPath);

        CsvReader nodes(nodesFile);
        CsvReader candidates(candidatesFile);

        std::string readError;

        if (!nodes.readHeader(readError)
            || !candidates.readHeader(readError))
        {
            throw std::runtime_error(readError);
        }

        const std::string nodeTable = "node";
        const std::string candidateTable = "candidate";
        const Columns nodeColumns(nodes, nodeTable);
        const Columns candidateColumns(candidates, candidateTable);

        const int nSampleIndex = nodeColumns.require("sample_index");
        const int nDepth = nodeColumns.require("node_depth");
        const int nLpObjective = nodeColumns.require("lp_objective");
        const int nPrimalBound = nodeColumns.require("primal_bound");
        const int nPrimalBoundValid =
            nodeColumns.require("primal_bound_valid");
        const int nCandidates = nodeColumns.require("number_of_candidates");
        const int nSelectionValid =
            nodeColumns.require("expert_selection_valid");
        const int nCutoff = nodeColumns.require("node_cutoff");
        const int nVertices = nodeColumns.require("number_of_vertices");
        const int nStates = nodeColumns.require("vertex_states");

        const int cSampleIndex = candidateColumns.require("sample_index");
        const int cPosition = candidateColumns.require("candidate_position");
        const int cVertex = candidateColumns.require("vertex_id");
        const int cLpValue = candidateColumns.require("lp_value");
        const int cStatus = candidateColumns.require("evaluation_status");
        const int cScore = candidateColumns.require("score");
        const int cRank = candidateColumns.require("expert_rank");
        const int cSelected = candidateColumns.require("expert_selected");

        // Columns of MISCandidateHistory, in the order of its fields.
        const char* const historyNames[] = {
            "pseudocost_down", "pseudocost_up",
            "pseudocost_count_down", "pseudocost_count_up",
            "inferences_down", "inferences_up",
            "cutoffs_down", "cutoffs_up", "conflict_score"};
        int cHistory[9];

        for (int h = 0; h < 9; ++h)
            cHistory[h] = candidateColumns.require(historyNames[h]);

        const std::string evaluatedName =
            misStrongBranchingEvaluationStatusName(MIS_SB_EVALUATED);

        const MISAdjacency adjacency = buildMISAdjacency(instance);
        const MISFeatureGraph graph = adjacency.view();

        // From here on, the output file is removed on failure.
        OutputCleanup cleanup(temporaryPath);
        std::ofstream output(temporaryPath);

        if (!output)
        {
            throw std::runtime_error(
                "Cannot open output file: " + temporaryPath.string()
            );
        }

        output << misFeatureTableHeader() << '\n';

        MISFeaturizationResult counted;

        std::vector<std::string> nodeFields;
        std::vector<std::string> candidateFields;

        std::vector<MISVertexState> states(instance.n);
        std::vector<double> work(MISbranchingFeatureWorkSize(instance.n));
        std::vector<int> vertices;
        std::vector<double> lpValues;
        std::vector<MISCandidateHistory> histories;
        std::vector<double> features;
        std::vector<std::string> labels;
        std::string rows;

        while (nodes.readRow(nodeFields, readError))
        {
            const long long sampleIndex =
                integerField(nodeFields, nSampleIndex, "sample_index");
            const std::string where =
                "Sample " + std::to_string(sampleIndex) + ": ";

            if (sampleIndex
                != static_cast<long long>(counted.numberOfSamples))
            {
                throw std::runtime_error(
                    where + "sample_index is not sequential."
                );
            }

            const std::string& stateCodes = nodeFields[nStates];

            if (integerField(nodeFields, nVertices, "number_of_vertices")
                    != instance.n
                || stateCodes.size()
                    != static_cast<std::size_t>(instance.n))
            {
                throw std::runtime_error(
                    where + "the sample does not belong to this instance."
                );
            }

            for (int v = 0; v < instance.n; ++v)
            {
                states[v] = stateFromCode(stateCodes[v]);
            }

            const long long numberOfCandidates = integerField(
                nodeFields, nCandidates, "number_of_candidates");

            if (numberOfCandidates < 0 || numberOfCandidates > instance.n)
            {
                throw std::runtime_error(
                    where + "invalid number of candidates."
                );
            }

            const long long nodeCutoff =
                integerField(nodeFields, nCutoff, "node_cutoff");
            const long long selectionValid = integerField(
                nodeFields, nSelectionValid, "expert_selection_valid");

            // Candidates of this sample, which are consecutive rows.
            vertices.clear();
            lpValues.clear();
            histories.clear();
            labels.clear();

            bool allEvaluated = true;

            for (long long c = 0; c < numberOfCandidates; ++c)
            {
                if (!candidates.readRow(candidateFields, readError))
                {
                    throw std::runtime_error(
                        candidates.atEnd()
                        ? where + "candidate rows are missing."
                        : readError
                    );
                }

                if (integerField(
                        candidateFields, cSampleIndex, "sample_index")
                        != sampleIndex
                    || integerField(
                        candidateFields, cPosition, "candidate_position")
                        != c)
                {
                    throw std::runtime_error(
                        where + "candidate rows are out of order."
                    );
                }

                const long long vertex =
                    integerField(candidateFields, cVertex, "vertex_id");

                if (vertex < 0 || vertex >= instance.n)
                {
                    throw std::runtime_error(
                        where + "candidate vertex is out of range."
                    );
                }

                const bool evaluated =
                    candidateFields[cStatus] == evaluatedName;

                allEvaluated = allEvaluated && evaluated;

                vertices.push_back(static_cast<int>(vertex));
                lpValues.push_back(
                    realField(candidateFields, cLpValue, "lp_value"));

                double history[9];

                for (int h = 0; h < 9; ++h)
                {
                    history[h] = realField(
                        candidateFields, cHistory[h], historyNames[h]);
                }

                histories.push_back(MISCandidateHistory{
                    history[0], history[1], history[2], history[3],
                    history[4], history[5], history[6], history[7],
                    history[8]});

                // Identification and expert answer, copied as written.
                labels.push_back(
                    std::to_string(sampleIndex) + ','
                    + std::to_string(c) + ','
                    + std::to_string(vertex) + ','
                    + std::to_string(nodeCutoff) + ','
                    + std::to_string(selectionValid) + ','
                    + (evaluated ? '1' : '0') + ','
                    + candidateFields[cScore] + ','
                    + candidateFields[cRank] + ','
                    + candidateFields[cSelected]);
            }

            features.assign(
                static_cast<std::size_t>(numberOfCandidates)
                    * MIS_BRANCHING_FEATURE_COUNT,
                0.0
            );

            MISBranchingNode node;
            node.vertex_states = states.data();
            node.node_depth = static_cast<int>(
                integerField(nodeFields, nDepth, "node_depth"));
            node.number_of_candidates =
                static_cast<int>(numberOfCandidates);
            node.candidate_vertices = vertices.data();
            node.candidate_lp_values = lpValues.data();
            node.candidate_histories = histories.data();
            node.objective_gap = MISbranchingObjectiveGap(
                realField(nodeFields, nLpObjective, "lp_objective"),
                realField(nodeFields, nPrimalBound, "primal_bound"),
                integerField(
                    nodeFields, nPrimalBoundValid, "primal_bound_valid")
                    != 0);

            if (MIScomputeBranchingFeatures(
                    &graph, &node, work.data(), features.data()) != 0)
            {
                throw std::runtime_error(
                    where + "vertex states and candidates are inconsistent."
                );
            }

            rows.clear();

            for (long long c = 0; c < numberOfCandidates; ++c)
            {
                rows += labels[c];

                for (int f = 0; f < MIS_BRANCHING_FEATURE_COUNT; ++f)
                {
                    rows += ',';
                    appendDouble(
                        rows,
                        features[c * MIS_BRANCHING_FEATURE_COUNT + f]);
                }

                rows += '\n';
            }

            output << rows;

            ++counted.numberOfSamples;
            counted.numberOfCandidates +=
                static_cast<std::size_t>(numberOfCandidates);

            if (nodeCutoff == 0 && selectionValid == 1 && allEvaluated)
            {
                ++counted.numberOfUsableSamples;
            }
        }

        if (!nodes.atEnd())
        {
            throw std::runtime_error(readError);
        }

        // Every candidate row must belong to a sample.
        if (candidates.readRow(candidateFields, readError)
            || !candidates.atEnd())
        {
            throw std::runtime_error(
                "The candidate table has rows that belong to no sample."
            );
        }

        output.close();

        if (!output)
        {
            throw std::runtime_error(
                "Failed to finish writing: " + temporaryPath.string()
            );
        }

        std::filesystem::rename(temporaryPath, outputPath);
        cleanup.commit();

        result = counted;
        errorMessage.clear();
        return true;
    }
    catch (const std::exception& exception)
    {
        errorMessage = exception.what();
        return false;
    }
}
