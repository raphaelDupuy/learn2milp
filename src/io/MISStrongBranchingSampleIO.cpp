#include "io/MISStrongBranchingSampleIO.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <ostream>
#include <system_error>
#include <utility>

const char* const MIS_SB_NODES_SUFFIX = ".sb.nodes.csv";
const char* const MIS_SB_CANDIDATES_SUFFIX = ".sb.candidates.csv";
const char* const MIS_SB_METADATA_SUFFIX = ".sb.meta";
const char* const MIS_SB_FEATURES_SUFFIX = ".sb.features.csv";

bool isMISCollectionMetadataFile(const std::filesystem::path& path)
{
    const std::string name = path.filename().string();
    const std::string suffix = MIS_SB_METADATA_SUFFIX;

    return name.size() > suffix.size()
        && name.compare(name.size() - suffix.size(), suffix.size(), suffix)
            == 0;
}

std::vector<std::filesystem::path> listMISCollections(
    const std::filesystem::path& directory
)
{
    std::vector<std::filesystem::path> collections;

    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file() && isMISCollectionMetadataFile(entry.path()))
            collections.push_back(entry.path());
    }

    std::sort(collections.begin(), collections.end());
    return collections;
}

const char* const MIS_SB_NODE_CSV_HEADER =
    "sample_index,node_id,parent_node_id,parent_node_valid,node_depth,"
    "lp_objective,lp_objective_valid,primal_bound,primal_bound_valid,"
    "dual_bound,dual_bound_valid,open_nodes,processed_nodes,"
    "number_of_candidates,number_of_priority_candidates,"
    "expert_selected_vertex_id,expert_selection_valid,node_cutoff,"
    "number_of_vertices,vertex_states";

const char* const MIS_SB_CANDIDATE_CSV_HEADER =
    "sample_index,node_id,candidate_position,vertex_id,"
    "lp_value,lp_value_valid,fractionality,objective_coefficient,"
    "local_lower_bound,local_upper_bound,local_bounds_valid,"
    "reduced_cost,reduced_cost_valid,evaluation_status,evaluation_order,"
    "down_bound,down_gain,down_bound_valid,down_gain_valid,"
    "down_infeasible,down_conflict,down_skipped,"
    "up_bound,up_gain,up_bound_valid,up_gain_valid,"
    "up_infeasible,up_conflict,up_skipped,"
    "score,score_valid,expert_rank,expert_selected,"
    "gain_reference_lp_objective,gain_reference_lp_objective_valid,"
    "pseudocost_down,pseudocost_up,"
    "pseudocost_count_down,pseudocost_count_up,"
    "inferences_down,inferences_up,cutoffs_down,cutoffs_up,"
    "conflict_score";

char misVertexStateCode(MISVertexState state)
{
    switch (state)
    {
        case MIS_VERTEX_FIXED_ZERO:
            return 'Z';

        case MIS_VERTEX_FIXED_ONE:
            return 'O';

        case MIS_VERTEX_FREE_LP_ZERO:
            return 'z';

        case MIS_VERTEX_FREE_LP_ONE:
            return 'o';

        case MIS_VERTEX_FRACTIONAL:
            return 'f';

        case MIS_VERTEX_UNKNOWN:
            return 'u';
    }

    return '\0';
}

const char* misStrongBranchingEvaluationStatusName(
    MISStrongBranchingEvaluationStatus status
)
{
    switch (status)
    {
        case MIS_SB_NOT_EVALUATED:
            return "not_evaluated";

        case MIS_SB_EVALUATED:
            return "evaluated";

        case MIS_SB_REUSED:
            return "reused";

        case MIS_SB_LP_ERROR:
            return "lp_error";

        case MIS_SB_PARTIALLY_EVALUATED:
            return "partially_evaluated";
    }

    return nullptr;
}

namespace
{

// Appends the shortest decimal text that reads back to the same double.
bool appendDouble(std::string& row, double value)
{
    if (!std::isfinite(value))
    {
        return false;
    }

    // SCIP may return a negative zero; write it as a plain 0.
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
        return false;
    }

    row.append(buffer, conversion.ptr);
    return true;
}

// Appends every field of one direction, each preceded by a comma.
bool appendDirection(
    std::string& row,
    const MISStrongBranchingDirection& direction
)
{
    row += ',';

    if (!appendDouble(row, direction.bound))
    {
        return false;
    }

    row += ',';

    if (!appendDouble(row, direction.gain))
    {
        return false;
    }

    for (const int flag : {
            direction.bound_valid,
            direction.gain_valid,
            direction.infeasible,
            direction.conflict,
            direction.skipped})
    {
        row += ',';
        row += std::to_string(flag);
    }

    return true;
}

bool buildNodeRow(
    std::size_t sampleIndex,
    const MISStrongBranchingSample& sample,
    std::string& row,
    std::string& errorMessage
)
{
    if (sample.number_of_vertices < 0
        || (sample.number_of_vertices > 0
            && sample.vertex_states == nullptr))
    {
        errorMessage = "Sample has no valid vertex state array.";
        return false;
    }

    row = std::to_string(sampleIndex);
    row += ',' + std::to_string(sample.node_id);
    row += ',' + std::to_string(sample.parent_node_id);
    row += ',' + std::to_string(sample.parent_node_valid);
    row += ',' + std::to_string(sample.node_depth);

    const std::pair<double, int> bounds[] = {
        {sample.lp_objective, sample.lp_objective_valid},
        {sample.primal_bound, sample.primal_bound_valid},
        {sample.dual_bound, sample.dual_bound_valid}
    };

    for (const auto& [value, valid] : bounds)
    {
        row += ',';

        if (!appendDouble(row, value))
        {
            errorMessage = "Sample contains a non-finite objective value.";
            return false;
        }

        row += ',' + std::to_string(valid);
    }

    row += ',' + std::to_string(sample.open_nodes);
    row += ',' + std::to_string(sample.processed_nodes);
    row += ',' + std::to_string(sample.number_of_candidates);
    row += ',' + std::to_string(sample.number_of_priority_candidates);
    row += ',' + std::to_string(sample.expert_selected_vertex_id);
    row += ',' + std::to_string(sample.expert_selection_valid);
    row += ',' + std::to_string(sample.node_cutoff);
    row += ',' + std::to_string(sample.number_of_vertices);
    row += ',';

    for (int v = 0; v < sample.number_of_vertices; ++v)
    {
        const char code = misVertexStateCode(sample.vertex_states[v]);

        if (code == '\0')
        {
            errorMessage = "Sample contains an unknown vertex state.";
            return false;
        }

        row += code;
    }

    row += '\n';
    return true;
}

bool appendCandidateRow(
    std::size_t sampleIndex,
    const MISStrongBranchingSample& sample,
    const MISStrongBranchingCandidate& candidate,
    std::string& rows,
    std::string& errorMessage
)
{
    if (candidate.vertex_id < 0
        || candidate.vertex_id >= sample.number_of_vertices)
    {
        errorMessage = "Candidate vertex id is outside the vertex range.";
        return false;
    }

    const char* const statusName =
        misStrongBranchingEvaluationStatusName(candidate.evaluation_status);

    if (statusName == nullptr)
    {
        errorMessage = "Candidate has an unknown evaluation status.";
        return false;
    }

    bool finite = true;

    rows += std::to_string(sampleIndex);
    rows += ',' + std::to_string(sample.node_id);
    rows += ',' + std::to_string(candidate.candidate_position);
    rows += ',' + std::to_string(candidate.vertex_id);

    rows += ',';
    finite = appendDouble(rows, candidate.lp_value) && finite;
    rows += ',' + std::to_string(candidate.lp_value_valid);

    rows += ',';
    finite = appendDouble(rows, candidate.fractionality) && finite;
    rows += ',';
    finite = appendDouble(rows, candidate.objective_coefficient) && finite;
    rows += ',';
    finite = appendDouble(rows, candidate.local_lower_bound) && finite;
    rows += ',';
    finite = appendDouble(rows, candidate.local_upper_bound) && finite;
    rows += ',' + std::to_string(candidate.local_bounds_valid);

    rows += ',';
    finite = appendDouble(rows, candidate.reduced_cost) && finite;
    rows += ',' + std::to_string(candidate.reduced_cost_valid);

    rows += ',';
    rows += statusName;
    rows += ',' + std::to_string(candidate.evaluation_order);

    finite = appendDirection(rows, candidate.down) && finite;
    finite = appendDirection(rows, candidate.up) && finite;

    rows += ',';
    finite = appendDouble(rows, candidate.score) && finite;
    rows += ',' + std::to_string(candidate.score_valid);
    rows += ',' + std::to_string(candidate.expert_rank);
    rows += ',' + std::to_string(candidate.expert_selected);

    rows += ',';
    finite = appendDouble(rows, candidate.gain_reference_lp_objective)
        && finite;
    rows += ','
        + std::to_string(candidate.gain_reference_lp_objective_valid);

    for (const double value : {
            candidate.history.pseudocost_down,
            candidate.history.pseudocost_up,
            candidate.history.pseudocost_count_down,
            candidate.history.pseudocost_count_up,
            candidate.history.inferences_down,
            candidate.history.inferences_up,
            candidate.history.cutoffs_down,
            candidate.history.cutoffs_up,
            candidate.history.conflict_score})
    {
        rows += ',';
        finite = appendDouble(rows, value) && finite;
    }

    rows += '\n';

    if (!finite)
    {
        errorMessage = "Candidate contains a non-finite value.";
        return false;
    }

    return true;
}

} // namespace

MISStrongBranchingSampleWriter::MISStrongBranchingSampleWriter(
    std::ostream& nodeOutput,
    std::ostream& candidateOutput
)
    : nodeOutput_(nodeOutput)
    , candidateOutput_(candidateOutput)
{
}

bool MISStrongBranchingSampleWriter::writeHeaders(
    std::string& errorMessage
)
{
    if (headersWritten_)
    {
        errorMessage = "Sample table headers were already written.";
        return false;
    }

    nodeOutput_ << MIS_SB_NODE_CSV_HEADER << '\n';
    candidateOutput_ << MIS_SB_CANDIDATE_CSV_HEADER << '\n';

    if (!nodeOutput_ || !candidateOutput_)
    {
        errorMessage = "Failed to write sample table headers.";
        return false;
    }

    headersWritten_ = true;
    errorMessage.clear();
    return true;
}

bool MISStrongBranchingSampleWriter::writeSample(
    const MISStrongBranchingSample& sample,
    const MISStrongBranchingCandidate* candidates,
    std::size_t numberOfCandidates,
    std::string& errorMessage
)
{
    if (!headersWritten_)
    {
        errorMessage = "Sample table headers must be written first.";
        return false;
    }

    if (sample.number_of_candidates < 0
        || static_cast<std::size_t>(sample.number_of_candidates)
            != numberOfCandidates
        || (numberOfCandidates > 0 && candidates == nullptr))
    {
        errorMessage =
            "Sample candidate count does not match the candidate array.";
        return false;
    }

    // Build both tables' rows in memory first, so that a rejected sample
    // leaves no partial row behind.
    std::string nodeRow;

    if (!buildNodeRow(numberOfSamples_, sample, nodeRow, errorMessage))
    {
        return false;
    }

    std::string candidateRows;

    for (std::size_t c = 0; c < numberOfCandidates; ++c)
    {
        if (!appendCandidateRow(
                numberOfSamples_,
                sample,
                candidates[c],
                candidateRows,
                errorMessage))
        {
            return false;
        }
    }

    nodeOutput_ << nodeRow;
    candidateOutput_ << candidateRows;

    if (!nodeOutput_ || !candidateOutput_)
    {
        errorMessage = "Failed to write strong branching sample.";
        return false;
    }

    ++numberOfSamples_;
    numberOfCandidates_ += numberOfCandidates;

    errorMessage.clear();
    return true;
}

std::size_t MISStrongBranchingSampleWriter::numberOfSamples() const
{
    return numberOfSamples_;
}

std::size_t MISStrongBranchingSampleWriter::numberOfCandidates() const
{
    return numberOfCandidates_;
}
