#pragma once

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

#include "branching/MISStrongBranchingCollector.h"

/*
 * Writes strong branching samples as two CSV tables:
 *
 * - the node table has one row per sample (one branching decision);
 * - the candidate table has one row per branching candidate.
 *
 * Both tables are joined on sample_index, which numbers the samples of
 * one collection run in the order they were produced (0, 1, 2, ...).
 *
 * The columns mirror MISStrongBranchingSample and
 * MISStrongBranchingCandidate field by field (see
 * branching/MISStrongBranchingCollector.h for their meaning), except that:
 *
 * - down.* and up.* are flattened into down_* and up_* columns, and
 *   history.* into columns named after its fields;
 * - evaluation_status is written by name (see
 *   misStrongBranchingEvaluationStatusName());
 * - vertex_states is written as one string with one character per vertex,
 *   indexed by vertex id (see misVertexStateCode()).
 *
 * Floating-point values use the shortest decimal representation that
 * reads back to the same double.
 */
class MISStrongBranchingSampleWriter
{
public:
    MISStrongBranchingSampleWriter(
        std::ostream& nodeOutput,
        std::ostream& candidateOutput
    );

    // Writes the header row of both tables. Must be called once, first.
    bool writeHeaders(std::string& errorMessage);

    // Validates and writes one sample with its candidates. Nothing is
    // written when the sample is rejected.
    bool writeSample(
        const MISStrongBranchingSample& sample,
        const MISStrongBranchingCandidate* candidates,
        std::size_t numberOfCandidates,
        std::string& errorMessage
    );

    std::size_t numberOfSamples() const;
    std::size_t numberOfCandidates() const;

private:
    std::ostream& nodeOutput_;
    std::ostream& candidateOutput_;

    bool headersWritten_ = false;
    std::size_t numberOfSamples_ = 0;
    std::size_t numberOfCandidates_ = 0;
};

// Suffixes appended to an instance name to form the file names of one
// collection: the two sample tables, the collection metadata and the
// feature table derived from the samples.
extern const char* const MIS_SB_NODES_SUFFIX;
extern const char* const MIS_SB_CANDIDATES_SUFFIX;
extern const char* const MIS_SB_METADATA_SUFFIX;
extern const char* const MIS_SB_FEATURES_SUFFIX;

// True for a collection metadata file, NAME.sb.meta.
bool isMISCollectionMetadataFile(const std::filesystem::path& path);

// The collection metadata files of a directory (not of its
// subdirectories), in name order. Throws
// std::filesystem::filesystem_error if it cannot be read.
std::vector<std::filesystem::path> listMISCollections(
    const std::filesystem::path& directory
);

// Header rows, without the trailing line break.
extern const char* const MIS_SB_NODE_CSV_HEADER;
extern const char* const MIS_SB_CANDIDATE_CSV_HEADER;

/*
 * Character used for a vertex state in the vertex_states column:
 *
 *   Z  fixed to 0 by the local bounds     z  free, LP value 0
 *   O  fixed to 1 by the local bounds     o  free, LP value 1
 *   f  free, fractional LP value (a branching candidate)
 *   u  unknown (multi-aggregated by presolving)
 *
 * Returns '\0' for a value outside MISVertexState.
 */
char misVertexStateCode(MISVertexState state);

// Lowercase name of an evaluation status, or nullptr for an unknown value.
const char* misStrongBranchingEvaluationStatusName(
    MISStrongBranchingEvaluationStatus status
);
