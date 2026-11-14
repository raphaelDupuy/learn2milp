#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include "problems/MIS.hpp"

/*
 * Turns the strong branching samples collected on one instance into a
 * training table: one row per branching candidate, with the features of
 * learning/MISBranchingFeatures.h and the expert's answer.
 *
 * The features are computed by the function that the learned branching
 * rule calls inside SCIP, from the same inputs: training and inference
 * cannot disagree on what a feature means.
 *
 * Columns of the table:
 *
 *   sample_index, candidate_position, vertex_id   identify the candidate
 *                                                 (as in the sample tables)
 *   node_cutoff, expert_selection_valid           copied from the node
 *   evaluated                                     1 if strong branching
 *                                                 evaluated the candidate
 *   score, expert_rank, expert_selected           the expert's answer
 *   then one column per feature, named after the feature
 *
 * Every sample is kept. For imitation learning, use the samples with
 * node_cutoff = 0, expert_selection_valid = 1 and all candidates
 * evaluated: in the others the expert's choice is arbitrary or missing.
 */

struct MISFeaturizationResult
{
    std::size_t numberOfSamples = 0;
    std::size_t numberOfCandidates = 0;

    // Samples in which the expert's choice is meaningful (see above).
    std::size_t numberOfUsableSamples = 0;
};

// Header row of the feature table, without the trailing line break.
std::string misFeatureTableHeader();

/*
 * Reads the node and candidate tables written by the data collection for
 * the given instance, and writes the feature table.
 *
 * The output file is never overwritten. On failure, no output file is
 * left behind and errorMessage describes the problem.
 */
bool featurizeMISStrongBranchingSamples(
    const MISInstance& instance,
    const std::filesystem::path& nodesPath,
    const std::filesystem::path& candidatesPath,
    const std::filesystem::path& outputPath,
    MISFeaturizationResult& result,
    std::string& errorMessage
);
