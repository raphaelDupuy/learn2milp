#pragma once

#include <vector>

#include "learning/MISBranchingFeatures.h"
#include "problems/MIS.hpp"

/*
 * Adjacency lists of a MIS instance in the compressed form expected by
 * the branching features: neighbours sorted by increasing vertex id.
 */
struct MISAdjacency
{
    int numberOfVertices = 0;
    std::vector<int> offsets;
    std::vector<int> neighbours;

    // The same edges as a bit matrix (MISFeatureGraph::adjacency_matrix),
    // empty above MIS_ADJACENCY_MATRIX_MAX_VERTICES vertices.
    std::vector<unsigned char> matrix;

    // View for the C feature functions. It points into this object and
    // must not outlive it.
    MISFeatureGraph view() const;
};

// Largest graph given a bit matrix: 8 MB at most.
constexpr int MIS_ADJACENCY_MATRIX_MAX_VERTICES = 8192;

// The instance must be valid (see validateMISInstance()).
MISAdjacency buildMISAdjacency(const MISInstance& instance);
