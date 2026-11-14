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

    // View for the C feature functions. It points into this object and
    // must not outlive it.
    MISFeatureGraph view() const;
};

// The instance must be valid (see validateMISInstance()).
MISAdjacency buildMISAdjacency(const MISInstance& instance);
