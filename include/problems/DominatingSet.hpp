#pragma once

#include <vector>

#include <scip/scip.h>

#include "problems/MIS.hpp"

/*
 * Creates the minimum dominating set problem of a graph in scip:
 *
 *     min  sum of x_v
 *     s.t. x_v + sum of x_u over the neighbours u of v >= 1   for every v
 *          x_v binary
 *
 * and stores the vertex variables in vars (owned by the problem, not
 * captured for the caller). Duplicate edges are counted once. Returns
 * SCIP_INVALIDDATA, without creating anything, if n is negative or an
 * edge endpoint is outside [0, n).
 */
SCIP_RETCODE buildDominatingSetModel(
    SCIP* scip,
    const MISInstance& graph,
    std::vector<SCIP_VAR*>& vars
);
