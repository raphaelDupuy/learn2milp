#pragma once

#include <utility>
#include <vector>

#include <scip/scip.h>
#include <scip/cons_linear.h>

struct MISInstance
{
    int n;
    std::vector<std::pair<int, int>> edges;
};

// Creates the MIS problem in scip and stores the vertex variables in vars
// (owned by the problem, not captured for the caller). Returns
// SCIP_INVALIDDATA, without creating anything, if n is negative or an
// edge endpoint is outside [0, n).
SCIP_RETCODE buildMISModel(
    SCIP* scip,
    const MISInstance& instance,
    std::vector<SCIP_VAR*>& vars
);
