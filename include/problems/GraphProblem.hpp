#pragma once

#include <string>
#include <vector>

#include <scip/scip.h>

#include "problems/MIS.hpp"

/*
 * The problems posed on a graph (a .mis instance), with one binary
 * variable x_v per vertex. The branching rules, the features and the
 * samples only see the graph and these variables, so the same chain
 * serves every problem of this list.
 */
enum class GraphProblem
{
    // Maximum independent set: max sum x_v, x_u + x_v <= 1 on every edge.
    MaximumIndependentSet,
    // Minimum dominating set: min sum x_v, and every vertex is in the set
    // or has a neighbour in it: x_v + sum of x_u over N(v) >= 1.
    MinimumDominatingSet
};

// Names used on the command line and in result tables: "mis" and
// "dominating".
const char* graphProblemName(GraphProblem problem);

bool parseGraphProblem(const std::string& name, GraphProblem& problem);

/*
 * Creates the problem in scip and stores the vertex variables in vars
 * (owned by the problem, not captured for the caller). Returns
 * SCIP_INVALIDDATA, without creating anything, if n is negative or an
 * edge endpoint is outside [0, n).
 */
SCIP_RETCODE buildGraphProblemModel(
    SCIP* scip,
    GraphProblem problem,
    const MISInstance& graph,
    std::vector<SCIP_VAR*>& vars
);
