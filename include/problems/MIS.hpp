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

SCIP_RETCODE buildMISModel(
    SCIP* scip,
    const MISInstance& instance,
    std::vector<SCIP_VAR*>& vars
);
