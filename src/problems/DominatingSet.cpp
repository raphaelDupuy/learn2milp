#include "problems/DominatingSet.hpp"

#include <algorithm>
#include <string>

#include <scip/cons_linear.h>

SCIP_RETCODE buildDominatingSetModel(
    SCIP* scip,
    const MISInstance& graph,
    std::vector<SCIP_VAR*>& vars
)
{
    if (graph.n < 0)
        return SCIP_INVALIDDATA;

    // Closed neighbourhoods, without duplicates.
    std::vector<std::vector<int>> closed(graph.n);

    for (int v = 0; v < graph.n; ++v)
        closed[v].push_back(v);

    for (const auto& [u, v] : graph.edges)
    {
        if (u < 0 || u >= graph.n || v < 0 || v >= graph.n)
            return SCIP_INVALIDDATA;

        closed[u].push_back(v);
        closed[v].push_back(u);
    }

    for (auto& neighbourhood : closed)
    {
        std::sort(neighbourhood.begin(), neighbourhood.end());
        neighbourhood.erase(
            std::unique(neighbourhood.begin(), neighbourhood.end()),
            neighbourhood.end());
    }

    SCIP_CALL(SCIPcreateProbBasic(scip, "minimum_dominating_set"));
    SCIP_CALL(SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE));

    vars.assign(graph.n, nullptr);

    for (int v = 0; v < graph.n; ++v)
    {
        const std::string name = "x_" + std::to_string(v);
        SCIP_VAR* var = nullptr;

        SCIP_CALL(SCIPcreateVarBasic(
            scip, &var, name.c_str(), 0.0, 1.0, 1.0, SCIP_VARTYPE_BINARY));
        SCIP_CALL(SCIPaddVar(scip, var));

        // The problem keeps the variable; vars only points to it.
        vars[v] = var;
        SCIP_CALL(SCIPreleaseVar(scip, &var));
    }

    for (int v = 0; v < graph.n; ++v)
    {
        const std::string name = "dominate_" + std::to_string(v);
        SCIP_CONS* cons = nullptr;

        SCIP_CALL(SCIPcreateConsBasicLinear(
            scip, &cons, name.c_str(), 0, nullptr, nullptr, 1.0,
            SCIPinfinity(scip)));

        for (const int u : closed[v])
            SCIP_CALL(SCIPaddCoefLinear(scip, cons, vars[u], 1.0));

        SCIP_CALL(SCIPaddCons(scip, cons));
        SCIP_CALL(SCIPreleaseCons(scip, &cons));
    }

    return SCIP_OKAY;
}
