#include "problems/GraphProblem.hpp"

#include "problems/DominatingSet.hpp"

const char* graphProblemName(GraphProblem problem)
{
    switch (problem)
    {
    case GraphProblem::MaximumIndependentSet:
        return "mis";
    case GraphProblem::MinimumDominatingSet:
        return "dominating";
    }

    return "unknown";
}

bool parseGraphProblem(const std::string& name, GraphProblem& problem)
{
    for (const GraphProblem candidate :
         {GraphProblem::MaximumIndependentSet,
          GraphProblem::MinimumDominatingSet})
    {
        if (name == graphProblemName(candidate))
        {
            problem = candidate;
            return true;
        }
    }

    return false;
}

SCIP_RETCODE buildGraphProblemModel(
    SCIP* scip,
    GraphProblem problem,
    const MISInstance& graph,
    std::vector<SCIP_VAR*>& vars
)
{
    switch (problem)
    {
    case GraphProblem::MaximumIndependentSet:
        return buildMISModel(scip, graph, vars);
    case GraphProblem::MinimumDominatingSet:
        return buildDominatingSetModel(scip, graph, vars);
    }

    return SCIP_INVALIDDATA;
}
