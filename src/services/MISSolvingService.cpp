
#include "services/MISSolvingService.hpp"

#include "io/MISInstanceIO.hpp"
#include "services/SCIPStatusName.hpp"

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include <exception>
#include <string>

bool solveMIS(
    const MISInstance& instance,
    MISResult& result
)
{
    result = MISResult{};

    // Same definition of a valid instance as the file format and the
    // data collection.
    if (!validateMISInstance(instance, result.errorMessage))
    {
        result.errorMessage =
            "Invalid MIS instance: " + result.errorMessage;
        return false;
    }

    SCIP* scip = nullptr;
    SCIP_RETCODE retcode = SCIP_OKAY;

    // Initialize SCIP.
    retcode = SCIPcreate(&scip);

    if (retcode != SCIP_OKAY)
    {
        result.errorMessage = "Failed to create SCIP instance.";
        return false;
    }

    // Include default SCIP plugins.
    retcode = SCIPincludeDefaultPlugins(scip);

    if (retcode != SCIP_OKAY)
    {
        result.errorMessage = "Failed to include SCIP default plugins.";
        SCIPfree(&scip);
        return false;
    }

    try
    {
        // Build the MIS MILP model.
        std::vector<SCIP_VAR*> vars;

        retcode = buildMISModel(scip, instance, vars);

        if (retcode != SCIP_OKAY)
        {
            result.errorMessage = "Failed to build the MIS model.";
            SCIPfree(&scip);
            return false;
        }

        // Disable SCIP console output.
        retcode = SCIPsetIntParam(
            scip,
            "display/verblevel",
            0
        );

        if (retcode != SCIP_OKAY)
        {
            result.errorMessage = "Failed to disable SCIP console output.";
            SCIPfree(&scip);
            return false;

        }

        // Solve the model.
        retcode = SCIPsolve(scip);

        if (retcode != SCIP_OKAY)
        {
            result.errorMessage = "SCIP failed during optimization.";
            SCIPfree(&scip);
            return false;
        }

        // Retrieve solver statistics.
        const SCIP_STATUS status = SCIPgetStatus(scip);

        result.status = getSCIPStatusName(status);
        result.solvingTimeSeconds = SCIPgetSolvingTime(scip);
        result.exploredNodes =
            static_cast<long long>(SCIPgetNNodes(scip));

        // Retrieve the best feasible solution, if one exists.
        SCIP_SOL* solution = SCIPgetBestSol(scip);

        if (solution != nullptr)
        {
            result.hasSolution = true;

            result.objectiveValue =
                SCIPgetSolOrigObj(scip, solution);

            for (std::size_t v = 0; v < vars.size(); ++v)
            {
                const SCIP_Real value =
                    SCIPgetSolVal(scip, solution, vars[v]);

                if (value > 0.5)
                {
                    result.selectedVertices.push_back(
                        static_cast<int>(v)
                    );
                }
            }
        }
    }
    catch (const std::exception& exception)
    {
        result.errorMessage = exception.what();
        SCIPfree(&scip);
        return false;
    }

    // Release SCIP resources.
    retcode = SCIPfree(&scip);

    if (retcode != SCIP_OKAY)
    {
        result.errorMessage = "Failed to release SCIP resources.";
        return false;
    }

    return true;
}
