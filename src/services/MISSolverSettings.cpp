#include "services/MISSolverSettings.hpp"

SCIP_RETCODE applyBranchingStudySettings(SCIP* scip)
{
    SCIP_CALL(
        SCIPsetIntParam(scip, "presolving/maxrestarts", 0)
    );

    SCIP_CALL(
        SCIPsetIntParam(scip, "separating/maxrounds", 0)
    );

    return SCIP_OKAY;
}
