#pragma once

#include <scip/scip.h>

/*
 * SCIP settings of the branching study, shared by data collection and
 * by the evaluation of branching rules so that both always run in the
 * same conditions:
 *
 * - presolving stays enabled;
 * - restarts are disabled (presolving/maxrestarts = 0), so that the
 *   variable space does not change during the search;
 * - cutting planes are separated at the root node only
 *   (separating/maxrounds = 0), so that the effect of branching is
 *   isolated.
 */
SCIP_RETCODE applyBranchingStudySettings(SCIP* scip);
