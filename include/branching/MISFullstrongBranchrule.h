
#ifndef LEARN2MILP_MIS_FULLSTRONG_BRANCHRULE_H
#define LEARN2MILP_MIS_FULLSTRONG_BRANCHRULE_H

#include <scip/scip.h>

#include "branching/MISStrongBranchingCollector.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Includes the MIS full strong branching rule.
 *
 * vertexvars[v] must be the original variable of vertex v. The plugin
 * stores these pointers WITHOUT capturing them: they must stay valid,
 * i.e. the original problem must not be freed or replaced, for as long
 * as the plugin may be initialized by a solve. The transformed variables
 * are captured internally for the lifetime of each transformed problem.
 *
 * The callback receives one sample per branching decision. No sample is
 * emitted for a node reached while the solver is stopping (time limit or
 * interrupt), because its strong branching evaluation is not reliable.
 *
 * Pass (NULL, 0, NULL, NULL) for the empty configuration used by copies.
 */
SCIP_RETCODE SCIPincludeBranchruleMISFullstrong(
    SCIP* scip,
    SCIP_VAR** vertexvars,
    int nvertices,
    MISStrongBranchingSampleCallback callback,
    void* userdata
);

#ifdef __cplusplus
}
#endif

#endif
