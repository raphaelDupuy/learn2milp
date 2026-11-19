#ifndef LEARN2MILP_MIS_NODE_LOG_H
#define LEARN2MILP_MIS_NODE_LOG_H

#include <scip/scip.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIS_NODE_LOG_EVENTHDLR_NAME "misnodelog"

/*
 * Includes an event handler that writes one CSV line per node solved, to see
 * where in the tree the separation of cuts pays: the LP bound before and after
 * the node's cut loop, what the loop cost, and how the node ended. The columns
 * are described in docs/data-format.md.
 *
 * The file at path is created, with its header, when the handler is included
 * (an existing file is replaced: the caller checks), and closed when the
 * solver is freed. Bounds are in SCIP's transformed space, a minimization: a
 * node is pruned when its lower bound reaches the cutoff bound.
 *
 * The handler only reads the solver's state, so the search is the same with
 * or without it.
 */
SCIP_RETCODE SCIPincludeEventHdlrMISNodeLog(SCIP* scip, const char* path);

#ifdef __cplusplus
}
#endif

#endif
