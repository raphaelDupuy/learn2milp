#ifndef LEARN2MILP_MIS_SEPARATION_GATE_H
#define LEARN2MILP_MIS_SEPARATION_GATE_H

#include <scip/scip.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIS_SEPARATION_GATE_NAME "misgate"

/* What the gate decided during a solve, over the nodes below the root. */
typedef struct
{
    long long separated;  /* nodes where the gated separators were let run */
    long long skipped;    /* nodes where they were kept off */
} MISSeparationGateStatistics;

/*
 * Includes a separator that decides, at each node below the root, whether
 * other separators run at that node: they run only when the gap between the
 * cutoff bound and the LP bound of the node is at most a threshold. Measured
 * on the node logs (experiments/cuts/analyze_node_logs.py), the cut loops
 * that prune a node are the ones that start close to the cutoff bound.
 *
 * The gate finds no cuts. It has the highest separator priority, so that it
 * is called first in the node's first separation round, when the LP bound is
 * that of the node's first LP. It then sets the frequency of each gated
 * separator, for the rest of the node, to the frequency the separator had
 * when the solve began (the separator runs as configured) or to 0 (root
 * only, so it is not called at this node). At the root, it restores the
 * frequencies and decides nothing. The cut pool and the separators it does
 * not gate are untouched. (SCIP separates the cut pool first in each round,
 * without solving the LP again; only when the pool alone gives enough cuts
 * to skip the separators of the first round does the gate decide in a later
 * round, on a bound already raised by those cuts.)
 *
 * Parameters (SCIP's own mechanism, for instance in a settings file):
 *   separating/misgate/freq        -1 (default): the gate is never called
 *                                  and changes nothing; 1: it decides at
 *                                  every node (its expbackoff is 1)
 *   separating/misgate/maxgap      separate when cutoff bound - LP bound is
 *                                  at most this value (default 0.5; in the
 *                                  transformed problem, a minimization)
 *   separating/misgate/separators  comma-separated names of the gated
 *                                  separators (default "oddcycle")
 *
 * A name that is not a separator of scip is an error when the solve begins.
 */
SCIP_RETCODE SCIPincludeSepaMISSeparationGate(SCIP* scip);

/* Statistics of the gate included in scip. */
SCIP_RETCODE SCIPgetMISSeparationGateStatistics(
    SCIP* scip,
    MISSeparationGateStatistics* statistics
);

#ifdef __cplusplus
}
#endif

#endif
