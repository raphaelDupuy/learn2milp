#ifndef LEARN2MILP_MIS_LOOKAHEAD_CUTSEL_H
#define LEARN2MILP_MIS_LOOKAHEAD_CUTSEL_H

#include <scip/scip.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MIS_LOOKAHEAD_CUTSEL_NAME "mislookahead"

/* What the lookahead selector did at the root during a solve. */
typedef struct
{
    long long rounds;           /* selections made at the root */
    long long scored;           /* candidate cuts scored by a dive */
    long long improving;        /* of which strictly raised the LP bound */
    long long selected;         /* cuts selected */
    long long dive_iterations;  /* LP iterations of the dives */
    double seconds;             /* time spent selecting */
} MISLookaheadStatistics;

/*
 * Includes the lookahead cut selector, the expert of Paulus et al. (2022),
 * with a priority above SCIP's own selectors.
 *
 * At the root, every candidate cut is scored by the increase of the LP
 * bound when it alone is added to the LP, solved in diving mode (one LP per
 * candidate). The cuts are then taken in decreasing order of score, ties
 * broken by efficacy, skipping a cut too parallel to a forced cut or to a
 * cut already taken, with the parallelism limits of SCIP's hybrid selector
 * (cutselection/hybrid/minorthoroot). At the other nodes, the selector
 * leaves the selection to SCIP's own selectors.
 */
SCIP_RETCODE SCIPincludeCutselMISLookahead(SCIP* scip);

/* Statistics of the lookahead selector included in scip. */
SCIP_RETCODE SCIPgetMISLookaheadStatistics(
    SCIP* scip,
    MISLookaheadStatistics* statistics
);

#ifdef __cplusplus
}
#endif

#endif
