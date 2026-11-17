#include "cuts/MISLookaheadCutsel.h"

#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CUTSEL_DESC \
    "root cuts ranked by the LP bound increase of each cut added alone"

/* Above SCIP's selectors (hybrid: 8000). */
#define CUTSEL_PRIORITY 1000000

/*
 * As in SCIP's hybrid selector: a cut whose score reaches GOOD_SCORE times
 * the best score may be more parallel to the cuts already taken.
 */
#define GOOD_SCORE 0.9
#define GOOD_MAX_PARALLELISM 0.5

struct SCIP_CutselData
{
    MISLookaheadStatistics statistics;
};

typedef struct
{
    SCIP_ROW* cut;
    SCIP_Real gain;
    SCIP_Real efficacy;
    int position;
} ScoredCut;

/* Decreasing gain, then decreasing efficacy, then the original order. */
static int compareScoredCuts(const void* a, const void* b)
{
    const ScoredCut* x = (const ScoredCut*) a;
    const ScoredCut* y = (const ScoredCut*) b;

    if( x->gain != y->gain )
        return x->gain > y->gain ? -1 : 1;

    if( x->efficacy != y->efficacy )
        return x->efficacy > y->efficacy ? -1 : 1;

    return x->position - y->position;
}

static double seconds(void)
{
    struct timespec now;
    timespec_get(&now, TIME_UTC);
    return (double) now.tv_sec + 1e-9 * (double) now.tv_nsec;
}

/*
 * Increase of the LP bound when the cut alone is added to the current LP,
 * solved in diving mode: infinity when the LP becomes infeasible or reaches
 * the cutoff bound, 0 when the dive fails.
 */
static SCIP_RETCODE scoreCut(
    SCIP* scip,
    SCIP_ROW* cut,
    SCIP_Real bound,
    SCIP_Real* gain
)
{
    SCIP_Bool lperror = FALSE;
    SCIP_Bool cutoff = FALSE;

    SCIP_CALL( SCIPstartDive(scip) );
    SCIP_CALL( SCIPaddRowDive(scip, cut) );
    SCIP_CALL( SCIPsolveDiveLP(scip, -1, &lperror, &cutoff) );

    if( cutoff )
        *gain = SCIPinfinity(scip);
    else if( lperror || SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
        *gain = 0.0;
    else
        *gain = MAX(0.0, SCIPgetLPObjval(scip) - bound);

    SCIP_CALL( SCIPendDive(scip) );

    return SCIP_OKAY;
}

/* Whether the cut is more parallel than limit to one of the given rows. */
static SCIP_Bool tooParallel(
    SCIP_ROW* cut,
    SCIP_ROW** rows,
    int nrows,
    SCIP_Real limit
)
{
    for( int i = 0; i < nrows; ++i )
    {
        if( SCIProwGetParallelism(cut, rows[i], 'e') > limit )
            return TRUE;
    }

    return FALSE;
}

static SCIP_DECL_CUTSELSELECT(cutselSelectMISLookahead)
{
    SCIP_CUTSELDATA* data = SCIPcutselGetData(cutsel);

    *nselectedcuts = 0;

    /* Outside the root, or without an optimal LP to dive from, the next
     * selector decides. */
    if( !root || SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
    {
        *result = SCIP_DIDNOTFIND;
        return SCIP_OKAY;
    }

    *result = SCIP_SUCCESS;

    if( ncuts == 0 || maxnselectedcuts <= 0 )
        return SCIP_OKAY;

    const double start = seconds();
    const SCIP_Longint iterationsBefore = SCIPgetNDivingLPIterations(scip);
    const SCIP_Real bound = SCIPgetLPObjval(scip);

    SCIP_Real minortho;
    SCIP_CALL( SCIPgetRealParam(scip, "cutselection/hybrid/minorthoroot", &minortho) );

    const SCIP_Real maxParallelism = 1.0 - minortho;
    const SCIP_Real goodMaxParallelism = MAX(GOOD_MAX_PARALLELISM, maxParallelism);

    ScoredCut* scored;
    SCIP_ROW** selected;
    SCIP_Bool* taken;

    SCIP_CALL( SCIPallocBufferArray(scip, &scored, ncuts) );
    SCIP_CALL( SCIPallocBufferArray(scip, &selected, ncuts) );
    SCIP_CALL( SCIPallocClearBufferArray(scip, &taken, ncuts) );

    for( int i = 0; i < ncuts; ++i )
    {
        scored[i].cut = cuts[i];
        scored[i].position = i;
        scored[i].efficacy = SCIPgetCutEfficacy(scip, NULL, cuts[i]);
        SCIP_CALL( scoreCut(scip, cuts[i], bound, &scored[i].gain) );

        if( SCIPisPositive(scip, scored[i].gain) )
            ++data->statistics.improving;
    }

    qsort(scored, (size_t) ncuts, sizeof(ScoredCut), compareScoredCuts);

    const SCIP_Real best = scored[0].gain;
    int nselected = 0;

    for( int k = 0; k < ncuts && nselected < maxnselectedcuts; ++k )
    {
        const SCIP_Bool good =
            SCIPisPositive(scip, best) && scored[k].gain >= GOOD_SCORE * best;
        const SCIP_Real limit = good ? goodMaxParallelism : maxParallelism;

        if( tooParallel(scored[k].cut, forcedcuts, nforcedcuts, limit)
            || tooParallel(scored[k].cut, selected, nselected, limit) )
        {
            continue;
        }

        selected[nselected++] = scored[k].cut;
        taken[scored[k].position] = TRUE;
    }

    /* The selected cuts first, then the others in their original order. */
    int next = nselected;

    for( int i = 0; i < ncuts; ++i )
    {
        if( !taken[i] )
            selected[next++] = cuts[i];
    }

    memcpy(cuts, selected, (size_t) ncuts * sizeof(SCIP_ROW*));
    *nselectedcuts = nselected;

    data->statistics.rounds += 1;
    data->statistics.scored += ncuts;
    data->statistics.selected += nselected;
    data->statistics.dive_iterations +=
        SCIPgetNDivingLPIterations(scip) - iterationsBefore;
    data->statistics.seconds += seconds() - start;

    SCIPfreeBufferArray(scip, &taken);
    SCIPfreeBufferArray(scip, &selected);
    SCIPfreeBufferArray(scip, &scored);

    return SCIP_OKAY;
}

static SCIP_DECL_CUTSELFREE(cutselFreeMISLookahead)
{
    SCIP_CUTSELDATA* data = SCIPcutselGetData(cutsel);

    SCIPfreeBlockMemory(scip, &data);
    SCIPcutselSetData(cutsel, NULL);

    return SCIP_OKAY;
}

SCIP_RETCODE SCIPincludeCutselMISLookahead(SCIP* scip)
{
    SCIP_CUTSELDATA* data;
    SCIP_CUTSEL* cutsel = NULL;

    SCIP_CALL( SCIPallocBlockMemory(scip, &data) );
    memset(&data->statistics, 0, sizeof(data->statistics));

    SCIP_CALL( SCIPincludeCutselBasic(scip, &cutsel, MIS_LOOKAHEAD_CUTSEL_NAME,
            CUTSEL_DESC, CUTSEL_PRIORITY, cutselSelectMISLookahead, data) );
    SCIP_CALL( SCIPsetCutselFree(scip, cutsel, cutselFreeMISLookahead) );

    return SCIP_OKAY;
}

SCIP_RETCODE SCIPgetMISLookaheadStatistics(
    SCIP* scip,
    MISLookaheadStatistics* statistics
)
{
    SCIP_CUTSEL* cutsel = SCIPfindCutsel(scip, MIS_LOOKAHEAD_CUTSEL_NAME);

    if( cutsel == NULL )
        return SCIP_PLUGINNOTFOUND;

    *statistics = SCIPcutselGetData(cutsel)->statistics;

    return SCIP_OKAY;
}
