#include "cuts/MISNodeLog.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#define EVENTHDLR_DESC \
    "one CSV line per solved node: LP bound before and after the cut loop, and its cost"

#define NODE_EVENTS (SCIP_EVENTTYPE_NODEFOCUSED | SCIP_EVENTTYPE_FIRSTLPSOLVED \
    | SCIP_EVENTTYPE_LPSOLVED | SCIP_EVENTTYPE_NODESOLVED)

/* Separators whose cuts and time are also reported on their own. */
static const char* const TRACKED[] = { "zerohalf", "oddcycle", "clique" };
#define NTRACKED 3

/* Cumulative counters of the separation, read at events. */
typedef struct
{
    SCIP_Real seconds;                  /* every separator and the cut pool */
    SCIP_Longint cuts;                  /* cuts applied to the LP */
    SCIP_Longint poolCuts;              /* of which taken from the global cut pool */
    SCIP_Real trackedSeconds[NTRACKED];
    SCIP_Longint trackedCuts[NTRACKED];
} Separation;

struct SCIP_EventhdlrData
{
    FILE* file;
    SCIP_SEPA* tracked[NTRACKED];       /* NULL when not included */

    /* The node being solved. Its LP is solved once without cuts (first LP),
     * then with its cut loop (LPSOLVED event); it can be solved again later,
     * when the branching rule tightens bounds (strong branching), which
     * fires LPSOLVED again: the cut loop is the first one. */
    SCIP_Longint number;
    SCIP_Real focusTime;
    SCIP_Real firstLPTime;
    SCIP_Real cutLoopTime;
    SCIP_Longint focusIterations;
    SCIP_Longint firstLPIterations;
    SCIP_Longint cutLoopIterations;
    SCIP_Real globalLower;
    SCIP_Real cutoffFirst;
    SCIP_Real lpFirst;
    SCIP_Real lpCutLoop;
    SCIP_Real lpFinal;
    int fractional;
    int half;
    int rounds;
    int lpSolves;               /* LPSOLVED events */
    SCIP_Bool firstLPSolved;
    Separation atFocus;
    Separation atCutLoop;
};

static void readSeparation(SCIP* scip, SCIP_EVENTHDLRDATA* data, Separation* separation)
{
    SCIP_SEPA** sepas = SCIPgetSepas(scip);
    const int nsepas = SCIPgetNSepas(scip);

    separation->seconds = SCIPcutpoolGetTime(SCIPgetGlobalCutpool(scip));
    separation->cuts = SCIPgetNCutsApplied(scip);
    separation->poolCuts = 0;

    for( int i = 0; i < nsepas; ++i )
    {
        separation->seconds += SCIPsepaGetTime(sepas[i]);
        separation->poolCuts += SCIPsepaGetNCutsAppliedViaPool(sepas[i]);
    }

    for( int k = 0; k < NTRACKED; ++k )
    {
        SCIP_SEPA* sepa = data->tracked[k];

        separation->trackedSeconds[k] = sepa != NULL ? SCIPsepaGetTime(sepa) : 0.0;
        separation->trackedCuts[k] = sepa != NULL ? SCIPsepaGetNCutsApplied(sepa) : 0;
    }
}

/* Objective of the current LP: infinity when it reached the cutoff bound or
 * is infeasible, NaN when it was not solved to optimality otherwise. */
static SCIP_Real lpObjective(SCIP* scip)
{
    switch( SCIPgetLPSolstat(scip) )
    {
    case SCIP_LPSOLSTAT_OPTIMAL:
        return SCIPgetLPObjval(scip);
    case SCIP_LPSOLSTAT_INFEASIBLE:
    case SCIP_LPSOLSTAT_OBJLIMIT:
        return SCIPinfinity(scip);
    default:
        return NAN;
    }
}

/* A value, "inf" or "-inf" for SCIP's infinities, nothing for NaN. */
static void writeReal(FILE* file, SCIP* scip, SCIP_Real value)
{
    if( isnan(value) )
        return;

    if( SCIPisInfinity(scip, value) )
        fputs("inf", file);
    else if( SCIPisInfinity(scip, -value) )
        fputs("-inf", file);
    else
        fprintf(file, "%.10g", value);
}

static void focus(SCIP* scip, SCIP_EVENTHDLRDATA* data, SCIP_NODE* node)
{
    data->number = SCIPnodeGetNumber(node);
    data->focusTime = SCIPgetSolvingTime(scip);
    data->firstLPTime = data->focusTime;
    data->cutLoopTime = data->focusTime;
    data->focusIterations = SCIPgetNLPIterations(scip);
    data->firstLPIterations = data->focusIterations;
    data->cutLoopIterations = data->focusIterations;
    data->globalLower = SCIPgetLowerbound(scip);
    data->cutoffFirst = SCIPgetCutoffbound(scip);
    data->lpFirst = NAN;
    data->lpCutLoop = NAN;
    data->lpFinal = NAN;
    data->fractional = -1;
    data->half = -1;
    data->rounds = 0;
    data->lpSolves = 0;
    data->firstLPSolved = FALSE;
    readSeparation(scip, data, &data->atFocus);
    data->atCutLoop = data->atFocus;
}

static SCIP_RETCODE firstLPSolved(SCIP* scip, SCIP_EVENTHDLRDATA* data)
{
    data->firstLPSolved = TRUE;
    data->firstLPTime = SCIPgetSolvingTime(scip);
    data->firstLPIterations = SCIPgetNLPIterations(scip);
    data->cutoffFirst = SCIPgetCutoffbound(scip);
    data->lpFirst = lpObjective(scip);

    if( SCIPgetLPSolstat(scip) == SCIP_LPSOLSTAT_OPTIMAL )
    {
        SCIP_Real* fractions;
        int ncands;

        SCIP_CALL( SCIPgetLPBranchCands(scip, NULL, NULL, &fractions, &ncands, NULL, NULL) );

        data->fractional = ncands;
        data->half = 0;

        for( int i = 0; i < ncands; ++i )
        {
            if( fabs(fractions[i] - 0.5) < 1e-6 )
                ++data->half;
        }
    }

    return SCIP_OKAY;
}

/* The LP of the node was solved with its cut loop: the first time ends the
 * node's cut loop, the last one gives its final LP bound. */
static void lpSolved(SCIP* scip, SCIP_EVENTHDLRDATA* data)
{
    data->lpFinal = lpObjective(scip);

    if( data->lpSolves++ > 0 )
        return;

    data->cutLoopTime = SCIPgetSolvingTime(scip);
    data->cutLoopIterations = SCIPgetNLPIterations(scip);
    data->lpCutLoop = data->lpFinal;
    data->rounds = SCIPgetNSepaRounds(scip);
    readSeparation(scip, data, &data->atCutLoop);
}

static void writeNode(SCIP* scip, SCIP_EVENTHDLRDATA* data, SCIP_NODE* node, SCIP_EVENTTYPE type)
{
    FILE* file = data->file;
    const SCIP_Real now = SCIPgetSolvingTime(scip);
    const SCIP_Longint iterations = SCIPgetNLPIterations(scip);
    SCIP_NODE* parent = SCIPnodeGetParent(node);
    Separation atEnd;

    readSeparation(scip, data, &atEnd);

    /* A node pruned during its cut loop may end without LPSOLVED event: its
     * loop then lasted until now. */
    const SCIP_Bool loopEnded = data->lpSolves > 0;
    const SCIP_Real cutLoopEnd = loopEnded ? data->cutLoopTime : now;
    const SCIP_Longint cutLoopEndIterations = loopEnded ? data->cutLoopIterations : iterations;
    const Separation* atLoopEnd = loopEnded ? &data->atCutLoop : &atEnd;
    const int rounds = loopEnded ? data->rounds : SCIPgetNSepaRounds(scip);
    const char* status =
        type == SCIP_EVENTTYPE_NODEBRANCHED ? "branched"
        : type == SCIP_EVENTTYPE_NODEINFEASIBLE ? "infeasible" : "feasible";

    fprintf(file, "%d,%" SCIP_LONGINT_FORMAT ",%" SCIP_LONGINT_FORMAT ",%d,%s,",
        SCIPgetNRuns(scip), data->number,
        parent != NULL ? SCIPnodeGetNumber(parent) : (SCIP_Longint) 0,
        SCIPnodeGetDepth(node), status);

    writeReal(file, scip, data->globalLower);
    fputc(',', file);
    writeReal(file, scip, data->cutoffFirst);
    fputc(',', file);
    writeReal(file, scip, SCIPgetCutoffbound(scip));
    fputc(',', file);
    writeReal(file, scip, data->lpFirst);
    fputc(',', file);
    writeReal(file, scip, data->lpCutLoop);
    fputc(',', file);
    writeReal(file, scip, data->lpFinal);
    fputc(',', file);
    writeReal(file, scip, SCIPnodeGetLowerbound(node));

    fprintf(file, ",%d,%d,%d,%" SCIP_LONGINT_FORMAT ",%" SCIP_LONGINT_FORMAT,
        data->fractional, data->half, data->firstLPSolved ? rounds : 0,
        atLoopEnd->cuts - data->atFocus.cuts, atLoopEnd->poolCuts - data->atFocus.poolCuts);

    for( int k = 0; k < NTRACKED; ++k )
        fprintf(file, ",%" SCIP_LONGINT_FORMAT, atLoopEnd->trackedCuts[k] - data->atFocus.trackedCuts[k]);

    fprintf(file, ",%.6f,%.6f,%.6f,%.6f",
        now - data->focusTime,
        data->firstLPTime - data->focusTime,
        data->firstLPSolved ? cutLoopEnd - data->firstLPTime : 0.0,
        atLoopEnd->seconds - data->atFocus.seconds);

    for( int k = 0; k < NTRACKED; ++k )
        fprintf(file, ",%.6f", atLoopEnd->trackedSeconds[k] - data->atFocus.trackedSeconds[k]);

    fprintf(file, ",%.6f,%" SCIP_LONGINT_FORMAT ",%" SCIP_LONGINT_FORMAT ",%d\n",
        atEnd.seconds - data->atFocus.seconds,
        data->firstLPIterations - data->focusIterations,
        data->firstLPSolved ? cutLoopEndIterations - data->firstLPIterations : (SCIP_Longint) 0,
        data->lpSolves > 1 ? data->lpSolves - 1 : 0);
}

static SCIP_DECL_EVENTEXEC(eventExecMISNodeLog)
{
    SCIP_EVENTHDLRDATA* data = SCIPeventhdlrGetData(eventhdlr);
    const SCIP_EVENTTYPE type = SCIPeventGetType(event);

    if( type == SCIP_EVENTTYPE_NODEFOCUSED )
        focus(scip, data, SCIPeventGetNode(event));
    else if( type == SCIP_EVENTTYPE_FIRSTLPSOLVED )
        SCIP_CALL( firstLPSolved(scip, data) );
    else if( type == SCIP_EVENTTYPE_LPSOLVED )
        lpSolved(scip, data);
    else if( type & SCIP_EVENTTYPE_NODESOLVED )
        writeNode(scip, data, SCIPeventGetNode(event), type);

    return SCIP_OKAY;
}

static SCIP_DECL_EVENTINITSOL(eventInitsolMISNodeLog)
{
    SCIP_EVENTHDLRDATA* data = SCIPeventhdlrGetData(eventhdlr);

    for( int k = 0; k < NTRACKED; ++k )
        data->tracked[k] = SCIPfindSepa(scip, TRACKED[k]);

    SCIP_CALL( SCIPcatchEvent(scip, NODE_EVENTS, eventhdlr, NULL, NULL) );

    return SCIP_OKAY;
}

static SCIP_DECL_EVENTEXITSOL(eventExitsolMISNodeLog)
{
    SCIP_EVENTHDLRDATA* data = SCIPeventhdlrGetData(eventhdlr);

    SCIP_CALL( SCIPdropEvent(scip, NODE_EVENTS, eventhdlr, NULL, -1) );

    if( fflush(data->file) != 0 )
    {
        SCIPerrorMessage("cannot write the node log\n");
        return SCIP_WRITEERROR;
    }

    return SCIP_OKAY;
}

static SCIP_DECL_EVENTFREE(eventFreeMISNodeLog)
{
    SCIP_EVENTHDLRDATA* data = SCIPeventhdlrGetData(eventhdlr);

    fclose(data->file);
    SCIPfreeBlockMemory(scip, &data);
    SCIPeventhdlrSetData(eventhdlr, NULL);

    return SCIP_OKAY;
}

SCIP_RETCODE SCIPincludeEventHdlrMISNodeLog(SCIP* scip, const char* path)
{
    SCIP_EVENTHDLRDATA* data;
    SCIP_EVENTHDLR* eventhdlr = NULL;
    FILE* file = fopen(path, "w");

    if( file == NULL )
    {
        SCIPerrorMessage("cannot create the node log <%s>\n", path);
        return SCIP_FILECREATEERROR;
    }

    fputs("run,node,parent,depth,status,global_lower,cutoff_first,cutoff_final,"
        "lp_first,lp_cut_loop,lp_final,lower_bound,fractional,half,"
        "separation_rounds,cuts,pool_cuts,zerohalf_cuts,oddcycle_cuts,clique_cuts,"
        "node_seconds,first_lp_seconds,cut_loop_seconds,separation_seconds,"
        "zerohalf_seconds,oddcycle_seconds,clique_seconds,"
        "node_separation_seconds,first_lp_iterations,cut_loop_iterations,"
        "lp_resolves\n", file);

    SCIP_CALL( SCIPallocBlockMemory(scip, &data) );
    memset(data, 0, sizeof(*data));
    data->file = file;

    SCIP_CALL( SCIPincludeEventhdlrBasic(scip, &eventhdlr, MIS_NODE_LOG_EVENTHDLR_NAME,
            EVENTHDLR_DESC, eventExecMISNodeLog, data) );
    SCIP_CALL( SCIPsetEventhdlrInitsol(scip, eventhdlr, eventInitsolMISNodeLog) );
    SCIP_CALL( SCIPsetEventhdlrExitsol(scip, eventhdlr, eventExitsolMISNodeLog) );
    SCIP_CALL( SCIPsetEventhdlrFree(scip, eventhdlr, eventFreeMISNodeLog) );

    return SCIP_OKAY;
}
