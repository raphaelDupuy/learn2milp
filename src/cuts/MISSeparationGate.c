#include "cuts/MISSeparationGate.h"

#include <limits.h>
#include <string.h>

#define SEPA_DESC "runs other separators only at nodes close to the cutoff bound"

/* Called before every other separator (SCIP's highest: 1000000). */
#define SEPA_PRIORITY (INT_MAX / 4)
#define SEPA_FREQ -1
#define SEPA_MAXBOUNDDIST 1.0
#define SEPA_USESSUBSCIP FALSE
#define SEPA_DELAY FALSE

#define DEFAULT_MAXGAP 0.5
#define DEFAULT_SEPARATORS "oddcycle"
#define MAX_GATED 16

struct SCIP_SepaData
{
    SCIP_Real maxgap;
    char* separators;  /* owned by SCIP's parameter */

    SCIP_SEPA* gated[MAX_GATED];
    int frequencies[MAX_GATED];  /* when the solve began */
    int ngated;

    /* Node decided last: number and run (node numbers restart each run). */
    SCIP_Longint node;
    int run;

    MISSeparationGateStatistics statistics;
};

static void setFrequencies(SCIP_SEPADATA* data, SCIP_Bool separate)
{
    for( int i = 0; i < data->ngated; ++i )
        SCIPsepaSetFreq(data->gated[i], separate ? data->frequencies[i] : 0);
}

static SCIP_DECL_SEPAINITSOL(sepaInitsolMISGate)
{
    SCIP_SEPADATA* data = SCIPsepaGetData(sepa);
    char names[SCIP_MAXSTRLEN];
    char* next = names;

    data->ngated = 0;
    data->node = -1;
    data->run = -1;
    data->statistics.separated = 0;
    data->statistics.skipped = 0;

    (void) SCIPsnprintf(names, SCIP_MAXSTRLEN, "%s", data->separators);

    while( *next != '\0' )
    {
        char* name = next + strspn(next, ", ");
        SCIP_SEPA* gated;

        if( *name == '\0' )
            break;

        next = name + strcspn(name, ", ");
        if( *next != '\0' )
            *next++ = '\0';

        gated = SCIPfindSepa(scip, name);

        if( gated == NULL || gated == sepa || data->ngated == MAX_GATED )
        {
            SCIPerrorMessage("separating/%s/separators: cannot gate <%s>\n", SCIPsepaGetName(sepa), name);
            return SCIP_PARAMETERWRONGVAL;
        }

        data->gated[data->ngated] = gated;
        data->frequencies[data->ngated] = SCIPsepaGetFreq(gated);
        ++data->ngated;
    }

    return SCIP_OKAY;
}

static SCIP_DECL_SEPAEXITSOL(sepaExitsolMISGate)
{
    SCIP_SEPADATA* data = SCIPsepaGetData(sepa);

    (void) scip;
    setFrequencies(data, TRUE);
    data->ngated = 0;

    return SCIP_OKAY;
}

static SCIP_DECL_SEPAEXECLP(sepaExeclpMISGate)
{
    SCIP_SEPADATA* data = SCIPsepaGetData(sepa);
    SCIP_NODE* node = SCIPgetCurrentNode(scip);
    const SCIP_Longint number = SCIPnodeGetNumber(node);
    const int run = SCIPgetNRuns(scip);
    SCIP_Bool separate = TRUE;

    (void) allowlocal;
    *result = SCIP_DIDNOTRUN;

    /* One decision per node, on its first separation round. */
    if( number == data->node && run == data->run )
        return SCIP_OKAY;

    data->node = number;
    data->run = run;

    if( depth == 0 )
    {
        setFrequencies(data, TRUE);
        return SCIP_OKAY;
    }

    if( SCIPgetLPSolstat(scip) == SCIP_LPSOLSTAT_OPTIMAL && !SCIPisInfinity(scip, SCIPgetCutoffbound(scip)) )
        separate = SCIPgetCutoffbound(scip) - SCIPgetLPObjval(scip) <= data->maxgap;

    setFrequencies(data, separate);

    if( separate )
        ++data->statistics.separated;
    else
        ++data->statistics.skipped;

    return SCIP_OKAY;
}

static SCIP_DECL_SEPAFREE(sepaFreeMISGate)
{
    SCIP_SEPADATA* data = SCIPsepaGetData(sepa);

    SCIPfreeBlockMemory(scip, &data);
    SCIPsepaSetData(sepa, NULL);

    return SCIP_OKAY;
}

SCIP_RETCODE SCIPincludeSepaMISSeparationGate(SCIP* scip)
{
    SCIP_SEPADATA* data = NULL;
    SCIP_SEPA* sepa = NULL;

    SCIP_CALL( SCIPallocBlockMemory(scip, &data) );
    BMSclearMemory(data);

    SCIP_CALL( SCIPincludeSepaBasic(scip, &sepa, MIS_SEPARATION_GATE_NAME, SEPA_DESC, SEPA_PRIORITY, SEPA_FREQ,
            SEPA_MAXBOUNDDIST, SEPA_USESSUBSCIP, SEPA_DELAY, sepaExeclpMISGate, NULL, data) );

    /* Called at every depth: SCIP's default backoff (4) would call it at
     * depths 1, 4, 16... only, leaving the last decision in force elsewhere. */
    SCIP_CALL( SCIPsetIntParam(scip, "separating/" MIS_SEPARATION_GATE_NAME "/expbackoff", 1) );

    SCIP_CALL( SCIPsetSepaInitsol(scip, sepa, sepaInitsolMISGate) );
    SCIP_CALL( SCIPsetSepaExitsol(scip, sepa, sepaExitsolMISGate) );
    SCIP_CALL( SCIPsetSepaFree(scip, sepa, sepaFreeMISGate) );

    SCIP_CALL( SCIPaddRealParam(scip, "separating/" MIS_SEPARATION_GATE_NAME "/maxgap",
            "the gated separators run at a node when cutoff bound - LP bound is at most this value",
            &data->maxgap, FALSE, DEFAULT_MAXGAP, -SCIPinfinity(scip), SCIPinfinity(scip), NULL, NULL) );
    SCIP_CALL( SCIPaddStringParam(scip, "separating/" MIS_SEPARATION_GATE_NAME "/separators",
            "comma-separated names of the separators the gate decides for",
            &data->separators, FALSE, DEFAULT_SEPARATORS, NULL, NULL) );

    return SCIP_OKAY;
}

SCIP_RETCODE SCIPgetMISSeparationGateStatistics(
    SCIP* scip,
    MISSeparationGateStatistics* statistics
)
{
    SCIP_SEPA* sepa = SCIPfindSepa(scip, MIS_SEPARATION_GATE_NAME);

    if( sepa == NULL )
        return SCIP_PLUGINNOTFOUND;

    *statistics = SCIPsepaGetData(sepa)->statistics;
    return SCIP_OKAY;
}
