#include "branching/MISLearnedBranchrule.h"

#include "MISVertexMapping.h"

#include <scip/scip.h>
#include <scip/scip_branch.h>
#include <scip/scip_mem.h>
#include <scip/scip_var.h>
#include <scip/pub_var.h>

#include <limits.h>

typedef struct
{
    /* Link between graph vertices and SCIP variables. */
    MISVertexMapping mapping;

    /* Graph and model, owned by the caller. */
    const MISFeatureGraph* graph;
    const MISBranchingModel* model;

    /* Optional observer and its context. */
    MISLearnedBranchingDecisionCallback callback;
    void* userdata;

    /* Candidates evaluated by strong branching (0: none). */
    int sbcandidates;

    /*
     * Time spent computing the features and evaluating the model: clocks
     * of the current solve (NULL outside of it), and their totals, kept
     * when the solve ends.
     */
    SCIP_CLOCK* featureclock;
    SCIP_CLOCK* modelclock;
    SCIP_Real featureseconds;
    SCIP_Real modelseconds;

} MISLearnedBranchruleData;

/*
 * Strong branching on the candidates of the node with the highest scores,
 * evaluated as the misfullstrong expert does (vanillafullstrong with
 * idempotent = TRUE): *chosen receives the best of them by strong
 * branching score (first best in candidate order) and *cutoff whether
 * strong branching proves that the node can be pruned. *chosen is left
 * unchanged if no candidate could be evaluated.
 */
static
SCIP_RETCODE strongBranchTopCandidates(
    SCIP* scip,
    SCIP_VAR** cands,
    int ncands,
    const double* scores,
    int count,
    SCIP_Bool* selected,
    int* chosen,
    SCIP_Bool* cutoff
)
{
    const SCIP_Real lpobjval = SCIPgetLPObjval(scip);
    SCIP_Real bestscore = -SCIPinfinity(scip);
    SCIP_Real bestdown = lpobjval;
    SCIP_Real bestup = lpobjval;
    SCIP_Real provedbound = lpobjval;
    SCIP_Bool bestdownvalid = FALSE;
    SCIP_Bool bestupvalid = FALSE;
    int best = -1;
    int picked;
    int c;

    *cutoff = FALSE;

    if( SCIPisStopped(scip) )
        return SCIP_OKAY;

    /* The count candidates with the highest scores (first ones on ties). */
    for( c = 0; c < ncands; ++c )
        selected[c] = FALSE;

    for( picked = 0; picked < count && picked < ncands; ++picked )
    {
        int top = -1;

        for( c = 0; c < ncands; ++c )
        {
            if( !selected[c] && (top < 0 || scores[c] > scores[top]) )
                top = c;
        }

        selected[top] = TRUE;
    }

    SCIP_CALL( SCIPstartStrongbranch(scip, FALSE) );

    for( c = 0; c < ncands; ++c )
    {
        SCIP_Real down;
        SCIP_Real up;
        SCIP_Bool downvalid;
        SCIP_Bool upvalid;
        SCIP_Bool downinf;
        SCIP_Bool upinf;
        SCIP_Bool downconflict;
        SCIP_Bool upconflict;
        SCIP_Bool lperror;
        SCIP_Real gains[2];
        SCIP_Real score;

        if( !selected[c] )
            continue;

        SCIP_CALL( SCIPgetVarStrongbranchFrac(scip, cands[c], INT_MAX, TRUE,
            &down, &up, &downvalid, &upvalid, &downinf, &upinf,
            &downconflict, &upconflict, &lperror) );

        /* As the expert: stop at the first LP error. */
        if( lperror )
            break;

        gains[0] = MAX(down, lpobjval) - lpobjval;
        gains[1] = MAX(up, lpobjval) - lpobjval;
        score = SCIPgetBranchScoreMultiple(scip, cands[c], 2, gains);

        if( score > bestscore )
        {
            best = c;
            bestscore = score;
            bestdown = MAX(down, lpobjval);
            bestup = MAX(up, lpobjval);
            bestdownvalid = downvalid;
            bestupvalid = upvalid;
        }
    }

    SCIP_CALL( SCIPendStrongbranch(scip) );

    if( best < 0 )
        return SCIP_OKAY;

    *chosen = best;

    if( bestdownvalid && bestupvalid )
        provedbound = MAX(provedbound, MIN(bestdown, bestup));

    *cutoff = SCIPisGE(scip, provedbound, SCIPgetCutoffbound(scip));

    return SCIP_OKAY;
}

/*
 * Scores the LP branching candidates of the current node and stores the
 * position of the best one. The arrays are caller-allocated buffers.
 */
static
SCIP_RETCODE scoreCandidates(
    SCIP* scip,
    const MISLearnedBranchruleData* data,
    SCIP_VAR** cands,
    int ncands,
    MISVertexState* states,
    int* vertices,
    double* lpvalues,
    MISCandidateHistory* histories,
    double* features,
    double* scores,
    double* featurework,
    double* modelwork,
    int* positions,
    int* best
)
{
    MISBranchingNode node;
    SCIP_Real upperbound;
    int c;

    /* The feature time includes gathering the candidates' data. */
    SCIP_CALL( SCIPstartClock(scip, data->featureclock) );

    /* Candidates in MIS-vertex space (x_v rather than LP column). */
    for( c = 0; c < ncands; ++c )
    {
        const int index = MISvertexMappingFind(&data->mapping, cands[c]);
        SCIP_Real value;

        /* Every LP candidate must map to exactly one MIS vertex. */
        if( index < 0 )
            return SCIP_INVALIDDATA;

        value = SCIPvarGetLPSol(cands[c]);

        vertices[c] = data->mapping.vertexids[index];
        lpvalues[c] = data->mapping.complemented[index] ? 1.0 - value : value;

        MISvertexMappingFillHistory(
            scip, cands[c], data->mapping.complemented[index], &histories[c]
        );
    }

    MISvertexMappingFillStates(scip, &data->mapping, states);

    /* Same quantities as in a collected sample. */
    upperbound = SCIPgetUpperbound(scip);

    node.vertex_states = states;
    node.node_depth = SCIPgetDepth(scip);
    node.number_of_candidates = ncands;
    node.candidate_vertices = vertices;
    node.candidate_lp_values = lpvalues;
    node.candidate_histories = histories;
    node.objective_gap = MISbranchingObjectiveGap(
        SCIPgetLPObjval(scip), upperbound, !SCIPisInfinity(scip, upperbound)
    );

    /* The same function builds the training tables. */
    if( MIScomputeBranchingFeatures(
            data->graph, &node, featurework, features) != 0 )
        return SCIP_INVALIDDATA;

    SCIP_CALL( SCIPstopClock(scip, data->featureclock) );
    SCIP_CALL( SCIPstartClock(scip, data->modelclock) );

    if( MISevaluateBranchingModelOnNode(
            data->model, data->graph, states, ncands, vertices, features,
            modelwork, positions, scores) != 0 )
        return SCIP_INVALIDDATA;

    SCIP_CALL( SCIPstopClock(scip, data->modelclock) );

    /* First candidate with the highest score. */
    *best = 0;

    for( c = 0; c < ncands; ++c )
    {
        if( scores[c] > scores[*best] )
            *best = c;
    }

    return SCIP_OKAY;
}

static
SCIP_DECL_BRANCHEXECLP(branchExecLpMISLearned)
{
    MISLearnedBranchruleData* data;
    SCIP_VAR** cands;
    MISVertexState* states;
    int* vertices;
    double* lpvalues;
    MISCandidateHistory* histories;
    double* features;
    double* scores;
    double* featurework;
    double* modelwork;
    int* positions;
    SCIP_Bool* selected;
    SCIP_RETCODE retcode;
    SCIP_Bool cutoff = FALSE;
    int ncands;
    int best;

    (void)allowaddcons;

    data = (MISLearnedBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);
    assert(result != NULL);

    *result = SCIP_DIDNOTRUN;

    /* An empty configuration is used when copying the plugin. */
    if( data->mapping.nvertices == 0 )
        return SCIP_OKAY;

    SCIP_CALL( MISvertexMappingResolveColumns(scip, &data->mapping) );

    SCIP_CALL( SCIPgetLPBranchCands(
        scip, &cands, NULL, NULL, &ncands, NULL, NULL
    ) );

    if( ncands == 0 )
        return SCIP_OKAY;

    SCIP_CALL( SCIPallocBufferArray(scip, &states, data->mapping.nvertices) );
    SCIP_CALL( SCIPallocBufferArray(scip, &vertices, ncands) );
    SCIP_CALL( SCIPallocBufferArray(scip, &lpvalues, ncands) );
    SCIP_CALL( SCIPallocBufferArray(scip, &histories, ncands) );
    SCIP_CALL( SCIPallocBufferArray(
        scip, &features, ncands * MIS_BRANCHING_FEATURE_COUNT
    ) );
    SCIP_CALL( SCIPallocBufferArray(scip, &scores, ncands) );
    SCIP_CALL( SCIPallocBufferArray(
        scip, &featurework,
        MISbranchingFeatureWorkSize(data->mapping.nvertices)
    ) );
    SCIP_CALL( SCIPallocBufferArray(
        scip, &modelwork,
        MISbranchingModelNodeWorkSize(data->model, data->graph, ncands)
    ) );
    SCIP_CALL( SCIPallocBufferArray(
        scip, &positions,
        MISbranchingModelNodeIntWorkSize(data->model, data->graph, ncands)
    ) );
    SCIP_CALL( SCIPallocBufferArray(scip, &selected, ncands) );

    retcode = scoreCandidates(
        scip, data, cands, ncands, states, vertices, lpvalues, histories,
        features, scores, featurework, modelwork, positions, &best
    );

    if( retcode == SCIP_OKAY && data->sbcandidates > 0 )
    {
        retcode = strongBranchTopCandidates(
            scip, cands, ncands, scores, data->sbcandidates, selected,
            &best, &cutoff
        );
    }

    if( retcode == SCIP_OKAY && cutoff )
    {
        *result = SCIP_CUTOFF;
    }
    else if( retcode == SCIP_OKAY )
    {
        SCIP_VAR* var = cands[best];

        if( data->callback != NULL )
        {
            SCIP_NODE* node = SCIPgetCurrentNode(scip);
            MISLearnedBranchingDecision decision;

            decision.node_id = SCIPnodeGetNumber(node);
            decision.node_depth = SCIPnodeGetDepth(node);
            decision.vertex_states = states;
            decision.number_of_vertices = data->mapping.nvertices;
            decision.number_of_candidates = ncands;
            decision.candidate_vertices = vertices;
            decision.candidate_lp_values = lpvalues;
            decision.candidate_histories = histories;
            decision.objective_gap = MISbranchingObjectiveGap(
                SCIPgetLPObjval(scip), SCIPgetUpperbound(scip),
                !SCIPisInfinity(scip, SCIPgetUpperbound(scip))
            );
            decision.features = features;
            decision.scores = scores;
            decision.selected_candidate = best;

            data->callback(data->userdata, &decision);
        }

        retcode = SCIPbranchVarVal(
            scip, var, SCIPvarGetLPSol(var), NULL, NULL, NULL
        );

        if( retcode == SCIP_OKAY )
            *result = SCIP_BRANCHED;
    }

    /* Buffers are freed in reverse order of allocation. */
    SCIPfreeBufferArray(scip, &selected);
    SCIPfreeBufferArray(scip, &positions);
    SCIPfreeBufferArray(scip, &modelwork);
    SCIPfreeBufferArray(scip, &featurework);
    SCIPfreeBufferArray(scip, &scores);
    SCIPfreeBufferArray(scip, &features);
    SCIPfreeBufferArray(scip, &histories);
    SCIPfreeBufferArray(scip, &lpvalues);
    SCIPfreeBufferArray(scip, &vertices);
    SCIPfreeBufferArray(scip, &states);

    return retcode;
}

static
SCIP_DECL_BRANCHFREE(branchFreeMISLearned)
{
    MISLearnedBranchruleData* data;

    data = (MISLearnedBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    MISvertexMappingFree(scip, &data->mapping);
    SCIPfreeBlockMemory(scip, &data);

    SCIPbranchruleSetData(branchrule, NULL);

    return SCIP_OKAY;
}

static
SCIP_DECL_BRANCHEXIT(branchExitMISLearned)
{
    MISLearnedBranchruleData* data;

    data = (MISLearnedBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    SCIP_CALL( MISvertexMappingExit(scip, &data->mapping) );

    data->featureseconds = SCIPgetClockTime(scip, data->featureclock);
    data->modelseconds = SCIPgetClockTime(scip, data->modelclock);
    SCIP_CALL( SCIPfreeClock(scip, &data->featureclock) );
    SCIP_CALL( SCIPfreeClock(scip, &data->modelclock) );

    return SCIP_OKAY;
}

static
SCIP_DECL_BRANCHINIT(branchInitMISLearned)
{
    MISLearnedBranchruleData* data;

    data = (MISLearnedBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    SCIP_CALL( MISvertexMappingInit(scip, &data->mapping) );

    SCIP_CALL( SCIPcreateClock(scip, &data->featureclock) );
    SCIP_CALL( SCIPcreateClock(scip, &data->modelclock) );
    data->featureseconds = 0.0;
    data->modelseconds = 0.0;

    return SCIP_OKAY;
}

static
SCIP_DECL_BRANCHCOPY(branchCopyMISLearned)
{
    assert(scip != NULL);
    assert(branchrule != NULL);

    SCIP_CALL(
        SCIPincludeBranchruleMISLearned(
            scip, NULL, 0, NULL, NULL, NULL, NULL
        )
    );

    return SCIP_OKAY;
}

SCIP_RETCODE SCIPincludeBranchruleMISLearned(
    SCIP* scip,
    SCIP_VAR** vertexvars,
    int nvertices,
    const MISFeatureGraph* graph,
    const MISBranchingModel* model,
    MISLearnedBranchingDecisionCallback callback,
    void* userdata
)
{
    MISLearnedBranchruleData* data;
    SCIP_BRANCHRULE* branchrule;
    SCIP_RETCODE retcode;
    int v;

    if( scip == NULL || nvertices < 0 )
        return SCIP_INVALIDDATA;

    /* Accept either an empty configuration (plugin copy)
     * or a complete configuration.
     */
    if( nvertices == 0 )
    {
        if( vertexvars != NULL || graph != NULL || model != NULL
            || callback != NULL || userdata != NULL )
            return SCIP_INVALIDDATA;
    }
    else
    {
        if( vertexvars == NULL || graph == NULL
            || graph->number_of_vertices != nvertices
            || !MISbranchingModelIsValid(model) )
            return SCIP_INVALIDDATA;

        for( v = 0; v < nvertices; ++v )
        {
            if( vertexvars[v] == NULL )
                return SCIP_INVALIDDATA;
        }
    }

    data = NULL;

    SCIP_CALL( SCIPallocBlockMemory(scip, &data) );

    data->graph = graph;
    data->model = model;
    data->callback = callback;
    data->userdata = userdata;
    data->sbcandidates = 0;
    data->featureclock = NULL;
    data->modelclock = NULL;
    data->featureseconds = 0.0;
    data->modelseconds = 0.0;

    retcode = MISvertexMappingCreate(
        scip, &data->mapping, vertexvars, nvertices
    );

    if( retcode != SCIP_OKAY )
        goto TERMINATE;

    retcode = SCIPincludeBranchruleBasic(
        scip,
        &branchrule,
        "mislearned",
        "MIS branching with a learned candidate ranking model",
        1000000,
        -1,
        1.0,
        (SCIP_BRANCHRULEDATA*)data
    );

    if( retcode != SCIP_OKAY )
        goto TERMINATE;

    /*
     * From here on, the branching rule owns data: on failure, SCIP frees
     * it through BRANCHFREE, so we must not jump to TERMINATE anymore.
     */
    SCIP_CALL( SCIPsetBranchruleFree(scip, branchrule, branchFreeMISLearned) );
    SCIP_CALL( SCIPsetBranchruleCopy(scip, branchrule, branchCopyMISLearned) );
    SCIP_CALL( SCIPsetBranchruleInit(scip, branchrule, branchInitMISLearned) );
    SCIP_CALL( SCIPsetBranchruleExit(scip, branchrule, branchExitMISLearned) );
    SCIP_CALL( SCIPsetBranchruleExecLp(
        scip, branchrule, branchExecLpMISLearned
    ) );

    return SCIP_OKAY;

TERMINATE:

    MISvertexMappingFree(scip, &data->mapping);
    SCIPfreeBlockMemory(scip, &data);

    return retcode;
}

SCIP_RETCODE SCIPsetMISLearnedStrongBranching(
    SCIP* scip,
    int candidates
)
{
    SCIP_BRANCHRULE* branchrule;
    MISLearnedBranchruleData* data;

    if( scip == NULL || candidates < 0 )
        return SCIP_INVALIDDATA;

    branchrule = SCIPfindBranchrule(scip, "mislearned");

    if( branchrule == NULL )
        return SCIP_INVALIDDATA;

    data = (MISLearnedBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    data->sbcandidates = candidates;

    return SCIP_OKAY;
}

SCIP_RETCODE SCIPgetMISLearnedTimes(
    SCIP* scip,
    SCIP_Real* featureseconds,
    SCIP_Real* modelseconds
)
{
    SCIP_BRANCHRULE* branchrule;
    MISLearnedBranchruleData* data;

    if( scip == NULL || featureseconds == NULL || modelseconds == NULL )
        return SCIP_INVALIDDATA;

    branchrule = SCIPfindBranchrule(scip, "mislearned");

    if( branchrule == NULL )
        return SCIP_INVALIDDATA;

    data = (MISLearnedBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    /* During or right after a solve, the clocks; once it ended, the totals. */
    *featureseconds = data->featureclock != NULL
        ? SCIPgetClockTime(scip, data->featureclock) : data->featureseconds;
    *modelseconds = data->modelclock != NULL
        ? SCIPgetClockTime(scip, data->modelclock) : data->modelseconds;

    return SCIP_OKAY;
}
