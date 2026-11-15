#include "branching/MISLearnedBranchrule.h"

#include "MISVertexMapping.h"

#include <scip/scip.h>
#include <scip/scip_branch.h>
#include <scip/scip_mem.h>
#include <scip/scip_var.h>
#include <scip/pub_var.h>

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

} MISLearnedBranchruleData;

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
    int* best
)
{
    MISBranchingNode node;
    SCIP_Real upperbound;
    int c;

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

    /* First candidate with the highest score. */
    *best = 0;

    for( c = 0; c < ncands; ++c )
    {
        scores[c] = MISevaluateBranchingModel(
            data->model,
            features + (size_t)c * MIS_BRANCHING_FEATURE_COUNT,
            modelwork
        );

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
    SCIP_RETCODE retcode;
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
        scip, &modelwork, MISbranchingModelWorkSize(data->model)
    ) );

    retcode = scoreCandidates(
        scip, data, cands, ncands, states, vertices, lpvalues, histories,
        features, scores, featurework, modelwork, &best
    );

    if( retcode == SCIP_OKAY )
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

    return SCIP_OKAY;
}

static
SCIP_DECL_BRANCHINIT(branchInitMISLearned)
{
    MISLearnedBranchruleData* data;

    data = (MISLearnedBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    SCIP_CALL( MISvertexMappingInit(scip, &data->mapping) );

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
