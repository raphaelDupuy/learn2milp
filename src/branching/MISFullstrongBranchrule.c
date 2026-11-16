
#include "branching/MISFullstrongBranchrule.h"

#include "MISVertexMapping.h"

#include <scip/scip.h>
#include <scip/scip_branch.h>
#include <scip/scip_mem.h>
#include <scip/scip_numerics.h>
#include <scip/scip_var.h>
#include <scip/pub_var.h>
#include <limits.h>
#include <math.h>

#define MIS_FULLSTRONG_NAME "misfullstrong"

typedef struct
{
    /* Link between graph vertices and SCIP variables. */
    MISVertexMapping mapping;

    /* User callback and its context. */
    MISStrongBranchingSampleCallback callback;
    void* userdata;

    /* Optional policy choosing the branching variable (NULL: expert). */
    MISBranchingSelector selector;
    void* selectoruserdata;

    /* Whether a node that strong branching proves prunable is pruned. */
    SCIP_Bool prune;

} MISFullstrongBranchruleData;

static
void initializeDirection(
    MISStrongBranchingDirection* direction
)
{
    direction->bound = 0.0;
    direction->gain = 0.0;
    direction->bound_valid = FALSE;
    direction->gain_valid = FALSE;
    direction->infeasible = FALSE;
    direction->conflict = FALSE;
    direction->skipped = TRUE;
}

/*
 * Stores the result of one strong branching direction, expressed on the
 * LP column. gain follows vanillafullstrong: MAX(bound, lpobjval) - lpobjval.
 */
static
void storeDirection(
    MISStrongBranchingDirection* direction,
    SCIP_Real bound,
    SCIP_Real lpobjval,
    SCIP_Bool valid,
    SCIP_Bool infeasible,
    SCIP_Bool conflict
)
{
    direction->bound = bound;
    direction->gain = MAX(bound, lpobjval) - lpobjval;
    direction->bound_valid = valid;
    direction->gain_valid = valid && !infeasible;
    direction->infeasible = infeasible;
    direction->conflict = conflict;
    direction->skipped = FALSE;
}

/*
 * Runs vanilla full strong branching on the LP candidates and fills the
 * per-candidate records.
 *
 * Reference: runVanillaStrongBranching() in SCIP 10.1.0
 * src/scip/branch_vanillafullstrong.c (ZIB, Apache License 2.0), with
 * idempotent = TRUE and scoreall = TRUE. Differences: every outcome of
 * every candidate is recorded instead of being kept in local variables.
 *
 * On return, *bestcand is the first candidate with the highest score
 * (strict comparison, as in the reference), or 0 if none was scored.
 */
static
SCIP_RETCODE runMISStrongBranching(
    SCIP* scip,
    SCIP_VAR** cands,
    int ncands,
    MISStrongBranchingCandidate* records,
    SCIP_Real lpobjval,
    int* bestcand,
    SCIP_Real* bestdown,
    SCIP_Real* bestup,
    SCIP_Bool* bestdownvalid,
    SCIP_Bool* bestupvalid,
    SCIP_Real* provedbound
)
{
    SCIP_Real bestscore;
    int c;

    *provedbound = lpobjval;
    *bestcand = 0;
    *bestdown = lpobjval;
    *bestup = lpobjval;
    *bestdownvalid = FALSE;
    *bestupvalid = FALSE;
    bestscore = -SCIPinfinity(scip);

    /* With scoreall = TRUE, only a stopping SCIP prevents evaluation. */
    if( SCIPisStopped(scip) )
        return SCIP_OKAY;

    assert(SCIPgetLPSolstat(scip) == SCIP_LPSOLSTAT_OPTIMAL);

    SCIP_CALL( SCIPstartStrongbranch(scip, FALSE) );

    for( c = 0; c < ncands; ++c )
    {
        MISStrongBranchingCandidate* record = &records[c];
        SCIP_VAR* var = cands[c];
        SCIP_Real val;
        SCIP_Bool integral;
        SCIP_Real down, up;
        SCIP_Bool downvalid, upvalid;
        SCIP_Bool downinf, upinf;
        SCIP_Bool downconflict, upconflict;
        SCIP_Bool lperror;
        SCIP_Real gains[3];
        SCIP_Real score;

        val = SCIPvarGetLPSol(var);
        integral = SCIPisFeasIntegral(scip, val);

        down = -SCIPinfinity(scip);
        up = -SCIPinfinity(scip);

        record->evaluation_order = c;

        if( integral )
        {
            SCIP_CALL( SCIPgetVarStrongbranchInt(scip, var, INT_MAX, TRUE,
                &down, &up, &downvalid, &upvalid, &downinf, &upinf,
                &downconflict, &upconflict, &lperror) );
        }
        else
        {
            SCIP_CALL( SCIPgetVarStrongbranchFrac(scip, var, INT_MAX, TRUE,
                &down, &up, &downvalid, &upvalid, &downinf, &upinf,
                &downconflict, &upconflict, &lperror) );
        }

        /* Same as the reference: stop at the first LP error. The remaining
         * candidates stay NOT_EVALUATED. */
        if( lperror )
        {
            record->evaluation_status = MIS_SB_LP_ERROR;
            break;
        }

        storeDirection(&record->down, down, lpobjval, downvalid, downinf, downconflict);
        storeDirection(&record->up, up, lpobjval, upvalid, upinf, upconflict);

        gains[0] = record->down.gain;
        gains[1] = record->up.gain;
        gains[2] = 0.0;
        score = SCIPgetBranchScoreMultiple(scip, var, integral ? 3 : 2, gains);

        record->evaluation_status = MIS_SB_EVALUATED;
        record->score = score;
        record->score_valid = TRUE;

        if( score > bestscore )
        {
            *bestcand = c;
            *bestdown = MAX(down, lpobjval);
            *bestup = MAX(up, lpobjval);
            *bestdownvalid = downvalid;
            *bestupvalid = upvalid;
            bestscore = score;
        }
    }

    SCIP_CALL( SCIPendStrongbranch(scip) );

    if( *bestdownvalid && *bestupvalid
        && !SCIPisFeasIntegral(scip, SCIPvarGetLPSol(cands[*bestcand])) )
    {
        *provedbound = MAX(*provedbound, MIN(*bestdown, *bestup));
    }

    return SCIP_OKAY;
}

/*
 * Converts a record from LP-column space to MIS-vertex space when
 * x_v = 1 - column: values, bounds, objective and reduced cost are
 * mirrored and the down/up directions are swapped.
 */
static
void complementRecord(
    MISStrongBranchingCandidate* record
)
{
    MISStrongBranchingDirection down;
    double lb;

    record->lp_value = 1.0 - record->lp_value;
    record->objective_coefficient = -record->objective_coefficient;
    record->reduced_cost = -record->reduced_cost;

    lb = record->local_lower_bound;
    record->local_lower_bound = 1.0 - record->local_upper_bound;
    record->local_upper_bound = 1.0 - lb;

    down = record->down;
    record->down = record->up;
    record->up = down;
}

/*
 * Expert rank = number of evaluated candidates with a strictly higher
 * score (0 = best). Tied candidates share the same rank.
 */
static
void computeExpertRanks(
    MISStrongBranchingCandidate* records,
    int ncands
)
{
    int i;
    int j;

    for( i = 0; i < ncands; ++i )
    {
        if( !records[i].score_valid )
            continue;

        records[i].expert_rank = 0;

        for( j = 0; j < ncands; ++j )
        {
            if( records[j].score_valid && records[j].score > records[i].score )
                ++records[i].expert_rank;
        }
    }
}

static
SCIP_DECL_BRANCHEXECLP(branchExecLpMISFullstrong)
{
    MISFullstrongBranchruleData* data;
    MISStrongBranchingCandidate* records;
    MISStrongBranchingSample sample;
    MISVertexState* vertexstates;
    SCIP_VAR** scipcands;
    SCIP_VAR** lpcands;
    SCIP_NODE* node;
    SCIP_NODE* parent;
    SCIP_Real lpobjval;
    SCIP_Real bestdown;
    SCIP_Real bestup;
    SCIP_Real provedbound;
    SCIP_Bool bestdownvalid;
    SCIP_Bool bestupvalid;
    SCIP_Bool anyscored;
    SCIP_Bool cutoff;
    int nlpcands;
    int npriocands;
    int bestcand;
    int branchcand;
    int c;

    (void)allowaddcons;

    data = (MISFullstrongBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);
    assert(result != NULL);

    *result = SCIP_DIDNOTRUN;

    /* An empty configuration is used when copying the plugin. */
    if( data->mapping.nvertices == 0 )
        return SCIP_OKAY;

    /* Initialize the original-to-transformed variable mapping. */
    SCIP_CALL( MISvertexMappingResolveColumns(scip, &data->mapping) );

    /* Retrieve the current LP branching candidates. */
    SCIP_CALL( SCIPgetLPBranchCands(
        scip, &scipcands, NULL, NULL, &nlpcands, &npriocands, NULL
    ) );

    if( nlpcands == 0 )
        return SCIP_OKAY;

    lpobjval = SCIPgetLPObjval(scip);

    /* Own copy, as in the reference: SCIP's candidate array may be
     * recomputed while strong branching solves LPs. */
    SCIP_CALL( SCIPduplicateBufferArray(scip, &lpcands, scipcands, nlpcands) );
    SCIP_CALL( SCIPallocBufferArray(scip, &records, nlpcands) );

    /* Static candidate data, in LP-column space. */
    for( c = 0; c < nlpcands; ++c )
    {
        MISStrongBranchingCandidate* record = &records[c];
        SCIP_VAR* var = lpcands[c];
        SCIP_Real val = SCIPvarGetLPSol(var);
        SCIP_Real frac = SCIPfeasFrac(scip, val);
        int mapping = MISvertexMappingFind(&data->mapping, var);

        /* Every LP candidate must map to exactly one MIS vertex. */
        if( mapping < 0 )
        {
            SCIPfreeBufferArray(scip, &records);
            SCIPfreeBufferArray(scip, &lpcands);
            return SCIP_INVALIDDATA;
        }

        record->vertex_id = data->mapping.vertexids[mapping];
        record->candidate_position = c;

        record->lp_value = val;
        record->lp_value_valid = TRUE;
        record->fractionality = MIN(frac, 1.0 - frac);
        record->objective_coefficient = SCIPvarGetObj(var);
        record->local_lower_bound = SCIPvarGetLbLocal(var);
        record->local_upper_bound = SCIPvarGetUbLocal(var);
        record->local_bounds_valid = TRUE;
        record->reduced_cost = SCIPgetVarRedcost(scip, var);
        record->reduced_cost_valid = TRUE;

        record->evaluation_status = MIS_SB_NOT_EVALUATED;
        initializeDirection(&record->down);
        initializeDirection(&record->up);

        record->score = 0.0;
        record->score_valid = FALSE;
        record->expert_rank = -1;
        record->expert_selected = FALSE;
        record->evaluation_order = -1;

        record->gain_reference_lp_objective = lpobjval;
        record->gain_reference_lp_objective_valid = TRUE;

        /* Already in MIS-vertex space, unlike the fields above. */
        MISvertexMappingFillHistory(
            scip, var, data->mapping.complemented[mapping],
            &record->history
        );
    }

    SCIP_CALL( runMISStrongBranching(
        scip, lpcands, nlpcands, records, lpobjval,
        &bestcand, &bestdown, &bestup, &bestdownvalid, &bestupvalid,
        &provedbound
    ) );

    computeExpertRanks(records, nlpcands);

    anyscored = records[bestcand].score_valid;

    if( anyscored )
        records[bestcand].expert_selected = TRUE;

    /* Express every record in MIS-vertex space (x_v rather than column). */
    for( c = 0; c < nlpcands; ++c )
    {
        if( data->mapping.complemented[MISvertexMappingFind(&data->mapping, lpcands[c])] )
            complementRecord(&records[c]);
    }

    /* Same test as vanillafullstrong: the node is pruned when the best
     * candidate proves a bound that reaches the cutoff bound. */
    cutoff = SCIPisGE(scip, provedbound, SCIPgetCutoffbound(scip));

    /* The expert's choice, unless a selector decides otherwise below. */
    branchcand = bestcand;

    /*
     * No sample is emitted when the solver is stopping (time limit or
     * interrupt): strong branching was skipped or may have been cut
     * short, so there is no reliable expert decision to record.
     */
    if( !SCIPisStopped(scip) )
    {
        /* Node-level sample. */
        node = SCIPgetCurrentNode(scip);
        parent = SCIPnodeGetParent(node);

        sample.node_id = SCIPnodeGetNumber(node);
        sample.parent_node_valid = (parent != NULL);
        sample.parent_node_id = (parent != NULL) ? SCIPnodeGetNumber(parent) : -1;
        sample.node_depth = SCIPnodeGetDepth(node);

        sample.lp_objective = lpobjval;
        sample.lp_objective_valid = TRUE;
        /* Global bounds in the same internal (minimization) space as the LP
         * objective: SCIPgetPrimalbound()/SCIPgetDualbound() would return
         * original-space values, which must not be mixed with lp_objective. */
        sample.primal_bound = SCIPgetUpperbound(scip);
        sample.primal_bound_valid = !SCIPisInfinity(scip, sample.primal_bound);
        sample.dual_bound = SCIPgetLowerbound(scip);
        sample.dual_bound_valid = !SCIPisInfinity(scip, -sample.dual_bound);

        sample.open_nodes = SCIPgetNNodesLeft(scip);
        sample.processed_nodes = SCIPgetNNodes(scip);

        sample.number_of_candidates = nlpcands;
        sample.number_of_priority_candidates = npriocands;
        sample.expert_selection_valid = anyscored;
        sample.expert_selected_vertex_id = anyscored ? records[bestcand].vertex_id : -1;
        sample.node_cutoff = cutoff;

        SCIP_CALL( SCIPallocBufferArray(scip, &vertexstates, data->mapping.nvertices) );
        MISvertexMappingFillStates(scip, &data->mapping, vertexstates);

        sample.vertex_states = vertexstates;
        sample.number_of_vertices = data->mapping.nvertices;

        data->callback(data->userdata, &sample, records, (size_t)nlpcands);

        if( data->selector != NULL && !cutoff )
        {
            const int selected = data->selector(
                data->selectoruserdata, &sample, records, (size_t)nlpcands
            );

            if( selected >= 0 && selected < nlpcands )
                branchcand = selected;
            else if( selected != -1 )
                branchcand = -1;
        }

        SCIPfreeBufferArray(scip, &vertexstates);
    }

    SCIPfreeBufferArray(scip, &records);

    if( branchcand < 0 )
    {
        SCIPfreeBufferArray(scip, &lpcands);
        return SCIP_INVALIDDATA;
    }

    /* Branching, identical to vanillafullstrong with idempotent = TRUE
     * when the expert is followed: no local bound update, no child lower
     * bound update. */
    if( cutoff && data->prune )
    {
        *result = SCIP_CUTOFF;
    }
    else
    {
        SCIP_VAR* var = lpcands[branchcand];

        SCIP_CALL( SCIPbranchVarVal(
            scip, var, SCIPvarGetLPSol(var), NULL, NULL, NULL
        ) );

        *result = SCIP_BRANCHED;
    }

    SCIPfreeBufferArray(scip, &lpcands);

    return SCIP_OKAY;
}

static
SCIP_DECL_BRANCHFREE(branchFreeMISFullstrong)
{
    MISFullstrongBranchruleData* data;

    data = (MISFullstrongBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    MISvertexMappingFree(scip, &data->mapping);

    /* Free the branchrule data. */
    SCIPfreeBlockMemory(scip, &data);

    SCIPbranchruleSetData(branchrule, NULL);

    return SCIP_OKAY;
}

static
SCIP_DECL_BRANCHEXIT(branchExitMISFullstrong)
{
    MISFullstrongBranchruleData* data;

    data = (MISFullstrongBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    SCIP_CALL( MISvertexMappingExit(scip, &data->mapping) );

    return SCIP_OKAY;
}

static
SCIP_DECL_BRANCHINIT(branchInitMISFullstrong)
{
    MISFullstrongBranchruleData* data;

    data = (MISFullstrongBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    SCIP_CALL( MISvertexMappingInit(scip, &data->mapping) );

    return SCIP_OKAY;
}

static
SCIP_DECL_BRANCHCOPY(branchCopyMISFullstrong)
{
    assert(scip != NULL);
    assert(branchrule != NULL);

    SCIP_CALL(
        SCIPincludeBranchruleMISFullstrong(
            scip,
            NULL,
            0,
            NULL,
            NULL
        )
    );

    return SCIP_OKAY;
}

SCIP_RETCODE SCIPincludeBranchruleMISFullstrong(
    SCIP* scip,
    SCIP_VAR** vertexvars,
    int nvertices,
    MISStrongBranchingSampleCallback callback,
    void* userdata
)
{
    MISFullstrongBranchruleData* data;
    SCIP_BRANCHRULE* branchrule;
    SCIP_RETCODE retcode;
    int v;

    if( scip == NULL || nvertices < 0 )
        return SCIP_INVALIDDATA;

    /* Accept either an empty configuration (plugin copy)
     * or a complete configuration (MIS data collection).
     */
    if( nvertices == 0 )
    {
        if( vertexvars != NULL || callback != NULL || userdata != NULL )
            return SCIP_INVALIDDATA;
    }
    else
    {
        if( vertexvars == NULL || callback == NULL )
            return SCIP_INVALIDDATA;

        for( v = 0; v < nvertices; ++v )
        {
            if( vertexvars[v] == NULL )
                return SCIP_INVALIDDATA;
        }
    }

    data = NULL;

    SCIP_CALL( SCIPallocBlockMemory(scip, &data) );

    data->callback = callback;
    data->userdata = userdata;
    data->selector = NULL;
    data->selectoruserdata = NULL;
    data->prune = TRUE;

    /* Copy the original MIS variable pointers (not captured, see
     * MISVertexMapping). */
    retcode = MISvertexMappingCreate(
        scip, &data->mapping, vertexvars, nvertices
    );

    if( retcode != SCIP_OKAY )
        goto TERMINATE;

    /* Include the branching rule. */
    retcode = SCIPincludeBranchruleBasic(
        scip,
        &branchrule,
        MIS_FULLSTRONG_NAME,
        "MIS full strong branching with data collection",
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

    /* Register lifecycle callbacks. */
    SCIP_CALL( SCIPsetBranchruleFree(
        scip, branchrule, branchFreeMISFullstrong
    ) );

    SCIP_CALL( SCIPsetBranchruleCopy(
        scip, branchrule, branchCopyMISFullstrong
    ) );

    SCIP_CALL( SCIPsetBranchruleInit(
        scip, branchrule, branchInitMISFullstrong
    ) );

    SCIP_CALL( SCIPsetBranchruleExit(
        scip, branchrule, branchExitMISFullstrong
    ) );

    /* Register LP execution callback. */
    SCIP_CALL( SCIPsetBranchruleExecLp(
        scip, branchrule, branchExecLpMISFullstrong
    ) );

    return SCIP_OKAY;

TERMINATE:

    MISvertexMappingFree(scip, &data->mapping);
    SCIPfreeBlockMemory(scip, &data);

    return retcode;
}

SCIP_RETCODE SCIPsetMISFullstrongSelector(
    SCIP* scip,
    MISBranchingSelector selector,
    void* userdata
)
{
    SCIP_BRANCHRULE* branchrule;
    MISFullstrongBranchruleData* data;

    if( scip == NULL )
        return SCIP_INVALIDDATA;

    branchrule = SCIPfindBranchrule(scip, MIS_FULLSTRONG_NAME);

    if( branchrule == NULL )
        return SCIP_INVALIDDATA;

    data = (MISFullstrongBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    data->selector = selector;
    data->selectoruserdata = selector != NULL ? userdata : NULL;

    return SCIP_OKAY;
}

SCIP_RETCODE SCIPsetMISFullstrongPruning(
    SCIP* scip,
    SCIP_Bool prune
)
{
    SCIP_BRANCHRULE* branchrule;
    MISFullstrongBranchruleData* data;

    if( scip == NULL )
        return SCIP_INVALIDDATA;

    branchrule = SCIPfindBranchrule(scip, MIS_FULLSTRONG_NAME);

    if( branchrule == NULL )
        return SCIP_INVALIDDATA;

    data = (MISFullstrongBranchruleData*)SCIPbranchruleGetData(branchrule);
    assert(data != NULL);

    data->prune = prune;

    return SCIP_OKAY;
}
