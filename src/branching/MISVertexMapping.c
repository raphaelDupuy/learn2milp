#include "MISVertexMapping.h"

#include <scip/scip_mem.h>
#include <scip/scip_numerics.h>
#include <scip/scip_var.h>
#include <scip/pub_var.h>

SCIP_RETCODE MISvertexMappingCreate(
    SCIP* scip,
    MISVertexMapping* mapping,
    SCIP_VAR** vertexvars,
    int nvertices
)
{
    mapping->vertexvars = NULL;
    mapping->nvertices = nvertices;

    mapping->vertextransvars = NULL;
    mapping->transformedvars = NULL;
    mapping->vertexids = NULL;
    mapping->complemented = NULL;
    mapping->ntransformedvars = 0;
    mapping->mapping_capacity = 0;
    mapping->mapping_initialized = FALSE;

    if( nvertices > 0 )
    {
        SCIP_CALL( SCIPduplicateBlockMemoryArray(
            scip, &mapping->vertexvars, vertexvars, nvertices
        ) );
    }

    return SCIP_OKAY;
}

void MISvertexMappingFree(
    SCIP* scip,
    MISVertexMapping* mapping
)
{
    /* MISvertexMappingExit() has already released the transformed
     * variables. */
    assert(mapping->vertextransvars == NULL);

    /* Free the original-variable array (the pointers are not captured). */
    if( mapping->vertexvars != NULL )
    {
        SCIPfreeBlockMemoryArray(
            scip, &mapping->vertexvars, mapping->nvertices
        );
    }
}

SCIP_RETCODE MISvertexMappingInit(
    SCIP* scip,
    MISVertexMapping* mapping
)
{
    SCIP_RETCODE retcode;
    int v;

    assert(mapping->vertextransvars == NULL);
    assert(mapping->transformedvars == NULL);
    assert(mapping->vertexids == NULL);
    assert(mapping->complemented == NULL);

    mapping->ntransformedvars = 0;
    mapping->mapping_capacity = 0;
    mapping->mapping_initialized = FALSE;

    if( mapping->nvertices == 0 )
        return SCIP_OKAY;

    /* Capture the transformed vertex variables for the lifetime of the
     * transformed problem. */
    SCIP_CALL( SCIPallocClearBlockMemoryArray(
        scip, &mapping->vertextransvars, mapping->nvertices
    ) );

    for( v = 0; v < mapping->nvertices; ++v )
    {
        SCIP_VAR* transvar = NULL;

        retcode = SCIPgetTransformedVar(
            scip, mapping->vertexvars[v], &transvar
        );

        if( retcode == SCIP_OKAY && transvar == NULL )
            retcode = SCIP_INVALIDDATA;

        if( retcode == SCIP_OKAY )
            retcode = SCIPcaptureVar(scip, transvar);

        if( retcode != SCIP_OKAY )
            goto TERMINATE;

        mapping->vertextransvars[v] = transvar;
    }

    return SCIP_OKAY;

TERMINATE:

    /* Release only the variables captured so far (others are NULL). */
    for( v = 0; v < mapping->nvertices; ++v )
    {
        if( mapping->vertextransvars[v] != NULL )
            (void)SCIPreleaseVar(scip, &mapping->vertextransvars[v]);
    }

    SCIPfreeBlockMemoryArray(
        scip, &mapping->vertextransvars, mapping->nvertices
    );

    return retcode;
}

SCIP_RETCODE MISvertexMappingExit(
    SCIP* scip,
    MISVertexMapping* mapping
)
{
    int v;

    /* Free the column mapping arrays. */
    SCIPfreeBlockMemoryArrayNull(
        scip, &mapping->transformedvars, mapping->mapping_capacity
    );

    SCIPfreeBlockMemoryArrayNull(
        scip, &mapping->vertexids, mapping->mapping_capacity
    );

    SCIPfreeBlockMemoryArrayNull(
        scip, &mapping->complemented, mapping->mapping_capacity
    );

    mapping->ntransformedvars = 0;
    mapping->mapping_capacity = 0;
    mapping->mapping_initialized = FALSE;

    /* Release the transformed vertex variables captured at init. */
    if( mapping->vertextransvars != NULL )
    {
        for( v = 0; v < mapping->nvertices; ++v )
        {
            if( mapping->vertextransvars[v] != NULL )
            {
                SCIP_CALL( SCIPreleaseVar(
                    scip, &mapping->vertextransvars[v]
                ) );
            }
        }

        SCIPfreeBlockMemoryArray(
            scip, &mapping->vertextransvars, mapping->nvertices
        );
    }

    return SCIP_OKAY;
}

/*
 * Resolves a transformed MIS variable through presolving aggregations
 * and negations down to its LP column.
 *
 * On success:
 * - columnvar is the final LP column variable;
 * - complemented is FALSE if x_v = columnvar;
 * - complemented is TRUE if x_v = 1 - columnvar.
 *
 * If the transformation is unsupported or does not lead to a
 * COLUMN variable, supported is set to FALSE.
 */
static
SCIP_RETCODE resolveMISVertexVariable(
    SCIP* scip,
    SCIP_VAR* startvar,
    SCIP_VAR** columnvar,
    int* complemented,
    SCIP_Bool* supported
)
{
    SCIP_VAR* var;
    SCIP_VAR* nextvar;
    SCIP_VARSTATUS status;
    SCIP_Real scalar;
    SCIP_Real constant;
    SCIP_Real nextscalar;
    SCIP_Real nextconstant;

    if( scip == NULL || startvar == NULL || columnvar == NULL
        || complemented == NULL || supported == NULL )
        return SCIP_INVALIDDATA;

    *columnvar = NULL;
    *complemented = FALSE;
    *supported = FALSE;

    var = startvar;

    /* Current relation: x_v = scalar * var + constant. */
    scalar = 1.0;
    constant = 0.0;

    while( TRUE )
    {
        status = SCIPvarGetStatus(var);

        switch( status )
        {
        case SCIP_VARSTATUS_COLUMN:
            if( SCIPisEQ(scip, scalar, 1.0)
                && SCIPisEQ(scip, constant, 0.0) )
            {
                *columnvar = var;
                *complemented = FALSE;
                *supported = TRUE;
            }
            else if( SCIPisEQ(scip, scalar, -1.0)
                && SCIPisEQ(scip, constant, 1.0) )
            {
                *columnvar = var;
                *complemented = TRUE;
                *supported = TRUE;
            }

            return SCIP_OKAY;

        case SCIP_VARSTATUS_AGGREGATED:
            nextvar = SCIPvarGetAggrVar(var);
            nextscalar = SCIPvarGetAggrScalar(var);
            nextconstant = SCIPvarGetAggrConstant(var);
            break;

        case SCIP_VARSTATUS_NEGATED:
            nextvar = SCIPvarGetNegationVar(var);

            if( nextvar == NULL )
                return SCIP_INVALIDDATA;

            nextscalar = -1.0;
            nextconstant = SCIPvarGetNegationConstant(var);
            break;

        default:
            /* LOOSE, FIXED, MULTAGGR or unsupported status. */
            return SCIP_OKAY;
        }

        if( nextvar == NULL )
            return SCIP_INVALIDDATA;

        /* Compose x_v = scalar * (nextscalar * nextvar
         * + nextconstant) + constant. */
        constant += scalar * nextconstant;
        scalar *= nextscalar;

        var = nextvar;
    }
}

SCIP_RETCODE MISvertexMappingResolveColumns(
    SCIP* scip,
    MISVertexMapping* mapping
)
{
    SCIP_VAR* columnvar;
    SCIP_Bool supported;
    SCIP_RETCODE retcode;
    int complemented;
    int v;

    if( scip == NULL || mapping == NULL )
        return SCIP_INVALIDDATA;

    if( mapping->mapping_initialized )
        return SCIP_OKAY;

    mapping->ntransformedvars = 0;
    mapping->mapping_capacity = mapping->nvertices;

    if( mapping->nvertices == 0 )
    {
        mapping->mapping_initialized = TRUE;
        return SCIP_OKAY;
    }

    retcode = SCIPallocBlockMemoryArray(
        scip, &mapping->transformedvars, mapping->mapping_capacity
    );

    if( retcode != SCIP_OKAY )
        goto TERMINATE;

    retcode = SCIPallocBlockMemoryArray(
        scip, &mapping->vertexids, mapping->mapping_capacity
    );

    if( retcode != SCIP_OKAY )
        goto TERMINATE;

    retcode = SCIPallocBlockMemoryArray(
        scip, &mapping->complemented, mapping->mapping_capacity
    );

    if( retcode != SCIP_OKAY )
        goto TERMINATE;

    for( v = 0; v < mapping->nvertices; ++v )
    {
        retcode = resolveMISVertexVariable(
            scip,
            mapping->vertextransvars[v],
            &columnvar,
            &complemented,
            &supported
        );

        if( retcode != SCIP_OKAY )
            goto TERMINATE;

        if( !supported )
            continue;

        mapping->transformedvars[mapping->ntransformedvars] = columnvar;
        mapping->vertexids[mapping->ntransformedvars] = v;
        mapping->complemented[mapping->ntransformedvars] = complemented;

        ++mapping->ntransformedvars;
    }

    mapping->mapping_initialized = TRUE;

    return SCIP_OKAY;

TERMINATE:

    SCIPfreeBlockMemoryArrayNull(
        scip, &mapping->transformedvars, mapping->mapping_capacity
    );
    SCIPfreeBlockMemoryArrayNull(
        scip, &mapping->vertexids, mapping->mapping_capacity
    );
    SCIPfreeBlockMemoryArrayNull(
        scip, &mapping->complemented, mapping->mapping_capacity
    );

    mapping->mapping_capacity = 0;
    mapping->ntransformedvars = 0;

    return retcode;
}

/*
 * Returns the mapping index of an LP column, or -1 if the column does not
 * correspond to exactly one MIS vertex.
 */
int MISvertexMappingFind(
    const MISVertexMapping* mapping,
    SCIP_VAR* columnvar
)
{
    int found = -1;
    int index;

    for( index = 0; index < mapping->ntransformedvars; ++index )
    {
        if( mapping->transformedvars[index] != columnvar )
            continue;

        if( found >= 0 )
            return -1;

        found = index;
    }

    return found;
}

/*
 * Classifies every MIS vertex at the current node from the local bounds
 * and the LP value of its transformed variable. Fixed, aggregated and
 * negated variables are handled by SCIP; multi-aggregated ones, whose
 * local bounds are not maintained, are reported as unknown, except a
 * multi-aggregation over no variable, which is a fixing.
 */
void MISvertexMappingFillStates(
    SCIP* scip,
    const MISVertexMapping* mapping,
    MISVertexState* states
)
{
    int v;

    for( v = 0; v < mapping->nvertices; ++v )
    {
        SCIP_VAR* var = mapping->vertextransvars[v];
        SCIP_Real lb;
        SCIP_Real ub;
        SCIP_Real val;

        if( SCIPvarGetStatus(var) == SCIP_VARSTATUS_MULTAGGR )
        {
            /* A multi-aggregation over no variable fixes the vertex to
             * its constant: presolve with clique lifting can make one. */
            if( SCIPvarGetMultaggrNVars(var) == 0 )
                states[v] = (SCIPvarGetMultaggrConstant(var) < 0.5)
                    ? MIS_VERTEX_FIXED_ZERO : MIS_VERTEX_FIXED_ONE;
            else
                states[v] = MIS_VERTEX_UNKNOWN;
            continue;
        }

        lb = SCIPvarGetLbLocal(var);
        ub = SCIPvarGetUbLocal(var);
        val = SCIPvarGetLPSol(var);

        if( SCIPisFeasEQ(scip, lb, ub) )
            states[v] = (ub < 0.5) ? MIS_VERTEX_FIXED_ZERO : MIS_VERTEX_FIXED_ONE;
        else if( !SCIPisFeasIntegral(scip, val) )
            states[v] = MIS_VERTEX_FRACTIONAL;
        else
            states[v] = (val < 0.5) ? MIS_VERTEX_FREE_LP_ZERO : MIS_VERTEX_FREE_LP_ONE;
    }
}

void MISvertexMappingFillHistory(
    SCIP* scip,
    SCIP_VAR* column,
    int complemented,
    MISCandidateHistory* history
)
{
    const SCIP_BRANCHDIR down =
        complemented ? SCIP_BRANCHDIR_UPWARDS : SCIP_BRANCHDIR_DOWNWARDS;
    const SCIP_BRANCHDIR up =
        complemented ? SCIP_BRANCHDIR_DOWNWARDS : SCIP_BRANCHDIR_UPWARDS;

    history->pseudocost_down = SCIPgetVarPseudocost(scip, column, down);
    history->pseudocost_up = SCIPgetVarPseudocost(scip, column, up);
    history->pseudocost_count_down =
        SCIPgetVarPseudocostCount(scip, column, down);
    history->pseudocost_count_up =
        SCIPgetVarPseudocostCount(scip, column, up);
    history->inferences_down = SCIPgetVarAvgInferences(scip, column, down);
    history->inferences_up = SCIPgetVarAvgInferences(scip, column, up);
    history->cutoffs_down = SCIPgetVarAvgCutoffs(scip, column, down);
    history->cutoffs_up = SCIPgetVarAvgCutoffs(scip, column, up);
    history->conflict_score = SCIPgetVarConflictScore(scip, column);
}
