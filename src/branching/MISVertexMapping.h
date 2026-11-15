#ifndef LEARN2MILP_MIS_VERTEX_MAPPING_H
#define LEARN2MILP_MIS_VERTEX_MAPPING_H

/*
 * Internal helper shared by the MIS branching rules: the link between
 * the vertices of the graph and the variables SCIP branches on.
 *
 * A branching rule embeds one MISVertexMapping in its data and forwards
 * its lifecycle callbacks:
 *
 *   inclusion      MISvertexMappingCreate()
 *   BRANCHINIT     MISvertexMappingInit()
 *   BRANCHEXECLP   MISvertexMappingResolveColumns(), then the queries
 *   BRANCHEXIT     MISvertexMappingExit()
 *   BRANCHFREE     MISvertexMappingFree()
 */

#include <scip/scip.h>

#include "branching/MISStrongBranchingCollector.h"

typedef struct
{
    /*
     * Original MIS variables, indexed by vertex id.
     *
     * These pointers are NOT captured: the original problem owns them,
     * and SCIP frees the problem before the plugins (stage FREE), where
     * SCIPreleaseVar() is no longer allowed. A rule is therefore bound
     * to the problem that was current when it was included.
     */
    SCIP_VAR** vertexvars;
    int nvertices;

    /*
     * Transformed counterparts of vertexvars, indexed by vertex id.
     * Captured in BRANCHINIT and released in BRANCHEXIT, i.e. for the
     * lifetime of the transformed problem.
     */
    SCIP_VAR** vertextransvars;

    /* LP column mapping, built lazily at the first LP branching call:
     * entry i says that column transformedvars[i] carries vertex
     * vertexids[i], as x_v = column or, if complemented, 1 - column. */
    SCIP_VAR** transformedvars;
    int* vertexids;
    int* complemented;
    int ntransformedvars;
    int mapping_capacity;
    SCIP_Bool mapping_initialized;

} MISVertexMapping;

/* Stores the vertex variables (copies the array, not the variables). */
SCIP_RETCODE MISvertexMappingCreate(
    SCIP* scip,
    MISVertexMapping* mapping,
    SCIP_VAR** vertexvars,
    int nvertices
);

/* Frees what MISvertexMappingCreate() allocated. */
void MISvertexMappingFree(
    SCIP* scip,
    MISVertexMapping* mapping
);

/* Captures the transformed vertex variables. */
SCIP_RETCODE MISvertexMappingInit(
    SCIP* scip,
    MISVertexMapping* mapping
);

/* Releases the transformed vertex variables and the column mapping. */
SCIP_RETCODE MISvertexMappingExit(
    SCIP* scip,
    MISVertexMapping* mapping
);

/* Builds the column mapping if it does not exist yet. */
SCIP_RETCODE MISvertexMappingResolveColumns(
    SCIP* scip,
    MISVertexMapping* mapping
);

/*
 * Returns the mapping index of an LP column, or -1 if the column does not
 * correspond to exactly one MIS vertex.
 */
int MISvertexMappingFind(
    const MISVertexMapping* mapping,
    SCIP_VAR* columnvar
);

/* Classifies every MIS vertex at the current node (nvertices entries). */
void MISvertexMappingFillStates(
    SCIP* scip,
    const MISVertexMapping* mapping,
    MISVertexState* states
);

/*
 * Reads the statistics SCIP keeps on the LP column of a vertex, expressed
 * on the vertex variable x_v: if complemented is non-zero (x_v = 1 -
 * column), the two directions are swapped.
 */
void MISvertexMappingFillHistory(
    SCIP* scip,
    SCIP_VAR* column,
    int complemented,
    MISCandidateHistory* history
);

#endif
