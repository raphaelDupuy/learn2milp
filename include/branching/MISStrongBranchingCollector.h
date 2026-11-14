
#ifndef LEARN2MILP_MIS_STRONG_BRANCHING_COLLECTOR_H
#define LEARN2MILP_MIS_STRONG_BRANCHING_COLLECTOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum
{
    MIS_SB_NOT_EVALUATED = 0,
    MIS_SB_EVALUATED,
    MIS_SB_REUSED,
    MIS_SB_LP_ERROR,
    MIS_SB_PARTIALLY_EVALUATED
} MISStrongBranchingEvaluationStatus;

/*
 * State of one MIS vertex variable x_v at a node (local bounds and LP
 * value). The numeric values are stable: they are part of the dataset.
 */
typedef enum
{
    MIS_VERTEX_FIXED_ZERO = 0,   /* local bounds fix x_v = 0 */
    MIS_VERTEX_FIXED_ONE = 1,    /* local bounds fix x_v = 1 */
    MIS_VERTEX_FREE_LP_ZERO = 2, /* not fixed, LP value 0 */
    MIS_VERTEX_FREE_LP_ONE = 3,  /* not fixed, LP value 1 */
    MIS_VERTEX_FRACTIONAL = 4,   /* not fixed, fractional LP value */
    MIS_VERTEX_UNKNOWN = 5       /* multi-aggregated by presolving */
} MISVertexState;

typedef struct
{
    double bound;
    double gain;

    int bound_valid;
    int gain_valid;

    int infeasible;
    int conflict;
    int skipped;
} MISStrongBranchingDirection;

/*
 * One branching candidate at one node.
 *
 * Conventions:
 * - Objective quantities (strong branching bounds, gains, and the sample's
 *   lp_objective, primal_bound and dual_bound) all live in SCIP's internal
 *   transformed space, which is a MINIMIZATION: dual_bound <= lp_objective
 *   <= primal_bound, and the MIS maximum appears negated. Use
 *   SCIPretransformObj() to map a value back to the original objective.
 * - Variable quantities are expressed on the MIS vertex variable x_v: if
 *   presolving replaced x_v by 1 - column, lp_value, bounds, objective
 *   coefficient and reduced cost are mirrored and down/up are swapped, so
 *   "down" always means x_v <= 0 and "up" means x_v >= 1.
 * - fractionality is min(f, 1 - f), f being the fractional part.
 * - infeasible is SCIP's strong branching flag: the LP of that branch is
 *   infeasible or its objective reaches the cutoff bound, i.e. the branch
 *   cannot contain a solution better than the incumbent. Such a branch
 *   has gain_valid = 0, and several candidates then often share the same
 *   gain and score.
 * - gain = MAX(bound, lpobjval) - lpobjval, and score is
 *   SCIPgetBranchScoreMultiple() on (down.gain, up.gain), exactly as in
 *   SCIP's vanillafullstrong rule.
 * - expert_rank is the number of evaluated candidates with a strictly
 *   higher score (0 = best, ties share a rank); -1 if not evaluated.
 * - expert_selected marks the first candidate with the best score, i.e.
 *   the variable the rule branches on.
 * - MIS_SB_REUSED and MIS_SB_PARTIALLY_EVALUATED are reserved, not set yet.
 */
typedef struct
{
    int vertex_id;
    int candidate_position;

    double lp_value;
    double fractionality;
    double objective_coefficient;
    double local_lower_bound;
    double local_upper_bound;

    int local_bounds_valid;
    double reduced_cost;

    int lp_value_valid;
    int reduced_cost_valid;

    MISStrongBranchingEvaluationStatus evaluation_status;

    MISStrongBranchingDirection down;
    MISStrongBranchingDirection up;

    double score;
    int score_valid;

    /* -1 means that no expert rank is available. */
    int expert_rank;
    int expert_selected;

    int evaluation_order;

    /* LP objective used as reference for the recorded gains. */
    double gain_reference_lp_objective;
    int gain_reference_lp_objective_valid;

} MISStrongBranchingCandidate;

/*
 * One branching decision (one node).
 *
 * - primal_bound is SCIP's global upper bound and dual_bound its global
 *   lower bound, in the same internal space as lp_objective.
 * - node_cutoff is 1 when strong branching proved that the node cannot
 *   contain a solution better than the incumbent (both branches of the
 *   selected candidate are infeasible or cut off). The rule then prunes
 *   the node instead of branching, and the expert selection is arbitrary:
 *   every candidate with two cut-off branches has the same score.
 * - vertex_states has number_of_vertices entries, indexed by vertex id.
 *   The vertices in state MIS_VERTEX_FRACTIONAL are exactly the candidates.
 *   The array is only valid during the callback.
 */
typedef struct
{
    long long node_id;
    long long parent_node_id;
    int parent_node_valid;

    int node_depth;

    double lp_objective;
    double primal_bound;
    double dual_bound;

    int lp_objective_valid;
    int primal_bound_valid;
    int dual_bound_valid;

    int open_nodes;
    long long processed_nodes;

    int number_of_candidates;
    int number_of_priority_candidates;
    int expert_selected_vertex_id;
    int expert_selection_valid;

    int node_cutoff;

    const MISVertexState* vertex_states;
    int number_of_vertices;
} MISStrongBranchingSample;

typedef void (*MISStrongBranchingSampleCallback)(
    void* userdata,
    const MISStrongBranchingSample* sample,
    const MISStrongBranchingCandidate* candidates,
    size_t number_of_candidates
);

#ifdef __cplusplus
}
#endif

#endif
