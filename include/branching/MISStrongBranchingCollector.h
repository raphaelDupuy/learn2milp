
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
 * - LP quantities (bounds, gains, lp_objective) live in SCIP's internal
 *   transformed space, which is a MINIMIZATION (the MIS maximum is negated).
 * - Variable quantities are expressed on the MIS vertex variable x_v: if
 *   presolving replaced x_v by 1 - column, lp_value, bounds, objective
 *   coefficient and reduced cost are mirrored and down/up are swapped, so
 *   "down" always means x_v <= 0 and "up" means x_v >= 1.
 * - fractionality is min(f, 1 - f), f being the fractional part.
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
