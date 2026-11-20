#ifndef LEARN2MILP_MIS_LEARNED_BRANCHRULE_H
#define LEARN2MILP_MIS_LEARNED_BRANCHRULE_H

#include <stddef.h>

#include <scip/scip.h>

#include "learning/MISBranchingFeatures.h"
#include "learning/MISBranchingModel.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * What the learned rule saw and decided at one node. Every array is only
 * valid during the callback.
 *
 * features has number_of_candidates rows of MIS_BRANCHING_FEATURE_COUNT
 * values; scores has one value per candidate. The rule branches on
 * candidate selected_candidate, the first one with the highest score.
 */
typedef struct
{
    long long node_id;
    int node_depth;

    const MISVertexState* vertex_states;
    int number_of_vertices;

    int number_of_candidates;
    const int* candidate_vertices;
    const double* candidate_lp_values;
    const MISCandidateHistory* candidate_histories;

    /* See MISbranchingObjectiveGap(). */
    double objective_gap;

    const double* features;
    const double* scores;

    int selected_candidate;
} MISLearnedBranchingDecision;

typedef void (*MISLearnedBranchingDecisionCallback)(
    void* userdata,
    const MISLearnedBranchingDecision* decision
);

/*
 * Includes the learned MIS branching rule: at every node, it computes the
 * features of the LP branching candidates, scores them with the model and
 * branches on the candidate with the highest score.
 *
 * vertexvars[v] must be the original variable of vertex v, and graph the
 * adjacency of the same graph. vertexvars, graph and model are NOT
 * copied: they must stay valid for as long as the rule may run. The model
 * must be valid (MISbranchingModelIsValid()).
 *
 * callback may be NULL. When given, it is called once per branching
 * decision, before branching; it is meant for tests and analysis.
 *
 * The solve fails with SCIP_INVALIDDATA if presolving merged two
 * vertices into one LP column: the features would not be defined.
 *
 * Pass (NULL, 0, NULL, NULL, NULL, NULL) for the empty configuration
 * used by copies.
 */
SCIP_RETCODE SCIPincludeBranchruleMISLearned(
    SCIP* scip,
    SCIP_VAR** vertexvars,
    int nvertices,
    const MISFeatureGraph* graph,
    const MISBranchingModel* model,
    MISLearnedBranchingDecisionCallback callback,
    void* userdata
);

/*
 * Strong branching on the candidates the model ranks first.
 *
 * With candidates > 0, strong branching evaluates, at every node, the (at
 * most) that many candidates with the highest model scores, as the
 * misfullstrong expert does for all of them: the rule then branches on
 * the best of them by strong branching score, or prunes the node when
 * strong branching proves that its bound reaches the cutoff. They are
 * evaluated in the order of the LP candidates and the first best one is
 * chosen, so that with as many candidates as the node has, the rule makes
 * the expert's decisions. 0, the default, branches on the model's best
 * candidate without strong branching. The rule must have been included.
 */
SCIP_RETCODE SCIPsetMISLearnedStrongBranching(
    SCIP* scip,
    int candidates
);

/*
 * Time the learned rule spent in the current or last solve computing the
 * features of the candidates (gathering their data included) and
 * evaluating the model, in seconds, measured with SCIP's clocks
 * (timing/clocktype). Strong branching on the model's first candidates is
 * not included: SCIP counts it as strong branching. The rule must have
 * been included.
 */
SCIP_RETCODE SCIPgetMISLearnedTimes(
    SCIP* scip,
    SCIP_Real* featureseconds,
    SCIP_Real* modelseconds
);

#ifdef __cplusplus
}
#endif

#endif
