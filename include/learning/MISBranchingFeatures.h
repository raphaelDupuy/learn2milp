#ifndef LEARN2MILP_MIS_BRANCHING_FEATURES_H
#define LEARN2MILP_MIS_BRANCHING_FEATURES_H

#include "branching/MISStrongBranchingCollector.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Features describing the branching candidates of one node.
 *
 * This is the single implementation of the features: the same function
 * is used offline, to turn collected samples into training data, and
 * online, inside the learned branching rule. A model must only be used
 * with the feature set it was trained on, identified by
 * MIS_BRANCHING_FEATURE_SET_VERSION and by the feature names.
 *
 * Inputs are limited to what a collected sample contains: the graph, the
 * state of every vertex, the depth of the node and the LP value of each
 * candidate. Notation for a candidate vertex v with LP value x:
 *
 *   N     neighbours of v in the graph
 *   free  vertices not fixed by the local bounds
 *   frac  vertices with a fractional LP value (the candidates)
 *   x_u   LP value of vertex u (0 or 1 when u is not fractional)
 *
 * Features, in column order:
 *
 *    0 lp_value                         x
 *    1 fractionality                    min(x, 1 - x)
 *    2 degree                           |N|
 *    3 free_degree                      |N & free|
 *    4 fractional_degree                |N & frac|
 *    5 neighbour_lp_sum                 sum of x_u over N
 *    6 neighbour_lp_max                 max of x_u over N (0 if N is empty)
 *    7 tight_edges                      free neighbours u with x_u + x >= 1
 *    8 up_mass_change                   neighbour_lp_sum - (1 - x): LP value
 *                                       lost at first order when x is set
 *                                       to 1 and its neighbours to 0
 *    9 neighbour_fractional_degree_sum  sum of fractional_degree(u) over
 *                                       N & frac
 *   10 fractional_triangles             edges between two vertices of
 *                                       N & frac
 *   11 lp_value_times_neighbour_sum     x * neighbour_lp_sum
 *   12 lp_value_relative                x / largest x among candidates
 *   13 neighbour_lp_sum_relative        neighbour_lp_sum / largest one
 *                                       among candidates (0 if all are 0)
 *   14 fractional_degree_relative       fractional_degree / largest one
 *                                       among candidates (0 if all are 0)
 *   15 node_depth                       depth of the node in the tree
 *   16 candidate_fraction               candidates / vertices
 *   17 fixed_fraction                   fixed vertices / vertices
 *
 *   18 two_hop_lp_sum                   sum of neighbour_lp_sum(u) over N
 *   19 neighbour_lp_min                 min of x_u over N & frac (1 if empty)
 *   20 down_room                        sum over N & free of the room left
 *                                       for u to grow if x were set to 0:
 *                                       max(0, 1 - x_u - largest LP value
 *                                       among the other neighbours of u)
 *   21 blocked_neighbours               vertices u of N & free for which x
 *                                       is larger than the LP value of
 *                                       every other neighbour of u
 *   22 clique_weight                    LP weight (sum of LP values, x
 *                                       included) of a clique grown
 *                                       greedily from v through N & frac
 *                                       by decreasing LP value
 *   23 clique_size                      number of vertices of that clique
 *   24 tight_cliques                    among the cliques grown the same
 *                                       way from each vertex of N & frac,
 *                                       those with an LP weight >= 1
 *   25 clique_weight_mean               mean LP weight of those cliques
 *                                       (x if N & frac is empty)
 *   26 clique_weight_min                smallest LP weight of those cliques
 *   27 lp_value_rank                    share of the candidates with a
 *                                       smaller x
 *   28 neighbour_lp_sum_rank            same for neighbour_lp_sum
 *   29 up_mass_change_rank              same for up_mass_change
 *
 * Features 15 to 17 are the same for every candidate of a node.
 * Features 0 to 17 are feature set version 1: their definition must not
 * change, since models may use any subset of the features.
 */

#define MIS_BRANCHING_FEATURE_SET_VERSION 2
#define MIS_BRANCHING_FEATURE_COUNT 30

/* Tolerance used to decide that an edge constraint is tight. */
#define MIS_BRANCHING_FEATURE_TIGHT_TOLERANCE 1e-6

/*
 * Graph in compressed adjacency form: the neighbours of vertex v are
 * adjacency[adjacency_offsets[v]] ... adjacency[adjacency_offsets[v + 1] - 1],
 * sorted by increasing vertex id, without v itself and without duplicates.
 */
typedef struct
{
    int number_of_vertices;
    const int* adjacency_offsets; /* number_of_vertices + 1 entries */
    const int* adjacency;
} MISFeatureGraph;

/* Name of a feature, or NULL if the index is out of range. */
const char* MISbranchingFeatureName(int feature);

/* Number of doubles needed in the work array for a graph of n vertices. */
int MISbranchingFeatureWorkSize(int number_of_vertices);

/*
 * Computes the features of every candidate of one node.
 *
 * - vertex_states has one entry per vertex. Exactly the candidates must
 *   be in state MIS_VERTEX_FRACTIONAL.
 * - candidate_vertices and candidate_lp_values have number_of_candidates
 *   entries; LP values are expressed on the vertex variable x_v and lie
 *   strictly between 0 and 1.
 * - work is a scratch array of MISbranchingFeatureWorkSize() doubles.
 * - features receives number_of_candidates rows of
 *   MIS_BRANCHING_FEATURE_COUNT values, one row per candidate, in the
 *   order of candidate_vertices.
 *
 * Returns 0 on success, and a non-zero value, without a usable result,
 * if the inputs are inconsistent.
 */
int MIScomputeBranchingFeatures(
    const MISFeatureGraph* graph,
    const MISVertexState* vertex_states,
    int node_depth,
    int number_of_candidates,
    const int* candidate_vertices,
    const double* candidate_lp_values,
    double* work,
    double* features
);

#ifdef __cplusplus
}
#endif

#endif
