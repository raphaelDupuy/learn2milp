#ifndef LEARN2MILP_MIS_BRANCHING_MODEL_H
#define LEARN2MILP_MIS_BRANCHING_MODEL_H

#include "learning/MISBranchingFeatures.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A small neural network that scores the branching candidates of a node
 * from their features (learning/MISBranchingFeatures.h).
 *
 * A model reads all the computed features or a subset of them, so that
 * a model stays usable when features are added. The features of every
 * candidate are first standardized, (x - mean) / std, then go through the
 * layers in order:
 *
 * - a dense layer transforms each candidate on its own:
 *       h'_v = W h_v + b
 * - a graph layer also looks at the other candidates of the node, through
 *   the edges of the graph between them:
 *       h'_v = W h_v + W_neighbours mean(h_u, u candidate neighbour of v)
 *              + W_node mean(h_u, u candidate of the node) + b
 *   where the mean over no neighbour is 0.
 *
 * Either is followed by max(y, 0) if the layer uses the ReLU activation.
 * The last layer has a single output: the score. Among the candidates of
 * a node, the one with the highest score is chosen. Without graph layer,
 * the model is a multilayer perceptron and every candidate is scored on
 * its own.
 *
 * The structures only point to arrays owned by the caller (see
 * io/MISBranchingModelIO.hpp for loading a model from a file).
 */

typedef struct
{
    int number_of_inputs;
    int number_of_outputs;

    /* Non-zero if the layer is followed by the ReLU activation. */
    int relu;

    /* number_of_outputs rows of number_of_inputs values: W. */
    const double* weights;

    /* number_of_outputs values. */
    const double* biases;

    /* Non-zero for a graph layer, which also has W_neighbours and W_node,
     * of the same shape as W; both are NULL for a dense layer. */
    int graph;
    const double* neighbour_weights;
    const double* node_weights;
} MISBranchingModelLayer;

typedef struct
{
    /* Features read by the model, in the order of its first layer:
     * feature_indices[i] is a column of the computed feature row. A NULL
     * feature_indices stands for every computed feature, in order. */
    int number_of_features;
    const int* feature_indices;
    const double* feature_means;
    const double* feature_stds;

    int number_of_layers;
    const MISBranchingModelLayer* layers;
} MISBranchingModel;

/*
 * Whether the model can be evaluated: it reads existing features (all of
 * them if feature_indices is NULL), its layers chain correctly, its graph
 * layers have their three weight matrices, and it ends with a single
 * output.
 */
int MISbranchingModelIsValid(const MISBranchingModel* model);

/* Whether the model has at least one graph layer. */
int MISbranchingModelHasGraphLayers(const MISBranchingModel* model);

/*
 * Number of doubles needed in the work array of
 * MISevaluateBranchingModel(), for a model without graph layer.
 */
int MISbranchingModelWorkSize(const MISBranchingModel* model);

/*
 * Score of one candidate, for a model without graph layer. features is a
 * full row of MIS_BRANCHING_FEATURE_COUNT computed features, of which the
 * model reads its own; work is a scratch array of
 * MISbranchingModelWorkSize() doubles. The model must be valid.
 */
double MISevaluateBranchingModel(
    const MISBranchingModel* model,
    const double* features,
    double* work
);

/*
 * Number of doubles needed in the work array of
 * MISevaluateBranchingModelOnNode() for a node with that many candidates.
 */
int MISbranchingModelNodeWorkSize(
    const MISBranchingModel* model,
    int number_of_candidates
);

/*
 * Scores of all the candidates of one node, for any valid model.
 *
 * - candidate_vertices gives the graph vertex of each candidate, all
 *   distinct;
 * - features holds number_of_candidates full rows of
 *   MIS_BRANCHING_FEATURE_COUNT computed features, in the same order;
 * - work is a scratch array of MISbranchingModelNodeWorkSize() doubles,
 *   and vertex_positions one of graph->number_of_vertices ints;
 * - scores receives one score per candidate.
 *
 * Without graph layer, every score is the one MISevaluateBranchingModel()
 * gives, bit for bit, and graph and vertex_positions are not used.
 *
 * Returns 0 on success, and a non-zero value, without a usable result,
 * if a candidate vertex is out of range or repeated.
 */
int MISevaluateBranchingModelOnNode(
    const MISBranchingModel* model,
    const MISFeatureGraph* graph,
    int number_of_candidates,
    const int* candidate_vertices,
    const double* features,
    double* work,
    int* vertex_positions,
    double* scores
);

#ifdef __cplusplus
}
#endif

#endif
