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
 * A bipartite model (after Gasse et al. 2019) first runs an encoder on
 * the bipartite graph of the free vertices and of the edge constraints
 * x_u + x_v <= 1 between two free vertices (an edge with a fixed end
 * constrains nothing). A vertex starts from its standardized candidate
 * features (0 if it is not a candidate) followed by its standardized
 * vertex inputs, a constraint from its standardized constraint inputs
 * (MISbranchingVertexInputName(), MISbranchingConstraintInputName()). Dense
 * layers transform each side, then each message layer updates one side
 * from the other:
 *     t' = activation(P mean(relu(L t + R s + b_hidden), s source of t)
 *                     + Q t + b)
 * where the sources of a constraint are its two vertices and those of a
 * vertex its constraints (mean 0 if none). The layers of the model then
 * score each candidate from its vertex state.
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

/* Inputs of a bipartite model beyond the candidate features. */
#define MIS_BRANCHING_VERTEX_INPUT_COUNT 6
#define MIS_BRANCHING_CONSTRAINT_INPUT_COUNT 3

/* Name of a vertex or constraint input, or NULL if out of range. */
const char* MISbranchingVertexInputName(int input);
const char* MISbranchingConstraintInputName(int input);

typedef struct
{
    /* Non-zero to update the vertices from their constraints; zero to
     * update the constraints from their two vertices. */
    int to_vertices;

    int number_of_target_inputs;
    int number_of_source_inputs;
    int number_of_hidden;
    int number_of_outputs;
    int relu;

    const double* target_weights;  /* L: hidden rows of target inputs */
    const double* source_weights;  /* R: hidden rows of source inputs */
    const double* hidden_biases;   /* hidden values */
    const double* message_weights; /* P: output rows of hidden values */
    const double* own_weights;     /* Q: output rows of target inputs */
    const double* biases;          /* output values */
} MISBranchingMessageLayer;

typedef struct
{
    const double* vertex_input_means;     /* MIS_BRANCHING_VERTEX_INPUT_COUNT */
    const double* vertex_input_stds;
    const double* constraint_input_means; /* MIS_BRANCHING_CONSTRAINT_INPUT_COUNT */
    const double* constraint_input_stds;

    /* Dense layers only. */
    int number_of_vertex_layers;
    const MISBranchingModelLayer* vertex_layers;
    int number_of_constraint_layers;
    const MISBranchingModelLayer* constraint_layers;

    int number_of_message_layers;
    const MISBranchingMessageLayer* message_layers;
} MISBranchingBipartiteEncoder;

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

    /* Bipartite encoder run before the layers, or NULL. The layers of a
     * bipartite model are dense. */
    const MISBranchingBipartiteEncoder* encoder;
} MISBranchingModel;

/*
 * Whether the model can be evaluated: it reads existing features (all of
 * them if feature_indices is NULL), its layers and those of its encoder
 * chain correctly, its graph layers have their three weight matrices, and
 * it ends with a single output.
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
 * Numbers of doubles and of ints needed in the work arrays of
 * MISevaluateBranchingModelOnNode() for a node of that graph with that
 * many candidates (at least 1).
 */
int MISbranchingModelNodeWorkSize(
    const MISBranchingModel* model,
    const MISFeatureGraph* graph,
    int number_of_candidates
);
int MISbranchingModelNodeIntWorkSize(
    const MISBranchingModel* model,
    const MISFeatureGraph* graph,
    int number_of_candidates
);

/*
 * Scores of all the candidates of one node, for any valid model.
 *
 * - vertex_states gives the state of every vertex (needed by bipartite
 *   models only, may be NULL otherwise); the candidates must then be
 *   exactly the vertices in state MIS_VERTEX_FRACTIONAL;
 * - candidate_vertices gives the graph vertex of each candidate, all
 *   distinct;
 * - features holds number_of_candidates full rows of
 *   MIS_BRANCHING_FEATURE_COUNT computed features, in the same order
 *   (feature 0, lp_value, gives the LP values of a bipartite model);
 * - work and int_work are scratch arrays of the sizes above;
 * - scores receives one score per candidate.
 *
 * Without graph layer or encoder, every score is the one
 * MISevaluateBranchingModel() gives, bit for bit, and graph, vertex_states
 * and int_work are not used.
 *
 * Returns 0 on success, and a non-zero value, without a usable result,
 * if a candidate vertex is out of range or repeated, or if the vertex
 * states do not match the candidates.
 */
int MISevaluateBranchingModelOnNode(
    const MISBranchingModel* model,
    const MISFeatureGraph* graph,
    const MISVertexState* vertex_states,
    int number_of_candidates,
    const int* candidate_vertices,
    const double* features,
    double* work,
    int* int_work,
    double* scores
);

#ifdef __cplusplus
}
#endif

#endif
