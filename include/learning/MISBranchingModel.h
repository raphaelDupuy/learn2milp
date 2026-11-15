#ifndef LEARN2MILP_MIS_BRANCHING_MODEL_H
#define LEARN2MILP_MIS_BRANCHING_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A small fully connected network that scores one branching candidate
 * from its features (learning/MISBranchingFeatures.h).
 *
 * A model reads all the computed features or a subset of them, so that
 * a model stays usable when features are added. The features are first
 * standardized, (x - mean) / std, then every
 * layer computes y = W x + b, followed by max(y, 0) if the layer uses the
 * ReLU activation. The last layer has a single output: the score. Among
 * the candidates of a node, the one with the highest score is chosen.
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

    /* number_of_outputs rows of number_of_inputs values. */
    const double* weights;

    /* number_of_outputs values. */
    const double* biases;
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
 * them if feature_indices is NULL), its layers chain correctly and end
 * with a single output.
 */
int MISbranchingModelIsValid(const MISBranchingModel* model);

/* Number of doubles needed in the work array of an evaluation. */
int MISbranchingModelWorkSize(const MISBranchingModel* model);

/*
 * Score of one candidate. features is a full row of
 * MIS_BRANCHING_FEATURE_COUNT computed features, of which the model
 * reads its own; work is
 * a scratch array of MISbranchingModelWorkSize() doubles. The model must
 * be valid.
 */
double MISevaluateBranchingModel(
    const MISBranchingModel* model,
    const double* features,
    double* work
);

#ifdef __cplusplus
}
#endif

#endif
