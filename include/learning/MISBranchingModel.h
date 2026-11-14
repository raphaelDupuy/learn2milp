#ifndef LEARN2MILP_MIS_BRANCHING_MODEL_H
#define LEARN2MILP_MIS_BRANCHING_MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * A small fully connected network that scores one branching candidate
 * from its features (learning/MISBranchingFeatures.h).
 *
 * The features are first standardized, (x - mean) / std, then every
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
    int number_of_features;
    const double* feature_means;
    const double* feature_stds;

    int number_of_layers;
    const MISBranchingModelLayer* layers;
} MISBranchingModel;

/*
 * Whether the model can be evaluated: it expects exactly the compiled
 * features, its layers chain correctly and end with a single output.
 */
int MISbranchingModelIsValid(const MISBranchingModel* model);

/* Number of doubles needed in the work array of an evaluation. */
int MISbranchingModelWorkSize(const MISBranchingModel* model);

/*
 * Score of one candidate. features has number_of_features values; work is
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
