#include "learning/MISBranchingModel.h"

#include <stddef.h>

#include "learning/MISBranchingFeatures.h"

int MISbranchingModelIsValid(const MISBranchingModel* model)
{
    int expected_inputs;
    int l;
    int f;

    if( model == NULL
        || model->number_of_features < 1
        || model->number_of_features > MIS_BRANCHING_FEATURE_COUNT
        || (model->feature_indices == NULL
            && model->number_of_features != MIS_BRANCHING_FEATURE_COUNT)
        || model->feature_means == NULL
        || model->feature_stds == NULL
        || model->number_of_layers < 1
        || model->layers == NULL )
        return 0;

    for( f = 0; f < model->number_of_features; ++f )
    {
        /* Also false for a NaN standard deviation. */
        if( !(model->feature_stds[f] > 0.0) )
            return 0;

        if( model->feature_indices != NULL
            && (model->feature_indices[f] < 0
                || model->feature_indices[f] >= MIS_BRANCHING_FEATURE_COUNT) )
            return 0;
    }

    expected_inputs = model->number_of_features;

    for( l = 0; l < model->number_of_layers; ++l )
    {
        const MISBranchingModelLayer* layer = &model->layers[l];

        if( layer->number_of_inputs != expected_inputs
            || layer->number_of_outputs < 1
            || layer->weights == NULL
            || layer->biases == NULL )
            return 0;

        if( layer->graph
            && (layer->neighbour_weights == NULL || layer->node_weights == NULL) )
            return 0;

        expected_inputs = layer->number_of_outputs;
    }

    /* A single score, without activation. */
    return expected_inputs == 1
        && !model->layers[model->number_of_layers - 1].relu;
}

int MISbranchingModelHasGraphLayers(const MISBranchingModel* model)
{
    int l;

    if( model == NULL || model->layers == NULL )
        return 0;

    for( l = 0; l < model->number_of_layers; ++l )
    {
        if( model->layers[l].graph )
            return 1;
    }

    return 0;
}

/* Largest of the input and layer widths. */
static
int widestStage(const MISBranchingModel* model)
{
    int widest = model->number_of_features;
    int l;

    for( l = 0; l < model->number_of_layers; ++l )
    {
        if( model->layers[l].number_of_outputs > widest )
            widest = model->layers[l].number_of_outputs;
    }

    return widest;
}

int MISbranchingModelWorkSize(const MISBranchingModel* model)
{
    if( model == NULL )
        return 0;

    /* Input and output of the current layer. */
    return 2 * widestStage(model);
}

/* Standardized features of one candidate, in the model's input order. */
static
void standardize(
    const MISBranchingModel* model,
    const double* features,
    double* input
)
{
    int i;

    for( i = 0; i < model->number_of_features; ++i )
    {
        const double value =
            features[model->feature_indices != NULL
                ? model->feature_indices[i] : i];

        input[i] = (value - model->feature_means[i]) / model->feature_stds[i];
    }
}

/* output = activation(W input + b) for one candidate. */
static
void applyDense(
    const MISBranchingModelLayer* layer,
    const double* input,
    double* output
)
{
    int i;
    int o;

    for( o = 0; o < layer->number_of_outputs; ++o )
    {
        const double* row = layer->weights + (size_t)o * layer->number_of_inputs;
        double sum = layer->biases[o];

        for( i = 0; i < layer->number_of_inputs; ++i )
            sum += row[i] * input[i];

        output[o] = (layer->relu && sum < 0.0) ? 0.0 : sum;
    }
}

double MISevaluateBranchingModel(
    const MISBranchingModel* model,
    const double* features,
    double* work
)
{
    const int half = MISbranchingModelWorkSize(model) / 2;
    double* input = work;
    double* output = work + half;
    int l;

    standardize(model, features, input);

    for( l = 0; l < model->number_of_layers; ++l )
    {
        double* swap;

        applyDense(&model->layers[l], input, output);

        swap = input;
        input = output;
        output = swap;
    }

    return input[0];
}

int MISbranchingModelNodeWorkSize(
    const MISBranchingModel* model,
    int number_of_candidates
)
{
    int widest;

    if( model == NULL || number_of_candidates < 0 )
        return 0;

    if( !MISbranchingModelHasGraphLayers(model) )
        return MISbranchingModelWorkSize(model);

    /* States before and after the layer and neighbour means, one row per
     * candidate, then the mean state of the node. */
    widest = widestStage(model);

    return (3 * number_of_candidates + 1) * widest;
}

int MISevaluateBranchingModelOnNode(
    const MISBranchingModel* model,
    const MISFeatureGraph* graph,
    int number_of_candidates,
    const int* candidate_vertices,
    const double* features,
    double* work,
    int* vertex_positions,
    double* scores
)
{
    const int n = number_of_candidates;
    int widest;
    double* current;
    double* next;
    double* neighbours;
    double* node;
    int l;
    int c;
    int i;
    int o;

    if( !MISbranchingModelHasGraphLayers(model) )
    {
        for( c = 0; c < n; ++c )
        {
            scores[c] = MISevaluateBranchingModel(
                model, features + (size_t)c * MIS_BRANCHING_FEATURE_COUNT, work
            );
        }

        return 0;
    }

    if( n == 0 )
        return 0;

    /* Position of each candidate vertex, -1 for the other vertices. */
    for( i = 0; i < graph->number_of_vertices; ++i )
        vertex_positions[i] = -1;

    for( c = 0; c < n; ++c )
    {
        const int vertex = candidate_vertices[c];

        if( vertex < 0 || vertex >= graph->number_of_vertices
            || vertex_positions[vertex] >= 0 )
            return 1;

        vertex_positions[vertex] = c;
    }

    widest = widestStage(model);
    current = work;
    next = work + (size_t)n * widest;
    neighbours = next + (size_t)n * widest;
    node = neighbours + (size_t)n * widest;

    for( c = 0; c < n; ++c )
    {
        standardize(
            model,
            features + (size_t)c * MIS_BRANCHING_FEATURE_COUNT,
            current + (size_t)c * widest
        );
    }

    for( l = 0; l < model->number_of_layers; ++l )
    {
        const MISBranchingModelLayer* layer = &model->layers[l];
        const int inputs = layer->number_of_inputs;
        double* swap;

        if( !layer->graph )
        {
            for( c = 0; c < n; ++c )
            {
                applyDense(
                    layer,
                    current + (size_t)c * widest,
                    next + (size_t)c * widest
                );
            }
        }
        else
        {
            /* Mean state of the node. */
            for( i = 0; i < inputs; ++i )
            {
                double sum = 0.0;

                for( c = 0; c < n; ++c )
                    sum += current[(size_t)c * widest + i];

                node[i] = sum / n;
            }

            /* Mean state of the candidate neighbours of each candidate. */
            for( c = 0; c < n; ++c )
            {
                const int vertex = candidate_vertices[c];
                double* mean = neighbours + (size_t)c * widest;
                int count = 0;
                int k;

                for( i = 0; i < inputs; ++i )
                    mean[i] = 0.0;

                for( k = graph->adjacency_offsets[vertex];
                     k < graph->adjacency_offsets[vertex + 1]; ++k )
                {
                    const int position = vertex_positions[graph->adjacency[k]];
                    const double* state;

                    if( position < 0 )
                        continue;

                    state = current + (size_t)position * widest;

                    for( i = 0; i < inputs; ++i )
                        mean[i] += state[i];

                    ++count;
                }

                for( i = 0; count > 0 && i < inputs; ++i )
                    mean[i] /= count;
            }

            for( c = 0; c < n; ++c )
            {
                const double* own = current + (size_t)c * widest;
                const double* mean = neighbours + (size_t)c * widest;
                double* output = next + (size_t)c * widest;

                for( o = 0; o < layer->number_of_outputs; ++o )
                {
                    const size_t row = (size_t)o * inputs;
                    double sum = layer->biases[o];

                    for( i = 0; i < inputs; ++i )
                    {
                        sum += layer->weights[row + i] * own[i]
                            + layer->neighbour_weights[row + i] * mean[i]
                            + layer->node_weights[row + i] * node[i];
                    }

                    output[o] = (layer->relu && sum < 0.0) ? 0.0 : sum;
                }
            }
        }

        swap = current;
        current = next;
        next = swap;
    }

    for( c = 0; c < n; ++c )
        scores[c] = current[(size_t)c * widest];

    return 0;
}
