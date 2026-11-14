#include "learning/MISBranchingModel.h"

#include <stddef.h>

#include "learning/MISBranchingFeatures.h"

int MISbranchingModelIsValid(const MISBranchingModel* model)
{
    int expected_inputs;
    int l;
    int f;

    if( model == NULL
        || model->number_of_features != MIS_BRANCHING_FEATURE_COUNT
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

        expected_inputs = layer->number_of_outputs;
    }

    /* A single score, without activation. */
    return expected_inputs == 1
        && !model->layers[model->number_of_layers - 1].relu;
}

int MISbranchingModelWorkSize(const MISBranchingModel* model)
{
    int widest;
    int l;

    if( model == NULL )
        return 0;

    widest = model->number_of_features;

    for( l = 0; l < model->number_of_layers; ++l )
    {
        if( model->layers[l].number_of_outputs > widest )
            widest = model->layers[l].number_of_outputs;
    }

    /* Input and output of the current layer. */
    return 2 * widest;
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
    int i;
    int o;

    for( i = 0; i < model->number_of_features; ++i )
    {
        input[i] =
            (features[i] - model->feature_means[i]) / model->feature_stds[i];
    }

    for( l = 0; l < model->number_of_layers; ++l )
    {
        const MISBranchingModelLayer* layer = &model->layers[l];
        double* swap;

        for( o = 0; o < layer->number_of_outputs; ++o )
        {
            const double* row =
                layer->weights + (size_t)o * layer->number_of_inputs;
            double sum = layer->biases[o];

            for( i = 0; i < layer->number_of_inputs; ++i )
                sum += row[i] * input[i];

            output[o] = (layer->relu && sum < 0.0) ? 0.0 : sum;
        }

        swap = input;
        input = output;
        output = swap;
    }

    return input[0];
}
