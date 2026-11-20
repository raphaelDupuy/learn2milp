#include "learning/MISBranchingModel.h"

#include <stddef.h>

#include "learning/MISBranchingFeatures.h"

static const char* const VERTEX_INPUT_NAMES[MIS_BRANCHING_VERTEX_INPUT_COUNT] = {
    "free_lp_zero", "free_lp_one", "fractional", "lp_value", "degree", "free_degree"
};

static const char* const CONSTRAINT_INPUT_NAMES[MIS_BRANCHING_CONSTRAINT_INPUT_COUNT] = {
    "slack", "tight", "fractional_endpoints"
};

const char* MISbranchingVertexInputName(int input)
{
    if( input < 0 || input >= MIS_BRANCHING_VERTEX_INPUT_COUNT )
        return NULL;

    return VERTEX_INPUT_NAMES[input];
}

const char* MISbranchingConstraintInputName(int input)
{
    if( input < 0 || input >= MIS_BRANCHING_CONSTRAINT_INPUT_COUNT )
        return NULL;

    return CONSTRAINT_INPUT_NAMES[input];
}

/* Positive standard deviations (also false for NaN). */
static
int positive(const double* values, int count)
{
    int i;

    if( values == NULL )
        return 0;

    for( i = 0; i < count; ++i )
    {
        if( !(values[i] > 0.0) )
            return 0;
    }

    return 1;
}

/* Dense layers chaining from *width, which receives the final width. */
static
int denseLayersValid(const MISBranchingModelLayer* layers, int count, int* width)
{
    int l;

    if( count < 0 || (count > 0 && layers == NULL) )
        return 0;

    for( l = 0; l < count; ++l )
    {
        if( layers[l].graph
            || layers[l].number_of_inputs != *width
            || layers[l].number_of_outputs < 1
            || layers[l].weights == NULL
            || layers[l].biases == NULL )
            return 0;

        *width = layers[l].number_of_outputs;
    }

    return 1;
}

/* The encoder chains, from the inputs of both sides; *vertex_width
 * receives the width of the final vertex states. */
static
int encoderValid(
    const MISBranchingBipartiteEncoder* encoder,
    int number_of_features,
    int* vertex_width
)
{
    int constraint_width = MIS_BRANCHING_CONSTRAINT_INPUT_COUNT;
    int l;

    *vertex_width = number_of_features + MIS_BRANCHING_VERTEX_INPUT_COUNT;

    if( encoder->vertex_input_means == NULL
        || encoder->constraint_input_means == NULL
        || !positive(encoder->vertex_input_stds, MIS_BRANCHING_VERTEX_INPUT_COUNT)
        || !positive(encoder->constraint_input_stds, MIS_BRANCHING_CONSTRAINT_INPUT_COUNT)
        || !denseLayersValid(encoder->vertex_layers, encoder->number_of_vertex_layers, vertex_width)
        || !denseLayersValid(encoder->constraint_layers, encoder->number_of_constraint_layers, &constraint_width)
        || encoder->number_of_message_layers < 0
        || (encoder->number_of_message_layers > 0 && encoder->message_layers == NULL) )
        return 0;

    for( l = 0; l < encoder->number_of_message_layers; ++l )
    {
        const MISBranchingMessageLayer* layer = &encoder->message_layers[l];
        int* target_width = layer->to_vertices ? vertex_width : &constraint_width;
        const int source_width = layer->to_vertices ? constraint_width : *vertex_width;

        if( layer->number_of_target_inputs != *target_width
            || layer->number_of_source_inputs != source_width
            || layer->number_of_hidden < 1
            || layer->number_of_outputs < 1
            || layer->target_weights == NULL
            || layer->source_weights == NULL
            || layer->hidden_biases == NULL
            || layer->message_weights == NULL
            || layer->own_weights == NULL
            || layer->biases == NULL )
            return 0;

        *target_width = layer->number_of_outputs;
    }

    return 1;
}

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

    if( model->encoder != NULL
        && !encoderValid(model->encoder, model->number_of_features, &expected_inputs) )
        return 0;

    for( l = 0; l < model->number_of_layers; ++l )
    {
        const MISBranchingModelLayer* layer = &model->layers[l];

        /* The layers of a bipartite model score each candidate. */
        if( layer->graph && model->encoder != NULL )
            return 0;

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

static
int largest(int a, int b)
{
    return a > b ? a : b;
}

/* Largest of the input and layer widths, encoder included. */
static
int widestStage(const MISBranchingModel* model)
{
    const MISBranchingBipartiteEncoder* encoder = model->encoder;
    int widest = model->number_of_features;
    int l;

    for( l = 0; l < model->number_of_layers; ++l )
        widest = largest(widest, model->layers[l].number_of_outputs);

    if( encoder == NULL )
        return widest;

    widest = largest(widest, model->number_of_features + MIS_BRANCHING_VERTEX_INPUT_COUNT);
    widest = largest(widest, MIS_BRANCHING_CONSTRAINT_INPUT_COUNT);

    for( l = 0; l < encoder->number_of_vertex_layers; ++l )
        widest = largest(widest, encoder->vertex_layers[l].number_of_outputs);

    for( l = 0; l < encoder->number_of_constraint_layers; ++l )
        widest = largest(widest, encoder->constraint_layers[l].number_of_outputs);

    for( l = 0; l < encoder->number_of_message_layers; ++l )
    {
        widest = largest(widest, encoder->message_layers[l].number_of_hidden);
        widest = largest(widest, encoder->message_layers[l].number_of_outputs);
    }

    return widest;
}

/* Number of edges of the graph. */
static
int numberOfEdges(const MISFeatureGraph* graph)
{
    return graph->adjacency_offsets[graph->number_of_vertices] / 2;
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

/*
 * output = activation(W input + b) for one candidate. Four outputs at a
 * time, each summed in the same order as alone (bias first, then inputs
 * in order): the same values to the bit, with every input read once for
 * the four rows.
 */
static
void applyDense(
    const MISBranchingModelLayer* layer,
    const double* input,
    double* output
)
{
    const int inputs = layer->number_of_inputs;
    int i;
    int o;

    for( o = 0; o + 4 <= layer->number_of_outputs; o += 4 )
    {
        const double* row0 = layer->weights + (size_t)o * inputs;
        const double* row1 = row0 + inputs;
        const double* row2 = row1 + inputs;
        const double* row3 = row2 + inputs;
        double sum0 = layer->biases[o];
        double sum1 = layer->biases[o + 1];
        double sum2 = layer->biases[o + 2];
        double sum3 = layer->biases[o + 3];

        for( i = 0; i < inputs; ++i )
        {
            const double x = input[i];

            sum0 += row0[i] * x;
            sum1 += row1[i] * x;
            sum2 += row2[i] * x;
            sum3 += row3[i] * x;
        }

        output[o] = (layer->relu && sum0 < 0.0) ? 0.0 : sum0;
        output[o + 1] = (layer->relu && sum1 < 0.0) ? 0.0 : sum1;
        output[o + 2] = (layer->relu && sum2 < 0.0) ? 0.0 : sum2;
        output[o + 3] = (layer->relu && sum3 < 0.0) ? 0.0 : sum3;
    }

    for( ; o < layer->number_of_outputs; ++o )
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
    const MISFeatureGraph* graph,
    int number_of_candidates
)
{
    int widest;

    if( model == NULL || number_of_candidates < 0 )
        return 0;

    widest = widestStage(model);

    if( model->encoder != NULL )
    {
        const int n = graph->number_of_vertices;
        const int m = numberOfEdges(graph);

        /* Vertex and constraint states before and after a layer; target
         * projections, source projections and means of a message layer;
         * the LP values; two rows for the final layers. */
        return (2 * n + 2 * m + 3 * largest(n, m) + 2) * widest + n;
    }

    if( !MISbranchingModelHasGraphLayers(model) )
        return MISbranchingModelWorkSize(model);

    /* States before and after the layer and neighbour means, one row per
     * candidate, then the mean state of the node. */
    return (3 * number_of_candidates + 1) * widest;
}

int MISbranchingModelNodeIntWorkSize(
    const MISBranchingModel* model,
    const MISFeatureGraph* graph,
    int number_of_candidates
)
{
    (void)number_of_candidates;

    if( model == NULL )
        return 1;

    if( model->encoder != NULL )
    {
        const int n = graph->number_of_vertices;
        const int m = numberOfEdges(graph);

        /* Row of each vertex, the two rows of each constraint, incidence
         * offsets and lists, and one counter per vertex. */
        return 3 * n + 4 * m + 1;
    }

    if( MISbranchingModelHasGraphLayers(model) )
        return largest(1, graph->number_of_vertices);

    return 1;
}

/* Value standardized with a mean and a standard deviation. */
static
double standardized(double value, double mean, double std)
{
    return (value - mean) / std;
}

/*
 * Scores of the candidates under a bipartite model. Work arrays as sized
 * by MISbranchingModelNodeWorkSize() and MISbranchingModelNodeIntWorkSize().
 */
static
int evaluateBipartite(
    const MISBranchingModel* model,
    const MISFeatureGraph* graph,
    const MISVertexState* states,
    int number_of_candidates,
    const int* candidate_vertices,
    const double* features,
    double* work,
    int* int_work,
    double* scores
)
{
    const MISBranchingBipartiteEncoder* encoder = model->encoder;
    const int n = graph->number_of_vertices;
    const int m = numberOfEdges(graph);
    const int big = largest(n, m);
    const int widest = widestStage(model);
    const int features_width = model->number_of_features;
    int* row = int_work;
    int* ends = row + n;
    int* offsets = ends + 2 * (size_t)m;
    int* incident = offsets + n + 1;
    int* counter = incident + 2 * (size_t)m;
    double* vertices = work;
    double* vertices_next = vertices + (size_t)n * widest;
    double* constraints = vertices_next + (size_t)n * widest;
    double* constraints_next = constraints + (size_t)m * widest;
    double* hidden = constraints_next + (size_t)m * widest;
    double* projected = hidden + (size_t)big * widest;
    double* means = projected + (size_t)big * widest;
    double* lp = means + (size_t)big * widest;
    double* input = lp + n;
    double* output = input + widest;
    int vertex_width = features_width + MIS_BRANCHING_VERTEX_INPUT_COUNT;
    int constraint_width = MIS_BRANCHING_CONSTRAINT_INPUT_COUNT;
    int free_vertices = 0;
    int fractional = 0;
    int active = 0;
    int v;
    int c;
    int e;
    int l;
    int i;
    int k;

    if( states == NULL )
        return 1;

    /* Free vertices, numbered in vertex order; LP values. */
    for( v = 0; v < n; ++v )
    {
        const MISVertexState state = states[v];

        if( state == MIS_VERTEX_FREE_LP_ZERO || state == MIS_VERTEX_FREE_LP_ONE
            || state == MIS_VERTEX_FRACTIONAL )
        {
            counter[free_vertices] = -1;
            row[v] = free_vertices++;
        }
        else
            row[v] = -1;

        lp[v] = state == MIS_VERTEX_FREE_LP_ONE ? 1.0 : 0.0;
        fractional += state == MIS_VERTEX_FRACTIONAL;
    }

    /* The candidates are exactly the fractional vertices, each once. */
    if( fractional != number_of_candidates )
        return 1;

    for( c = 0; c < number_of_candidates; ++c )
    {
        const int vertex = candidate_vertices[c];

        if( vertex < 0 || vertex >= n || states[vertex] != MIS_VERTEX_FRACTIONAL
            || counter[row[vertex]] >= 0 )
            return 1;

        counter[row[vertex]] = c;
        lp[vertex] = features[(size_t)c * MIS_BRANCHING_FEATURE_COUNT];
    }

    /* Active constraints: edges between two free vertices. */
    for( i = 0; i < free_vertices; ++i )
        counter[i] = 0;

    for( v = 0; v < n; ++v )
    {
        if( row[v] < 0 )
            continue;

        for( k = graph->adjacency_offsets[v]; k < graph->adjacency_offsets[v + 1]; ++k )
        {
            const int u = graph->adjacency[k];
            const double* means_in = encoder->constraint_input_means;
            const double* stds_in = encoder->constraint_input_stds;
            double* state;
            double slack;

            if( u <= v || row[u] < 0 )
                continue;

            e = active++;
            ends[2 * e] = row[v];
            ends[2 * e + 1] = row[u];
            ++counter[row[v]];
            ++counter[row[u]];

            slack = 1.0 - lp[v] - lp[u];
            state = constraints + (size_t)e * widest;
            state[0] = standardized(slack, means_in[0], stds_in[0]);
            state[1] = standardized(
                slack <= MIS_BRANCHING_FEATURE_TIGHT_TOLERANCE ? 1.0 : 0.0, means_in[1], stds_in[1]);
            state[2] = standardized(
                (double)(states[v] == MIS_VERTEX_FRACTIONAL) + (double)(states[u] == MIS_VERTEX_FRACTIONAL),
                means_in[2], stds_in[2]);
        }
    }

    /* Vertex inputs: candidate features (0 for the other vertices), then
     * the vertex inputs. */
    for( v = 0; v < n; ++v )
    {
        const int r = row[v];
        const double* means_in = encoder->vertex_input_means;
        const double* stds_in = encoder->vertex_input_stds;
        double* state;
        double values[MIS_BRANCHING_VERTEX_INPUT_COUNT];

        if( r < 0 )
            continue;

        state = vertices + (size_t)r * widest;

        for( i = 0; i < features_width; ++i )
            state[i] = 0.0;

        values[0] = states[v] == MIS_VERTEX_FREE_LP_ZERO;
        values[1] = states[v] == MIS_VERTEX_FREE_LP_ONE;
        values[2] = states[v] == MIS_VERTEX_FRACTIONAL;
        values[3] = lp[v];
        values[4] = graph->adjacency_offsets[v + 1] - graph->adjacency_offsets[v];
        values[5] = counter[r];

        for( i = 0; i < MIS_BRANCHING_VERTEX_INPUT_COUNT; ++i )
            state[features_width + i] = standardized(values[i], means_in[i], stds_in[i]);
    }

    for( c = 0; c < number_of_candidates; ++c )
    {
        standardize(
            model,
            features + (size_t)c * MIS_BRANCHING_FEATURE_COUNT,
            vertices + (size_t)row[candidate_vertices[c]] * widest
        );
    }

    /* Constraints of each free vertex, in constraint order. */
    offsets[0] = 0;

    for( i = 0; i < free_vertices; ++i )
    {
        offsets[i + 1] = offsets[i] + counter[i];
        counter[i] = offsets[i];
    }

    for( e = 0; e < active; ++e )
    {
        incident[counter[ends[2 * e]]++] = e;
        incident[counter[ends[2 * e + 1]]++] = e;
    }

    /* Dense layers of each side. */
    for( l = 0; l < encoder->number_of_vertex_layers; ++l )
    {
        double* swap;

        for( i = 0; i < free_vertices; ++i )
            applyDense(&encoder->vertex_layers[l], vertices + (size_t)i * widest, vertices_next + (size_t)i * widest);

        swap = vertices;
        vertices = vertices_next;
        vertices_next = swap;
        vertex_width = encoder->vertex_layers[l].number_of_outputs;
    }

    for( l = 0; l < encoder->number_of_constraint_layers; ++l )
    {
        double* swap;

        for( e = 0; e < active; ++e )
            applyDense(&encoder->constraint_layers[l], constraints + (size_t)e * widest, constraints_next + (size_t)e * widest);

        swap = constraints;
        constraints = constraints_next;
        constraints_next = swap;
        constraint_width = encoder->constraint_layers[l].number_of_outputs;
    }

    /* Message layers. */
    for( l = 0; l < encoder->number_of_message_layers; ++l )
    {
        const MISBranchingMessageLayer* layer = &encoder->message_layers[l];
        const int to_vertices = layer->to_vertices;
        const int targets = to_vertices ? free_vertices : active;
        const int sources = to_vertices ? active : free_vertices;
        const int target_inputs = layer->number_of_target_inputs;
        const int source_inputs = layer->number_of_source_inputs;
        const int hidden_width = layer->number_of_hidden;
        double* target_states = to_vertices ? vertices : constraints;
        double* next_states = to_vertices ? vertices_next : constraints_next;
        const double* source_states = to_vertices ? constraints : vertices;
        int t;
        int h;
        int o;

        /* L t + b_hidden for each target, R s for each source. */
        for( t = 0; t < targets; ++t )
        {
            const double* state = target_states + (size_t)t * widest;

            for( h = 0; h < hidden_width; ++h )
            {
                const double* weights = layer->target_weights + (size_t)h * target_inputs;
                double sum = layer->hidden_biases[h];

                for( i = 0; i < target_inputs; ++i )
                    sum += weights[i] * state[i];

                hidden[(size_t)t * widest + h] = sum;
            }
        }

        for( t = 0; t < sources; ++t )
        {
            const double* state = source_states + (size_t)t * widest;

            for( h = 0; h < hidden_width; ++h )
            {
                const double* weights = layer->source_weights + (size_t)h * source_inputs;
                double sum = 0.0;

                for( i = 0; i < source_inputs; ++i )
                    sum += weights[i] * state[i];

                projected[(size_t)t * widest + h] = sum;
            }
        }

        /* Mean message of each target. */
        for( t = 0; t < targets; ++t )
        {
            const double* own = hidden + (size_t)t * widest;
            double* mean = means + (size_t)t * widest;
            const int first = to_vertices ? offsets[t] : 0;
            const int count = to_vertices ? offsets[t + 1] - offsets[t] : 2;

            for( h = 0; h < hidden_width; ++h )
                mean[h] = 0.0;

            for( k = 0; k < count; ++k )
            {
                const int source = to_vertices ? incident[first + k] : ends[2 * t + k];
                const double* other = projected + (size_t)source * widest;

                for( h = 0; h < hidden_width; ++h )
                {
                    const double message = own[h] + other[h];

                    if( message > 0.0 )
                        mean[h] += message;
                }
            }

            for( h = 0; count > 0 && h < hidden_width; ++h )
                mean[h] /= count;
        }

        /* t' = activation(P mean + Q t + b). */
        for( t = 0; t < targets; ++t )
        {
            const double* mean = means + (size_t)t * widest;
            const double* state = target_states + (size_t)t * widest;
            double* next = next_states + (size_t)t * widest;

            for( o = 0; o < layer->number_of_outputs; ++o )
            {
                const double* message_weights = layer->message_weights + (size_t)o * hidden_width;
                const double* own_weights = layer->own_weights + (size_t)o * target_inputs;
                double sum = layer->biases[o];

                for( h = 0; h < hidden_width; ++h )
                    sum += message_weights[h] * mean[h];

                for( i = 0; i < target_inputs; ++i )
                    sum += own_weights[i] * state[i];

                next[o] = (layer->relu && sum < 0.0) ? 0.0 : sum;
            }
        }

        if( to_vertices )
        {
            vertices_next = vertices;
            vertices = next_states;
            vertex_width = layer->number_of_outputs;
        }
        else
        {
            constraints_next = constraints;
            constraints = next_states;
            constraint_width = layer->number_of_outputs;
        }
    }

    (void)constraint_width;

    /* Final layers on the state of each candidate. */
    for( c = 0; c < number_of_candidates; ++c )
    {
        const double* state = vertices + (size_t)row[candidate_vertices[c]] * widest;
        double* in = input;
        double* out = output;

        for( i = 0; i < vertex_width; ++i )
            in[i] = state[i];

        for( l = 0; l < model->number_of_layers; ++l )
        {
            double* swap;

            applyDense(&model->layers[l], in, out);
            swap = in;
            in = out;
            out = swap;
        }

        scores[c] = in[0];
    }

    return 0;
}

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
)
{
    const int n = number_of_candidates;
    int* vertex_positions = int_work;
    int widest;
    double* current;
    double* next;
    double* neighbours;
    double* node;
    int l;
    int c;
    int i;
    int o;

    if( model->encoder != NULL )
    {
        if( n == 0 )
            return 0;

        return evaluateBipartite(
            model, graph, vertex_states, number_of_candidates, candidate_vertices,
            features, work, int_work, scores
        );
    }

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
