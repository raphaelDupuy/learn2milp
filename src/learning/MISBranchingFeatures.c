#include "learning/MISBranchingFeatures.h"

#include <stddef.h>

static const char* const FEATURE_NAMES[MIS_BRANCHING_FEATURE_COUNT] = {
    "lp_value",
    "fractionality",
    "degree",
    "free_degree",
    "fractional_degree",
    "neighbour_lp_sum",
    "neighbour_lp_max",
    "tight_edges",
    "up_mass_change",
    "neighbour_fractional_degree_sum",
    "fractional_triangles",
    "lp_value_times_neighbour_sum",
    "lp_value_relative",
    "neighbour_lp_sum_relative",
    "fractional_degree_relative",
    "node_depth",
    "candidate_fraction",
    "fixed_fraction",
    "two_hop_lp_sum",
    "neighbour_lp_min",
    "down_room",
    "blocked_neighbours",
    "clique_weight",
    "clique_size",
    "tight_cliques",
    "clique_weight_mean",
    "clique_weight_min",
    "lp_value_rank",
    "neighbour_lp_sum_rank",
    "up_mass_change_rank",
    "pseudocost_down",
    "pseudocost_up",
    "pseudocost_count_down",
    "pseudocost_count_up",
    "pseudocost_score",
    "pseudocost_score_relative",
    "pseudocost_score_rank",
    "inferences_down",
    "inferences_up",
    "cutoffs_down",
    "cutoffs_up",
    "conflict_score_relative",
    "objective_gap",
    "has_incumbent"
};

enum
{
    F_LP_VALUE = 0,
    F_FRACTIONALITY,
    F_DEGREE,
    F_FREE_DEGREE,
    F_FRACTIONAL_DEGREE,
    F_NEIGHBOUR_LP_SUM,
    F_NEIGHBOUR_LP_MAX,
    F_TIGHT_EDGES,
    F_UP_MASS_CHANGE,
    F_NEIGHBOUR_FRACTIONAL_DEGREE_SUM,
    F_FRACTIONAL_TRIANGLES,
    F_LP_VALUE_TIMES_NEIGHBOUR_SUM,
    F_LP_VALUE_RELATIVE,
    F_NEIGHBOUR_LP_SUM_RELATIVE,
    F_FRACTIONAL_DEGREE_RELATIVE,
    F_NODE_DEPTH,
    F_CANDIDATE_FRACTION,
    F_FIXED_FRACTION,
    F_TWO_HOP_LP_SUM,
    F_NEIGHBOUR_LP_MIN,
    F_DOWN_ROOM,
    F_BLOCKED_NEIGHBOURS,
    F_CLIQUE_WEIGHT,
    F_CLIQUE_SIZE,
    F_TIGHT_CLIQUES,
    F_CLIQUE_WEIGHT_MEAN,
    F_CLIQUE_WEIGHT_MIN,
    F_LP_VALUE_RANK,
    F_NEIGHBOUR_LP_SUM_RANK,
    F_UP_MASS_CHANGE_RANK,
    F_PSEUDOCOST_DOWN,
    F_PSEUDOCOST_UP,
    F_PSEUDOCOST_COUNT_DOWN,
    F_PSEUDOCOST_COUNT_UP,
    F_PSEUDOCOST_SCORE,
    F_PSEUDOCOST_SCORE_RELATIVE,
    F_PSEUDOCOST_SCORE_RANK,
    F_INFERENCES_DOWN,
    F_INFERENCES_UP,
    F_CUTOFFS_DOWN,
    F_CUTOFFS_UP,
    F_CONFLICT_SCORE_RELATIVE,
    F_OBJECTIVE_GAP,
    F_HAS_INCUMBENT
};

double MISbranchingObjectiveGap(
    double lp_objective,
    double primal_bound,
    int primal_bound_valid
)
{
    double gap;

    if( !primal_bound_valid )
        return -1.0;

    gap = primal_bound - lp_objective;

    return (gap > 0.0) ? gap : 0.0;
}

const char* MISbranchingFeatureName(int feature)
{
    if( feature < 0 || feature >= MIS_BRANCHING_FEATURE_COUNT )
        return NULL;

    return FEATURE_NAMES[feature];
}

int MISbranchingFeatureWorkSize(int number_of_vertices)
{
    /* Per vertex: LP value, fractional degree, neighbour LP sum, the two
     * largest neighbour LP values, and two lists used to grow cliques. */
    return number_of_vertices > 0 ? 7 * number_of_vertices : 0;
}

/* Whether u and w are adjacent: in the bit matrix when the graph has one,
 * else by binary search in the sorted list of u. */
static
int areAdjacent(
    const MISFeatureGraph* graph,
    int u,
    int w
)
{
    int low;
    int high;

    if( graph->adjacency_matrix != NULL )
    {
        const unsigned char* row = graph->adjacency_matrix
            + (size_t)u * MIS_ADJACENCY_MATRIX_ROW_BYTES(graph->number_of_vertices);

        return (row[w >> 3] >> (w & 7)) & 1;
    }

    low = graph->adjacency_offsets[u];
    high = graph->adjacency_offsets[u + 1] - 1;

    while( low <= high )
    {
        const int middle = low + (high - low) / 2;
        const int neighbour = graph->adjacency[middle];

        if( neighbour == w )
            return 1;

        if( neighbour < w )
            low = middle + 1;
        else
            high = middle - 1;
    }

    return 0;
}

/*
 * LP weight and size of a clique containing the candidate and one of its
 * fractional neighbours, grown greedily. ordered holds the fractional
 * neighbours of the candidate by decreasing LP value; the clique starts
 * with ordered[seed], then takes, in the order of the list, every other
 * vertex that is adjacent to all its members. The candidate itself is
 * counted in both results.
 */
static
double growClique(
    const MISFeatureGraph* graph,
    const double* lp_values,
    double candidate_value,
    const double* ordered,
    int number_of_ordered,
    int seed,
    double* members,
    int* size
)
{
    double weight = candidate_value;
    int count = 0;
    int i;
    int m;

    members[count++] = ordered[seed];
    weight += lp_values[(int)ordered[seed]];

    for( i = 0; i < number_of_ordered; ++i )
    {
        const int vertex = (int)ordered[i];
        int adjacent = (i != seed);

        for( m = 0; adjacent && m < count; ++m )
            adjacent = areAdjacent(graph, (int)members[m], vertex);

        if( adjacent )
        {
            members[count++] = ordered[i];
            weight += lp_values[vertex];
        }
    }

    *size = count + 1;

    return weight;
}

static
int isFreeState(MISVertexState state)
{
    return state == MIS_VERTEX_FREE_LP_ZERO
        || state == MIS_VERTEX_FREE_LP_ONE
        || state == MIS_VERTEX_FRACTIONAL;
}

int MIScomputeBranchingFeatures(
    const MISFeatureGraph* graph,
    const MISBranchingNode* node,
    double* work,
    double* features
)
{
    const MISVertexState* vertex_states;
    const int* candidate_vertices;
    const double* candidate_lp_values;
    const MISCandidateHistory* candidate_histories;
    int node_depth;
    int number_of_candidates;
    double max_pseudocost_score = 0.0;
    double max_conflict_score = 0.0;
    double* lp_values;
    double* fractional_degrees;
    double* neighbour_sums;
    double* largest;
    double* second_largest;
    double* ordered;
    double* members;
    double max_lp_value = 0.0;
    double max_neighbour_lp_sum = 0.0;
    double max_fractional_degree = 0.0;
    double candidate_fraction;
    double fixed_fraction;
    int number_of_fractional = 0;
    int number_of_fixed = 0;
    int n;
    int v;
    int c;
    int e;

    if( graph == NULL || node == NULL || node->number_of_candidates < 0 )
        return 1;

    vertex_states = node->vertex_states;
    candidate_vertices = node->candidate_vertices;
    candidate_lp_values = node->candidate_lp_values;
    candidate_histories = node->candidate_histories;
    node_depth = node->node_depth;
    number_of_candidates = node->number_of_candidates;

    n = graph->number_of_vertices;

    if( n < 0 )
        return 1;

    if( n > 0
        && (graph->adjacency_offsets == NULL || vertex_states == NULL
            || work == NULL) )
        return 1;

    if( number_of_candidates > 0
        && (candidate_vertices == NULL || candidate_lp_values == NULL
            || candidate_histories == NULL || features == NULL) )
        return 1;

    lp_values = work;
    fractional_degrees = (n > 0) ? work + n : work;
    neighbour_sums = (n > 0) ? work + 2 * n : work;
    largest = (n > 0) ? work + 3 * n : work;
    second_largest = (n > 0) ? work + 4 * n : work;
    ordered = (n > 0) ? work + 5 * n : work;
    members = (n > 0) ? work + 6 * n : work;

    /* LP value of every vertex, from its state. Fractional vertices get
     * a negative marker until their candidate provides the value. */
    for( v = 0; v < n; ++v )
    {
        switch( vertex_states[v] )
        {
        case MIS_VERTEX_FIXED_ZERO:
        case MIS_VERTEX_FREE_LP_ZERO:
            lp_values[v] = 0.0;
            break;

        case MIS_VERTEX_FIXED_ONE:
        case MIS_VERTEX_FREE_LP_ONE:
            lp_values[v] = 1.0;
            break;

        case MIS_VERTEX_FRACTIONAL:
            lp_values[v] = -1.0;
            ++number_of_fractional;
            break;

        default:
            /* Unknown state: the features are not defined. */
            return 1;
        }

        if( vertex_states[v] == MIS_VERTEX_FIXED_ZERO
            || vertex_states[v] == MIS_VERTEX_FIXED_ONE )
            ++number_of_fixed;
    }

    /* The candidates must be exactly the fractional vertices. */
    if( number_of_fractional != number_of_candidates )
        return 1;

    for( c = 0; c < number_of_candidates; ++c )
    {
        const int vertex = candidate_vertices[c];
        const double value = candidate_lp_values[c];

        if( vertex < 0 || vertex >= n
            || vertex_states[vertex] != MIS_VERTEX_FRACTIONAL
            || lp_values[vertex] >= 0.0
            || !(value > 0.0 && value < 1.0) )
            return 1;

        lp_values[vertex] = value;
    }

    /* Fractional degree of every vertex. */
    for( v = 0; v < n; ++v )
    {
        int count = 0;

        for( e = graph->adjacency_offsets[v];
             e < graph->adjacency_offsets[v + 1];
             ++e )
        {
            const int neighbour = graph->adjacency[e];

            if( neighbour < 0 || neighbour >= n || neighbour == v )
                return 1;

            if( vertex_states[neighbour] == MIS_VERTEX_FRACTIONAL )
                ++count;
        }

        fractional_degrees[v] = (double)count;
    }

    /* Without candidates there is nothing to compute (and n may be 0). */
    if( number_of_candidates == 0 )
        return 0;

    /* LP values around every vertex: their sum and the two largest. */
    for( v = 0; v < n; ++v )
    {
        double sum = 0.0;
        double first_value = 0.0;
        double second_value = 0.0;

        for( e = graph->adjacency_offsets[v];
             e < graph->adjacency_offsets[v + 1];
             ++e )
        {
            const double value = lp_values[graph->adjacency[e]];

            sum += value;

            if( value > first_value )
            {
                second_value = first_value;
                first_value = value;
            }
            else if( value > second_value )
                second_value = value;
        }

        neighbour_sums[v] = sum;
        largest[v] = first_value;
        second_largest[v] = second_value;
    }

    candidate_fraction = (double)number_of_candidates / (double)n;
    fixed_fraction = (double)number_of_fixed / (double)n;

    /* Features that only depend on the candidate and its neighbourhood. */
    for( c = 0; c < number_of_candidates; ++c )
    {
        const int vertex = candidate_vertices[c];
        const int first = graph->adjacency_offsets[vertex];
        const int last = graph->adjacency_offsets[vertex + 1];
        const double value = lp_values[vertex];
        double* row = features + (size_t)c * MIS_BRANCHING_FEATURE_COUNT;
        double neighbour_lp_sum = 0.0;
        double neighbour_lp_max = 0.0;
        double neighbour_fractional_degree_sum = 0.0;
        double two_hop_lp_sum = 0.0;
        double neighbour_lp_min = 1.0;
        double down_room = 0.0;
        double clique_weight = value;
        double clique_weight_sum = 0.0;
        double clique_weight_min = value;
        int free_degree = 0;
        int tight_edges = 0;
        int triangles = 0;
        int blocked_neighbours = 0;
        int number_of_ordered = 0;
        int clique_size = 1;
        int tight_cliques = 0;
        int f;

        for( e = first; e < last; ++e )
        {
            const int neighbour = graph->adjacency[e];
            const double neighbour_value = lp_values[neighbour];

            neighbour_lp_sum += neighbour_value;
            two_hop_lp_sum += neighbour_sums[neighbour];

            if( neighbour_value > neighbour_lp_max )
                neighbour_lp_max = neighbour_value;

            if( isFreeState(vertex_states[neighbour]) )
            {
                /* Largest LP value around the neighbour, the candidate
                 * excluded: it bounds how much the neighbour can grow
                 * when the candidate is set to 0. */
                const double others =
                    (value >= largest[neighbour])
                    ? second_largest[neighbour] : largest[neighbour];
                const double room = 1.0 - neighbour_value - others;

                if( room > 0.0 )
                    down_room += room;

                if( value > others )
                    ++blocked_neighbours;

                ++free_degree;

                if( neighbour_value + value
                    >= 1.0 - MIS_BRANCHING_FEATURE_TIGHT_TOLERANCE )
                    ++tight_edges;
            }

            if( vertex_states[neighbour] == MIS_VERTEX_FRACTIONAL )
            {
                int position = number_of_ordered++;

                /* Insert by decreasing LP value; ties keep the vertex
                 * order of the adjacency list. */
                while( position > 0
                    && lp_values[(int)ordered[position - 1]]
                        < neighbour_value )
                {
                    ordered[position] = ordered[position - 1];
                    --position;
                }

                ordered[position] = (double)neighbour;

                if( neighbour_value < neighbour_lp_min )
                    neighbour_lp_min = neighbour_value;

                neighbour_fractional_degree_sum +=
                    fractional_degrees[neighbour];
            }
        }

        /* Edges between two fractional neighbours, each counted once:
         * every pair of the list is tested, f^2 / 2 tests instead of one
         * per fractional neighbour and neighbour. */
        for( f = 0; f < number_of_ordered; ++f )
        {
            int g;

            for( g = f + 1; g < number_of_ordered; ++g )
            {
                if( areAdjacent(graph, (int)ordered[f], (int)ordered[g]) )
                    ++triangles;
            }
        }

        /* Cliques through the candidate, one per fractional neighbour:
         * a clique whose LP values add up to 1 is a tight clique
         * inequality, whether or not the solver separated it. */
        for( f = 0; f < number_of_ordered; ++f )
        {
            int size;
            const double weight = growClique(
                graph, lp_values, value, ordered, number_of_ordered, f,
                members, &size
            );

            if( f == 0 )
            {
                clique_weight = weight;
                clique_weight_min = weight;
                clique_size = size;
            }

            if( weight < clique_weight_min )
                clique_weight_min = weight;

            if( weight >= 1.0 - MIS_BRANCHING_FEATURE_TIGHT_TOLERANCE )
                ++tight_cliques;

            clique_weight_sum += weight;
        }

        row[F_TWO_HOP_LP_SUM] = two_hop_lp_sum;
        row[F_NEIGHBOUR_LP_MIN] = neighbour_lp_min;
        row[F_DOWN_ROOM] = down_room;
        row[F_BLOCKED_NEIGHBOURS] = (double)blocked_neighbours;
        row[F_CLIQUE_WEIGHT] = clique_weight;
        row[F_CLIQUE_SIZE] = (double)clique_size;
        row[F_TIGHT_CLIQUES] = (double)tight_cliques;
        row[F_CLIQUE_WEIGHT_MEAN] =
            (number_of_ordered > 0)
            ? clique_weight_sum / (double)number_of_ordered : value;
        row[F_CLIQUE_WEIGHT_MIN] = clique_weight_min;

        {
            const MISCandidateHistory* history = &candidate_histories[c];
            double down_gain = history->pseudocost_down * value;
            double up_gain = history->pseudocost_up * (1.0 - value);

            if( !(down_gain > MIS_BRANCHING_FEATURE_PSEUDOCOST_EPSILON) )
                down_gain = MIS_BRANCHING_FEATURE_PSEUDOCOST_EPSILON;

            if( !(up_gain > MIS_BRANCHING_FEATURE_PSEUDOCOST_EPSILON) )
                up_gain = MIS_BRANCHING_FEATURE_PSEUDOCOST_EPSILON;

            row[F_PSEUDOCOST_DOWN] = history->pseudocost_down;
            row[F_PSEUDOCOST_UP] = history->pseudocost_up;
            row[F_PSEUDOCOST_COUNT_DOWN] = history->pseudocost_count_down;
            row[F_PSEUDOCOST_COUNT_UP] = history->pseudocost_count_up;
            row[F_PSEUDOCOST_SCORE] = down_gain * up_gain;
            row[F_INFERENCES_DOWN] = history->inferences_down;
            row[F_INFERENCES_UP] = history->inferences_up;
            row[F_CUTOFFS_DOWN] = history->cutoffs_down;
            row[F_CUTOFFS_UP] = history->cutoffs_up;

            if( row[F_PSEUDOCOST_SCORE] > max_pseudocost_score )
                max_pseudocost_score = row[F_PSEUDOCOST_SCORE];

            if( history->conflict_score > max_conflict_score )
                max_conflict_score = history->conflict_score;
        }

        row[F_OBJECTIVE_GAP] =
            (node->objective_gap > 0.0) ? node->objective_gap : 0.0;
        row[F_HAS_INCUMBENT] = (node->objective_gap >= 0.0) ? 1.0 : 0.0;

        row[F_LP_VALUE] = value;
        row[F_FRACTIONALITY] = (value < 0.5) ? value : 1.0 - value;
        row[F_DEGREE] = (double)(last - first);
        row[F_FREE_DEGREE] = (double)free_degree;
        row[F_FRACTIONAL_DEGREE] = fractional_degrees[vertex];
        row[F_NEIGHBOUR_LP_SUM] = neighbour_lp_sum;
        row[F_NEIGHBOUR_LP_MAX] = neighbour_lp_max;
        row[F_TIGHT_EDGES] = (double)tight_edges;
        row[F_UP_MASS_CHANGE] = neighbour_lp_sum - (1.0 - value);
        row[F_NEIGHBOUR_FRACTIONAL_DEGREE_SUM] =
            neighbour_fractional_degree_sum;
        row[F_FRACTIONAL_TRIANGLES] = (double)triangles;
        row[F_LP_VALUE_TIMES_NEIGHBOUR_SUM] = value * neighbour_lp_sum;
        row[F_NODE_DEPTH] = (double)node_depth;
        row[F_CANDIDATE_FRACTION] = candidate_fraction;
        row[F_FIXED_FRACTION] = fixed_fraction;

        if( value > max_lp_value )
            max_lp_value = value;

        if( neighbour_lp_sum > max_neighbour_lp_sum )
            max_neighbour_lp_sum = neighbour_lp_sum;

        if( fractional_degrees[vertex] > max_fractional_degree )
            max_fractional_degree = fractional_degrees[vertex];
    }

    /* Features relative to the other candidates of the node. */
    for( c = 0; c < number_of_candidates; ++c )
    {
        double* row = features + (size_t)c * MIS_BRANCHING_FEATURE_COUNT;

        row[F_LP_VALUE_RELATIVE] = row[F_LP_VALUE] / max_lp_value;

        row[F_NEIGHBOUR_LP_SUM_RELATIVE] =
            (max_neighbour_lp_sum > 0.0)
            ? row[F_NEIGHBOUR_LP_SUM] / max_neighbour_lp_sum : 0.0;

        row[F_FRACTIONAL_DEGREE_RELATIVE] =
            (max_fractional_degree > 0.0)
            ? row[F_FRACTIONAL_DEGREE] / max_fractional_degree : 0.0;

        row[F_PSEUDOCOST_SCORE_RELATIVE] =
            row[F_PSEUDOCOST_SCORE] / max_pseudocost_score;

        row[F_CONFLICT_SCORE_RELATIVE] =
            (max_conflict_score > 0.0)
            ? candidate_histories[c].conflict_score / max_conflict_score
            : 0.0;
    }

    /* Ranks: share of the candidates of the node with a smaller value. */
    for( c = 0; c < number_of_candidates; ++c )
    {
        double* row = features + (size_t)c * MIS_BRANCHING_FEATURE_COUNT;
        int smaller_lp_value = 0;
        int smaller_neighbour_lp_sum = 0;
        int smaller_up_mass_change = 0;
        int smaller_pseudocost_score = 0;
        int other;

        for( other = 0; other < number_of_candidates; ++other )
        {
            const double* other_row =
                features + (size_t)other * MIS_BRANCHING_FEATURE_COUNT;

            if( other_row[F_LP_VALUE] < row[F_LP_VALUE] )
                ++smaller_lp_value;

            if( other_row[F_NEIGHBOUR_LP_SUM] < row[F_NEIGHBOUR_LP_SUM] )
                ++smaller_neighbour_lp_sum;

            if( other_row[F_UP_MASS_CHANGE] < row[F_UP_MASS_CHANGE] )
                ++smaller_up_mass_change;

            if( other_row[F_PSEUDOCOST_SCORE] < row[F_PSEUDOCOST_SCORE] )
                ++smaller_pseudocost_score;
        }

        row[F_LP_VALUE_RANK] =
            (double)smaller_lp_value / (double)number_of_candidates;
        row[F_NEIGHBOUR_LP_SUM_RANK] =
            (double)smaller_neighbour_lp_sum / (double)number_of_candidates;
        row[F_UP_MASS_CHANGE_RANK] =
            (double)smaller_up_mass_change / (double)number_of_candidates;
        row[F_PSEUDOCOST_SCORE_RANK] =
            (double)smaller_pseudocost_score / (double)number_of_candidates;
    }

    return 0;
}
