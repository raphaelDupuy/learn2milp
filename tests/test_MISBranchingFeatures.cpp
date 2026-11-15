#include "learning/MISAdjacency.hpp"
#include "learning/MISBranchingFeatures.h"

#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace
{

int failures = 0;

void check(bool condition, const std::string& testName)
{
    if (condition)
    {
        std::cout << "[PASS] " << testName << '\n';
    }
    else
    {
        std::cerr << "[FAIL] " << testName << '\n';
        ++failures;
    }
}

/*
 * Reference node, with values computed by hand.
 *
 *   0 - 1        vertices 0, 1, 2: fractional (0.5, 0.25, 0.5)
 *   | \ |        vertex 3: free, LP value 0
 *   5   2 - 3    vertex 4: fixed to 1
 *   |       |    vertex 5: fixed to 0
 *   + - 4 - +
 */
struct Reference
{
    MISInstance instance{
        6,
        {{0, 1}, {0, 2}, {0, 5}, {1, 2}, {2, 3}, {3, 4}, {4, 5}}
    };

    MISAdjacency adjacency = buildMISAdjacency(instance);

    std::vector<MISVertexState> states{
        MIS_VERTEX_FRACTIONAL,
        MIS_VERTEX_FRACTIONAL,
        MIS_VERTEX_FRACTIONAL,
        MIS_VERTEX_FREE_LP_ZERO,
        MIS_VERTEX_FIXED_ONE,
        MIS_VERTEX_FIXED_ZERO
    };

    // Candidates deliberately not in vertex order.
    std::vector<int> vertices{2, 0, 1};
    std::vector<double> values{0.5, 0.5, 0.25};
    int depth = 3;

    std::vector<double> work = std::vector<double>(
        MISbranchingFeatureWorkSize(6), -7.0);
    std::vector<double> features = std::vector<double>(
        3 * MIS_BRANCHING_FEATURE_COUNT, -7.0);

    int compute()
    {
        const MISFeatureGraph graph = adjacency.view();

        return MIScomputeBranchingFeatures(
            &graph,
            states.data(),
            depth,
            static_cast<int>(vertices.size()),
            vertices.data(),
            values.data(),
            work.data(),
            features.data()
        );
    }
};

bool rowEquals(
    const std::vector<double>& features,
    int row,
    const std::vector<double>& expected
)
{
    bool equal = expected.size() == MIS_BRANCHING_FEATURE_COUNT;

    for (int f = 0; equal && f < MIS_BRANCHING_FEATURE_COUNT; ++f)
    {
        const double value =
            features[row * MIS_BRANCHING_FEATURE_COUNT + f];

        if (std::fabs(value - expected[f]) > 1e-12)
        {
            std::cerr << "  row " << row << ", "
                      << MISbranchingFeatureName(f) << ": got " << value
                      << ", expected " << expected[f] << '\n';
            equal = false;
        }
    }

    return equal;
}

void testAdjacency()
{
    const Reference reference;
    const auto& adjacency = reference.adjacency;

    check(
        adjacency.offsets == std::vector<int>({0, 3, 5, 8, 10, 12, 14})
        && adjacency.neighbours == std::vector<int>(
            {1, 2, 5, 0, 2, 0, 1, 3, 2, 4, 3, 5, 0, 4}),
        "Adjacency lists are complete and sorted"
    );

    const MISAdjacency empty = buildMISAdjacency(MISInstance{0, {}});

    check(
        empty.offsets == std::vector<int>({0}) && empty.neighbours.empty(),
        "Empty graph has an empty adjacency"
    );
}

void testReferenceValues()
{
    Reference reference;

    check(reference.compute() == 0, "Reference node is accepted");

    // Columns: lp_value, fractionality, degree, free_degree,
    // fractional_degree, neighbour_lp_sum, neighbour_lp_max, tight_edges,
    // up_mass_change, neighbour_fractional_degree_sum,
    // fractional_triangles, lp_value_times_neighbour_sum,
    // lp_value_relative, neighbour_lp_sum_relative,
    // fractional_degree_relative, node_depth, candidate_fraction,
    // fixed_fraction; then two_hop_lp_sum, neighbour_lp_min, down_room,
    // blocked_neighbours, clique_weight, clique_size, tight_cliques,
    // clique_weight_mean, clique_weight_min, lp_value_rank,
    // neighbour_lp_sum_rank, up_mass_change_rank.
    //
    // Vertices 0, 1 and 2 form a triangle of LP weight 1.25: every
    // candidate finds it from each of its two fractional neighbours.
    const double third = 1.0 / 3.0;

    check(
        rowEquals(reference.features, 0, {
            0.5, 0.5, 3, 3, 2, 0.75, 0.5, 1, 0.25, 4, 1, 0.375,
            1.0, 0.75, 1.0, 3, 0.5, third,
            3.25, 0.25, 0.5, 1, 1.25, 3, 2, 1.25, 1.25, third, 0, 0}),
        "Vertex 2: free neighbour at 0 counts as free, not fractional"
    );

    check(
        rowEquals(reference.features, 1, {
            0.5, 0.5, 3, 2, 2, 0.75, 0.5, 1, 0.25, 4, 1, 0.375,
            1.0, 0.75, 1.0, 3, 0.5, third,
            3.25, 0.25, 0.5, 1, 1.25, 3, 2, 1.25, 1.25, third, 0, 0}),
        "Vertex 0: fixed neighbour is neither free nor fractional"
    );

    check(
        rowEquals(reference.features, 2, {
            0.25, 0.25, 2, 2, 2, 1.0, 0.5, 0, 0.25, 4, 1, 0.25,
            0.5, 1.0, 1.0, 3, 0.5, third,
            1.5, 0.5, 0, 0, 1.25, 3, 2, 1.25, 1.25, 0, 2 * third, 0}),
        "Vertex 1: no tight edge, largest neighbour sum, no room"
    );
}

void testOrderIndependence()
{
    // Reordering the candidates only reorders the rows.
    Reference first;
    Reference second;

    second.vertices = {0, 1, 2};
    second.values = {0.5, 0.25, 0.5};

    const bool computed = first.compute() == 0 && second.compute() == 0;

    const auto row = [](const Reference& reference, int index)
    {
        return std::vector<double>(
            reference.features.begin()
                + index * MIS_BRANCHING_FEATURE_COUNT,
            reference.features.begin()
                + (index + 1) * MIS_BRANCHING_FEATURE_COUNT);
    };

    check(
        computed
        && row(first, 0) == row(second, 2)
        && row(first, 1) == row(second, 0)
        && row(first, 2) == row(second, 1),
        "Candidate order only changes the row order"
    );
}

void testNames()
{
    std::set<std::string> names;
    bool valid = true;

    for (int f = 0; f < MIS_BRANCHING_FEATURE_COUNT; ++f)
    {
        const char* name = MISbranchingFeatureName(f);

        valid = valid && name != nullptr && std::string(name) != "";

        if (name != nullptr)
            names.insert(name);
    }

    check(
        valid
        && names.size() == MIS_BRANCHING_FEATURE_COUNT
        && MISbranchingFeatureName(-1) == nullptr
        && MISbranchingFeatureName(MIS_BRANCHING_FEATURE_COUNT) == nullptr
        && std::string(MISbranchingFeatureName(0)) == "lp_value",
        "Feature names are distinct and bounded"
    );
}

template <typename Mutation>
void testRejection(const std::string& testName, Mutation mutate)
{
    Reference reference;
    mutate(reference);

    check(reference.compute() != 0, testName);
}

void testRejections()
{
    testRejection(
        "Fractional vertex without candidate is rejected",
        [](Reference& r) { r.states[3] = MIS_VERTEX_FRACTIONAL; }
    );

    testRejection(
        "Candidate that is not fractional is rejected",
        [](Reference& r) { r.vertices[0] = 3; }
    );

    testRejection(
        "Duplicate candidate is rejected",
        [](Reference& r) { r.vertices[1] = 2; }
    );

    testRejection(
        "Candidate outside the vertex range is rejected",
        [](Reference& r) { r.vertices[2] = 6; }
    );

    testRejection(
        "Integral candidate LP value is rejected",
        [](Reference& r) { r.values[0] = 1.0; }
    );

    testRejection(
        "NaN candidate LP value is rejected",
        [](Reference& r) { r.values[0] = std::nan(""); }
    );

    testRejection(
        "Unknown vertex state is rejected",
        [](Reference& r) { r.states[5] = MIS_VERTEX_UNKNOWN; }
    );

    Reference reference;
    const MISFeatureGraph graph = reference.adjacency.view();

    check(
        MIScomputeBranchingFeatures(
            nullptr, reference.states.data(), 0, 0,
            nullptr, nullptr, reference.work.data(), nullptr) != 0
        && MIScomputeBranchingFeatures(
            &graph, nullptr, 0, 3, reference.vertices.data(),
            reference.values.data(), reference.work.data(),
            reference.features.data()) != 0,
        "Missing arrays are rejected"
    );
}

void testNoCandidate()
{
    // A node without fractional vertex has no row to compute.
    Reference reference;

    reference.states = {
        MIS_VERTEX_FIXED_ONE, MIS_VERTEX_FIXED_ZERO, MIS_VERTEX_FIXED_ZERO,
        MIS_VERTEX_FREE_LP_ONE, MIS_VERTEX_FIXED_ZERO, MIS_VERTEX_FIXED_ZERO
    };
    reference.vertices.clear();
    reference.values.clear();

    check(
        reference.compute() == 0 && reference.features[0] == -7.0,
        "Node without candidate is accepted and writes nothing"
    );

    const MISAdjacency empty = buildMISAdjacency(MISInstance{0, {}});
    const MISFeatureGraph graph = empty.view();

    check(
        MIScomputeBranchingFeatures(
            &graph, nullptr, 0, 0, nullptr, nullptr, nullptr, nullptr) == 0
        && MISbranchingFeatureWorkSize(0) == 0,
        "Empty graph is accepted"
    );
}

} // namespace

int main()
{
    testAdjacency();
    testReferenceValues();
    testOrderIndependence();
    testNames();
    testRejections();
    testNoCandidate();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS branching feature tests passed.\n";
    return 0;
}
