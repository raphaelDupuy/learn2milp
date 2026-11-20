#include "learning/MISAdjacency.hpp"

#include <algorithm>
#include <cstddef>

MISFeatureGraph MISAdjacency::view() const
{
    MISFeatureGraph graph;

    graph.number_of_vertices = numberOfVertices;
    graph.adjacency_offsets = offsets.data();
    graph.adjacency = neighbours.data();
    graph.adjacency_matrix = matrix.empty() ? nullptr : matrix.data();

    return graph;
}

MISAdjacency buildMISAdjacency(const MISInstance& instance)
{
    MISAdjacency adjacency;

    adjacency.numberOfVertices = instance.n;
    adjacency.offsets.assign(instance.n + 1, 0);

    // Count the neighbours of each vertex, then turn counts into offsets.
    for (const auto& [u, v] : instance.edges)
    {
        ++adjacency.offsets[u + 1];
        ++adjacency.offsets[v + 1];
    }

    for (int v = 0; v < instance.n; ++v)
    {
        adjacency.offsets[v + 1] += adjacency.offsets[v];
    }

    adjacency.neighbours.resize(2 * instance.edges.size());

    std::vector<int> next(
        adjacency.offsets.begin(),
        adjacency.offsets.end() - 1
    );

    for (const auto& [u, v] : instance.edges)
    {
        adjacency.neighbours[next[u]++] = v;
        adjacency.neighbours[next[v]++] = u;
    }

    for (int v = 0; v < instance.n; ++v)
    {
        std::sort(
            adjacency.neighbours.begin() + adjacency.offsets[v],
            adjacency.neighbours.begin() + adjacency.offsets[v + 1]
        );
    }

    if (instance.n > 0 && instance.n <= MIS_ADJACENCY_MATRIX_MAX_VERTICES)
    {
        const std::size_t row = MIS_ADJACENCY_MATRIX_ROW_BYTES(instance.n);

        adjacency.matrix.assign(row * instance.n, 0);

        for (const auto& [u, v] : instance.edges)
        {
            adjacency.matrix[u * row + v / 8] |= static_cast<unsigned char>(1u << (v % 8));
            adjacency.matrix[v * row + u / 8] |= static_cast<unsigned char>(1u << (u % 8));
        }
    }

    return adjacency;
}
