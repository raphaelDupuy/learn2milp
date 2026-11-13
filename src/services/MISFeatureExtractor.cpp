
#include "services/MISFeatureExtractor.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

bool extractMISFeatures(
    const MISInstance& instance,
    MISFeatures& features,
    std::string& errorMessage
)
{
    // Validate the number of vertices.
    if (instance.n < 0)
    {
        errorMessage = "The number of vertices cannot be negative.";
        return false;
    }

    // Validate the maximum possible number of edges.
    const long long n = instance.n;
    const long long maxEdges = n * (n - 1) / 2;

    if (static_cast<long long>(instance.edges.size()) > maxEdges)
    {
        errorMessage = "The number of edges exceeds the maximum possible "
                       "for a simple undirected graph.";
        return false;
    }

    // Build the adjacency lists.
    std::vector<std::vector<int>> adjacency(instance.n);
    std::set<std::pair<int, int>> uniqueEdges;

    for (const auto& edge : instance.edges)
    {
        const int u = edge.first;
        const int v = edge.second;

        if (u < 0 || u >= instance.n
            || v < 0 || v >= instance.n)
        {
            errorMessage = "An edge endpoint is outside the valid vertex range.";
            return false;
        }

        if (u == v)
        {
            errorMessage = "Self-loops are not allowed in a MIS instance.";
            return false;
        }

        const auto normalizedEdge = std::minmax(u, v);
        const std::pair<int, int> canonicalEdge(
            normalizedEdge.first,
            normalizedEdge.second
        );

        if (!uniqueEdges.insert(canonicalEdge).second)
        {
            errorMessage = "Duplicate edges are not allowed.";
            return false;
        }

        adjacency[u].push_back(v);
        adjacency[v].push_back(u);
    }

    // Initialize the feature structure.
    MISFeatures computedFeatures;

    computedFeatures.numberOfVertices = instance.n;
    computedFeatures.numberOfEdges = instance.edges.size();

    // Compute graph density.
    if (instance.n >= 2)
    {
        computedFeatures.density =
            (2.0 * static_cast<double>(instance.edges.size()))
            / (static_cast<double>(instance.n)
               * static_cast<double>(instance.n - 1));
    }

    // Compute vertex degrees.
    if (instance.n > 0)
    {
        std::vector<int> degrees(instance.n);

        long long degreeSum = 0;

        computedFeatures.minimumDegree = instance.n;
        computedFeatures.maximumDegree = 0;

        for (int vertex = 0; vertex < instance.n; ++vertex)
        {
            const int degree =
                static_cast<int>(adjacency[vertex].size());

            degrees[vertex] = degree;
            degreeSum += degree;

            computedFeatures.minimumDegree =
                std::min(computedFeatures.minimumDegree, degree);

            computedFeatures.maximumDegree =
                std::max(computedFeatures.maximumDegree, degree);
        }

        computedFeatures.averageDegree =
            static_cast<double>(degreeSum) / instance.n;

        // Compute population standard deviation.
        double squaredDifferenceSum = 0.0;

        for (const int degree : degrees)
        {
            const double difference =
                degree - computedFeatures.averageDegree;

            squaredDifferenceSum += difference * difference;
        }

        computedFeatures.degreeStdDev =
            std::sqrt(squaredDifferenceSum / instance.n);
    }

    // Compute the number of connected components using iterative DFS.
    std::vector<bool> visited(instance.n, false);
    std::vector<int> stack;

    for (int vertex = 0; vertex < instance.n; ++vertex)
    {
        if (visited[vertex])
            continue;

        ++computedFeatures.numberOfConnectedComponents;

        stack.push_back(vertex);
        visited[vertex] = true;

        while (!stack.empty())
        {
            const int current = stack.back();
            stack.pop_back();

            for (const int neighbor : adjacency[current])
            {
                if (!visited[neighbor])
                {
                    visited[neighbor] = true;
                    stack.push_back(neighbor);
                }
            }
        }
    }

    features = computedFeatures;
    errorMessage.clear();
    return true;

}
