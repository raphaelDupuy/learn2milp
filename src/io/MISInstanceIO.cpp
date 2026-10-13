#include "io/MISInstanceIO.hpp"

#include <istream>
#include <ostream>
#include <set>
#include <string>
#include <utility>

namespace
{
bool validateMISInstance(
    const MISInstance& instance,
    std::string& errorMessage
)
{
    if (instance.n < 0)
    {
        errorMessage = "Number of vertices must be non-negative.";
        return false;
    }

    std::set<std::pair<int, int>> uniqueEdges;

    for (const auto& [u, v] : instance.edges)
    {
        if (u < 0 || v < 0 || u >= instance.n || v >= instance.n)
        {
            errorMessage = "Edge endpoint is outside the valid vertex range.";
            return false;
        }

        if (u >= v)
        {
            errorMessage = "Each edge must satisfy u < v; loops are not allowed.";
            return false;
        }

        if (!uniqueEdges.emplace(u, v).second)
        {
            errorMessage = "Duplicate edge detected.";
            return false;
        }
    }

    return true;
}
}

bool writeMISInstance(
    std::ostream& output,
    const MISInstance& instance,
    std::string& errorMessage
)
{
    if (!validateMISInstance(instance, errorMessage))
    {
        return false;
    }

    output << "MIS_INSTANCE 1\n";
    output << "n " << instance.n << '\n';
    output << "m " << instance.edges.size() << '\n';

    for (const auto& [u, v] : instance.edges)
    {
        output << u << ' ' << v << '\n';
    }

    if (!output)
    {
        errorMessage = "Failed to write MIS instance.";
        return false;
    }

    errorMessage.clear();
    return true;
}

bool readMISInstance(
    std::istream& input,
    MISInstance& instance,
    std::string& errorMessage
)
{
    MISInstance parsed;
    std::string token;
    int version = 0;

    if (!(input >> token >> version)
        || token != "MIS_INSTANCE"
        || version != 1)
    {
        errorMessage = "Invalid header or unsupported MIS format version.";
        return false;
    }

    if (!(input >> token >> parsed.n) || token != "n")
    {
        errorMessage = "Expected vertex-count field: n <integer>.";
        return false;
    }

    if (parsed.n < 0)
    {
        errorMessage = "Number of vertices must be non-negative.";
        return false;
    }

    long long edgeCountInput = -1;

    if (!(input >> token >> edgeCountInput) || token != "m")
    {
        errorMessage = "Expected edge-count field: m <integer>.";
        return false;
    }

    if (edgeCountInput < 0)
    {
        errorMessage = "Number of edges must be non-negative.";
        return false;
    }

    const long long n = parsed.n;
    const long long maxPossibleEdges = n * (n - 1) / 2;

    if (edgeCountInput > maxPossibleEdges)
    {
        errorMessage = "Edge count exceeds the maximum for a simple graph.";
        return false;
    }

    const auto edgeCount = static_cast<std::size_t>(edgeCountInput);
    parsed.edges.reserve(edgeCount);

    for (std::size_t i = 0; i < edgeCount; ++i)
    {
        int u = 0;
        int v = 0;

        if (!(input >> u >> v))
        {
            errorMessage = "Unexpected end of file while reading edges.";
            return false;
        }

        parsed.edges.emplace_back(u, v);
    }

    input >> std::ws;

    if (input.peek() != std::char_traits<char>::eof())
    {
        errorMessage = "Unexpected content after the declared edges.";
        return false;
    }

    if (!validateMISInstance(parsed, errorMessage))
    {
        return false;
    }

    instance = std::move(parsed);
    errorMessage.clear();
    return true;
}
