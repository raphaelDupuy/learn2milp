#include "generators/BarabasiAlbertGenerator.hpp"

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

/*
 * Uniform integer in [0, bound), by rejection sampling on the 32-bit
 * output of the generator. 0 < bound <= 2^32.
 */
std::uint32_t uniformBelow(std::mt19937& rng, std::uint64_t bound)
{
    const std::uint64_t range = std::uint64_t{1} << 32;

    // Largest multiple of bound that does not exceed 2^32: values at or
    // above it would make the small residues more likely.
    const std::uint64_t limit = range - range % bound;

    for (;;)
    {
        const std::uint64_t value = static_cast<std::uint32_t>(rng());

        if (value < limit)
            return static_cast<std::uint32_t>(value % bound);
    }
}

} // namespace

void BarabasiAlbertGenerator::validate(const BarabasiAlbertConfig& config)
{
    if (config.k < 1)
    {
        throw std::invalid_argument(
            "Barabasi-Albert parameter k must be at least 1"
        );
    }

    if (config.n <= config.k)
    {
        throw std::invalid_argument(
            "Barabasi-Albert parameter n must be larger than k"
        );
    }

    // The degree list holds two entries per edge and is indexed by
    // 32-bit draws.
    const std::uint64_t edges =
        static_cast<std::uint64_t>(config.k)
        * static_cast<std::uint64_t>(config.n - config.k);

    if (2 * edges >= (std::uint64_t{1} << 32))
    {
        throw std::invalid_argument(
            "Barabasi-Albert graph is too large: k (n - k) must be "
            "below 2^31"
        );
    }
}

GeneratedMISInstance BarabasiAlbertGenerator::generate(
    const BarabasiAlbertConfig& config
) const
{
    validate(config);

    const int n = config.n;
    const int k = config.k;

    std::mt19937 rng(config.seed);

    MISInstance graph;
    graph.n = n;
    graph.edges.reserve(
        static_cast<std::size_t>(k) * static_cast<std::size_t>(n - k)
    );

    // Each vertex appears once per incident edge: a uniform draw from
    // this list is a draw proportional to the degree.
    std::vector<int> endpoints;
    endpoints.reserve(2 * graph.edges.capacity());

    std::vector<int> targets(k);

    for (int v = 0; v < k; ++v)
        targets[v] = v;

    for (int source = k; source < n; ++source)
    {
        for (const int target : targets)
        {
            graph.edges.emplace_back(target, source);
            endpoints.push_back(target);
            endpoints.push_back(source);
        }

        if (source + 1 == n)
            break;

        // Targets of the next vertex: k distinct vertices.
        std::set<int> chosen;

        while (static_cast<int>(chosen.size()) < k)
        {
            chosen.insert(endpoints[uniformBelow(rng, endpoints.size())]);
        }

        targets.assign(chosen.begin(), chosen.end());
    }

    // Same edge order as the other generators: by first, then second
    // endpoint.
    std::sort(graph.edges.begin(), graph.edges.end());

    InstanceMetadata metadata{
        "MIS",
        "barabasi_albert",
        "1.0",
        config.seed,
        {
            {"k", std::to_string(k)},
            {"n", std::to_string(n)}
        }
    };

    return GeneratedMISInstance{
        std::move(graph),
        std::move(metadata)
    };
}
