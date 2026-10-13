#include "generators/ErdosRenyiGenerator.hpp"

#include <random>
#include <stdexcept>
#include <utility>

GeneratedMISInstance ErdosRenyiGenerator::generate(
    const ErdosRenyiConfig& config
) const
{
    if (config.n < 0)
    {
        throw std::invalid_argument(
            "Erdos-Renyi parameter n must be non-negative"
        );
    }

    if (config.p < 0.0 || config.p > 1.0)
    {
        throw std::invalid_argument(
            "Erdos-Renyi parameter p must be in [0, 1]"
        );
    }

    MISInstance graph;
    graph.n = config.n;

    std::mt19937 rng(config.seed);
    std::bernoulli_distribution edgeDistribution(config.p);

    for (int u = 0; u < config.n; ++u)
    {
        for (int v = u + 1; v < config.n; ++v)
        {
            if (edgeDistribution(rng))
            {
                graph.edges.emplace_back(u, v);
            }
        }
    }

    MISGenerationMetadata metadata{
        "erdos_renyi",
        config.seed,
        config.n,
        config.p,
        "1.0"
    };

    return GeneratedMISInstance{
        std::move(graph),
        std::move(metadata)
    };
}
