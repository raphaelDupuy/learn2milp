#include "generators/ErdosRenyiGenerator.hpp"

#include <random>
#include <stdexcept>

MISInstance ErdosRenyiGenerator::generate(
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

    MISInstance instance;
    instance.n = config.n;

    std::mt19937 rng(config.seed);
    std::bernoulli_distribution edge_distribution(config.p);

    // Generate each possible edge exactly once.
    for (int u = 0; u < config.n; ++u)
    {
        for (int v = u + 1; v < config.n; ++v)
        {
            if (edge_distribution(rng))
            {
                instance.edges.emplace_back(u, v);
            }
        }
    }

    return instance;
}
