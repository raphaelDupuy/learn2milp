#include "generators/ErdosRenyiGenerator.hpp"

#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace
{
    std::string formatDouble(double value)
    {
        std::ostringstream output;
        output << std::setprecision(
            std::numeric_limits<double>::max_digits10
        ) << value;

        return output.str();
    }
}

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

    InstanceMetadata metadata{
        "MIS",
        "erdos_renyi",
        "1.0",
        config.seed,
        {
            {"n", std::to_string(config.n)},
            {"p", formatDouble(config.p)}
        }
    };

    return GeneratedMISInstance{
        std::move(graph),
        std::move(metadata)
    };
}
