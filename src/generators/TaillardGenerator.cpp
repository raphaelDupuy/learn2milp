#include "generators/TaillardGenerator.hpp"

#include <cstdint>
#include <random>
#include <stdexcept>
#include <utility>

namespace
{

constexpr int TAILLARD_MIN_DURATION = 1;
constexpr int TAILLARD_MAX_DURATION = 99;

/*
 * Taillard's unif(seed, low, high), as published in Pascal: one step of
 * the Lehmer generator (Schrage's method, no overflow on 32 bits), then a
 * value in [low, high]. The seed stays in [1, 2^31 - 2].
 */
int uniform(int& seed, int low, int high)
{
    constexpr std::int64_t m = 2147483647;
    constexpr std::int64_t a = 16807;
    constexpr std::int64_t b = 127773;
    constexpr std::int64_t c = 2836;

    const std::int64_t k = seed / b;
    std::int64_t next = a * (seed % b) - k * c;

    if (next < 0)
        next += m;

    seed = static_cast<int>(next);

    const double value01 = static_cast<double>(next) / static_cast<double>(m);
    return low + static_cast<int>(value01 * (high - low + 1));
}

// A seed of Taillard's generator, uniform in [1, TAILLARD_SEED_MAX].
int drawTaillardSeed(std::mt19937& rng)
{
    for (;;)
    {
        const std::uint32_t value = static_cast<std::uint32_t>(rng()) & 0x7fffffffu;

        if (value >= 1 && value <= static_cast<std::uint32_t>(TAILLARD_SEED_MAX))
            return static_cast<int>(value);
    }
}

} // namespace

TaillardConfig taillardConfigFromSeed(
    int jobs,
    int machines,
    unsigned int seed
)
{
    std::mt19937 rng(seed);

    TaillardConfig config;
    config.jobs = jobs;
    config.machines = machines;
    config.timeSeed = drawTaillardSeed(rng);
    config.machineSeed = drawTaillardSeed(rng);
    config.seed = seed;

    return config;
}

void TaillardGenerator::validate(const TaillardConfig& config)
{
    if (config.jobs < 1)
        throw std::invalid_argument("Taillard parameter jobs must be at least 1");

    if (config.machines < 1)
    {
        throw std::invalid_argument(
            "Taillard parameter machines must be at least 1");
    }

    if (config.timeSeed < 1 || config.timeSeed > TAILLARD_SEED_MAX
        || config.machineSeed < 1 || config.machineSeed > TAILLARD_SEED_MAX)
    {
        throw std::invalid_argument(
            "Taillard seeds must lie in [1, "
            + std::to_string(TAILLARD_SEED_MAX) + "]");
    }
}

GeneratedJSSPInstance TaillardGenerator::generate(
    const TaillardConfig& config
) const
{
    validate(config);

    const int jobs = config.jobs;
    const int machines = config.machines;
    int timeSeed = config.timeSeed;
    int machineSeed = config.machineSeed;

    GeneratedJSSPInstance generated;
    JSSPInstance& instance = generated.instance;
    instance.numberOfMachines = machines;
    instance.jobs.assign(jobs, std::vector<JSSPOperation>(machines));

    // As published: every duration first, job by job, then the machine
    // orders, each a shuffle of 1..machines by successive swaps.
    for (int j = 0; j < jobs; ++j)
    {
        for (int k = 0; k < machines; ++k)
        {
            instance.jobs[j][k].duration = uniform(
                timeSeed, TAILLARD_MIN_DURATION, TAILLARD_MAX_DURATION);
        }
    }

    for (int j = 0; j < jobs; ++j)
    {
        std::vector<int> order(machines);

        for (int k = 0; k < machines; ++k)
            order[k] = k;

        for (int k = 0; k < machines; ++k)
        {
            // Taillard draws a position in [k + 1, machines], 1-based.
            const int other = uniform(machineSeed, k + 1, machines) - 1;
            std::swap(order[k], order[other]);
        }

        for (int k = 0; k < machines; ++k)
            instance.jobs[j][k].machine = order[k];
    }

    InstanceMetadata& metadata = generated.metadata;
    metadata.problemType = "JSSP";
    metadata.generatorType = "taillard";
    metadata.generatorVersion = "1.0";
    metadata.generatorSeed = config.seed;
    metadata.generatorParameters = {
        {"jobs", std::to_string(jobs)},
        {"machines", std::to_string(machines)},
        {"time_seed", std::to_string(config.timeSeed)},
        {"machine_seed", std::to_string(config.machineSeed)},
        {"min_duration", std::to_string(TAILLARD_MIN_DURATION)},
        {"max_duration", std::to_string(TAILLARD_MAX_DURATION)}};

    return generated;
}

std::string taillardInstanceId(const TaillardConfig& config)
{
    const std::string size = "taillard_j" + std::to_string(config.jobs)
        + "_m" + std::to_string(config.machines);

    if (config.seed)
        return size + "_seed" + std::to_string(*config.seed);

    return size + "_ts" + std::to_string(config.timeSeed)
        + "_ms" + std::to_string(config.machineSeed);
}
