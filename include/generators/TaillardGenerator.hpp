#pragma once

#include <optional>
#include <string>
#include <vector>

#include "instances/InstanceMetadata.hpp"
#include "problems/JSSP.hpp"

/*
 * Job-shop instances generated as in Taillard (1993), "Benchmarks for
 * basic scheduling problems", EJOR 64: durations uniform in [1, 99], and
 * for every job a random order of the machines, both drawn with
 * Taillard's own generator (a Lehmer generator, multiplier 16807, modulus
 * 2^31 - 1) from a time seed and a machine seed. With the published seeds,
 * it gives the benchmark instances ta01 to ta80 exactly.
 */

// Seeds of Taillard's generator must lie in [1, TAILLARD_SEED_MAX].
constexpr int TAILLARD_SEED_MAX = 2147483646;

struct TaillardConfig
{
    int jobs = 0;
    int machines = 0;
    int timeSeed = 1;
    int machineSeed = 1;

    // Seed the two seeds were derived from (taillardConfigFromSeed()),
    // none when they were given directly.
    std::optional<unsigned int> seed;
};

/*
 * Derives the time and machine seeds from one seed, through std::mt19937:
 * consecutive seeds then give unrelated instances, which consecutive
 * seeds of Taillard's generator would not (its first values grow with the
 * seed).
 */
TaillardConfig taillardConfigFromSeed(
    int jobs,
    int machines,
    unsigned int seed
);

struct GeneratedJSSPInstance
{
    JSSPInstance instance;
    InstanceMetadata metadata;
};

class TaillardGenerator
{
public:
    // Throws std::invalid_argument unless jobs >= 1, machines >= 1 and
    // both seeds lie in [1, TAILLARD_SEED_MAX].
    static void validate(const TaillardConfig& config);

    GeneratedJSSPInstance generate(const TaillardConfig& config) const;
};

/*
 * Identifier of an instance, used as its file name without the .jsp
 * extension:
 *
 *   taillard_j10_m10_seed7                     (seeds derived from 7)
 *   taillard_j15_m15_ts840612802_ms398197754   (seeds given: ta01)
 */
std::string taillardInstanceId(const TaillardConfig& config);
