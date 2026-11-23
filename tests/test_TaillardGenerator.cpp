#include "generators/TaillardGenerator.hpp"
#include "io/JSSPInstanceIO.hpp"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{
int failures = 0;

void check(bool condition, const std::string& message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

// ta01 as published (JSPLIB), from the seeds 840612802 and 398197754.
const char* const TA01 =
    "15 15\n"
    "6 94 12 66 4 10 7 53 3 26 2 15 10 65 11 82 8 10 14 27 9 93 13 92 5 96 0 70 1 83\n"
    "4 74 5 31 7 88 14 51 13 57 8 78 11 8 9 7 6 91 10 79 0 18 3 51 12 18 1 99 2 33\n"
    "1 4 8 82 9 40 12 86 6 50 11 54 13 21 5 6 0 54 2 68 7 82 10 20 4 39 3 35 14 68\n"
    "5 73 2 23 9 30 6 30 10 53 0 94 13 58 4 93 7 32 14 91 11 30 8 56 12 27 1 92 3 9\n"
    "7 78 8 23 6 21 10 60 4 36 9 29 2 95 14 99 12 79 5 76 1 93 13 42 11 52 0 42 3 96\n"
    "5 29 3 61 12 88 13 70 11 16 4 31 14 65 7 83 2 78 1 26 10 50 0 87 9 62 6 14 8 30\n"
    "12 18 3 75 7 20 8 4 14 91 6 68 1 19 11 54 4 85 5 73 2 43 10 24 0 37 13 87 9 66\n"
    "11 32 5 52 0 9 7 49 12 61 13 35 14 99 1 62 2 6 8 62 4 7 3 80 9 3 6 57 10 7\n"
    "10 85 11 30 6 96 14 91 0 13 1 87 2 82 5 83 12 78 4 56 8 85 7 8 9 66 13 88 3 15\n"
    "6 5 11 59 9 30 2 60 8 41 0 17 13 66 3 89 10 78 7 88 1 69 12 45 14 82 4 6 5 13\n"
    "4 90 7 27 13 1 0 8 5 91 12 80 6 89 8 49 14 32 10 28 3 90 1 93 11 6 9 35 2 73\n"
    "2 47 14 43 0 75 12 8 6 51 10 3 7 84 5 34 8 28 9 60 13 69 1 45 3 67 11 58 4 87\n"
    "5 65 8 62 10 97 2 20 3 31 6 33 9 33 0 77 13 50 4 80 1 48 11 90 12 75 7 96 14 44\n"
    "8 28 14 21 4 51 13 75 5 17 6 89 9 59 1 56 12 63 7 18 11 17 10 30 3 16 2 7 0 35\n"
    "10 57 8 16 12 42 6 34 4 37 1 26 13 68 14 73 11 5 0 8 7 12 3 87 2 83 9 20 5 97\n";

std::string written(const JSSPInstance& instance)
{
    std::ostringstream output;
    std::string error;
    writeJSSPInstance(output, instance, error);
    return output.str();
}

void testPublishedInstance()
{
    TaillardConfig config;
    config.jobs = 15;
    config.machines = 15;
    config.timeSeed = 840612802;
    config.machineSeed = 398197754;

    const GeneratedJSSPInstance generated = TaillardGenerator{}.generate(config);
    check(written(generated.instance) == TA01, "the published seeds give ta01");

    const InstanceMetadata& metadata = generated.metadata;
    check(metadata.problemType == "JSSP", "problem type");
    check(metadata.generatorType == "taillard", "generator type");
    check(!metadata.generatorSeed, "no seed when the two seeds are given");
    check(metadata.generatorParameters.at("time_seed") == "840612802", "time seed kept");
    check(metadata.generatorParameters.at("machine_seed") == "398197754",
          "machine seed kept");
    check(metadata.generatorParameters.at("jobs") == "15", "jobs kept");
    check(taillardInstanceId(config) == "taillard_j15_m15_ts840612802_ms398197754",
          "identifier with given seeds");
}

void testDerivedSeeds()
{
    const TaillardConfig first = taillardConfigFromSeed(10, 10, 1);
    const TaillardConfig again = taillardConfigFromSeed(10, 10, 1);
    const TaillardConfig second = taillardConfigFromSeed(10, 10, 2);

    check(first.timeSeed == again.timeSeed && first.machineSeed == again.machineSeed,
          "derived seeds are reproducible");
    check(first.timeSeed != second.timeSeed, "consecutive seeds give unrelated time seeds");
    check(first.timeSeed >= 1 && first.timeSeed <= TAILLARD_SEED_MAX
              && first.machineSeed >= 1 && first.machineSeed <= TAILLARD_SEED_MAX,
          "derived seeds lie in Taillard's range");

    const GeneratedJSSPInstance a = TaillardGenerator{}.generate(first);
    const GeneratedJSSPInstance b = TaillardGenerator{}.generate(again);
    const GeneratedJSSPInstance c = TaillardGenerator{}.generate(second);

    check(written(a.instance) == written(b.instance), "same seed, same instance");
    check(written(a.instance) != written(c.instance), "other seed, other instance");
    check(a.metadata.generatorSeed == 1u, "the seed is recorded");
    check(taillardInstanceId(first) == "taillard_j10_m10_seed1", "identifier with a seed");

    std::string error;
    check(validateJSSPInstance(a.instance, error), "a generated instance is valid");

    // Durations as in Taillard's paper.
    bool inRange = true;

    for (const auto& job : a.instance.jobs)
    {
        for (const JSSPOperation& operation : job)
            inRange = inRange && operation.duration >= 1 && operation.duration <= 99;
    }

    check(inRange, "durations lie in [1, 99]");
}

void testValidation()
{
    const auto rejected = [](TaillardConfig config) {
        try
        {
            TaillardGenerator{}.generate(config);
        }
        catch (const std::invalid_argument&)
        {
            return true;
        }

        return false;
    };

    TaillardConfig config = taillardConfigFromSeed(3, 3, 5);
    config.jobs = 0;
    check(rejected(config), "no job is refused");

    config = taillardConfigFromSeed(3, 3, 5);
    config.machines = 0;
    check(rejected(config), "no machine is refused");

    config = taillardConfigFromSeed(3, 3, 5);
    config.timeSeed = 0;
    check(rejected(config), "a zero seed is refused");

    config = taillardConfigFromSeed(3, 3, 5);
    config.machineSeed = TAILLARD_SEED_MAX + 1;
    check(rejected(config), "a seed above the range is refused");
}

} // namespace

int main()
{
    testPublishedInstance();
    testDerivedSeeds();
    testValidation();

    if (failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }

    std::cout << "All Taillard generator tests passed\n";
    return 0;
}
