#include "services/MISSolvingService.hpp"

#include "services/MISBenchmarkService.hpp"

namespace
{

// SCIP's own default for limits/time: no time limit.
const double NO_TIME_LIMIT = 1e20;

} // namespace

bool solveMIS(
    const MISInstance& instance,
    MISResult& result
)
{
    result = MISResult{};

    // SCIP as shipped: its own branching rules and settings.
    MISBenchmarkConfig config;
    config.settings = MISBenchmarkSettings::Default;
    config.timeLimitSeconds = NO_TIME_LIMIT;

    MISBenchmarkResult solved;

    if (!benchmarkMIS(instance, config, solved, result.errorMessage))
        return false;

    result.status = solved.status;
    result.hasSolution = solved.hasSolution;
    result.objectiveValue = solved.primalBound;
    result.selectedVertices = solved.selectedVertices;
    result.solvingTimeSeconds = solved.solvingTimeSeconds;
    result.exploredNodes = solved.exploredNodes;

    return true;
}
