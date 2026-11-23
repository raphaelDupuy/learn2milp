#include "problems/JSSP.hpp"

#include <algorithm>
#include <string>
#include <vector>


bool validateJSSPInstance(
    const JSSPInstance& instance,
    std::string& errorMessage
)
{
    if (instance.numberOfMachines < 1)
    {
        errorMessage = "A job shop needs at least one machine.";
        return false;
    }

    if (instance.jobs.empty())
    {
        errorMessage = "A job shop needs at least one job.";
        return false;
    }

    for (std::size_t j = 0; j < instance.jobs.size(); ++j)
    {
        const auto& job = instance.jobs[j];
        const std::string where = "Job " + std::to_string(j) + ": ";

        if (static_cast<int>(job.size()) != instance.numberOfMachines)
        {
            errorMessage = where + "expected one operation per machine.";
            return false;
        }

        std::vector<bool> visited(instance.numberOfMachines, false);

        for (const JSSPOperation& operation : job)
        {
            if (operation.machine < 0
                || operation.machine >= instance.numberOfMachines)
            {
                errorMessage = where + "machine outside the valid range.";
                return false;
            }

            if (visited[operation.machine])
            {
                errorMessage = where + "visits a machine twice.";
                return false;
            }

            visited[operation.machine] = true;

            if (operation.duration < 1)
            {
                errorMessage = where + "durations must be at least 1.";
                return false;
            }
        }
    }

    return true;
}

int jsspDispatchMakespan(const JSSPInstance& instance)
{
    const int jobs = static_cast<int>(instance.jobs.size());
    const int machines = instance.numberOfMachines;

    std::vector<int> next(jobs, 0);
    std::vector<int> jobReady(jobs, 0);
    std::vector<int> machineReady(machines, 0);
    std::vector<int> remaining(jobs, 0);

    for (int j = 0; j < jobs; ++j)
    {
        for (const JSSPOperation& operation : instance.jobs[j])
            remaining[j] += operation.duration;
    }

    for (int step = 0; step < jobs * machines; ++step)
    {
        int best = -1;
        int bestStart = 0;

        for (int j = 0; j < jobs; ++j)
        {
            if (next[j] == machines)
                continue;

            const JSSPOperation& operation = instance.jobs[j][next[j]];
            const int start = std::max(jobReady[j], machineReady[operation.machine]);

            if (best < 0
                || start < bestStart
                || (start == bestStart && remaining[j] > remaining[best]))
            {
                best = j;
                bestStart = start;
            }
        }

        const JSSPOperation& operation = instance.jobs[best][next[best]];
        jobReady[best] = bestStart + operation.duration;
        machineReady[operation.machine] = jobReady[best];
        remaining[best] -= operation.duration;
        ++next[best];
    }

    return *std::max_element(jobReady.begin(), jobReady.end());
}

int jsspOneMachineBound(const JSSPInstance& instance)
{
    std::vector<int> heads;
    std::vector<int> tails;
    jsspHeadsAndTails(instance, heads, tails);

    const int machines = instance.numberOfMachines;
    int bound = 0;

    for (std::size_t j = 0; j < instance.jobs.size(); ++j)
        bound = std::max(bound, tails[j * machines]);

    for (const auto& operations : jsspOperationsByMachine(instance))
    {
        int head = heads[operations[0].job * machines + operations[0].position];
        int tail = tails[operations[0].job * machines + operations[0].position]
            - operations[0].duration;
        int load = 0;

        for (const JSSPMachineOperation& operation : operations)
        {
            const int index = operation.job * machines + operation.position;
            head = std::min(head, heads[index]);
            tail = std::min(tail, tails[index] - operation.duration);
            load += operation.duration;
        }

        bound = std::max(bound, head + load + tail);
    }

    return bound;
}

void jsspHeadsAndTails(
    const JSSPInstance& instance,
    std::vector<int>& heads,
    std::vector<int>& tails
)
{
    const int machines = instance.numberOfMachines;
    const int jobs = static_cast<int>(instance.jobs.size());

    heads.assign(static_cast<std::size_t>(jobs) * machines, 0);
    tails.assign(static_cast<std::size_t>(jobs) * machines, 0);

    for (int j = 0; j < jobs; ++j)
    {
        int before = 0;

        for (int k = 0; k < machines; ++k)
        {
            heads[j * machines + k] = before;
            before += instance.jobs[j][k].duration;
        }

        int after = 0;

        for (int k = machines - 1; k >= 0; --k)
        {
            after += instance.jobs[j][k].duration;
            tails[j * machines + k] = after;
        }
    }
}

std::vector<std::vector<JSSPMachineOperation>> jsspOperationsByMachine(
    const JSSPInstance& instance
)
{
    std::vector<std::vector<JSSPMachineOperation>> on(instance.numberOfMachines);

    for (int j = 0; j < static_cast<int>(instance.jobs.size()); ++j)
    {
        for (int k = 0; k < instance.numberOfMachines; ++k)
        {
            const JSSPOperation& operation = instance.jobs[j][k];
            on[operation.machine].push_back({j, k, operation.duration});
        }
    }

    return on;
}
