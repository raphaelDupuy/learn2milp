#include "generators/TaillardGenerator.hpp"
#include "io/JSSPInstanceIO.hpp"
#include "problems/JSSPModel.hpp"

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

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

/*
 * Optimal makespan by enumeration: every order of the jobs on every
 * machine, each scheduled as early as possible (longest path in the graph
 * of the job and machine orders); an order with a cycle is infeasible.
 */
int bruteForceMakespan(const JSSPInstance& instance)
{
    const int jobs = static_cast<int>(instance.jobs.size());
    const int machines = instance.numberOfMachines;
    const int operations = jobs * machines;

    // position[j][m]: index of the operation of job j on machine m.
    std::vector<std::vector<int>> position(jobs, std::vector<int>(machines));

    for (int j = 0; j < jobs; ++j)
    {
        for (int k = 0; k < machines; ++k)
            position[j][instance.jobs[j][k].machine] = k;
    }

    std::vector<std::vector<int>> orders(machines, std::vector<int>(jobs));

    for (auto& order : orders)
        std::iota(order.begin(), order.end(), 0);

    int best = -1;

    for (;;)
    {
        // Successors of every operation j * machines + k.
        std::vector<std::vector<int>> next(operations);
        std::vector<int> indegree(operations, 0);

        const auto arc = [&](int from, int to) {
            next[from].push_back(to);
            ++indegree[to];
        };

        for (int j = 0; j < jobs; ++j)
        {
            for (int k = 0; k + 1 < machines; ++k)
                arc(j * machines + k, j * machines + k + 1);
        }

        for (int m = 0; m < machines; ++m)
        {
            for (int i = 0; i + 1 < jobs; ++i)
            {
                const int a = orders[m][i];
                const int b = orders[m][i + 1];
                arc(a * machines + position[a][m], b * machines + position[b][m]);
            }
        }

        std::vector<int> start(operations, 0);
        std::vector<int> ready;

        for (int o = 0; o < operations; ++o)
        {
            if (indegree[o] == 0)
                ready.push_back(o);
        }

        int visited = 0;
        int makespan = 0;

        while (!ready.empty())
        {
            const int o = ready.back();
            ready.pop_back();
            ++visited;

            const int end = start[o] + instance.jobs[o / machines][o % machines].duration;
            makespan = std::max(makespan, end);

            for (const int to : next[o])
            {
                start[to] = std::max(start[to], end);

                if (--indegree[to] == 0)
                    ready.push_back(to);
            }
        }

        if (visited == operations && (best < 0 || makespan < best))
            best = makespan;

        // Next combination of machine orders, like an odometer.
        int m = 0;

        while (m < machines && !std::next_permutation(orders[m].begin(), orders[m].end()))
            ++m;

        if (m == machines)
            break;
    }

    return best;
}

// Optimal makespan found by SCIP on the model, or -1 on failure.
double solveWithSCIP(const JSSPInstance& instance, JSSPModel& model)
{
    SCIP* scip = nullptr;

    if (SCIPcreate(&scip) != SCIP_OKAY)
        return -1.0;

    double value = -1.0;

    if (SCIPincludeDefaultPlugins(scip) == SCIP_OKAY
        && SCIPsetIntParam(scip, "display/verblevel", 0) == SCIP_OKAY
        && buildJSSPModel(scip, instance, model) == SCIP_OKAY
        && SCIPsolve(scip) == SCIP_OKAY
        && SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL)
    {
        value = SCIPgetPrimalbound(scip);
    }

    SCIPfree(&scip);
    return value;
}

void testAgainstEnumeration()
{
    // 3 jobs x 3 machines (216 combinations) and 4 x 3 (13,824).
    for (const auto& [jobs, machines] : std::vector<std::pair<int, int>>{{3, 3}, {4, 3}})
    {
        for (unsigned int seed = 1; seed <= 5; ++seed)
        {
            const JSSPInstance instance = TaillardGenerator{}
                .generate(taillardConfigFromSeed(jobs, machines, seed))
                .instance;
            const std::string name = std::to_string(jobs) + "x"
                + std::to_string(machines) + " seed " + std::to_string(seed);

            const int optimum = bruteForceMakespan(instance);
            JSSPModel model;
            const double solved = solveWithSCIP(instance, model);

            check(std::fabs(solved - optimum) < 1e-6,
                  name + ": SCIP finds the optimum of the enumeration ("
                      + std::to_string(solved) + " vs " + std::to_string(optimum) + ")");
            check(model.lowerBound <= optimum && optimum <= model.horizon,
                  name + ": the bounds of the formulation hold");
            check(model.lowerBound == jsspOneMachineBound(instance)
                      && model.horizon == jsspDispatchMakespan(instance),
                  name + ": the model reports its bounds");
        }
    }
}

void testFt06()
{
    // Fisher and Thompson (1963), optimum 55.
    const std::string ft06 =
        "6 6\n"
        "2 1 0 3 1 6 3 7 5 3 4 6\n"
        "1 8 2 5 4 10 5 10 0 10 3 4\n"
        "2 5 3 4 5 8 0 9 1 1 4 7\n"
        "1 5 0 5 2 5 3 3 4 8 5 9\n"
        "2 9 1 3 4 5 5 4 0 3 3 1\n"
        "1 3 3 3 5 9 0 10 4 4 2 1\n";

    std::istringstream input(ft06);
    JSSPInstance instance;
    std::string error;
    check(readJSSPInstance(input, instance, error), "ft06 is read");

    JSSPModel model;
    check(std::fabs(solveWithSCIP(instance, model) - 55.0) < 1e-6, "ft06 has makespan 55");

    // 6 machines x C(6, 2) pairs, 36 start times.
    check(model.orders.size() == 90, "one order variable per pair of jobs and machine");
    check(model.startTimes.size() == 36, "one start time per operation");
    check(model.makespan != nullptr, "makespan variable");

    bool ordered = true;

    for (const JSSPOrderVariable& order : model.orders)
        ordered = ordered && order.firstJob < order.secondJob;

    check(ordered, "order variables name the smaller job first");
}

void testInvalidInstance()
{
    JSSPInstance instance;
    instance.numberOfMachines = 2;
    instance.jobs = {{{0, 3}, {0, 2}}};

    SCIP* scip = nullptr;
    SCIPcreate(&scip);
    SCIPincludeDefaultPlugins(scip);

    JSSPModel model;
    check(buildJSSPModel(scip, instance, model) == SCIP_INVALIDDATA,
          "an invalid instance is refused");
    check(model.orders.empty() && model.makespan == nullptr, "the model is left empty");

    SCIPfree(&scip);
}

} // namespace

int main()
{
    testAgainstEnumeration();
    testFt06();
    testInvalidInstance();

    if (failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }

    std::cout << "All JSSP model tests passed\n";
    return 0;
}
