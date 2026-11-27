#pragma once

#include <vector>

#include <scip/scip.h>

#include "problems/JSSP.hpp"

// One disjunction: on machine, job firstJob runs before job secondJob
// (firstJob < secondJob) if var = 1, after it if var = 0.
struct JSSPOrderVariable
{
    int firstJob = 0;
    int secondJob = 0;
    int machine = 0;
    SCIP_VAR* var = nullptr;
};

// The variables of the model, owned by the problem, not captured.
struct JSSPModel
{
    // Start time of operation k of job j at j * numberOfMachines + k.
    std::vector<SCIP_VAR*> startTimes;

    SCIP_VAR* makespan = nullptr;

    // Machine by machine, then pairs of jobs in lexicographic order.
    std::vector<JSSPOrderVariable> orders;

    // Upper bound H (jsspDispatchMakespan) and lower bound
    // (jsspOneMachineBound) used by the formulation.
    int horizon = 0;
    int lowerBound = 0;
};

/*
 * Creates the disjunctive formulation of Manne (1960) in scip:
 *
 *   minimize    C
 *   subject to  s(j, k+1) >= s(j, k) + p(j, k)       (order of a job)
 *               C >= s(j, last) + p(j, last)
 *               s(l, h) >= s(j, k) + p(j, k) - H (1 - y)
 *               s(j, k) >= s(l, h) + p(l, h) - H y   for the operations
 *                                     (j, k) and (l, h) of one machine
 *               head(j, k) <= s(j, k) <= H - tail(j, k)
 *               lower bound <= C <= H
 *
 * where y is binary, H is the makespan of the greedy schedule and serves
 * as big-M, head(j, k) is the work of job j before operation k and
 * tail(j, k) the work from operation k to the end of the job. The bounds
 * on C carry the one-machine bound, which the LP of the big-M formulation
 * does not see. Start times and C are integers (the durations are), so
 * that scip knows the objective is integral: a new solution must improve
 * by at least 1, which prunes far more nodes than continuous times (5
 * times less time on the 10x10 pilot instances). They get a branching
 * priority below that of the order variables, on which scip branches
 * first.
 *
 * Returns SCIP_INVALIDDATA, without creating anything, if the instance is
 * not valid (validateJSSPInstance()).
 */
SCIP_RETCODE buildJSSPModel(
    SCIP* scip,
    const JSSPInstance& instance,
    JSSPModel& model
);
