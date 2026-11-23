#pragma once

#include <string>
#include <vector>

/*
 * Job-shop scheduling (JSSP): every job is a sequence of operations, each
 * run on one machine for a given duration, in the order of the sequence;
 * a machine runs one operation at a time, without preemption; the
 * objective is to minimize the makespan, the end of the last operation.
 * As in the classical problem and its benchmarks, every job visits every
 * machine exactly once.
 */

struct JSSPOperation
{
    int machine = 0;  // in [0, numberOfMachines)
    int duration = 0; // at least 1
};

struct JSSPInstance
{
    int numberOfMachines = 0;

    // jobs[j] lists the operations of job j in processing order.
    std::vector<std::vector<JSSPOperation>> jobs;
};

/*
 * Checks that the instance is a classical job shop: at least one job and
 * one machine, every job has exactly one operation on every machine, and
 * every duration is at least 1. On failure, errorMessage describes the
 * first problem found; it is left untouched on success.
 */
bool validateJSSPInstance(
    const JSSPInstance& instance,
    std::string& errorMessage
);

/*
 * Makespan of a greedy schedule, an upper bound on the optimum: among the
 * next operations of the jobs, start the one that can start first, ties
 * broken by the most work left in its job, then by the smallest job. The
 * instance must be valid.
 */
int jsspDispatchMakespan(const JSSPInstance& instance);

/*
 * Lower bound on the makespan: the longest job, and on every machine the
 * smallest head (work of a job before its operation on the machine), the
 * load of the machine and the smallest tail (work of a job after it). The
 * instance must be valid.
 */
int jsspOneMachineBound(const JSSPInstance& instance);

/*
 * Work of job j before its operation k (heads) and from operation k to
 * the end of the job, k included (tails), both at j * numberOfMachines + k.
 * The instance must be valid.
 */
void jsspHeadsAndTails(
    const JSSPInstance& instance,
    std::vector<int>& heads,
    std::vector<int>& tails
);

// An operation seen from its machine.
struct JSSPMachineOperation
{
    int job = 0;
    int position = 0; // in the job
    int duration = 0;
};

// The operations of every machine, in job order. The instance must be
// valid.
std::vector<std::vector<JSSPMachineOperation>> jsspOperationsByMachine(
    const JSSPInstance& instance
);
