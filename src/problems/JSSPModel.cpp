#include "problems/JSSPModel.hpp"

#include <string>
#include <utility>
#include <vector>

#include <scip/cons_linear.h>

namespace
{

SCIP_RETCODE addVariable(
    SCIP* scip,
    const std::string& name,
    double lower,
    double upper,
    double objective,
    SCIP_VARTYPE type,
    SCIP_VAR*& kept
)
{
    SCIP_VAR* var = nullptr;

    SCIP_CALL(SCIPcreateVarBasic(
        scip, &var, name.c_str(), lower, upper, objective, type));
    SCIP_CALL(SCIPaddVar(scip, var));

    // The problem keeps the variable; the local reference is released.
    kept = var;
    SCIP_CALL(SCIPreleaseVar(scip, &var));

    return SCIP_OKAY;
}

// Adds lhs <= sum of coefficients * variables, with no upper side.
SCIP_RETCODE addGreaterEqual(
    SCIP* scip,
    const std::string& name,
    const std::vector<SCIP_VAR*>& vars,
    const std::vector<double>& coefficients,
    double lhs
)
{
    SCIP_CONS* cons = nullptr;

    SCIP_CALL(SCIPcreateConsBasicLinear(
        scip,
        &cons,
        name.c_str(),
        static_cast<int>(vars.size()),
        const_cast<SCIP_VAR**>(vars.data()),
        const_cast<double*>(coefficients.data()),
        lhs,
        SCIPinfinity(scip)));
    SCIP_CALL(SCIPaddCons(scip, cons));
    SCIP_CALL(SCIPreleaseCons(scip, &cons));

    return SCIP_OKAY;
}

} // namespace

SCIP_RETCODE buildJSSPModel(
    SCIP* scip,
    const JSSPInstance& instance,
    JSSPModel& model
)
{
    std::string ignored;

    if (!validateJSSPInstance(instance, ignored))
        return SCIP_INVALIDDATA;

    const int jobs = static_cast<int>(instance.jobs.size());
    const int machines = instance.numberOfMachines;

    std::vector<int> heads;
    std::vector<int> tails;
    jsspHeadsAndTails(instance, heads, tails);

    JSSPModel built;
    built.horizon = jsspDispatchMakespan(instance);
    built.lowerBound = jsspOneMachineBound(instance);
    const double horizon = built.horizon;

    SCIP_CALL(SCIPcreateProbBasic(scip, "job_shop_scheduling"));
    SCIP_CALL(SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE));

    SCIP_CALL(addVariable(
        scip, "cmax", built.lowerBound, horizon, 1.0,
        SCIP_VARTYPE_INTEGER, built.makespan));
    SCIP_CALL(SCIPchgVarBranchPriority(scip, built.makespan, -1));

    built.startTimes.resize(static_cast<std::size_t>(jobs) * machines, nullptr);

    for (int j = 0; j < jobs; ++j)
    {
        for (int k = 0; k < machines; ++k)
        {
            const int index = j * machines + k;

            SCIP_CALL(addVariable(
                scip,
                "s_" + std::to_string(j) + "_" + std::to_string(k),
                heads[index],
                horizon - tails[index],
                0.0,
                SCIP_VARTYPE_INTEGER,
                built.startTimes[index]));
            SCIP_CALL(SCIPchgVarBranchPriority(scip, built.startTimes[index], -1));
        }
    }

    const auto onMachine = jsspOperationsByMachine(instance);

    for (int m = 0; m < machines; ++m)
    {
        const auto& operations = onMachine[m];

        for (std::size_t a = 0; a < operations.size(); ++a)
        {
            for (std::size_t b = a + 1; b < operations.size(); ++b)
            {
                JSSPOrderVariable order;
                order.firstJob = operations[a].job;
                order.secondJob = operations[b].job;
                order.machine = m;

                SCIP_CALL(addVariable(
                    scip,
                    "y_" + std::to_string(order.firstJob) + "_"
                        + std::to_string(order.secondJob) + "_"
                        + std::to_string(m),
                    0.0, 1.0, 0.0, SCIP_VARTYPE_BINARY, order.var));

                built.orders.push_back(order);
            }
        }
    }

    for (int j = 0; j < jobs; ++j)
    {
        for (int k = 0; k + 1 < machines; ++k)
        {
            SCIP_CALL(addGreaterEqual(
                scip,
                "prec_" + std::to_string(j) + "_" + std::to_string(k),
                {built.startTimes[j * machines + k + 1],
                 built.startTimes[j * machines + k]},
                {1.0, -1.0},
                instance.jobs[j][k].duration));
        }

        SCIP_CALL(addGreaterEqual(
            scip,
            "end_" + std::to_string(j),
            {built.makespan, built.startTimes[j * machines + machines - 1]},
            {1.0, -1.0},
            instance.jobs[j][machines - 1].duration));
    }

    std::size_t next = 0;

    for (int m = 0; m < machines; ++m)
    {
        const auto& operations = onMachine[m];

        for (std::size_t a = 0; a < operations.size(); ++a)
        {
            for (std::size_t b = a + 1; b < operations.size(); ++b)
            {
                const JSSPOrderVariable& order = built.orders[next++];
                SCIP_VAR* first = built.startTimes[
                    operations[a].job * machines + operations[a].position];
                SCIP_VAR* second = built.startTimes[
                    operations[b].job * machines + operations[b].position];
                const std::string suffix = std::to_string(order.firstJob) + "_"
                    + std::to_string(order.secondJob) + "_" + std::to_string(m);

                // y = 1: the second starts after the first ends.
                SCIP_CALL(addGreaterEqual(
                    scip, "a_" + suffix,
                    {second, first, order.var}, {1.0, -1.0, -horizon},
                    operations[a].duration - horizon));

                // y = 0: the first starts after the second ends.
                SCIP_CALL(addGreaterEqual(
                    scip, "b_" + suffix,
                    {first, second, order.var}, {1.0, -1.0, horizon},
                    operations[b].duration));
            }
        }
    }

    model = std::move(built);
    return SCIP_OKAY;
}
