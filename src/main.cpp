#include <iostream>
#include <iomanip>
#include <vector>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "generators/ErdosRenyiGenerator.hpp"
#include "problems/MIS.hpp"

int main()
{
    SCIP* scip = nullptr;

    SCIP_CALL_ABORT(SCIPcreate(&scip));
    SCIP_CALL_ABORT(SCIPincludeDefaultPlugins(scip));

    std::cout << "SCIP version: "
              << SCIPversion()
              << '\n';

    // ------------------------------------------------------------
    // Generate an Erdos-Renyi random graph G(n, p)
    // ------------------------------------------------------------

    const ErdosRenyiConfig config{
        .n = 50,
        .p = 0.10,
        .seed = 42
    };

    ErdosRenyiGenerator generator;
    GeneratedMISInstance generated = generator.generate(config);

    const MISInstance& instance = generated.graph;
    const MISGenerationMetadata& metadata = generated.metadata;

    const std::size_t m = instance.edges.size();

    double density = 0.0;

    if (instance.n >= 2)
    {
        density =
            (2.0 * static_cast<double>(m))
            / (static_cast<double>(instance.n)
               * static_cast<double>(instance.n - 1));
    }

    std::cout << '\n';
    std::cout << "=== Erdos-Renyi MIS Instance ===\n";
    std::cout << "n: " << instance.n << '\n';
    std::cout << "m: " << m << '\n';
    std::cout << "p (expected): " << config.p << '\n';
    std::cout << "density (actual): "
              << std::fixed
              << std::setprecision(4)
              << density
              << '\n';
    std::cout << "seed: " << config.seed << '\n';

    // ------------------------------------------------------------
    // Build the MIS MILP model
    // ------------------------------------------------------------

    std::vector<SCIP_VAR*> vars;

    SCIP_CALL_ABORT(
        buildMISModel(
            scip,
            instance,
            vars
        )
    );

    // ------------------------------------------------------------
    // Solve
    // ------------------------------------------------------------

    SCIP_CALL_ABORT(SCIPsolve(scip));

    SCIP_SOL* solution = SCIPgetBestSol(scip);

    if (solution == nullptr)
    {
        std::cout << "\nNo feasible solution found.\n";

        SCIP_CALL_ABORT(SCIPfree(&scip));
        return 0;
    }

    // ------------------------------------------------------------
    // Extract the independent set
    // ------------------------------------------------------------

    std::vector<int> independentSet;

    for (int v = 0; v < instance.n; ++v)
    {
        const double value =
            SCIPgetSolVal(
                scip,
                solution,
                vars[v]
            );

        if (value > 0.5)
        {
            independentSet.push_back(v);
        }
    }

    // ------------------------------------------------------------
    // Results
    // ------------------------------------------------------------

    std::cout << '\n';
    std::cout << "=== Solution ===\n";
    std::cout << "Objective value: "
              << independentSet.size()
              << '\n';

    std::cout << "Independent set: {";

    for (std::size_t i = 0; i < independentSet.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }

        std::cout << independentSet[i];
    }

    std::cout << "}\n";

    std::cout << "B&B nodes: "
              << SCIPgetNNodes(scip)
              << '\n';

    std::cout << "Solving time: "
              << SCIPgetSolvingTime(scip)
              << " s\n";

    // ------------------------------------------------------------
    // Cleanup
    // ------------------------------------------------------------

    SCIP_CALL_ABORT(SCIPfree(&scip));

    return 0;
}
