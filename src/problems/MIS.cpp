#include "problems/MIS.hpp"

#include <string>

SCIP_RETCODE buildMISModel(
    SCIP* scip,
    const MISInstance& instance,
    std::vector<SCIP_VAR*>& vars
)
{
    SCIP_CALL(
        SCIPcreateProbBasic(
            scip,
            "maximum_independent_set"
        )
    );

    SCIP_CALL(
        SCIPsetObjsense(
            scip,
            SCIP_OBJSENSE_MAXIMIZE
        )
    );

    vars.resize(instance.n, nullptr);

    // Create one binary variable x_v for each vertex.
    for (int v = 0; v < instance.n; ++v)
    {
        std::string name = "x_" + std::to_string(v);

        SCIP_VAR* var = nullptr;

        SCIP_CALL(
            SCIPcreateVarBasic(
                scip,
                &var,
                name.c_str(),
                0.0,
                1.0,
                1.0,
                SCIP_VARTYPE_BINARY
            )
        );

        SCIP_CALL(
            SCIPaddVar(scip, var)
        );

        // Keep the pointer for later access, while releasing
        // the local reference created by SCIPcreateVarBasic().
        vars[v] = var;

        SCIP_CALL(
            SCIPreleaseVar(scip, &var)
        );
    }

    // For every edge (u,v):
    //
    //     x_u + x_v <= 1
    //
    // This guarantees that adjacent vertices
    // cannot both belong to the independent set.
    for (std::size_t e = 0; e < instance.edges.size(); ++e)
    {
        const auto [u, v] = instance.edges[e];

        std::string name =
            "edge_" + std::to_string(u) + "_" + std::to_string(v);

        SCIP_CONS* cons = nullptr;

        SCIP_CALL(
            SCIPcreateConsBasicLinear(
                scip,
                &cons,
                name.c_str(),
                0,
                nullptr,
                nullptr,
                -SCIPinfinity(scip),
                1.0
            )
        );

        SCIP_CALL(
            SCIPaddCoefLinear(
                scip,
                cons,
                vars[u],
                1.0
            )
        );

        SCIP_CALL(
            SCIPaddCoefLinear(
                scip,
                cons,
                vars[v],
                1.0
            )
        );

        SCIP_CALL(
            SCIPaddCons(scip, cons)
        );

        SCIP_CALL(
            SCIPreleaseCons(scip, &cons)
        );
    }

    return SCIP_OKAY;
}
