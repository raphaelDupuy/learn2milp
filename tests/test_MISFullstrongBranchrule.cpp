
#include <iostream>
#include <cstdio>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <scip/scip_var.h>
#include <scip/cons_linear.h>

#include "branching/MISFullstrongBranchrule.h"

static void collectSample(
    void* userdata,
    const MISStrongBranchingSample* sample,
    const MISStrongBranchingCandidate* candidates,
    size_t numberOfCandidates)
{
    (void)userdata;
    (void)sample;
    (void)candidates;
    (void)numberOfCandidates;
}

int main()
{
    SCIP* scip = nullptr;
    SCIP_VAR* vars[3] = {nullptr, nullptr, nullptr};
    SCIP_CONS* cons = nullptr;
    SCIP_RETCODE retcode;

    retcode = SCIPcreate(&scip);
    if( retcode != SCIP_OKAY )
    {
        std::cerr << "SCIPcreate failed: " << retcode << '\n';
        return 1;
    }

    retcode = SCIPincludeDefaultPlugins(scip);
    if( retcode != SCIP_OKAY )
    {
        std::cerr << "SCIPincludeDefaultPlugins failed\n";
        SCIPfree(&scip);
        return 1;
    }

    retcode = SCIPcreateProbBasic(scip, "mis_triangle");
    if( retcode != SCIP_OKAY )
    {
        std::cerr << "SCIPcreateProbBasic failed\n";
        SCIPfree(&scip);
        return 1;
    }

    SCIPsetObjsense(scip, SCIP_OBJSENSE_MAXIMIZE);

    for( int v = 0; v < 3; ++v )
    {
        char name[16];
        std::snprintf(name, sizeof(name), "x_%d", v);

        retcode = SCIPcreateVarBasic(
            scip, &vars[v], name, 0.0, 1.0, 1.0, SCIP_VARTYPE_BINARY
        );
        if( retcode != SCIP_OKAY )
        {
            std::cerr << "SCIPcreateVarBasic failed\n";
            SCIPfree(&scip);
            return 1;
        }

        retcode = SCIPaddVar(scip, vars[v]);
        if( retcode != SCIP_OKAY )
        {
            std::cerr << "SCIPaddVar failed\n";
            SCIPfree(&scip);
            return 1;
        }
    }

    const int edges[3][2] = {
        {0, 1},
        {1, 2},
        {0, 2}
    };

    for( int e = 0; e < 3; ++e )
    {
        SCIP_VAR* consvars[2] = {
            vars[edges[e][0]],
            vars[edges[e][1]]
        };
        SCIP_Real vals[2] = {1.0, 1.0};

        char name[16];
        std::snprintf(name, sizeof(name), "edge_%d", e);

        retcode = SCIPcreateConsBasicLinear(
            scip, &cons, name, 2, consvars, vals, -SCIPinfinity(scip), 1.0
        );
        if( retcode != SCIP_OKAY )
        {
            std::cerr << "SCIPcreateConsBasicLinear failed\n";
            SCIPfree(&scip);
            return 1;
        }

        retcode = SCIPaddCons(scip, cons);
        if( retcode != SCIP_OKAY )
        {
            std::cerr << "SCIPaddCons failed\n";
            SCIPfree(&scip);
            return 1;
        }

        SCIPreleaseCons(scip, &cons);
    }

    retcode = SCIPincludeBranchruleMISFullstrong(
        scip,
        vars,
        3,
        collectSample,
        nullptr
    );

    if( retcode != SCIP_OKAY )
    {
        std::cerr << "Plugin inclusion failed\n";
        SCIPfree(&scip);
        return 1;
    }

    /* Disable presolving and separation to preserve the fractional root LP. */
    retcode = SCIPsetIntParam(scip, "presolving/maxrounds", 0);
    if( retcode != SCIP_OKAY )
    {
        std::cerr << "Could not disable presolving\n";
        SCIPfree(&scip);
        return 1;
    }

    retcode = SCIPsetIntParam(scip, "separating/maxrounds", 0);
    if( retcode != SCIP_OKAY )
    {
        std::cerr << "Could not disable separation\n";
        SCIPfree(&scip);
        return 1;
    }

    /* Release the caller's variable captures. The problem and plugin retain theirs. */
    for( int v = 0; v < 3; ++v )
        SCIPreleaseVar(scip, &vars[v]);

    /* Two full solve cycles: exercises BRANCHINIT/BRANCHEXIT re-entry
     * (capture and release of the transformed variables each time). */
    for( int cycle = 0; cycle < 2; ++cycle )
    {
        retcode = SCIPsolve(scip);
        if( retcode != SCIP_OKAY )
        {
            std::cerr << "SCIPsolve failed (cycle " << cycle << "): " << retcode << '\n';
            SCIPfree(&scip);
            return 1;
        }

        retcode = SCIPfreeTransform(scip);
        if( retcode != SCIP_OKAY )
        {
            std::cerr << "SCIPfreeTransform failed (cycle " << cycle << "): " << retcode << '\n';
            SCIPfree(&scip);
            return 1;
        }
    }

    /* Freeing SCIP must not touch already-freed original variables. */
    retcode = SCIPfree(&scip);
    if( retcode != SCIP_OKAY )
    {
        std::cerr << "SCIPfree failed: " << retcode << '\n';
        return 1;
    }

    std::cout << "MISFullstrongBranchrule integration test passed.\n";
    return 0;
}
