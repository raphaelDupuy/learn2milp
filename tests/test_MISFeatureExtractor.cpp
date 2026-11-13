#include "services/MISFeatureExtractor.hpp"

#include <cmath>
#include <iostream>
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

void checkNear(
    double actual,
    double expected,
    const std::string& message
)
{
    check(
        std::abs(actual - expected) < 1e-9,
        message
    );
}

void testConnectedGraph()
{
    MISInstance instance;
    instance.n = 4;
    instance.edges = {
        {0, 1},
        {1, 2},
        {2, 3}
    };

    MISFeatures features;
    std::string error;

    const bool success =
        extractMISFeatures(instance, features, error);

    check(success, "connected graph should be accepted");

    if (!success)
        return;

    check(features.numberOfVertices == 4, "vertex count should be 4");
    check(features.numberOfEdges == 3, "edge count should be 3");
    checkNear(features.density, 0.5, "density should be 0.5");
    checkNear(features.averageDegree, 1.5, "average degree should be 1.5");
    check(features.minimumDegree == 1, "minimum degree should be 1");
    check(features.maximumDegree == 2, "maximum degree should be 2");
    checkNear(features.degreeStdDev, 0.5, "degree standard deviation should be 0.5");
    check(
        features.numberOfConnectedComponents == 1,
        "connected graph should have one component"
    );
    check(error.empty(), "error message should be empty on success");
}

void testDisconnectedGraph()
{
    MISInstance instance;
    instance.n = 5;
    instance.edges = {
        {0, 1},
        {2, 3}
    };

    MISFeatures features;
    std::string error;

    const bool success =
        extractMISFeatures(instance, features, error);

    check(success, "disconnected graph should be accepted");

    if (!success)
        return;

    checkNear(features.density, 0.2, "density should be 0.2");
    checkNear(features.averageDegree, 0.8, "average degree should be 0.8");
    check(features.minimumDegree == 0, "minimum degree should be 0");
    check(features.maximumDegree == 1, "maximum degree should be 1");
    checkNear(features.degreeStdDev, 0.4, "degree standard deviation should be 0.4");
    check(
        features.numberOfConnectedComponents == 3,
        "graph should have three connected components"
    );
}

void testGraphWithoutEdges()
{
    MISInstance instance;
    instance.n = 5;

    MISFeatures features;
    std::string error;

    const bool success =
        extractMISFeatures(instance, features, error);

    check(success, "graph without edges should be accepted");

    if (!success)
        return;

    check(features.numberOfEdges == 0, "edge count should be zero");
    checkNear(features.density, 0.0, "density should be zero");
    checkNear(features.averageDegree, 0.0, "average degree should be zero");
    check(features.minimumDegree == 0, "minimum degree should be zero");
    check(features.maximumDegree == 0, "maximum degree should be zero");
    checkNear(features.degreeStdDev, 0.0, "degree standard deviation should be zero");
    check(
        features.numberOfConnectedComponents == 5,
        "graph should have five connected components"
    );
}

void testEmptyGraph()
{
    MISInstance instance;
    instance.n = 0;

    MISFeatures features;
    std::string error;

    const bool success =
        extractMISFeatures(instance, features, error);

    check(success, "empty graph should be accepted");

    if (!success)
        return;

    check(features.numberOfVertices == 0, "vertex count should be zero");
    check(features.numberOfEdges == 0, "edge count should be zero");
    checkNear(features.density, 0.0, "empty graph density should be zero");
    checkNear(features.averageDegree, 0.0, "empty graph average degree should be zero");
    check(features.numberOfConnectedComponents == 0, "empty graph should have zero components");
}

void testSingleVertexGraph()
{
    MISInstance instance;
    instance.n = 1;

    MISFeatures features;
    std::string error;

    const bool success =
        extractMISFeatures(instance, features, error);

    check(success, "single-vertex graph should be accepted");

    if (!success)
        return;

    checkNear(features.density, 0.0, "single-vertex graph density should be zero");
    checkNear(features.averageDegree, 0.0, "single-vertex graph average degree should be zero");
    check(features.minimumDegree == 0, "single-vertex graph minimum degree should be zero");
    check(features.maximumDegree == 0, "single-vertex graph maximum degree should be zero");
    check(
        features.numberOfConnectedComponents == 1,
        "single-vertex graph should have one component"
    );
}

void testRejectsNegativeVertexCount()
{
    MISInstance instance;
    instance.n = -1;

    MISFeatures features;
    std::string error;

    check(
        !extractMISFeatures(instance, features, error),
        "negative vertex count should be rejected"
    );
    check(!error.empty(), "negative vertex count should produce an error");
}

void testRejectsInvalidEndpoint()
{
    MISInstance instance;
    instance.n = 3;
    instance.edges = {{0, 3}};

    MISFeatures features;
    std::string error;

    check(
        !extractMISFeatures(instance, features, error),
        "out-of-range endpoint should be rejected"
    );
    check(!error.empty(), "invalid endpoint should produce an error");
}

void testRejectsSelfLoop()
{
    MISInstance instance;
    instance.n = 3;
    instance.edges = {{1, 1}};

    MISFeatures features;
    std::string error;

    check(
        !extractMISFeatures(instance, features, error),
        "self-loop should be rejected"
    );
    check(!error.empty(), "self-loop should produce an error");
}

void testRejectsDuplicateEdge()
{
    MISInstance instance;
    instance.n = 3;
    instance.edges = {
        {0, 1},
        {1, 0}
    };

    MISFeatures features;
    std::string error;

    check(
        !extractMISFeatures(instance, features, error),
        "duplicate edge in reverse order should be rejected"
    );
    check(!error.empty(), "duplicate edge should produce an error");
}

void testPreservesFeaturesOnFailure()
{
    MISInstance instance;
    instance.n = 2;
    instance.edges = {{0, 2}};

    MISFeatures features;
    features.numberOfVertices = 42;
    features.numberOfEdges = 17;
    features.density = 0.75;

    std::string error;

    const bool success =
        extractMISFeatures(instance, features, error);

    check(!success, "invalid graph should fail extraction");
    check(
        features.numberOfVertices == 42,
        "vertex count should remain unchanged after failure"
    );
    check(
        features.numberOfEdges == 17,
        "edge count should remain unchanged after failure"
    );
    checkNear(
        features.density,
        0.75,
        "density should remain unchanged after failure"
    );
}
}

int main()
{
    testConnectedGraph();
    testDisconnectedGraph();
    testGraphWithoutEdges();
    testEmptyGraph();
    testSingleVertexGraph();
    testRejectsNegativeVertexCount();
    testRejectsInvalidEndpoint();
    testRejectsSelfLoop();
    testRejectsDuplicateEdge();
    testPreservesFeaturesOnFailure();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS feature extractor tests passed.\n";
    return 0;
}
