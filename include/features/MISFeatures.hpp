#pragma once

#include <cstddef>

struct MISFeatures
{
    int numberOfVertices = 0;
    std::size_t numberOfEdges = 0;

    double density = 0.0;

    double averageDegree = 0.0;
    int minimumDegree = 0;
    int maximumDegree = 0;
    double degreeStdDev = 0.0;

    int numberOfConnectedComponents = 0;
};