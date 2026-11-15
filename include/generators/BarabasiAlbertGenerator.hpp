#pragma once

#include "generators/GeneratedMISInstance.hpp"

/*
 * Barabasi-Albert preferential attachment graphs.
 *
 * The graph starts with k isolated vertices 0, ..., k-1. Vertices k, ...,
 * n-1 are then added one at a time, each joined to k distinct existing
 * vertices: vertex k to all the initial ones, every later vertex to
 * vertices drawn with probability proportional to their degree. The graph
 * has k (n - k) edges and is connected.
 *
 * k is often written m in the literature; in this repository m is the
 * number of edges, as in the instance files.
 *
 * The graph only depends on (n, k, seed), on every platform: random
 * numbers are drawn from the raw output of std::mt19937, which the C++
 * standard fixes, and not through the standard distributions, whose
 * algorithms are left to each library.
 */
struct BarabasiAlbertConfig
{
    int n;
    int k;
    unsigned int seed;
};

class BarabasiAlbertGenerator
{
public:
    // Throws std::invalid_argument unless 1 <= k < n, with a number of
    // edges small enough to be generated.
    static void validate(const BarabasiAlbertConfig& config);

    GeneratedMISInstance generate(
        const BarabasiAlbertConfig& config
    ) const;
};
