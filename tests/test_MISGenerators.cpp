/*
 * Unit tests of the graph generators and of the family interface
 * (generators/MISGenerator.hpp).
 */

#include "generators/MISGenerator.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace
{

int failures = 0;

void check(bool condition, const std::string& testName)
{
    if (condition)
    {
        std::cout << "[PASS] " << testName << '\n';
    }
    else
    {
        std::cerr << "[FAIL] " << testName << '\n';
        ++failures;
    }
}

bool throwsInvalidArgument(const std::function<void()>& action)
{
    try
    {
        action();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    catch (...)
    {
        return false;
    }

    return false;
}

std::vector<std::vector<int>> adjacency(const MISInstance& graph)
{
    std::vector<std::vector<int>> neighbours(graph.n);

    for (const auto& [u, v] : graph.edges)
    {
        neighbours[u].push_back(v);
        neighbours[v].push_back(u);
    }

    return neighbours;
}

bool isConnected(const MISInstance& graph)
{
    if (graph.n == 0)
        return true;

    const auto neighbours = adjacency(graph);
    std::vector<bool> seen(graph.n, false);
    std::queue<int> queue;
    int reached = 1;

    seen[0] = true;
    queue.push(0);

    while (!queue.empty())
    {
        const int u = queue.front();
        queue.pop();

        for (const int v : neighbours[u])
        {
            if (!seen[v])
            {
                seen[v] = true;
                ++reached;
                queue.push(v);
            }
        }
    }

    return reached == graph.n;
}

// Edges are pairs u < v inside [0, n), strictly increasing: sorted, no
// duplicate and no loop.
bool hasCanonicalEdges(const MISInstance& graph)
{
    for (std::size_t e = 0; e < graph.edges.size(); ++e)
    {
        const auto& [u, v] = graph.edges[e];

        if (u < 0 || u >= v || v >= graph.n)
            return false;

        if (e > 0 && !(graph.edges[e - 1] < graph.edges[e]))
            return false;
    }

    return true;
}

void testBarabasiAlbertStructure()
{
    const BarabasiAlbertGenerator generator;

    for (const unsigned int seed : {1u, 2u, 3u})
    {
        const int n = 300;
        const int k = 3;
        const MISInstance graph = generator.generate({n, k, seed}).graph;
        const auto neighbours = adjacency(graph);

        // Every vertex added after the initial ones brings exactly k edges
        // towards earlier vertices; the first one joins all of them.
        bool attachment = true;

        for (int v = k; v < n; ++v)
        {
            const auto earlier = std::count_if(
                neighbours[v].begin(), neighbours[v].end(),
                [v](int u) { return u < v; });

            attachment = attachment && earlier == k;
        }

        for (int v = 0; v < k; ++v)
        {
            attachment = attachment
                && std::count(neighbours[k].begin(), neighbours[k].end(), v)
                    == 1;
        }

        const std::string where = " (seed " + std::to_string(seed) + ")";

        check(
            graph.n == n
            && graph.edges.size() == static_cast<std::size_t>(k * (n - k)),
            "Barabasi-Albert graph has k (n - k) edges" + where
        );
        check(hasCanonicalEdges(graph),
            "Barabasi-Albert edges are sorted, simple pairs" + where);
        check(attachment,
            "Every new vertex is joined to k earlier vertices" + where);
        check(isConnected(graph),
            "Barabasi-Albert graph is connected" + where);
    }

    // Preferential attachment: hubs appear. With n = 2000 and k = 3, the
    // largest degree grows like k sqrt(n) (about 130), whereas a random
    // graph with as many edges has a largest degree around 15.
    const MISInstance large = generator.generate({2000, 3, 7}).graph;
    const auto neighbours = adjacency(large);
    std::size_t largest = 0;

    for (const auto& list : neighbours)
        largest = std::max(largest, list.size());

    check(largest >= 50,
        "Preferential attachment creates hubs (largest degree "
            + std::to_string(largest) + ")");

    // Smallest graph: one vertex joined to the k initial ones.
    const MISInstance star = generator.generate({4, 3, 1}).graph;

    check(
        star.edges == std::vector<std::pair<int, int>>({{0, 3}, {1, 3}, {2, 3}}),
        "Barabasi-Albert graph with n = k + 1 is a star"
    );
}

void testBarabasiAlbertReproducibility()
{
    const BarabasiAlbertGenerator generator;

    const auto first = generator.generate({200, 8, 11});
    const auto again = generator.generate({200, 8, 11});
    const auto other = generator.generate({200, 8, 12});

    check(first.graph.edges == again.graph.edges,
        "Same parameters and seed give the same graph");
    check(first.graph.edges != other.graph.edges,
        "Another seed gives another graph");

    // Pins the random number use: these edges must never change, or the
    // instances of earlier experiments could not be generated again. They
    // were checked against an independent implementation of the same
    // process on numpy's MT19937.
    const auto pinned = generator.generate({8, 2, 1}).graph.edges;

    check(
        pinned == std::vector<std::pair<int, int>>({
            {0, 2}, {0, 3}, {0, 4}, {0, 7}, {1, 2}, {2, 3},
            {2, 5}, {2, 7}, {3, 4}, {3, 6}, {4, 5}, {5, 6}}),
        "Barabasi-Albert graph of (8, 2, seed 1) is unchanged"
    );

    const auto& metadata = first.metadata;

    check(
        metadata.problemType == "MIS"
        && metadata.generatorType == "barabasi_albert"
        && metadata.generatorVersion == "1.0"
        && metadata.generatorSeed == 11u
        && metadata.generatorParameters
            == std::map<std::string, std::string>({{"k", "8"}, {"n", "200"}}),
        "Barabasi-Albert metadata names the generator and its parameters"
    );
}

void testValidation()
{
    check(
        throwsInvalidArgument([] { BarabasiAlbertGenerator::validate({10, 0, 1}); })
        && throwsInvalidArgument([] { BarabasiAlbertGenerator::validate({5, 5, 1}); })
        && throwsInvalidArgument([] { BarabasiAlbertGenerator::validate({3, 5, 1}); })
        && throwsInvalidArgument([] {
            BarabasiAlbertGenerator::validate(
                {std::numeric_limits<int>::max(), 1000, 1});
        })
        && !throwsInvalidArgument([] { BarabasiAlbertGenerator::validate({6, 5, 1}); }),
        "Barabasi-Albert parameters must satisfy 1 <= k < n"
    );

    check(
        throwsInvalidArgument([] { BarabasiAlbertGenerator{}.generate({4, 4, 1}); }),
        "Generation validates its parameters"
    );

    check(
        throwsInvalidArgument([] { ErdosRenyiGenerator::validate({-1, 0.5, 1}); })
        && throwsInvalidArgument([] { ErdosRenyiGenerator::validate({10, 1.5, 1}); })
        && throwsInvalidArgument([] {
            ErdosRenyiGenerator::validate({10, std::nan(""), 1});
        })
        && !throwsInvalidArgument([] { ErdosRenyiGenerator::validate({0, 0.0, 1}); }),
        "Erdos-Renyi parameters must satisfy n >= 0 and 0 <= p <= 1"
    );
}

void testFamilyInterface()
{
    const MISGeneratorConfig erdosRenyi = ErdosRenyiConfig{125, 0.1, 7};
    const MISGeneratorConfig barabasiAlbert = BarabasiAlbertConfig{200, 8, 7};

    bool roundTrip = true;

    for (const MISGraphFamily family : misGraphFamilies())
    {
        MISGraphFamily parsed{};
        roundTrip = roundTrip
            && parseMISGraphFamily(misGraphFamilyName(family), parsed)
            && parsed == family;
    }

    MISGraphFamily unused{};

    check(
        roundTrip && misGraphFamilies().size() == 2
        && !parseMISGraphFamily("watts_strogatz", unused),
        "Family names are parsed back"
    );

    check(
        misGraphFamily(erdosRenyi) == MISGraphFamily::ErdosRenyi
        && misGraphFamily(barabasiAlbert) == MISGraphFamily::BarabasiAlbert
        && misGeneratorSeed(barabasiAlbert) == 7u,
        "A configuration knows its family and seed"
    );

    check(
        misInstanceId(erdosRenyi) == "erdos_renyi_n125_p0_1_seed7"
        && misInstanceId(ErdosRenyiConfig{10, 0.05, 2})
            == "erdos_renyi_n10_p0_05_seed2"
        && misInstanceId(barabasiAlbert) == "barabasi_albert_n200_k8_seed7",
        "Instance identifiers name the family and every parameter"
    );

    check(
        misManifestParameterNames(MISGraphFamily::ErdosRenyi)
            == std::vector<std::string>({"n", "p"})
        && misManifestParameterValues(erdosRenyi)
            == std::vector<std::string>({"125", "0.10000000000000001"})
        && misManifestParameterNames(MISGraphFamily::BarabasiAlbert)
            == std::vector<std::string>({"n", "k"})
        && misManifestParameterValues(barabasiAlbert)
            == std::vector<std::string>({"200", "8"}),
        "Manifest columns hold the parameters of the family"
    );

    check(
        generateMISInstance(erdosRenyi).graph.edges
            == ErdosRenyiGenerator{}.generate({125, 0.1, 7}).graph.edges
        && generateMISInstance(barabasiAlbert).graph.edges
            == BarabasiAlbertGenerator{}.generate({200, 8, 7}).graph.edges,
        "Generation goes to the generator of the family"
    );

    check(
        throwsInvalidArgument([] {
            validateMISGeneratorConfig(BarabasiAlbertConfig{3, 3, 1});
        })
        && !throwsInvalidArgument([] {
            validateMISGeneratorConfig(ErdosRenyiConfig{3, 0.5, 1});
        }),
        "Validation goes to the generator of the family"
    );
}

} // namespace

int main()
{
    testBarabasiAlbertStructure();
    testBarabasiAlbertReproducibility();
    testValidation();
    testFamilyInterface();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS generator tests passed.\n";
    return 0;
}
