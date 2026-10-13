
#include "io/MISInstanceIO.hpp"

#include <iostream>
#include <sstream>
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

void testRoundTrip()
{
    MISInstance original;
    original.n = 5;
    original.edges = {
        {0, 1},
        {0, 2},
        {1, 3},
        {2, 4}
    };

    std::stringstream buffer;
    std::string error;

    const bool writeOk =
        writeMISInstance(buffer, original, error);

    check(writeOk, "valid instance should be writable");

    if (!writeOk)
    {
        std::cerr << error << '\n';
        return;
    }

    MISInstance restored;
    const bool readOk =
        readMISInstance(buffer, restored, error);

    check(readOk, "written instance should be readable");

    if (readOk)
    {
        check(restored.n == original.n, "vertex count should match");
        check(
            restored.edges == original.edges,
            "edges should match after round-trip"
        );
    }
}

void testRejectsInvalidEndpoint()
{
    std::stringstream input(
        "MIS_INSTANCE 1\n"
        "n 3\n"
        "m 1\n"
        "0 3\n"
    );

    MISInstance instance;
    std::string error;

    check(
        !readMISInstance(input, instance, error),
        "out-of-range endpoint should be rejected"
    );
}

void testRejectsDuplicateEdge()
{
    std::stringstream input(
        "MIS_INSTANCE 1\n"
        "n 3\n"
        "m 2\n"
        "0 1\n"
        "0 1\n"
    );

    MISInstance instance;
    std::string error;

    check(
        !readMISInstance(input, instance, error),
        "duplicate edges should be rejected"
    );
}

void testRejectsMalformedInput()
{
    std::stringstream badHeader(
        "MIS_INSTANCE 99\n"
        "n 3\n"
        "m 0\n"
    );

    MISInstance instance;
    std::string error;

    check(
        !readMISInstance(badHeader, instance, error),
        "unsupported format version should be rejected"
    );

    std::stringstream truncated(
        "MIS_INSTANCE 1\n"
        "n 3\n"
        "m 2\n"
        "0 1\n"
    );

    check(
        !readMISInstance(truncated, instance, error),
        "truncated edge list should be rejected"
    );
}
}

int main()
{
    testRoundTrip();
    testRejectsInvalidEndpoint();
    testRejectsDuplicateEdge();
    testRejectsMalformedInput();

    if (failures != 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS I/O tests passed.\n";
    return 0;
}
