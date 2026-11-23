#include "io/JSSPInstanceIO.hpp"
#include "problems/JSSP.hpp"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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

// Two jobs on two machines:
//   job 0: machine 0 for 3, then machine 1 for 2
//   job 1: machine 1 for 4, then machine 0 for 1
JSSPInstance smallInstance()
{
    JSSPInstance instance;
    instance.numberOfMachines = 2;
    instance.jobs = {
        {{0, 3}, {1, 2}},
        {{1, 4}, {0, 1}}};
    return instance;
}

bool sameInstance(const JSSPInstance& a, const JSSPInstance& b)
{
    if (a.numberOfMachines != b.numberOfMachines || a.jobs.size() != b.jobs.size())
        return false;

    for (std::size_t j = 0; j < a.jobs.size(); ++j)
    {
        for (std::size_t k = 0; k < a.jobs[j].size(); ++k)
        {
            if (a.jobs[j][k].machine != b.jobs[j][k].machine
                || a.jobs[j][k].duration != b.jobs[j][k].duration)
                return false;
        }
    }

    return true;
}

bool reads(const std::string& text, JSSPInstance& instance, std::string& error)
{
    std::istringstream input(text);
    return readJSSPInstance(input, instance, error);
}

void testRoundTrip()
{
    const JSSPInstance original = smallInstance();
    std::stringstream buffer;
    std::string error;

    check(writeJSSPInstance(buffer, original, error), "valid instance is written");
    check(buffer.str() == "2 2\n0 3 1 2\n1 4 0 1\n", "written in the OR-Library format");

    JSSPInstance restored;
    check(readJSSPInstance(buffer, restored, error), "written instance is read back");
    check(sameInstance(original, restored), "round trip keeps the instance");
}

void testComments()
{
    JSSPInstance instance;
    std::string error;
    const std::string text =
        "#++++\n# instance tiny\n#++++\n2 2\n0 3 1 2 # job 0\n  1 4   0 1\n";

    check(reads(text, instance, error), "comments and spaces are accepted: " + error);
    check(sameInstance(instance, smallInstance()), "comments do not change the data");
}

void testRejections()
{
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"", "empty file"},
        {"2 2\n0 3 1 2\n1 4\n", "missing operations"},
        {"2 2\n0 3 1 2\n1 4 0 1\n7\n", "trailing content"},
        {"2 2\n0 3 0 2\n1 4 0 1\n", "machine visited twice"},
        {"2 2\n0 3 2 2\n1 4 0 1\n", "machine out of range"},
        {"2 2\n0 0 1 2\n1 4 0 1\n", "zero duration"},
        {"0 2\n", "no job"},
        {"2 0\n", "no machine"},
        {"2 2\n0 3 1 2\n1 x 0 1\n", "non-numeric token"},
        {"1 2\n0 2147483647 1 1\n", "durations overflowing an int"}};

    for (const auto& [text, name] : cases)
    {
        JSSPInstance instance = smallInstance();
        std::string error;

        check(!reads(text, instance, error), "rejects: " + name);
        check(!error.empty(), "explains the rejection: " + name);
        check(sameInstance(instance, smallInstance()), "leaves the instance alone: " + name);
    }

    JSSPInstance invalid = smallInstance();
    invalid.jobs[1][1].machine = 1;
    std::ostringstream output;
    std::string error;
    check(!writeJSSPInstance(output, invalid, error), "an invalid instance is not written");
    check(output.str().empty(), "nothing is written for an invalid instance");
}

void testBounds()
{
    const JSSPInstance instance = smallInstance();

    std::vector<int> heads;
    std::vector<int> tails;
    jsspHeadsAndTails(instance, heads, tails);
    check((heads == std::vector<int>{0, 3, 0, 4}), "heads");
    check((tails == std::vector<int>{5, 2, 5, 1}), "tails");

    const auto onMachine = jsspOperationsByMachine(instance);
    check(onMachine.size() == 2 && onMachine[0].size() == 2, "operations by machine");
    check(onMachine[0][1].job == 1 && onMachine[0][1].position == 1
              && onMachine[0][1].duration == 1,
          "operations by machine, in job order");

    // Machine 1 carries 6 units with no head and no tail on one side.
    check(jsspOneMachineBound(instance) == 6, "one-machine bound");

    // Job 0 first on machine 0, job 1 first on machine 1, then both end
    // at 6, which is optimal.
    check(jsspDispatchMakespan(instance) == 6, "dispatch makespan");
}

} // namespace

int main()
{
    testRoundTrip();
    testComments();
    testRejections();
    testBounds();

    if (failures != 0)
    {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }

    std::cout << "All JSSP instance IO tests passed\n";
    return 0;
}
