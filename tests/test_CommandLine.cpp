#include "tools/CommandLine.hpp"

#include <climits>
#include <iostream>
#include <string>
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

// Arguments as main() receives them, the program name first.
struct Arguments
{
    explicit Arguments(std::vector<std::string> values)
        : storage(std::move(values))
    {
        for (std::string& value : storage)
            pointers.push_back(value.data());
    }

    int argc() { return static_cast<int>(pointers.size()); }
    char** argv() { return pointers.data(); }

    std::vector<std::string> storage;
    std::vector<char*> pointers;
};

const std::vector<std::string> KNOWN{"--a", "--list", "--number"};

// The message of the error thrown by parsing, empty if none.
std::string parseError(std::vector<std::string> values)
{
    values.insert(values.begin(), "program");
    Arguments arguments(values);

    try
    {
        CommandLineOptions(arguments.argc(), arguments.argv(), 1, KNOWN);
        return "";
    }
    catch (const CommandLineError& error)
    {
        return error.what();
    }
}

template <typename Action>
std::string errorOf(Action action)
{
    try
    {
        action();
        return "";
    }
    catch (const CommandLineError& error)
    {
        return error.what();
    }
}

} // namespace

int main()
{
    // Value parsers.
    {
        int i = 0;
        long long l = 0;
        double d = 0.0;
        unsigned int u = 0;

        check(parseNonNegativeInt("0", i) && i == 0
                && parseNonNegativeInt("125", i) && i == 125,
            "Non-negative integers");
        check(!parseNonNegativeInt("-1", i) && !parseNonNegativeInt("", i)
                && !parseNonNegativeInt("1.5", i)
                && !parseNonNegativeInt(" 5", i)
                && !parseNonNegativeInt("+5", i)
                && !parseNonNegativeInt("5x", i)
                && !parseNonNegativeInt("99999999999", i),
            "Malformed or out-of-range integers are rejected");
        check(!parsePositiveInt("0", i) && parsePositiveInt("8", i) && i == 8,
            "Positive integers");
        check(parsePositiveInteger("10000000000", l) && l == 10000000000LL
                && !parsePositiveInteger("0", l)
                && !parsePositiveInteger("abc", l),
            "Positive long integers");
        check(parseProbability("0", d) && parseProbability("1", d)
                && parseProbability("0.05", d) && d == 0.05
                && parseProbability("1e-3", d)
                && !parseProbability("1.5", d)
                && !parseProbability("-0.1", d)
                && !parseProbability("nan", d)
                && !parseProbability("0.1x", d)
                && !parseProbability(" 0.1", d),
            "Probabilities");
        check(parsePositiveDouble("0.5", d) && d == 0.5
                && !parsePositiveDouble("0", d)
                && !parsePositiveDouble("inf", d),
            "Positive numbers");
        check(parseSeed("4294967295", u) && u == 4294967295U
                && !parseSeed("4294967296", u) && !parseSeed("-1", u),
            "Generator seeds cover unsigned int");
        check(parseSCIPSeed(std::to_string(INT_MAX), u)
                && !parseSCIPSeed(std::to_string(1LL + INT_MAX), u),
            "SCIP seeds stop at INT_MAX");
    }

    // Options.
    check(parseError({}).empty(), "No option is accepted");
    check(parseError({"--a", "x", "--number", "3"}).empty(),
        "Known options are accepted");
    check(parseError({"--b", "x"}) == "Unknown option: --b",
        "Unknown option is rejected");
    check(parseError({"--a"}) == "Missing value for option: --a",
        "Missing value is rejected");
    check(parseError({"--a", ""}) == "Empty value for option: --a",
        "Empty value is rejected");
    check(parseError({"--a", "x", "--a", "y"}) == "Option given twice: --a",
        "Repeated option is rejected");

    {
        Arguments arguments({"program", "sub", "--number", "7",
                             "--list", "1,2,3"});
        const CommandLineOptions options(
            arguments.argc(), arguments.argv(), 2, KNOWN);

        check(options.has("--number") && !options.has("--a"),
            "has() tells given options");
        check(options.get<int>("--number", parseNonNegativeInt) == 7,
            "Required value is parsed");
        check(options.get("--a", parseText, std::string("default"))
                == "default",
            "Absent optional value gives the default");
        check(options.list<int>("--list", parseNonNegativeInt)
                == std::vector<int>({1, 2, 3}),
            "List values are parsed in order");
        check(errorOf([&] { options.text("--a"); })
                == "Missing required option: --a",
            "Absent required option is reported");
    }

    {
        Arguments arguments({"program", "--number", "x", "--list", "1,2,1",
                             "--a", "1,,2"});
        const CommandLineOptions options(
            arguments.argc(), arguments.argv(), 1, KNOWN);

        check(errorOf([&] { options.get<int>("--number", parseNonNegativeInt); })
                == "Invalid value for --number: 'x'",
            "Invalid value is reported");
        check(errorOf([&] { options.list<int>("--list", parseNonNegativeInt); })
                == "Duplicate value in --list: '1'",
            "Duplicate list value is reported");
        check(!errorOf([&] { options.list<int>("--a", parseNonNegativeInt); })
                   .empty(),
            "Empty list item is reported");
        check(!errorOf([] { splitCommaSeparated("1,2,", "--x"); }).empty()
                && !errorOf([] { splitCommaSeparated(",1", "--x"); }).empty(),
            "Leading and trailing commas are reported");
    }

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All command line tests passed.\n";
    return 0;
}
