#include "tools/CommandLine.hpp"

#include <cctype>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace
{

// Digits only: no sign, no space, at least one digit.
bool isDigits(const std::string& text)
{
    return !text.empty()
        && std::all_of(text.begin(), text.end(), [](unsigned char c) {
               return std::isdigit(c);
           });
}

// Non-negative integer written with digits only, at most maximum.
bool parseUnsigned(
    const std::string& text,
    unsigned long long maximum,
    unsigned long long& value
)
{
    if (!isDigits(text))
        return false;

    try
    {
        value = std::stoull(text);
        return value <= maximum;
    }
    catch (const std::out_of_range&)
    {
        return false;
    }
}

bool parseFiniteDouble(const std::string& text, double& value)
{
    if (text.empty() || std::isspace(static_cast<unsigned char>(text[0])))
        return false;

    char* end = nullptr;
    value = std::strtod(text.c_str(), &end);

    return end == text.c_str() + text.size() && std::isfinite(value);
}

} // namespace

bool isHelpRequested(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];

        if (argument == "-h" || argument == "--help")
            return true;
    }

    return false;
}

int reportCommandLineError(const char* program, const CommandLineError& error)
{
    std::cerr << error.what() << '\n'
              << "Run " << program << " --help for usage.\n";
    return 2;
}

bool parseNonNegativeInt(const std::string& text, int& value)
{
    unsigned long long parsed = 0;

    if (!parseUnsigned(text, std::numeric_limits<int>::max(), parsed))
        return false;

    value = static_cast<int>(parsed);
    return true;
}

bool parsePositiveInt(const std::string& text, int& value)
{
    return parseNonNegativeInt(text, value) && value > 0;
}

bool parsePositiveInteger(const std::string& text, long long& value)
{
    unsigned long long parsed = 0;

    if (!parseUnsigned(text, std::numeric_limits<long long>::max(), parsed)
        || parsed == 0)
    {
        return false;
    }

    value = static_cast<long long>(parsed);
    return true;
}

bool parseProbability(const std::string& text, double& value)
{
    return parseFiniteDouble(text, value) && value >= 0.0 && value <= 1.0;
}

bool parsePositiveDouble(const std::string& text, double& value)
{
    return parseFiniteDouble(text, value) && value > 0.0;
}

bool parseSeed(const std::string& text, unsigned int& value)
{
    unsigned long long parsed = 0;

    if (!parseUnsigned(
            text, std::numeric_limits<unsigned int>::max(), parsed))
    {
        return false;
    }

    value = static_cast<unsigned int>(parsed);
    return true;
}

bool parseSCIPSeed(const std::string& text, unsigned int& value)
{
    unsigned long long parsed = 0;

    if (!parseUnsigned(text, INT_MAX, parsed))
        return false;

    value = static_cast<unsigned int>(parsed);
    return true;
}

bool parseText(const std::string& text, std::string& value)
{
    value = text;
    return !text.empty();
}

std::vector<std::string> splitCommaSeparated(
    const std::string& text,
    const std::string& option
)
{
    std::vector<std::string> items;
    std::size_t start = 0;

    while (true)
    {
        const std::size_t comma = text.find(',', start);
        const std::size_t end =
            comma == std::string::npos ? text.size() : comma;

        if (end == start)
        {
            throw CommandLineError(
                "Empty item in the list of " + option + ": '" + text + "'"
            );
        }

        items.push_back(text.substr(start, end - start));

        if (comma == std::string::npos)
            return items;

        start = comma + 1;
    }
}

CommandLineOptions::CommandLineOptions(
    int argc,
    char* argv[],
    int first,
    const std::vector<std::string>& known
)
{
    for (int i = first; i < argc; i += 2)
    {
        const std::string option = argv[i];

        if (std::find(known.begin(), known.end(), option) == known.end())
            throw CommandLineError("Unknown option: " + option);

        if (i + 1 >= argc)
            throw CommandLineError("Missing value for option: " + option);

        const std::string value = argv[i + 1];

        if (value.empty())
            throw CommandLineError("Empty value for option: " + option);

        if (!values_.emplace(option, value).second)
            throw CommandLineError("Option given twice: " + option);
    }
}

bool CommandLineOptions::has(const std::string& option) const
{
    return values_.count(option) > 0;
}

const std::string& CommandLineOptions::text(const std::string& option) const
{
    const auto found = values_.find(option);

    if (found == values_.end())
        throw CommandLineError("Missing required option: " + option);

    return found->second;
}
