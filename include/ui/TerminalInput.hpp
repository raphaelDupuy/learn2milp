#pragma once

#include <cstddef>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "tools/CommandLine.hpp"
#include "util/Text.hpp"

/*
 * Answers typed in the interactive menu. Every reader asks again until the
 * answer is valid, and returns false only when the input ends, the caller
 * then going back to the menu. Values are read with the parsers of the
 * command line tools (tools/CommandLine.hpp), so that the menu and the
 * tools accept the same values.
 */
namespace terminal
{

/*
 * A typed comma-separated list without the spaces around its items
 * ("0, 1" becomes "0,1"), which the parsers of the tools do not accept.
 */
inline std::string withoutSpacesAroundItems(const std::string& text)
{
    std::string result;
    std::size_t start = 0;

    while (true)
    {
        const std::size_t comma = text.find(',', start);
        result += trimWhitespace(text.substr(start, comma - start));

        if (comma == std::string::npos)
            return result;

        result += ',';
        start = comma + 1;
    }
}

// Prints the prompt and reads one line, without its surrounding spaces.
bool readLine(const std::string& prompt, std::string& value);

bool readNonEmptyLine(const std::string& prompt, std::string& value);

/*
 * Reads one value with parser(text, value). requirement tells what a valid
 * answer is. An empty answer gives defaultValue when there is one.
 */
template <typename T, typename Parser>
bool readValue(
    const std::string& prompt,
    T& value,
    Parser parser,
    const std::string& requirement,
    std::optional<T> defaultValue = std::nullopt
)
{
    while (true)
    {
        std::string line;

        if (!readLine(prompt, line))
            return false;

        if (line.empty() && defaultValue.has_value())
        {
            value = *defaultValue;
            return true;
        }

        T parsed{};

        if (parser(line, parsed))
        {
            value = parsed;
            return true;
        }

        std::cout << "Invalid value: " << requirement << ".\n";
    }
}

/*
 * Reads a comma-separated list of distinct values, each read by parser,
 * as the list options of the tools (spaces around the items allowed).
 * An empty answer stands for defaultText when it is not empty.
 */
template <typename T, typename Parser>
bool readList(
    const std::string& prompt,
    std::vector<T>& values,
    Parser parser,
    const std::string& defaultText = ""
)
{
    while (true)
    {
        std::string line;

        if (!readLine(prompt, line))
            return false;

        if (line.empty())
            line = defaultText;

        try
        {
            values = parseCommaSeparated<T>(
                withoutSpacesAroundItems(line), parser, "the list");
            return true;
        }
        catch (const CommandLineError& error)
        {
            std::cout << error.what() << ".\n";
        }
    }
}

// Shows numbered options and reads one of them (0-based index).
bool readChoice(
    const std::string& title,
    const std::vector<std::string>& options,
    std::size_t& choice,
    std::size_t defaultChoice = 0
);

} // namespace terminal
