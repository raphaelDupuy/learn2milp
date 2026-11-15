#pragma once

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/*
 * Command line arguments of the mis-* tools: options written as pairs
 * "--option value", each given at most once.
 *
 * Every malformed argument is reported by a CommandLineError, which the
 * tools turn into exit code 2.
 */

class CommandLineError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

// True if -h or --help is one of the arguments.
bool isHelpRequested(int argc, char* argv[]);

// Prints the error and how to get help to stderr; returns exit code 2.
int reportCommandLineError(const char* program, const CommandLineError& error);

/*
 * Value parsers: true if the whole text is a valid value, which is then
 * stored in value; false otherwise, value being left unspecified.
 */
bool parseNonNegativeInt(const std::string& text, int& value);
bool parsePositiveInt(const std::string& text, int& value);
bool parsePositiveInteger(const std::string& text, long long& value);
bool parseProbability(const std::string& text, double& value);
bool parsePositiveDouble(const std::string& text, double& value);

// Any unsigned int: seeds of the instance generators.
bool parseSeed(const std::string& text, unsigned int& value);

// At most INT_MAX: SCIP's randomization/randomseedshift.
bool parseSCIPSeed(const std::string& text, unsigned int& value);

// Any non-empty text.
bool parseText(const std::string& text, std::string& value);

/*
 * Items of a comma-separated list. Throws CommandLineError, naming
 * option, if the list or one of its items is empty (as with a trailing
 * comma).
 */
std::vector<std::string> splitCommaSeparated(
    const std::string& text,
    const std::string& option
);

class CommandLineOptions
{
public:
    /*
     * Reads argv[first], ..., argv[argc - 1]. Throws CommandLineError for
     * an option that is not in known, an option given twice and a
     * missing or empty value.
     */
    CommandLineOptions(
        int argc,
        char* argv[],
        int first,
        const std::vector<std::string>& known
    );

    bool has(const std::string& option) const;

    // Text of a required option; throws CommandLineError if it is absent.
    const std::string& text(const std::string& option) const;

    // Value of a required option, read by parser(text, value).
    template <typename T, typename Parser>
    T get(const std::string& option, Parser parser) const
    {
        return parseOrThrow<T>(option, text(option), parser);
    }

    // Value of an optional option, defaultValue when it is absent.
    template <typename T, typename Parser>
    T get(const std::string& option, Parser parser, T defaultValue) const
    {
        return has(option) ? get<T>(option, parser) : defaultValue;
    }

    // Comma-separated values of a required option, all distinct.
    template <typename T, typename Parser>
    std::vector<T> list(const std::string& option, Parser parser) const
    {
        std::vector<T> values;

        for (const std::string& item :
                splitCommaSeparated(text(option), option))
        {
            T value = parseOrThrow<T>(option, item, parser);

            if (std::find(values.begin(), values.end(), value)
                != values.end())
            {
                throw CommandLineError(
                    "Duplicate value in " + option + ": '" + item + "'"
                );
            }

            values.push_back(std::move(value));
        }

        return values;
    }

private:
    template <typename T, typename Parser>
    static T parseOrThrow(
        const std::string& option,
        const std::string& text,
        Parser parser
    )
    {
        T value{};

        if (!parser(text, value))
        {
            throw CommandLineError(
                "Invalid value for " + option + ": '" + text + "'"
            );
        }

        return value;
    }

    std::map<std::string, std::string> values_;
};
