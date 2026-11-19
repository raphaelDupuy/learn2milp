#include "ui/TerminalInput.hpp"

#include "util/Text.hpp"

namespace terminal
{

bool readLine(const std::string& prompt, std::string& value)
{
    std::cout << prompt;

    std::string line;

    if (!std::getline(std::cin, line))
    {
        std::cout << "\nInput interrupted.\n";
        return false;
    }

    value = trimWhitespace(line);
    return true;
}

bool readNonEmptyLine(const std::string& prompt, std::string& value)
{
    while (true)
    {
        if (!readLine(prompt, value))
            return false;

        if (!value.empty())
            return true;

        std::cout << "The answer must not be empty.\n";
    }
}

bool readChoice(
    const std::string& title,
    const std::vector<std::string>& options,
    std::size_t& choice,
    std::size_t defaultChoice
)
{
    std::cout << title << '\n';

    for (std::size_t i = 0; i < options.size(); ++i)
        std::cout << "  " << i + 1 << ". " << options[i] << '\n';

    int number = 0;

    if (!readValue<int>(
            "Choice [" + std::to_string(defaultChoice + 1) + "]: ",
            number,
            [&](const std::string& text, int& value) {
                return parsePositiveInt(text, value)
                    && static_cast<std::size_t>(value) <= options.size();
            },
            "choose a number from 1 to " + std::to_string(options.size()),
            static_cast<int>(defaultChoice + 1)))
    {
        return false;
    }

    choice = static_cast<std::size_t>(number - 1);
    return true;
}

} // namespace terminal
