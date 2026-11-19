#pragma once

#include <algorithm>
#include <cctype>
#include <string>

// The text without its leading and trailing white space.
inline std::string trimWhitespace(const std::string& text)
{
    const auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };

    const auto first = std::find_if_not(text.begin(), text.end(), isSpace);
    const auto last = std::find_if_not(text.rbegin(), text.rend(), isSpace);

    return first < last.base() ? std::string(first, last.base()) : "";
}
