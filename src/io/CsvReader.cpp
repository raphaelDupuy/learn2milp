#include "io/CsvReader.hpp"

#include <charconv>
#include <istream>
#include <system_error>

namespace
{
void splitFields(const std::string& line, std::vector<std::string>& fields)
{
    fields.clear();

    std::size_t start = 0;

    while (true)
    {
        const std::size_t comma = line.find(',', start);

        if (comma == std::string::npos)
        {
            fields.emplace_back(line, start);
            return;
        }

        fields.emplace_back(line, start, comma - start);
        start = comma + 1;
    }
}
}

CsvReader::CsvReader(std::istream& input)
    : input_(input)
{
}

bool CsvReader::readHeader(std::string& errorMessage)
{
    if (headerRead_)
    {
        errorMessage = "CSV header was already read.";
        return false;
    }

    if (!std::getline(input_, line_) || line_.empty())
    {
        errorMessage = "CSV table has no header row.";
        return false;
    }

    splitFields(line_, columns_);

    for (std::size_t i = 0; i < columns_.size(); ++i)
    {
        if (columns_[i].empty())
        {
            errorMessage = "CSV header contains an empty column name.";
            return false;
        }

        for (std::size_t j = 0; j < i; ++j)
        {
            if (columns_[j] == columns_[i])
            {
                errorMessage =
                    "CSV header repeats the column: " + columns_[i];
                return false;
            }
        }
    }

    headerRead_ = true;
    errorMessage.clear();
    return true;
}

const std::vector<std::string>& CsvReader::columns() const
{
    return columns_;
}

int CsvReader::column(const std::string& name) const
{
    for (std::size_t i = 0; i < columns_.size(); ++i)
    {
        if (columns_[i] == name)
        {
            return static_cast<int>(i);
        }
    }

    return -1;
}

bool CsvReader::readRow(
    std::vector<std::string>& fields,
    std::string& errorMessage
)
{
    if (!headerRead_)
    {
        errorMessage = "CSV header must be read first.";
        return false;
    }

    if (atEnd_ || !std::getline(input_, line_))
    {
        atEnd_ = true;
        errorMessage.clear();
        return false;
    }

    splitFields(line_, fields);

    if (fields.size() != columns_.size())
    {
        errorMessage =
            "CSV row " + std::to_string(rowsRead_ + 1) + " has "
            + std::to_string(fields.size()) + " fields instead of "
            + std::to_string(columns_.size()) + ".";
        return false;
    }

    ++rowsRead_;
    errorMessage.clear();
    return true;
}

bool CsvReader::atEnd() const
{
    return atEnd_;
}

std::size_t CsvReader::rowsRead() const
{
    return rowsRead_;
}

bool parseCsvInteger(const std::string& field, long long& value)
{
    const char* const first = field.data();
    const char* const last = first + field.size();

    const auto conversion = std::from_chars(first, last, value);

    return conversion.ec == std::errc{} && conversion.ptr == last;
}

bool parseCsvReal(const std::string& field, double& value)
{
    const char* const first = field.data();
    const char* const last = first + field.size();

    const auto conversion = std::from_chars(first, last, value);

    return conversion.ec == std::errc{} && conversion.ptr == last;
}
