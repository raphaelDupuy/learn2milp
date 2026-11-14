#pragma once

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

/*
 * Streaming reader for the plain CSV tables written by this project: a
 * header row, then rows with the same number of comma-separated fields.
 * Quoting is not supported (the tables never need it).
 *
 * Rows are read one at a time, so that large tables are never held in
 * memory.
 */
class CsvReader
{
public:
    // The stream must outlive the reader.
    explicit CsvReader(std::istream& input);

    // Reads the header row. Must be called once, first.
    bool readHeader(std::string& errorMessage);

    const std::vector<std::string>& columns() const;

    // Index of a column, or -1 if the table has no such column.
    int column(const std::string& name) const;

    /*
     * Reads the next row into fields. Returns false at the end of the
     * table, or on a malformed row; atEnd() tells the two cases apart and
     * errorMessage describes a malformed row.
     */
    bool readRow(
        std::vector<std::string>& fields,
        std::string& errorMessage
    );

    bool atEnd() const;

    // Number of data rows read so far.
    std::size_t rowsRead() const;

private:
    std::istream& input_;
    std::vector<std::string> columns_;
    bool headerRead_ = false;
    bool atEnd_ = false;
    std::size_t rowsRead_ = 0;
    std::string line_;
};

// Strict conversions of a CSV field: the whole field must be a number.
bool parseCsvInteger(const std::string& field, long long& value);
bool parseCsvReal(const std::string& field, double& value);
