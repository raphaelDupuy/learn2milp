#include "io/CsvReader.hpp"

#include <iostream>
#include <sstream>
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

} // namespace

int main()
{
    std::string error;
    std::vector<std::string> fields;

    // Regular table, including empty fields.
    {
        std::istringstream input("a,b,c\n1,,x\n,2.5,\n");
        CsvReader reader(input);

        check(
            !reader.readRow(fields, error) && !error.empty(),
            "Row before header is rejected"
        );

        check(reader.readHeader(error), "Header is read");

        check(
            reader.columns() == std::vector<std::string>({"a", "b", "c"})
            && reader.column("a") == 0
            && reader.column("c") == 2
            && reader.column("missing") == -1,
            "Columns are found by name"
        );

        check(
            !reader.readHeader(error),
            "Header cannot be read twice"
        );

        check(
            reader.readRow(fields, error)
            && fields == std::vector<std::string>({"1", "", "x"}),
            "Row with an empty middle field"
        );

        check(
            reader.readRow(fields, error)
            && fields == std::vector<std::string>({"", "2.5", ""})
            && reader.rowsRead() == 2,
            "Row with empty first and last fields"
        );

        check(
            !reader.readRow(fields, error)
            && reader.atEnd()
            && error.empty()
            && reader.rowsRead() == 2,
            "End of table is not an error"
        );

        check(
            !reader.readRow(fields, error) && reader.atEnd(),
            "Reading past the end stays at the end"
        );
    }

    // Malformed rows.
    {
        std::istringstream input("a,b\n1,2\n1,2,3\n");
        CsvReader reader(input);

        reader.readHeader(error);
        reader.readRow(fields, error);

        check(
            !reader.readRow(fields, error)
            && !reader.atEnd()
            && error.find("row 2") != std::string::npos,
            "Row with too many fields is reported with its number"
        );
    }

    {
        std::istringstream input("a,b\n1\n");
        CsvReader reader(input);

        reader.readHeader(error);

        check(
            !reader.readRow(fields, error) && !reader.atEnd(),
            "Row with too few fields is rejected"
        );
    }

    // Malformed headers.
    {
        std::istringstream empty("");
        CsvReader emptyReader(empty);

        std::istringstream duplicate("a,b,a\n");
        CsvReader duplicateReader(duplicate);

        std::istringstream unnamed("a,,c\n");
        CsvReader unnamedReader(unnamed);

        check(
            !emptyReader.readHeader(error)
            && !duplicateReader.readHeader(error)
            && !unnamedReader.readHeader(error),
            "Empty, duplicate and unnamed headers are rejected"
        );
    }

    // Header only.
    {
        std::istringstream input("a,b\n");
        CsvReader reader(input);

        check(
            reader.readHeader(error)
            && !reader.readRow(fields, error)
            && reader.atEnd(),
            "Table with a header only has no row"
        );
    }

    // Number parsing.
    {
        long long integer = 0;
        double real = 0.0;

        check(
            parseCsvInteger("-42", integer) && integer == -42
            && !parseCsvInteger("", integer)
            && !parseCsvInteger("4.2", integer)
            && !parseCsvInteger("42 ", integer)
            && !parseCsvInteger("x", integer),
            "Integers must fill the whole field"
        );

        check(
            parseCsvReal("0.3333333333333333", real)
            && real == 1.0 / 3.0
            && parseCsvReal("1e+20", real) && real == 1e20
            && parseCsvReal("-34.96235521235521", real)
            && real == -34.96235521235521
            && !parseCsvReal("", real)
            && !parseCsvReal("1.5x", real)
            && !parseCsvReal("abc", real),
            "Reals round-trip and must fill the whole field"
        );
    }

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All CSV reader tests passed.\n";
    return 0;
}
