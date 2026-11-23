#include "io/JSSPInstanceIO.hpp"

#include <algorithm>
#include <istream>
#include <limits>
#include <ostream>
#include <sstream>
#include <utility>

const char* const JSSP_INSTANCE_EXTENSION = ".jsp";

std::vector<std::filesystem::path> listJSSPInstances(
    const std::filesystem::path& directory
)
{
    std::vector<std::filesystem::path> instances;

    for (const auto& entry : std::filesystem::directory_iterator(directory))
    {
        if (entry.is_regular_file()
            && entry.path().extension() == JSSP_INSTANCE_EXTENSION)
        {
            instances.push_back(entry.path());
        }
    }

    std::sort(instances.begin(), instances.end());
    return instances;
}

bool writeJSSPInstance(
    std::ostream& output,
    const JSSPInstance& instance,
    std::string& errorMessage
)
{
    if (!validateJSSPInstance(instance, errorMessage))
        return false;

    output << instance.jobs.size() << ' ' << instance.numberOfMachines << '\n';

    for (const auto& job : instance.jobs)
    {
        for (std::size_t k = 0; k < job.size(); ++k)
        {
            output << (k == 0 ? "" : " ")
                   << job[k].machine << ' ' << job[k].duration;
        }

        output << '\n';
    }

    if (!output)
    {
        errorMessage = "Failed to write JSSP instance.";
        return false;
    }

    errorMessage.clear();
    return true;
}

bool readJSSPInstance(
    std::istream& input,
    JSSPInstance& instance,
    std::string& errorMessage
)
{
    // The numbers of the file, comments removed.
    std::ostringstream numbers;
    std::string line;

    while (std::getline(input, line))
        numbers << line.substr(0, line.find('#')) << '\n';

    if (input.bad())
    {
        errorMessage = "Failed to read JSSP instance.";
        return false;
    }

    std::istringstream tokens(numbers.str());
    long long jobs = 0;
    long long machines = 0;

    if (!(tokens >> jobs >> machines))
    {
        errorMessage = "Expected the numbers of jobs and machines.";
        return false;
    }

    // Keeps jobs * machines in an int.
    constexpr long long maxCount = 1000;

    if (jobs < 1 || machines < 1 || jobs > maxCount || machines > maxCount)
    {
        errorMessage = "Numbers of jobs and machines must lie in [1, "
            + std::to_string(maxCount) + "].";
        return false;
    }

    JSSPInstance parsed;
    parsed.numberOfMachines = static_cast<int>(machines);
    parsed.jobs.assign(
        static_cast<std::size_t>(jobs),
        std::vector<JSSPOperation>(static_cast<std::size_t>(machines)));

    // Every time of a schedule stays below the sum of the durations,
    // which must fit in an int.
    long long totalDuration = 0;

    for (auto& job : parsed.jobs)
    {
        for (JSSPOperation& operation : job)
        {
            long long machine = 0;
            long long duration = 0;

            if (!(tokens >> machine >> duration))
            {
                errorMessage = "Unexpected end of file while reading operations.";
                return false;
            }

            if (machine < 0 || machine >= machines || duration < 1)
            {
                errorMessage = "Machine or duration outside the valid range.";
                return false;
            }

            totalDuration += std::min<long long>(
                duration, std::numeric_limits<int>::max());

            if (totalDuration > std::numeric_limits<int>::max())
            {
                errorMessage = "The sum of the durations is too large.";
                return false;
            }

            operation.machine = static_cast<int>(machine);
            operation.duration = static_cast<int>(duration);
        }
    }

    tokens >> std::ws;

    if (!tokens.eof())
    {
        errorMessage = "Unexpected content after the declared operations.";
        return false;
    }

    if (!validateJSSPInstance(parsed, errorMessage))
        return false;

    instance = std::move(parsed);
    errorMessage.clear();
    return true;
}
