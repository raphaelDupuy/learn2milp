#include "ui/FileBrowser.hpp"

#include "io/MISInstanceIO.hpp"
#include "io/MISStrongBranchingSampleIO.hpp"
#include "tools/CommandLine.hpp"
#include "ui/TerminalInput.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <system_error>
#include <vector>

namespace terminal
{

const BrowsedKind DIRECTORY_KIND;
const BrowsedKind INSTANCE_KIND = MIS_INSTANCE_EXTENSION;
const BrowsedKind MODEL_KIND = ".model";
const BrowsedKind SETTINGS_KIND = ".set";

namespace
{

const std::size_t BROWSER_PAGE_SIZE = 20;

std::map<BrowsedKind, std::filesystem::path>& lastBrowsedDirectories()
{
    static std::map<BrowsedKind, std::filesystem::path> directories;
    return directories;
}

// Absolute path without "." or "..", so that ".." always goes up.
std::filesystem::path normalized(const std::filesystem::path& path)
{
    std::error_code error;
    const std::filesystem::path canonical =
        std::filesystem::weakly_canonical(path, error);

    return error ? std::filesystem::absolute(path).lexically_normal()
                 : canonical;
}

bool hasExtension(const std::filesystem::path& path, const BrowsedKind& kind)
{
    return path.extension() == kind;
}

struct DirectoryListing
{
    std::vector<std::filesystem::path> directories;
    std::vector<std::filesystem::path> files;

    // Directory mode: instances and collections, to recognize a dataset.
    std::size_t instances = 0;
    std::size_t collections = 0;
};

DirectoryListing listDirectory(
    const std::filesystem::path& directory,
    const BrowsedKind& kind
)
{
    DirectoryListing listing;

    for (const auto& entry : std::filesystem::directory_iterator(
            directory,
            std::filesystem::directory_options::skip_permission_denied))
    {
        const std::filesystem::path& path = entry.path();
        const std::string name = path.filename().string();

        if (name.empty() || name[0] == '.')
            continue;

        std::error_code error;

        if (entry.is_directory(error))
        {
            listing.directories.push_back(path);
        }
        else if (entry.is_regular_file(error))
        {
            if (kind == DIRECTORY_KIND)
            {
                listing.instances +=
                    hasExtension(path, INSTANCE_KIND) ? 1 : 0;
                listing.collections +=
                    isMISCollectionMetadataFile(path) ? 1 : 0;
            }
            else if (hasExtension(path, kind))
            {
                listing.files.push_back(path);
            }
        }
    }

    std::sort(listing.directories.begin(), listing.directories.end());
    std::sort(listing.files.begin(), listing.files.end());
    return listing;
}

} // namespace

std::filesystem::path browserStart(const BrowsedKind& kind)
{
    const auto& last = lastBrowsedDirectories();
    const auto found = last.find(kind);

    if (found != last.end() && std::filesystem::is_directory(found->second))
        return found->second;

    const std::filesystem::path preferred =
        kind == INSTANCE_KIND ? "data/instances"
        : kind == MODEL_KIND ? "experiments"
        : kind == SETTINGS_KIND ? "experiments/cuts/settings"
        : "data";

    std::error_code error;

    return std::filesystem::is_directory(preferred, error)
        ? preferred : std::filesystem::path(".");
}

std::string displayPath(const std::filesystem::path& path)
{
    std::error_code error;
    const std::filesystem::path relative =
        path.lexically_relative(std::filesystem::current_path(error));

    if (error || relative.empty() || *relative.begin() == "..")
        return path.string();

    return relative.string();
}

bool browse(
    const BrowsedKind& kind,
    std::filesystem::path start,
    std::filesystem::path& selected
)
{
    const bool pickDirectory = kind == DIRECTORY_KIND;
    std::filesystem::path directory = normalized(start);
    std::size_t page = 0;

    // The listing is printed again only after a move, not after a
    // mistyped answer.
    bool showListing = true;

    selected.clear();

    while (true)
    {
        DirectoryListing listing;

        try
        {
            listing = listDirectory(directory, kind);
        }
        catch (const std::filesystem::filesystem_error& exception)
        {
            std::cout << "Cannot read " << displayPath(directory) << ": "
                      << exception.code().message() << '\n';

            if (directory == directory.parent_path())
                return true;

            directory = directory.parent_path();
            continue;
        }

        // Entries: subdirectories first, then the files to pick.
        std::vector<std::filesystem::path> entries = listing.directories;
        entries.insert(
            entries.end(), listing.files.begin(), listing.files.end());

        const std::size_t pages = std::max<std::size_t>(
            1, (entries.size() + BROWSER_PAGE_SIZE - 1) / BROWSER_PAGE_SIZE);
        page = std::min(page, pages - 1);

        if (showListing)
        {
            std::cout << "\n[" << displayPath(directory) << "]";

            if (pickDirectory && (listing.instances > 0 || listing.collections > 0))
            {
                std::cout << "  " << listing.instances << " .mis, "
                          << listing.collections << ' '
                          << MIS_SB_METADATA_SUFFIX;
            }
            else if (!pickDirectory)
            {
                std::cout << "  " << listing.files.size() << ' ' << kind
                          << " file(s)";
            }

            std::cout << '\n';

            const std::size_t first = page * BROWSER_PAGE_SIZE;
            const std::size_t last =
                std::min(entries.size(), first + BROWSER_PAGE_SIZE);

            for (std::size_t i = first; i < last; ++i)
            {
                const bool isDirectory = i < listing.directories.size();

                std::cout << std::setw(5) << i + 1 << ". "
                          << entries[i].filename().string()
                          << (isDirectory ? "/" : "") << '\n';
            }

            if (entries.empty())
            {
                std::cout << "      (no subdirectory"
                          << (pickDirectory ? "" : " or " + kind + " file")
                          << ")\n";
            }

            std::cout << "Number to open" << (pickDirectory ? "" : " or pick")
                      << ", .. to go up"
                      << (pickDirectory ? ", . to pick this directory" : "");

            if (pages > 1)
            {
                std::cout << ", n/p: next/previous page (" << page + 1 << '/'
                          << pages << ")";
            }

            std::cout << ", a path, q to cancel.\n";
        }

        showListing = true;

        std::string answer;

        if (!readLine("> ", answer))
            return false;

        if (answer == "q")
            return true;

        if (answer == "n" || answer == "p")
        {
            page = answer == "n"
                ? std::min(page + 1, pages - 1)
                : (page > 0 ? page - 1 : 0);
            continue;
        }

        if (answer == "." && pickDirectory)
        {
            selected = directory;
            lastBrowsedDirectories()[kind] = directory;
            return true;
        }

        std::filesystem::path target;
        int number = 0;

        if (answer.empty())
        {
            showListing = false;
            continue;
        }

        if (parseNonNegativeInt(answer, number))
        {
            if (entries.empty())
            {
                std::cout << "Nothing to open here: "
                          << (pickDirectory ? ". picks this directory, " : "")
                          << ".. goes up, q cancels.\n";
                showListing = false;
                continue;
            }

            if (number < 1 || static_cast<std::size_t>(number) > entries.size())
            {
                std::cout << "Choose a number from 1 to " << entries.size()
                          << ".\n";
                showListing = false;
                continue;
            }

            target = entries[static_cast<std::size_t>(number - 1)];
        }
        else
        {
            const std::filesystem::path typed(answer);
            target = normalized(
                typed.is_absolute() ? typed : directory / typed);
        }

        std::error_code error;

        if (std::filesystem::is_directory(target, error))
        {
            directory = normalized(target);
            page = 0;
        }
        else if (!pickDirectory
                 && std::filesystem::is_regular_file(target, error)
                 && hasExtension(target, kind))
        {
            selected = target;
            lastBrowsedDirectories()[kind] = directory;
            return true;
        }
        else
        {
            std::cout << "Not " << (pickDirectory ? "a directory" :
                          "a directory or a " + kind + " file")
                      << ": " << answer << '\n';
            showListing = false;
        }
    }
}

bool readPath(
    const std::string& what,
    const BrowsedKind& kind,
    std::filesystem::path& path,
    PathRequirement requirement
)
{
    const bool pickDirectory = kind == DIRECTORY_KIND;
    const bool optional = requirement == PathRequirement::Optional;

    while (true)
    {
        std::string answer;

        if (!readLine(what + (optional
                    ? " (empty for none, ? to browse, q to cancel): "
                    : " (empty or ? to browse, q to cancel): "),
                answer))
        {
            return false;
        }

        if (answer == "q")
            return false;

        if (answer.empty() && optional)
        {
            path.clear();
            return true;
        }

        std::error_code error;
        std::filesystem::path start = browserStart(kind);

        if (!answer.empty() && answer != "?")
        {
            const std::filesystem::path typed(answer);

            if (pickDirectory
                ? std::filesystem::is_directory(typed, error)
                : std::filesystem::is_regular_file(typed, error)
                    && hasExtension(typed, kind))
            {
                path = typed;
                return true;
            }

            if (pickDirectory || !std::filesystem::is_directory(typed, error))
            {
                std::cout << "No such "
                          << (pickDirectory ? "directory" : kind + " file")
                          << ": " << answer << '\n';
                continue;
            }

            start = typed;
        }

        std::filesystem::path selected;

        if (!browse(kind, start, selected))
            return false;

        if (!selected.empty())
        {
            path = displayPath(selected);
            std::cout << "Selected: " << path.string() << '\n';
            return true;
        }
    }
}

} // namespace terminal
