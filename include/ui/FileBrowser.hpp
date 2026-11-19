#pragma once

#include <filesystem>
#include <string>

/*
 * Choosing files and directories in the interactive menu: typed, or picked
 * in a browser. A file is picked among those of one extension; a directory
 * is picked by moving into it. Each kind of file remembers where it was
 * last browsed.
 */
namespace terminal
{

// Extension of the files to pick, or empty to pick a directory.
using BrowsedKind = std::string;

extern const BrowsedKind DIRECTORY_KIND;
extern const BrowsedKind INSTANCE_KIND;  // .mis
extern const BrowsedKind MODEL_KIND;     // .model
extern const BrowsedKind SETTINGS_KIND;  // .set, SCIP settings files

// Where the browser opens for a kind: where it was last used, or a
// directory of the project where such files are usually kept.
std::filesystem::path browserStart(const BrowsedKind& kind);

// Relative to the working directory when inside it, absolute otherwise.
std::string displayPath(const std::filesystem::path& path);

/*
 * Lets the user move through directories and pick a file of the given
 * kind, or a directory. selected stays empty if the user cancels.
 * Returns false only when the input ends.
 */
bool browse(
    const BrowsedKind& kind,
    std::filesystem::path start,
    std::filesystem::path& selected
);

enum class PathRequirement
{
    Required,  // an empty answer opens the browser
    Optional   // an empty answer means none
};

/*
 * Asks for a file of the given kind, or a directory: typed, or picked in
 * the browser on "?" (or on an empty answer when the path is required).
 * A typed directory, where a file is expected, opens the browser there.
 * An optional path left empty is cleared. Returns false when the user
 * cancels with "q" or the input ends.
 */
bool readPath(
    const std::string& what,
    const BrowsedKind& kind,
    std::filesystem::path& path,
    PathRequirement requirement = PathRequirement::Required
);

} // namespace terminal
