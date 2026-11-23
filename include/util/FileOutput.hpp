#pragma once

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

// Removes every tracked file, newest first, unless committed: a failed
// generation leaves no file behind.
class FileCleanup
{
public:
    FileCleanup() = default;
    FileCleanup(const FileCleanup&) = delete;
    FileCleanup& operator=(const FileCleanup&) = delete;

    void track(const std::filesystem::path& path)
    {
        paths_.push_back(path);
    }

    void commit()
    {
        committed_ = true;
    }

    ~FileCleanup()
    {
        if (committed_)
            return;

        for (auto it = paths_.rbegin(); it != paths_.rend(); ++it)
        {
            std::error_code ignored;
            std::filesystem::remove(*it, ignored);
        }
    }

private:
    std::vector<std::filesystem::path> paths_;
    bool committed_ = false;
};

// Closes a file and throws std::runtime_error if anything failed.
inline void closeOutput(std::ofstream& file, const std::filesystem::path& path)
{
    file.close();

    if (!file)
    {
        throw std::runtime_error(
            "Failed to finish writing: " + path.string()
        );
    }
}

// Throws std::runtime_error if the path exists: files are never
// overwritten.
inline void requireAbsent(const std::filesystem::path& path)
{
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error(
            "Output file already exists: " + path.string()
        );
    }
}
