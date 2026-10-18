
#include "generators/ErdosRenyiGenerator.hpp"
#include "io/MISInstanceIO.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace
{

void printUsage(const char* program)
{
    std::cerr
        << "Usage:\n"
        << "  " << program
        << " single --n N --p P --seed S --output PATH\n"
        << "  " << program
        << " batch --n-values N1,N2 --p-values P1,P2 "
        << "--seeds S1,S2 --output-dir DIR\n\n"
        << "Options for single:\n"
        << "  --n       Number of vertices (N >= 0)\n"
        << "  --p       Edge probability (0 <= P <= 1)\n"
        << "  --seed    Random seed (unsigned integer)\n"
        << "  --output  Output path for the .mis instance\n\n"
        << "Options for batch:\n"
        << "  --n-values   Comma-separated vertex counts\n"
        << "  --p-values   Comma-separated edge probabilities\n"
        << "  --seeds      Comma-separated random seeds\n"
        << "  --output-dir Output directory for the batch\n";
}


bool parseInt(const std::string& text, int& value)
{
    try
    {
        std::size_t position = 0;
        const long long parsed = std::stoll(text, &position);

        if (position != text.size()
            || parsed < 0
            || parsed > std::numeric_limits<int>::max())
        {
            return false;
        }

        value = static_cast<int>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool parseSeed(
    const std::string& text,
    unsigned int& value
)
{
    try
    {
        std::size_t position = 0;

        if (text.empty() || text.front() == '-')
        {
            return false;
        }

        const unsigned long long parsed =
            std::stoull(text, &position);

        if (position != text.size()
            || parsed > std::numeric_limits<unsigned int>::max())
        {
            return false;
        }

        value = static_cast<unsigned int>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

bool parseProbability(
    const std::string& text,
    double& value
)
{
    try
    {
        std::size_t position = 0;
        value = std::stod(text, &position);

        return position == text.size()
            && value >= 0.0
            && value <= 1.0;
    }
    catch (...)
    {
        return false;
    }
}

int runSingle(int argc, char* argv[])
{
    int n = -1;
    double p = -1.0;
    unsigned int seed = 0;
    std::string outputPath;

    bool hasN = false;
    bool hasP = false;
    bool hasSeed = false;
    bool hasOutput = false;

    for (int i = 2; i < argc; ++i)
    {
        const std::string option = argv[i];

        if (i + 1 >= argc)
        {
            std::cerr << "Missing value for option: "
                      << option << '\n';
            return 2;
        }

        const std::string value = argv[++i];

        if (option == "--n")
        {
            if (hasN || !parseInt(value, n))
            {
                std::cerr << "Invalid or duplicate --n value.\n";
                return 2;
            }
            hasN = true;
        }
        else if (option == "--p")
        {
            if (hasP || !parseProbability(value, p))
            {
                std::cerr << "Invalid or duplicate --p value.\n";
                return 2;
            }
            hasP = true;
        }
        else if (option == "--seed")
        {
            if (hasSeed || !parseSeed(value, seed))
            {
                std::cerr << "Invalid or duplicate --seed value.\n";
                return 2;
            }
            hasSeed = true;
        }
        else if (option == "--output")
        {
            if (hasOutput || value.empty())
            {
                std::cerr << "Invalid or duplicate --output value.\n";
                return 2;
            }
            outputPath = value;
            hasOutput = true;
        }
        else
        {
            std::cerr << "Unknown option: " << option << '\n';
            return 2;
        }
    }

    if (!hasN || !hasP || !hasSeed || !hasOutput)
    {
        printUsage(argv[0]);
        return 2;
    }

    try
    {
        ErdosRenyiGenerator generator;

        const GeneratedMISInstance generated =
            generator.generate(ErdosRenyiConfig{n, p, seed});

        std::ofstream graphFile(outputPath);

        if (!graphFile)
        {
            std::cerr << "Cannot open output file: "
                      << outputPath << '\n';
            return 1;
        }

        std::string errorMessage;

        if (!writeMISInstance(
                graphFile,
                generated.graph,
                errorMessage))
        {
            std::cerr << "Failed to write instance: "
                      << errorMessage << '\n';
            return 1;
        }

        graphFile.close();

        if (!graphFile)
        {
            std::cerr << "Failed to finish writing: "
                      << outputPath << '\n';
            return 1;
        }

        const std::string metadataPath = outputPath + ".meta";
        std::ofstream metadataFile(metadataPath);

        if (!metadataFile)
        {
            std::cerr << "Instance was written, but metadata file "
                      << "could not be opened: "
                      << metadataPath << '\n';
            return 1;
        }

        const auto& metadata = generated.metadata;

        metadataFile
            << "MIS_GENERATION_METADATA 1\n"
            << "family " << metadata.family << '\n'
            << "seed " << metadata.seed << '\n'
            << "n " << metadata.n << '\n'
            << "p " << std::setprecision(17)
            << metadata.p << '\n'
            << "generator_version "
            << metadata.generatorVersion << '\n';

        metadataFile.close();

        if (!metadataFile)
        {
            std::cerr << "Failed to finish writing metadata: "
                      << metadataPath << '\n';
            return 1;
        }

        std::cout
            << "Generated MIS instance\n"
            << "  vertices: " << generated.graph.n << '\n'
            << "  edges: " << generated.graph.edges.size() << '\n'
            << "  seed: " << metadata.seed << '\n'
            << "  graph: " << outputPath << '\n'
            << "  metadata: " << metadataPath << '\n';

        return 0;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "Generation failed: "
                  << exception.what() << '\n';
        return 1;
    }
}
}

int runMISGeneratorBatch(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printUsage(argv[0]);
        return 2;
    }

    const std::string command = argv[1];

    if (command == "single")
    {
        return runSingle(argc, argv);
    }

    if (command == "batch")
    {
        return runMISGeneratorBatch(argc, argv);
    }

    printUsage(argv[0]);
    return 2;
}
