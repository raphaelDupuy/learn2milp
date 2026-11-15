#include "generators/MISGenerator.hpp"

#include <algorithm>
#include <charconv>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace
{

// Helper to visit a variant with one lambda per alternative.
template <typename... Functions>
struct Overloaded : Functions...
{
    using Functions::operator()...;
};

template <typename... Functions>
Overloaded(Functions...) -> Overloaded<Functions...>;

// Shortest text that reads back to the same double (0.1 -> "0_1"), made
// safe for a file name.
std::string probabilityToken(double p)
{
    char buffer[64];

    const auto conversion = std::to_chars(
        buffer,
        buffer + sizeof(buffer),
        p,
        std::chars_format::general
    );

    if (conversion.ec != std::errc{})
    {
        throw std::runtime_error("Failed to format probability value.");
    }

    std::string token(buffer, conversion.ptr);

    std::replace(token.begin(), token.end(), '.', '_');
    std::replace(token.begin(), token.end(), '-', 'm');
    std::replace(token.begin(), token.end(), '+', 'p');

    return token;
}

// Exact text of a probability, as in the instance metadata.
std::string probabilityText(double p)
{
    std::ostringstream stream;
    stream << std::setprecision(
        std::numeric_limits<double>::max_digits10
    ) << p;

    return stream.str();
}

} // namespace

const std::vector<MISGraphFamily>& misGraphFamilies()
{
    static const std::vector<MISGraphFamily> families{
        MISGraphFamily::ErdosRenyi,
        MISGraphFamily::BarabasiAlbert
    };

    return families;
}

const char* misGraphFamilyName(MISGraphFamily family)
{
    switch (family)
    {
        case MISGraphFamily::ErdosRenyi:
            return "erdos_renyi";

        case MISGraphFamily::BarabasiAlbert:
            return "barabasi_albert";
    }

    return "unknown";
}

bool parseMISGraphFamily(const std::string& name, MISGraphFamily& family)
{
    for (const MISGraphFamily candidate : misGraphFamilies())
    {
        if (name == misGraphFamilyName(candidate))
        {
            family = candidate;
            return true;
        }
    }

    return false;
}

MISGraphFamily misGraphFamily(const MISGeneratorConfig& config)
{
    return std::visit(
        Overloaded{
            [](const ErdosRenyiConfig&) {
                return MISGraphFamily::ErdosRenyi;
            },
            [](const BarabasiAlbertConfig&) {
                return MISGraphFamily::BarabasiAlbert;
            }},
        config
    );
}

unsigned int misGeneratorSeed(const MISGeneratorConfig& config)
{
    return std::visit(
        [](const auto& parameters) { return parameters.seed; },
        config
    );
}

void validateMISGeneratorConfig(const MISGeneratorConfig& config)
{
    std::visit(
        Overloaded{
            [](const ErdosRenyiConfig& parameters) {
                ErdosRenyiGenerator::validate(parameters);
            },
            [](const BarabasiAlbertConfig& parameters) {
                BarabasiAlbertGenerator::validate(parameters);
            }},
        config
    );
}

GeneratedMISInstance generateMISInstance(const MISGeneratorConfig& config)
{
    return std::visit(
        Overloaded{
            [](const ErdosRenyiConfig& parameters) {
                return ErdosRenyiGenerator{}.generate(parameters);
            },
            [](const BarabasiAlbertConfig& parameters) {
                return BarabasiAlbertGenerator{}.generate(parameters);
            }},
        config
    );
}

std::string misInstanceId(const MISGeneratorConfig& config)
{
    const std::string family = misGraphFamilyName(misGraphFamily(config));

    const std::string parameters = std::visit(
        Overloaded{
            [](const ErdosRenyiConfig& parameters) {
                return "_n" + std::to_string(parameters.n)
                    + "_p" + probabilityToken(parameters.p);
            },
            [](const BarabasiAlbertConfig& parameters) {
                return "_n" + std::to_string(parameters.n)
                    + "_k" + std::to_string(parameters.k);
            }},
        config
    );

    return family + parameters
        + "_seed" + std::to_string(misGeneratorSeed(config));
}

std::vector<std::string> misManifestParameterNames(MISGraphFamily family)
{
    switch (family)
    {
        case MISGraphFamily::ErdosRenyi:
            return {"n", "p"};

        case MISGraphFamily::BarabasiAlbert:
            return {"n", "k"};
    }

    return {};
}

std::vector<std::string> misManifestParameterValues(
    const MISGeneratorConfig& config
)
{
    return std::visit(
        Overloaded{
            [](const ErdosRenyiConfig& parameters) {
                return std::vector<std::string>{
                    std::to_string(parameters.n),
                    probabilityText(parameters.p)};
            },
            [](const BarabasiAlbertConfig& parameters) {
                return std::vector<std::string>{
                    std::to_string(parameters.n),
                    std::to_string(parameters.k)};
            }},
        config
    );
}
