#pragma once

#include <string>
#include <variant>
#include <vector>

#include "generators/BarabasiAlbertGenerator.hpp"
#include "generators/ErdosRenyiGenerator.hpp"
#include "generators/GeneratedMISInstance.hpp"

/*
 * The graph families of the MIS instances, behind one interface: what a
 * family is called, how its instances are named and listed, and how they
 * are generated. Adding a family means adding it here; the generation
 * service and the command line tool only go through these functions.
 */

enum class MISGraphFamily
{
    ErdosRenyi,
    BarabasiAlbert
};

// Parameters of one instance; the alternative held gives the family.
using MISGeneratorConfig = std::variant<ErdosRenyiConfig, BarabasiAlbertConfig>;

// Every family, in a fixed order.
const std::vector<MISGraphFamily>& misGraphFamilies();

// Name of a family: "erdos_renyi" or "barabasi_albert", which is also the
// generator_type of its instances.
const char* misGraphFamilyName(MISGraphFamily family);

bool parseMISGraphFamily(const std::string& name, MISGraphFamily& family);

MISGraphFamily misGraphFamily(const MISGeneratorConfig& config);

unsigned int misGeneratorSeed(const MISGeneratorConfig& config);

// Throws std::invalid_argument if the parameters are not valid for the
// family (see the generator of the family).
void validateMISGeneratorConfig(const MISGeneratorConfig& config);

// Throws std::invalid_argument for invalid parameters.
GeneratedMISInstance generateMISInstance(const MISGeneratorConfig& config);

/*
 * Identifier of an instance, used as its file name without the .mis
 * extension. It names the family and every parameter:
 *
 *   erdos_renyi_n125_p0_1_seed7      (p written with the shortest text
 *                                     that reads back to the same value,
 *                                     '.' replaced by '_')
 *   barabasi_albert_n200_k8_seed7
 */
std::string misInstanceId(const MISGeneratorConfig& config);

/*
 * Columns of a batch manifest that hold the parameters of a family, and
 * their values for one instance (in the same order): n and p for
 * Erdos-Renyi, n and k for Barabasi-Albert.
 */
std::vector<std::string> misManifestParameterNames(MISGraphFamily family);
std::vector<std::string> misManifestParameterValues(
    const MISGeneratorConfig& config
);
