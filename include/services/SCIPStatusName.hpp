#pragma once

#include <string>

#include <scip/scip.h>

// Stable upper-case name of a SCIP solving status (e.g. "OPTIMAL",
// "TIMELIMIT"), or "UNKNOWN" for a status without a dedicated name.
std::string getSCIPStatusName(SCIP_STATUS status);
