#include "services/SCIPStatusName.hpp"

std::string getSCIPStatusName(SCIP_STATUS status)
{
    switch (status)
    {
        case SCIP_STATUS_OPTIMAL:
            return "OPTIMAL";

        case SCIP_STATUS_INFEASIBLE:
            return "INFEASIBLE";

        case SCIP_STATUS_UNBOUNDED:
            return "UNBOUNDED";

        case SCIP_STATUS_INFORUNBD:
            return "INFORUNBD";

        case SCIP_STATUS_USERINTERRUPT:
            return "USERINTERRUPT";

        case SCIP_STATUS_TERMINATE:
            return "TERMINATE";

        case SCIP_STATUS_NODELIMIT:
            return "NODELIMIT";

        case SCIP_STATUS_TOTALNODELIMIT:
            return "TOTALNODELIMIT";

        case SCIP_STATUS_STALLNODELIMIT:
            return "STALLNODELIMIT";

        case SCIP_STATUS_TIMELIMIT:
            return "TIMELIMIT";

        case SCIP_STATUS_MEMLIMIT:
            return "MEMLIMIT";

        case SCIP_STATUS_GAPLIMIT:
            return "GAPLIMIT";

        case SCIP_STATUS_PRIMALLIMIT:
            return "PRIMALLIMIT";

        case SCIP_STATUS_DUALLIMIT:
            return "DUALLIMIT";

        case SCIP_STATUS_SOLLIMIT:
            return "SOLLIMIT";

        case SCIP_STATUS_BESTSOLLIMIT:
            return "BESTSOLLIMIT";

        case SCIP_STATUS_RESTARTLIMIT:
            return "RESTARTLIMIT";

        default:
            return "UNKNOWN";
    }
}
