#include "io/MISStrongBranchingSampleIO.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <set>
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

MISStrongBranchingDirection makeDirection(
    double bound,
    double gain,
    bool infeasible
)
{
    MISStrongBranchingDirection direction{};

    direction.bound = bound;
    direction.gain = gain;
    direction.bound_valid = 1;
    direction.gain_valid = infeasible ? 0 : 1;
    direction.infeasible = infeasible ? 1 : 0;
    direction.conflict = 0;
    direction.skipped = 0;

    return direction;
}

MISStrongBranchingCandidate makeCandidate(int position, int vertex)
{
    MISStrongBranchingCandidate candidate{};

    candidate.vertex_id = vertex;
    candidate.candidate_position = position;
    candidate.lp_value = 0.5;
    candidate.fractionality = 0.5;
    candidate.objective_coefficient = -1.0;
    candidate.local_lower_bound = 0.0;
    candidate.local_upper_bound = 1.0;
    candidate.local_bounds_valid = 1;
    candidate.reduced_cost = 0.0;
    candidate.lp_value_valid = 1;
    candidate.reduced_cost_valid = 1;
    candidate.evaluation_status = MIS_SB_EVALUATED;
    candidate.down = makeDirection(-1.5, 0.0, false);
    candidate.up = makeDirection(-1.0, 0.5, false);
    candidate.score = 0.1;
    candidate.score_valid = 1;
    candidate.expert_rank = 0;
    candidate.expert_selected = 0;
    candidate.evaluation_order = position;
    candidate.gain_reference_lp_objective = -1.5;
    candidate.gain_reference_lp_objective_valid = 1;

    return candidate;
}

struct Fixture
{
    std::vector<MISVertexState> states{
        MIS_VERTEX_FRACTIONAL,
        MIS_VERTEX_FRACTIONAL,
        MIS_VERTEX_FIXED_ZERO,
        MIS_VERTEX_FIXED_ONE,
        MIS_VERTEX_FREE_LP_ZERO,
        MIS_VERTEX_FREE_LP_ONE,
        MIS_VERTEX_UNKNOWN
    };

    std::vector<MISStrongBranchingCandidate> candidates{
        makeCandidate(0, 0),
        makeCandidate(1, 1)
    };

    MISStrongBranchingSample sample{};

    Fixture()
    {
        candidates[0].expert_selected = 1;

        candidates[1].lp_value = 1.0 / 3.0;
        candidates[1].fractionality = 1.0 / 3.0;
        candidates[1].up = makeDirection(1e20, 1e20, true);
        candidates[1].expert_rank = 1;
        candidates[1].score = 1e-6;

        sample.node_id = 7;
        sample.parent_node_id = 3;
        sample.parent_node_valid = 1;
        sample.node_depth = 2;
        sample.lp_objective = -1.5;
        sample.primal_bound = -1.0;
        sample.dual_bound = -1.75;
        sample.lp_objective_valid = 1;
        sample.primal_bound_valid = 1;
        sample.dual_bound_valid = 1;
        sample.open_nodes = 4;
        sample.processed_nodes = 9;
        sample.number_of_candidates = 2;
        sample.number_of_priority_candidates = 2;
        sample.expert_selected_vertex_id = 0;
        sample.expert_selection_valid = 1;
        sample.node_cutoff = 1;
        sample.vertex_states = states.data();
        sample.number_of_vertices = static_cast<int>(states.size());
    }
};

// A writer with its two in-memory tables.
struct Tables
{
    std::ostringstream nodes;
    std::ostringstream candidates;
    MISStrongBranchingSampleWriter writer{nodes, candidates};
    std::string error;
};

void testHeaders()
{
    Tables tables;

    check(
        !tables.writer.writeSample(
            Fixture{}.sample, nullptr, 0, tables.error),
        "Sample before headers is rejected"
    );

    check(tables.writer.writeHeaders(tables.error), "Headers are written");

    check(
        tables.nodes.str()
            == std::string(MIS_SB_NODE_CSV_HEADER) + "\n"
        && tables.candidates.str()
            == std::string(MIS_SB_CANDIDATE_CSV_HEADER) + "\n",
        "Header rows match the published constants"
    );

    check(
        !tables.writer.writeHeaders(tables.error),
        "Headers cannot be written twice"
    );
}

void testExactRows()
{
    Tables tables;
    Fixture fixture;

    tables.writer.writeHeaders(tables.error);

    const bool written = tables.writer.writeSample(
        fixture.sample,
        fixture.candidates.data(),
        fixture.candidates.size(),
        tables.error
    );

    check(written, "Valid sample is written");

    const std::string expectedNodes =
        std::string(MIS_SB_NODE_CSV_HEADER) + "\n"
        "0,7,3,1,2,-1.5,1,-1,1,-1.75,1,4,9,2,2,0,1,1,7,ffZOzou\n";

    check(tables.nodes.str() == expectedNodes, "Node row is exact");

    const std::string expectedCandidates =
        std::string(MIS_SB_CANDIDATE_CSV_HEADER) + "\n"
        "0,7,0,0,0.5,1,0.5,-1,0,1,1,0,1,evaluated,0,"
        "-1.5,0,1,1,0,0,0,-1,0.5,1,1,0,0,0,"
        "0.1,1,0,1,-1.5,1\n"
        "0,7,1,1,0.3333333333333333,1,0.3333333333333333,-1,0,1,1,0,1,"
        "evaluated,1,"
        "-1.5,0,1,1,0,0,0,1e+20,1e+20,1,0,1,0,0,"
        "1e-06,1,1,0,-1.5,1\n";

    check(
        tables.candidates.str() == expectedCandidates,
        "Candidate rows are exact"
    );

    if (tables.candidates.str() != expectedCandidates)
    {
        std::cerr << "--- got ---\n" << tables.candidates.str()
                  << "--- expected ---\n" << expectedCandidates;
    }

    check(
        tables.writer.numberOfSamples() == 1
        && tables.writer.numberOfCandidates() == 2,
        "Counters reflect the written sample"
    );

    // The second sample gets the next sample index.
    tables.writer.writeSample(
        fixture.sample,
        fixture.candidates.data(),
        fixture.candidates.size(),
        tables.error
    );

    check(
        tables.nodes.str().find("\n1,7,3,") != std::string::npos
        && tables.candidates.str().find("\n1,7,0,0,") != std::string::npos
        && tables.writer.numberOfSamples() == 2,
        "Sample index increases with each sample"
    );
}

void testColumnCounts()
{
    Tables tables;
    Fixture fixture;

    tables.writer.writeHeaders(tables.error);
    tables.writer.writeSample(
        fixture.sample,
        fixture.candidates.data(),
        fixture.candidates.size(),
        tables.error
    );

    const auto commasPerLine = [](const std::string& text)
    {
        std::set<std::size_t> counts;
        std::istringstream stream(text);
        std::string line;

        while (std::getline(stream, line))
        {
            std::size_t commas = 0;

            for (const char c : line)
            {
                if (c == ',')
                    ++commas;
            }

            counts.insert(commas);
        }

        return counts;
    };

    check(
        commasPerLine(tables.nodes.str()).size() == 1
        && commasPerLine(tables.candidates.str()).size() == 1,
        "Every row has as many fields as its header"
    );
}

// Runs one rejection case: the sample must be refused, and both tables
// and counters must be left untouched.
template <typename Mutation>
void testRejection(const std::string& testName, Mutation mutate)
{
    Tables tables;
    Fixture fixture;

    tables.writer.writeHeaders(tables.error);

    const std::string nodesBefore = tables.nodes.str();
    const std::string candidatesBefore = tables.candidates.str();

    std::size_t numberOfCandidates = fixture.candidates.size();
    mutate(fixture, numberOfCandidates);

    const bool written = tables.writer.writeSample(
        fixture.sample,
        fixture.candidates.data(),
        numberOfCandidates,
        tables.error
    );

    check(
        !written
        && !tables.error.empty()
        && tables.nodes.str() == nodesBefore
        && tables.candidates.str() == candidatesBefore
        && tables.writer.numberOfSamples() == 0
        && tables.writer.numberOfCandidates() == 0,
        testName
    );
}

void testRejections()
{
    testRejection(
        "Candidate count mismatch is rejected",
        [](Fixture&, std::size_t& count) { count = 1; }
    );

    testRejection(
        "Missing vertex states are rejected",
        [](Fixture& fixture, std::size_t&)
        {
            fixture.sample.vertex_states = nullptr;
        }
    );

    testRejection(
        "Unknown vertex state is rejected",
        [](Fixture& fixture, std::size_t&)
        {
            fixture.states[2] = static_cast<MISVertexState>(42);
        }
    );

    testRejection(
        "Candidate vertex outside the range is rejected",
        [](Fixture& fixture, std::size_t&)
        {
            fixture.candidates[1].vertex_id = 7;
        }
    );

    testRejection(
        "Unknown evaluation status is rejected",
        [](Fixture& fixture, std::size_t&)
        {
            fixture.candidates[0].evaluation_status =
                static_cast<MISStrongBranchingEvaluationStatus>(42);
        }
    );

    testRejection(
        "NaN candidate value is rejected",
        [](Fixture& fixture, std::size_t&)
        {
            // The second candidate: the first one must not be written.
            fixture.candidates[1].score =
                std::numeric_limits<double>::quiet_NaN();
        }
    );

    testRejection(
        "Infinite direction bound is rejected",
        [](Fixture& fixture, std::size_t&)
        {
            fixture.candidates[0].down.bound =
                std::numeric_limits<double>::infinity();
        }
    );

    testRejection(
        "NaN node objective is rejected",
        [](Fixture& fixture, std::size_t&)
        {
            fixture.sample.lp_objective =
                std::numeric_limits<double>::quiet_NaN();
        }
    );
}

void testNegativeZero()
{
    Tables tables;
    Fixture fixture;

    fixture.candidates.resize(1);
    fixture.states[1] = MIS_VERTEX_FREE_LP_ZERO;
    fixture.sample.number_of_candidates = 1;
    fixture.candidates[0].local_lower_bound = -0.0;
    fixture.candidates[0].reduced_cost = -0.0;

    tables.writer.writeHeaders(tables.error);
    tables.writer.writeSample(
        fixture.sample,
        fixture.candidates.data(),
        fixture.candidates.size(),
        tables.error
    );

    check(
        tables.candidates.str().find("-0,") == std::string::npos
        && tables.candidates.str().find(",-1,0,1,1,0,1,evaluated,")
            != std::string::npos,
        "Negative zero is written as 0"
    );
}

void testCodes()
{
    const MISVertexState states[] = {
        MIS_VERTEX_FIXED_ZERO,
        MIS_VERTEX_FIXED_ONE,
        MIS_VERTEX_FREE_LP_ZERO,
        MIS_VERTEX_FREE_LP_ONE,
        MIS_VERTEX_FRACTIONAL,
        MIS_VERTEX_UNKNOWN
    };

    std::set<char> codes;
    bool lettersOnly = true;

    for (const MISVertexState state : states)
    {
        const char code = misVertexStateCode(state);
        codes.insert(code);

        // Letters only: a state string must never look like a number.
        lettersOnly = lettersOnly
            && ((code >= 'a' && code <= 'z')
                || (code >= 'A' && code <= 'Z'));
    }

    check(
        codes.size() == 6 && lettersOnly,
        "Vertex state codes are distinct letters"
    );

    check(
        misVertexStateCode(static_cast<MISVertexState>(6)) == '\0',
        "Unknown vertex state has no code"
    );

    const MISStrongBranchingEvaluationStatus statuses[] = {
        MIS_SB_NOT_EVALUATED,
        MIS_SB_EVALUATED,
        MIS_SB_REUSED,
        MIS_SB_LP_ERROR,
        MIS_SB_PARTIALLY_EVALUATED
    };

    std::set<std::string> names;

    for (const auto status : statuses)
    {
        const char* name = misStrongBranchingEvaluationStatusName(status);

        if (name != nullptr)
            names.insert(name);
    }

    check(names.size() == 5, "Evaluation status names are distinct");
}

void testEmptySample()
{
    // A sample without candidates is unusual but well-formed.
    Tables tables;
    Fixture fixture;

    fixture.sample.number_of_candidates = 0;
    fixture.sample.expert_selection_valid = 0;
    fixture.sample.expert_selected_vertex_id = -1;

    tables.writer.writeHeaders(tables.error);

    const std::string candidatesBefore = tables.candidates.str();

    check(
        tables.writer.writeSample(
            fixture.sample, nullptr, 0, tables.error)
        && tables.candidates.str() == candidatesBefore
        && tables.writer.numberOfSamples() == 1,
        "Sample without candidates writes only a node row"
    );
}

} // namespace

int main()
{
    testHeaders();
    testExactRows();
    testColumnCounts();
    testRejections();
    testNegativeZero();
    testCodes();
    testEmptySample();

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All strong branching sample IO tests passed.\n";
    return 0;
}
