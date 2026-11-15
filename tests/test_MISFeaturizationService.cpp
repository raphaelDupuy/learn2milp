/*
 * Tests of the offline featurization.
 *
 * The key check is the parity between the two ways features are obtained:
 *
 * - online: computed inside the solve, from the sample handed to the
 *   branching callback (what a learned rule sees at inference time);
 * - offline: recomputed by the featurizer from the collected CSV tables
 *   (what the model is trained on).
 *
 * Both must be identical, bit for bit.
 */

#include "services/MISFeaturizationService.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <scip/scip.h>
#include <scip/scipdefplugins.h>

#include "branching/MISFullstrongBranchrule.h"
#include "generators/ErdosRenyiGenerator.hpp"
#include "io/CsvReader.hpp"
#include "io/MISStrongBranchingSampleIO.hpp"
#include "learning/MISAdjacency.hpp"
#include "learning/MISBranchingFeatures.h"
#include "problems/MIS.hpp"
#include "services/MISSolverSettings.hpp"

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

// What the callback saw for one candidate.
struct OnlineCandidate
{
    long long sampleIndex;
    int position;
    int vertex;
    int nodeCutoff;
    double score;
    int expertRank;
    int expertSelected;
    std::vector<double> features;
};

struct OnlineRecorder
{
    MISFeatureGraph graph;
    MISStrongBranchingSampleWriter* writer = nullptr;
    std::vector<OnlineCandidate> candidates;
    long long samples = 0;
    bool failed = false;
};

void recordOnline(
    void* userdata,
    const MISStrongBranchingSample* sample,
    const MISStrongBranchingCandidate* candidates,
    size_t numberOfCandidates)
{
    auto& recorder = *static_cast<OnlineRecorder*>(userdata);
    std::string error;

    // The collected tables, as the data collection writes them.
    if (!recorder.writer->writeSample(
            *sample, candidates, numberOfCandidates, error))
    {
        recorder.failed = true;
        return;
    }

    // The features, as a learned rule computes them at this node.
    std::vector<int> vertices;
    std::vector<double> lpValues;
    std::vector<MISCandidateHistory> histories;

    for (size_t c = 0; c < numberOfCandidates; ++c)
    {
        vertices.push_back(candidates[c].vertex_id);
        lpValues.push_back(candidates[c].lp_value);
        histories.push_back(candidates[c].history);
    }

    MISBranchingNode node;
    node.vertex_states = sample->vertex_states;
    node.node_depth = sample->node_depth;
    node.number_of_candidates = static_cast<int>(numberOfCandidates);
    node.candidate_vertices = vertices.data();
    node.candidate_lp_values = lpValues.data();
    node.candidate_histories = histories.data();
    node.objective_gap = MISbranchingObjectiveGap(
        sample->lp_objective,
        sample->primal_bound,
        sample->primal_bound_valid);

    std::vector<double> work(
        MISbranchingFeatureWorkSize(sample->number_of_vertices));
    std::vector<double> features(
        numberOfCandidates * MIS_BRANCHING_FEATURE_COUNT);

    if (MIScomputeBranchingFeatures(
            &recorder.graph, &node, work.data(), features.data()) != 0)
    {
        recorder.failed = true;
        return;
    }

    for (size_t c = 0; c < numberOfCandidates; ++c)
    {
        recorder.candidates.push_back({
            recorder.samples,
            static_cast<int>(c),
            candidates[c].vertex_id,
            sample->node_cutoff,
            candidates[c].score,
            candidates[c].expert_rank,
            candidates[c].expert_selected,
            std::vector<double>(
                features.begin() + c * MIS_BRANCHING_FEATURE_COUNT,
                features.begin() + (c + 1) * MIS_BRANCHING_FEATURE_COUNT)});
    }

    ++recorder.samples;
}

// Solves the instance with the expert, writing the sample tables and
// recording the online features.
bool collectOnline(
    const MISInstance& instance,
    const MISAdjacency& adjacency,
    const std::filesystem::path& nodesPath,
    const std::filesystem::path& candidatesPath,
    OnlineRecorder& recorder)
{
    std::ofstream nodesFile(nodesPath);
    std::ofstream candidatesFile(candidatesPath);
    MISStrongBranchingSampleWriter writer(nodesFile, candidatesFile);
    std::string error;

    recorder.graph = adjacency.view();
    recorder.writer = &writer;

    SCIP* scip = nullptr;
    std::vector<SCIP_VAR*> vars;

    const bool solved =
        writer.writeHeaders(error)
        && SCIPcreate(&scip) == SCIP_OKAY
        && SCIPincludeDefaultPlugins(scip) == SCIP_OKAY
        && buildMISModel(scip, instance, vars) == SCIP_OKAY
        && SCIPsetIntParam(scip, "display/verblevel", 0) == SCIP_OKAY
        && applyBranchingStudySettings(scip) == SCIP_OKAY
        && SCIPincludeBranchruleMISFullstrong(
            scip, vars.data(), static_cast<int>(vars.size()),
            recordOnline, &recorder) == SCIP_OKAY
        && SCIPsolve(scip) == SCIP_OKAY
        && SCIPgetStatus(scip) == SCIP_STATUS_OPTIMAL;

    if (scip != nullptr)
        SCIPfree(&scip);

    recorder.writer = nullptr;

    return solved && !recorder.failed && nodesFile && candidatesFile;
}

std::string readFile(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    std::ostringstream content;
    content << file.rdbuf();
    return content.str();
}

void writeFile(const std::filesystem::path& path, const std::string& text)
{
    std::ofstream file(path, std::ios::binary);
    file << text;
}

// Compares the feature table with what the callback recorded. Returns a
// description of the first difference, or an empty string.
std::string compareWithOnline(
    const std::filesystem::path& featuresPath,
    const OnlineRecorder& recorder)
{
    std::ifstream file(featuresPath);
    CsvReader reader(file);
    std::string error;

    if (!reader.readHeader(error))
        return error;

    std::ostringstream header;

    for (std::size_t i = 0; i < reader.columns().size(); ++i)
        header << (i > 0 ? "," : "") << reader.columns()[i];

    if (header.str() != misFeatureTableHeader())
        return "unexpected header";

    const int firstFeature = reader.column(MISbranchingFeatureName(0));

    if (firstFeature != 9)
        return "features do not start after the nine label columns";

    std::vector<std::string> fields;
    std::size_t row = 0;

    while (reader.readRow(fields, error))
    {
        if (row >= recorder.candidates.size())
            return "more rows than online candidates";

        const OnlineCandidate& online = recorder.candidates[row];
        const std::string where = "row " + std::to_string(row) + ": ";

        long long integer = 0;
        double real = 0.0;

        const auto integerAt = [&](const char* name)
        {
            parseCsvInteger(fields[reader.column(name)], integer);
            return integer;
        };

        if (integerAt("sample_index") != online.sampleIndex
            || integerAt("candidate_position") != online.position
            || integerAt("vertex_id") != online.vertex
            || integerAt("node_cutoff") != online.nodeCutoff
            || integerAt("expert_selection_valid") != 1
            || integerAt("evaluated") != 1
            || integerAt("expert_rank") != online.expertRank
            || integerAt("expert_selected") != online.expertSelected)
            return where + "identification or expert answer differs";

        if (!parseCsvReal(fields[reader.column("score")], real)
            || real != online.score)
            return where + "score differs";

        for (int f = 0; f < MIS_BRANCHING_FEATURE_COUNT; ++f)
        {
            // Exact comparison: the table must read back to the very
            // doubles computed online.
            if (!parseCsvReal(fields[firstFeature + f], real)
                || real != online.features[f])
                return where + MISbranchingFeatureName(f)
                    + " differs: offline " + fields[firstFeature + f]
                    + ", online " + std::to_string(online.features[f]);
        }

        ++row;
    }

    if (!reader.atEnd())
        return error;

    if (row != recorder.candidates.size())
        return "fewer rows than online candidates";

    return "";
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " WORK_DIR\n";
        return 2;
    }

    const std::filesystem::path workDir(argv[1]);
    std::filesystem::remove_all(workDir);
    std::filesystem::create_directories(workDir);

    const MISInstance instance =
        ErdosRenyiGenerator{}.generate({100, 0.10, 1}).graph;
    const MISAdjacency adjacency = buildMISAdjacency(instance);

    const auto nodesPath = workDir / "run.sb.nodes.csv";
    const auto candidatesPath = workDir / "run.sb.candidates.csv";
    const auto featuresPath = workDir / "run.sb.features.csv";

    OnlineRecorder recorder;

    const bool collected = collectOnline(
        instance, adjacency, nodesPath, candidatesPath, recorder);

    check(
        collected && recorder.samples > 1,
        "Samples are collected with their online features ("
            + std::to_string(recorder.candidates.size()) + " candidates)"
    );

    MISFeaturizationResult result;
    std::string error;

    const bool featurized = featurizeMISStrongBranchingSamples(
        instance, nodesPath, candidatesPath, featuresPath, result, error);

    check(featurized, "Feature table is written: " + error);

    std::size_t usable = 0;
    long long lastSample = -1;

    for (const OnlineCandidate& candidate : recorder.candidates)
    {
        if (candidate.sampleIndex != lastSample && candidate.nodeCutoff == 0)
            ++usable;

        lastSample = candidate.sampleIndex;
    }

    check(
        featurized
        && result.numberOfSamples
            == static_cast<std::size_t>(recorder.samples)
        && result.numberOfCandidates == recorder.candidates.size()
        && result.numberOfUsableSamples == usable
        && usable > 0 && usable < result.numberOfSamples,
        "Counts match the collection ("
            + std::to_string(result.numberOfUsableSamples) + " usable of "
            + std::to_string(result.numberOfSamples) + " samples)"
    );

    const std::string difference =
        compareWithOnline(featuresPath, recorder);

    check(
        difference.empty(),
        "Offline features are identical to online features"
            + (difference.empty() ? "" : ": " + difference)
    );

    check(
        !std::filesystem::exists(featuresPath.string() + ".tmp"),
        "No temporary file is left behind"
    );

    // Reproducible, and never overwritten.
    {
        const auto again = workDir / "again.sb.features.csv";
        MISFeaturizationResult other;

        check(
            featurizeMISStrongBranchingSamples(
                instance, nodesPath, candidatesPath, again, other, error)
            && readFile(again) == readFile(featuresPath),
            "Featurization is reproducible"
        );

        const std::string before = readFile(featuresPath);

        check(
            !featurizeMISStrongBranchingSamples(
                instance, nodesPath, candidatesPath, featuresPath, other,
                error)
            && readFile(featuresPath) == before,
            "Existing feature table is not overwritten"
        );
    }

    // Inconsistent inputs are rejected and leave no file behind.
    {
        const auto rejected = workDir / "rejected.sb.features.csv";
        MISFeaturizationResult other;

        const auto expectRejection = [&](
            const std::string& testName,
            const MISInstance& usedInstance,
            const std::filesystem::path& usedNodes,
            const std::filesystem::path& usedCandidates)
        {
            const bool ok = featurizeMISStrongBranchingSamples(
                usedInstance, usedNodes, usedCandidates, rejected, other,
                error);

            check(
                !ok && !error.empty()
                && !std::filesystem::exists(rejected)
                && !std::filesystem::exists(rejected.string() + ".tmp"),
                testName
            );
        };

        expectRejection(
            "Samples of another instance are rejected",
            ErdosRenyiGenerator{}.generate({90, 0.10, 1}).graph,
            nodesPath, candidatesPath);

        const std::string candidatesText = readFile(candidatesPath);
        const std::string nodesText = readFile(nodesPath);

        // Drop the last candidate row.
        const auto truncated = workDir / "truncated.csv";
        writeFile(
            truncated,
            candidatesText.substr(
                0, candidatesText.rfind('\n', candidatesText.size() - 2) + 1));

        expectRejection(
            "Missing candidate rows are rejected",
            instance, nodesPath, truncated);

        // Add a candidate row that belongs to no sample.
        const auto extra = workDir / "extra.csv";
        const std::size_t lastLine =
            candidatesText.rfind('\n', candidatesText.size() - 2) + 1;
        writeFile(extra, candidatesText + candidatesText.substr(lastLine));

        expectRejection(
            "Candidate rows without sample are rejected",
            instance, nodesPath, extra);

        // Swap the two tables: required columns are missing.
        expectRejection(
            "Table without the expected columns is rejected",
            instance, candidatesPath, nodesPath);

        // Mark a fractional vertex as fixed in the first sample.
        const auto corrupted = workDir / "corrupted.csv";
        std::string corruptedText = nodesText;
        const std::size_t firstRow = corruptedText.find('\n') + 1;
        const std::size_t firstState =
            corruptedText.rfind(',', corruptedText.find('\n', firstRow)) + 1;
        const std::size_t fractional =
            corruptedText.find('f', firstState);
        corruptedText[fractional] = 'Z';
        writeFile(corrupted, corruptedText);

        expectRejection(
            "States contradicting the candidates are rejected",
            instance, corrupted, candidatesPath);

        expectRejection(
            "Missing sample table is rejected",
            instance, workDir / "missing.csv", candidatesPath);

        check(
            !featurizeMISStrongBranchingSamples(
                instance, nodesPath, candidatesPath, "", other, error),
            "Empty output path is rejected"
        );
    }

    if (failures > 0)
    {
        std::cerr << failures << " test(s) failed.\n";
        return 1;
    }

    std::cout << "All MIS featurization service tests passed.\n";
    return 0;
}
