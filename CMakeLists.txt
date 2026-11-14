cmake_minimum_required(VERSION 3.20)

project(learn2milp
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Branching features and model evaluation: plain C without SCIP, shared
# by the offline featurizer and by the learned branching rule.
add_library(mis_learning STATIC
    src/learning/MISBranchingFeatures.c
    src/learning/MISBranchingModel.c
)

target_include_directories(mis_learning
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

add_library(mis_branching STATIC
    src/branching/MISFullstrongBranchrule.c
    src/branching/MISLearnedBranchrule.c
    src/branching/MISVertexMapping.c
)

target_include_directories(mis_branching
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mis_branching
    PUBLIC
    mis_learning
    scip
)

add_executable(learn2milp
    src/main.cpp
    src/problems/MIS.cpp
    src/generators/ErdosRenyiGenerator.cpp
    src/io/MISInstanceIO.cpp
    src/services/MISGenerationService.cpp
    src/services/MISInstanceLoadingService.cpp
    src/services/MISSolvingService.cpp
    src/services/SCIPStatusName.cpp
    src/services/MISFeatureExtractor.cpp
    src/ui/TerminalUI.cpp
    src/io/InstanceMetadataIO.cpp
)

target_include_directories(learn2milp
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(learn2milp
    PRIVATE
    scip
)

add_executable(mis-generator
    src/tools/mis_generator_main.cpp
    src/tools/mis_generator_batch.cpp
    src/generators/ErdosRenyiGenerator.cpp
    src/io/MISInstanceIO.cpp
    src/io/InstanceMetadataIO.cpp
)

target_include_directories(mis-generator
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

add_executable(mis-collector
    src/tools/mis_collector_main.cpp
    src/problems/MIS.cpp
    src/io/MISInstanceIO.cpp
    src/io/InstanceMetadataIO.cpp
    src/io/MISStrongBranchingSampleIO.cpp
    src/services/MISInstanceLoadingService.cpp
    src/services/MISCollectionService.cpp
    src/services/MISSolverSettings.cpp
    src/services/SCIPStatusName.cpp
)

target_link_libraries(mis-collector
    PRIVATE
    mis_branching
)

add_executable(mis-benchmark
    src/tools/mis_benchmark_main.cpp
    src/problems/MIS.cpp
    src/io/MISInstanceIO.cpp
    src/io/InstanceMetadataIO.cpp
    src/services/MISInstanceLoadingService.cpp
    src/io/MISBranchingModelIO.cpp
    src/learning/MISAdjacency.cpp
    src/services/MISBenchmarkService.cpp
    src/services/MISSolverSettings.cpp
    src/services/SCIPStatusName.cpp
)

target_link_libraries(mis-benchmark
    PRIVATE
    mis_branching
)

add_executable(mis-featurizer
    src/tools/mis_featurizer_main.cpp
    src/io/CsvReader.cpp
    src/io/MISInstanceIO.cpp
    src/io/InstanceMetadataIO.cpp
    src/io/MISStrongBranchingSampleIO.cpp
    src/learning/MISAdjacency.cpp
    src/services/MISInstanceLoadingService.cpp
    src/services/MISFeaturizationService.cpp
)

target_link_libraries(mis-featurizer
    PRIVATE
    mis_learning
    scip
)

include(CTest)

if(BUILD_TESTING)
    add_executable(test_MISFullstrongBranchrule
        tests/test_MISFullstrongBranchrule.cpp
    )

    target_link_libraries(test_MISFullstrongBranchrule
        PRIVATE
        mis_branching
    )

    add_test(
        NAME MISFullstrongBranchrule
        COMMAND test_MISFullstrongBranchrule
    )

    add_executable(test_MISFullstrongOracle
        tests/test_MISFullstrongOracle.cpp
        src/problems/MIS.cpp
        src/generators/ErdosRenyiGenerator.cpp
    )

    target_link_libraries(test_MISFullstrongOracle
        PRIVATE
        mis_branching
    )

    add_test(
        NAME MISFullstrongOracle
        COMMAND test_MISFullstrongOracle
    )

    add_executable(test_MISBranchingFeatures
        tests/test_MISBranchingFeatures.cpp
        src/learning/MISAdjacency.cpp
    )

    target_link_libraries(test_MISBranchingFeatures
        PRIVATE
        mis_learning
        scip
    )

    add_test(
        NAME MISBranchingFeatures
        COMMAND test_MISBranchingFeatures
    )

    add_executable(test_MISBranchingModelIO
        tests/test_MISBranchingModelIO.cpp
        src/io/MISBranchingModelIO.cpp
    )

    target_link_libraries(test_MISBranchingModelIO
        PRIVATE
        mis_learning
    )

    add_test(
        NAME MISBranchingModelIO
        COMMAND test_MISBranchingModelIO
    )

    add_executable(test_MISLearnedBranchrule
        tests/test_MISLearnedBranchrule.cpp
        src/problems/MIS.cpp
        src/generators/ErdosRenyiGenerator.cpp
        src/io/MISBranchingModelIO.cpp
        src/learning/MISAdjacency.cpp
        src/services/MISSolverSettings.cpp
    )

    target_link_libraries(test_MISLearnedBranchrule
        PRIVATE
        mis_branching
    )

    add_test(
        NAME MISLearnedBranchrule
        COMMAND test_MISLearnedBranchrule
    )

    add_executable(test_CsvReader
        tests/test_CsvReader.cpp
        src/io/CsvReader.cpp
    )

    target_include_directories(test_CsvReader
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME CsvReader
        COMMAND test_CsvReader
    )

    add_executable(test_MISFeaturizationService
        tests/test_MISFeaturizationService.cpp
        src/problems/MIS.cpp
        src/generators/ErdosRenyiGenerator.cpp
        src/io/CsvReader.cpp
        src/io/MISInstanceIO.cpp
        src/io/MISStrongBranchingSampleIO.cpp
        src/learning/MISAdjacency.cpp
        src/services/MISFeaturizationService.cpp
        src/services/MISSolverSettings.cpp
    )

    target_link_libraries(test_MISFeaturizationService
        PRIVATE
        mis_branching
        mis_learning
    )

    add_test(
        NAME MISFeaturizationService
        COMMAND test_MISFeaturizationService
            ${PROJECT_BINARY_DIR}/test_mis_featurization_service
    )

    add_executable(test_MISStrongBranchingSampleIO
        tests/test_MISStrongBranchingSampleIO.cpp
        src/io/MISStrongBranchingSampleIO.cpp
    )

    target_include_directories(test_MISStrongBranchingSampleIO
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME MISStrongBranchingSampleIO
        COMMAND test_MISStrongBranchingSampleIO
    )

    add_executable(test_MISCollectionService
        tests/test_MISCollectionService.cpp
        src/problems/MIS.cpp
        src/generators/ErdosRenyiGenerator.cpp
        src/io/MISInstanceIO.cpp
        src/io/MISStrongBranchingSampleIO.cpp
        src/services/MISCollectionService.cpp
        src/services/MISSolverSettings.cpp
        src/services/SCIPStatusName.cpp
    )

    target_link_libraries(test_MISCollectionService
        PRIVATE
        mis_branching
    )

    add_test(
        NAME MISCollectionService
        COMMAND test_MISCollectionService
            ${PROJECT_BINARY_DIR}/test_mis_collection_service
    )

    add_executable(test_MISBenchmarkService
        tests/test_MISBenchmarkService.cpp
        src/problems/MIS.cpp
        src/generators/ErdosRenyiGenerator.cpp
        src/io/MISInstanceIO.cpp
        src/io/MISStrongBranchingSampleIO.cpp
        src/io/MISBranchingModelIO.cpp
        src/learning/MISAdjacency.cpp
        src/services/MISBenchmarkService.cpp
        src/services/MISCollectionService.cpp
        src/services/MISSolverSettings.cpp
        src/services/SCIPStatusName.cpp
    )

    target_link_libraries(test_MISBenchmarkService
        PRIVATE
        mis_branching
    )

    add_test(
        NAME MISBenchmarkService
        COMMAND test_MISBenchmarkService
            ${PROJECT_BINARY_DIR}/test_mis_benchmark_service
    )

    add_test(
        NAME MISBenchmarkCLI
        COMMAND bash
            ${PROJECT_SOURCE_DIR}/tests/test_mis_benchmark.sh
            $<TARGET_FILE:mis-generator>
            $<TARGET_FILE:mis-benchmark>
            ${PROJECT_BINARY_DIR}/test_mis_benchmark
    )

    add_test(
        NAME MISCollectorCLI
        COMMAND bash
            ${PROJECT_SOURCE_DIR}/tests/test_mis_collector.sh
            $<TARGET_FILE:mis-generator>
            $<TARGET_FILE:mis-collector>
            ${PROJECT_BINARY_DIR}/test_mis_collector
    )

    add_executable(test_MISInstanceIO
        tests/test_MISInstanceIO.cpp
        src/io/MISInstanceIO.cpp
    )

    add_executable(test_InstanceMetadataIO
        tests/test_InstanceMetadataIO.cpp
        src/io/InstanceMetadataIO.cpp
    )

    target_include_directories(test_InstanceMetadataIO
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME InstanceMetadataIO
        COMMAND test_InstanceMetadataIO
    )

    add_executable(test_MISFeatureExtractor
        tests/test_MISFeatureExtractor.cpp
        src/services/MISFeatureExtractor.cpp
    )

    target_include_directories(test_MISFeatureExtractor
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(test_MISFeatureExtractor
        PRIVATE
        scip
    )

    add_test(
        NAME MISFeatureExtractor
        COMMAND test_MISFeatureExtractor
    )

    add_executable(test_MISInstanceLoadingService
        tests/test_MISInstanceLoadingService.cpp
        src/services/MISInstanceLoadingService.cpp
        src/io/MISInstanceIO.cpp
        src/io/InstanceMetadataIO.cpp
    )

    target_include_directories(test_MISInstanceLoadingService
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME MISInstanceLoadingService
        COMMAND test_MISInstanceLoadingService
    )

    add_executable(test_MISSolvingService
        tests/test_MISSolvingService.cpp
        src/services/MISSolvingService.cpp
        src/services/SCIPStatusName.cpp
        src/io/MISInstanceIO.cpp
        src/problems/MIS.cpp
    )

    target_include_directories(test_MISSolvingService
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(test_MISSolvingService
        PRIVATE
        scip
    )

    add_test(
        NAME MISSolvingService
        COMMAND test_MISSolvingService
    )

    add_executable(test_MISGenerationService
        tests/test_MISGenerationService.cpp
        src/services/MISGenerationService.cpp
        src/generators/ErdosRenyiGenerator.cpp
        src/io/MISInstanceIO.cpp
        src/io/InstanceMetadataIO.cpp
    )

    target_include_directories(test_MISGenerationService
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    target_link_libraries(test_MISGenerationService
        PRIVATE
        scip
    )

    add_test(
        NAME MISGenerationService
        COMMAND test_MISGenerationService
            ${PROJECT_BINARY_DIR}/test_mis_generation_service
    )

    add_test(
        NAME MISGeneratorCLI
        COMMAND bash
            ${PROJECT_SOURCE_DIR}/tests/test_mis_generator.sh
            $<TARGET_FILE:mis-generator>
            ${PROJECT_BINARY_DIR}/test_mis_generator
    )

    add_test(
        NAME MISGeneratorBatch
        COMMAND bash
            ${PROJECT_SOURCE_DIR}/tests/test_mis_generator_batch.sh
            $<TARGET_FILE:mis-generator>
            ${PROJECT_BINARY_DIR}/test_mis_generator_batch
    )

    target_include_directories(test_MISInstanceIO
        PRIVATE
        ${PROJECT_SOURCE_DIR}/include
    )

    add_test(
        NAME MISInstanceIO
        COMMAND test_MISInstanceIO
    )
endif()
