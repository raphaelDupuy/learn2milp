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

add_library(mis_branching STATIC
    src/branching/MISFullstrongBranchrule.c
)

target_include_directories(mis_branching
    PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mis_branching
    PUBLIC
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
