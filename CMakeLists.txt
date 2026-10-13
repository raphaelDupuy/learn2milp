cmake_minimum_required(VERSION 3.20)

project(learn2milp
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(learn2milp
    src/main.cpp
    src/problems/MIS.cpp
    src/generators/ErdosRenyiGenerator.cpp
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
    src/generators/ErdosRenyiGenerator.cpp
    src/io/MISInstanceIO.cpp
)

target_include_directories(mis-generator
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

include(CTest)

if(BUILD_TESTING)
    add_executable(test_MISInstanceIO
        tests/test_MISInstanceIO.cpp
        src/io/MISInstanceIO.cpp
    )

    add_test(
        NAME MISGeneratorCLI
        COMMAND bash
            ${PROJECT_SOURCE_DIR}/tests/test_mis_generator.sh
            $<TARGET_FILE:mis-generator>
            ${PROJECT_BINARY_DIR}/test_mis_generator
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
