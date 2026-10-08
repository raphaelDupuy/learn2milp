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
