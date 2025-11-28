cmake_minimum_required(VERSION 3.16)
project(cos888_tscfl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# Ambiente conda
# ------------------------------------------------------------
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()

# ------------------------------------------------------------
# CPLEX/Concert/CP Optimizer
# ------------------------------------------------------------
if(NOT DEFINED ENV{ILOG})
    message(FATAL_ERROR "ILOG environment variable not set.")
endif()

set(ILOG_ROOT $ENV{ILOG})

# Headers
include_directories(
    ${ARMADILLO_INCLUDE_DIRS}
    ${ILOG_ROOT}/cplex/include
    ${ILOG_ROOT}/concert/include
    ${ILOG_ROOT}/cpoptimizer/include
)

# Libraries
link_directories(
    ${ILOG_ROOT}/cplex/lib/x86-64_linux
    ${ILOG_ROOT}/concert/lib/x86-64_linux
    ${ILOG_ROOT}/cpoptimizer/lib/x86-64_linux
)

# ------------------------------------------------------------
# Executável dos experimentos
# ------------------------------------------------------------
add_executable(tscfl_experiments
    experiments.cpp
)

target_link_libraries(tscfl_experiments PRIVATE
    ilocplex
    cplex
    concert
    cp
    pthread
    dl
    m
)
