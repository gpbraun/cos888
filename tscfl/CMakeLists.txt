cmake_minimum_required(VERSION 3.16)
project(cos888_tscfl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------
# CPLEX/Concert/CP Optimizer
# ------------------------------------------------------------
if(DEFINED ENV{ILOG})
    set(ILOG_ROOT $ENV{ILOG})
else()
    message(STATUS "Variável ambiente ILOG não definida. Usando: /opt/ibm/ILOG/CPLEX_Studio2212")
    set(ILOG_ROOT "/opt/ibm/ILOG/CPLEX_Studio2212")
endif()

set(ILOG_ROOT $ENV{ILOG})

# Diretórios de headers do CPLEX/Concert/CP
set(CPLEX_INCLUDE_DIRS
    ${ILOG_ROOT}/cplex/include
    ${ILOG_ROOT}/concert/include
    ${ILOG_ROOT}/cpoptimizer/include
)

# Diretórios de libs do CPLEX/Concert/CP
set(CPLEX_LIBRARY_DIRS
    ${ILOG_ROOT}/cplex/lib/x86-64_linux
    ${ILOG_ROOT}/concert/lib/x86-64_linux
    ${ILOG_ROOT}/cpoptimizer/lib/x86-64_linux
)

# Ainda podemos usar include/link "globais" se quiser
include_directories(${CPLEX_INCLUDE_DIRS})
link_directories(${CPLEX_LIBRARY_DIRS})

# ------------------------------------------------------------
# Biblioteca TSCFL (todo o código da "lib" em src/)
# ------------------------------------------------------------
file(GLOB_RECURSE TSCFL_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(tscfl STATIC ${TSCFL_SOURCES})

# Headers da própria lib (tscfl_instance.hpp, tscfl_utils.hpp, solvers, etc.)
target_include_directories(tscfl PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Linkar CPLEX/Concert/CP na biblioteca
target_link_libraries(tscfl PUBLIC
    ilocplex
    cplex
    concert
    cp
    pthread
    dl
    m
)

# ------------------------------------------------------------
# Executável dos experimentos
# ------------------------------------------------------------
add_executable(tscfl_experiments
    src/experiments/tscfl_experiments.cpp
)

# Executável linka apenas com a biblioteca tscfl,
# que já está linkada com CPLEX/Concert/CP.
target_link_libraries(tscfl_experiments PRIVATE
    tscfl
)
