/*
COS888

Utilitários para solvers do TSCFL.

Gabriel Braun, 2025
*/

#pragma once

#include <utils/instance/instance.hpp>
#include <utils/relaxation/relaxation_balance.hpp>
#include <utils/relaxation/relaxation_capacity.hpp>
#include <utils/subproblem/subproblem_dual.hpp>
#include <utils/subproblem/subproblem_net.hpp>
#include <utils/subproblem/subproblem_primal.hpp>

// Cria um solver do subprolema do tipo especificado
std::unique_ptr<Subproblem> Subproblem::create(const TSCFLInstance& inst, Subproblem::Mode mode) {
    switch (mode) {
        case Subproblem::Mode::DUAL:
            return std::make_unique<SubproblemDual>(inst);
        case Subproblem::Mode::PRIMAL:
            return std::make_unique<SubproblemPrimal>(inst);
        case Subproblem::Mode::NET:
            return std::make_unique<SubproblemNet>(inst);
    }
    throw std::invalid_argument("Subproblema inválido.");
}

// Cria um solver de relaxação Lagrangeana do tipo especificado
std::unique_ptr<Relaxation> Relaxation::create(const TSCFLInstance& inst, Relaxation::Mode mode) {
    switch (mode) {
        case Relaxation::Mode::BALANCES:
            return std::make_unique<RelaxationBalance>(inst);
        case Relaxation::Mode::CAPACITIES:
            return std::make_unique<RelaxationCapacity>(inst);
    }
    throw std::invalid_argument("Relaxação Lagrangeana inválida.");
}
