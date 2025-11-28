#!/usr/bin/python
# ---------------------------------------------------------------------------
# File: conjind.py
# Version 0.0.1
# ---------------------------------------------------------------------------
# Licensed Materials - Property of DoCara
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright OCara.com Corporation 2024. All Rights Reserved.
# ---------------------------------------------------------------------------
# Solve o conjunto independente.

import sys

import cplex

# import traceback


def ler_instancia(nome_arq):
    # lê a instância do KPF
    n = b = p = w = None
    with open(nome_arq, "r") as arq:
        l1 = arq.readline()
        n, k, b = l1.split()
        n = int(n)
        b = int(b)
        k = int(k)
        if n <= 2 or b < 1:
            raise ValueError
        print(n, b, k)
        l1 = arq.readline()
        l2 = arq.readline()
        l1 = l1.split()
        l2 = l2.split()
        if len(l1) != n or len(l2) != n:
            raise ValueError
        p = [int(a) for a in l1]
        w = [int(a) for a in l2]
        F = {}
        for i in range(k):
            l1 = [int(a) for a in arq.readline().split()]
            l2 = [int(a) for a in arq.readline().split()]
            if len(l1) != 3 or len(l2) != 2:
                print(i, k, l1, l2)
                raise ValueError
            if l1[0] != 1 or l1[2] != 2 or l1[1] <= 0:
                raise ValueError
            if l2[0] < 0 or l2[0] >= n:
                raise ValueError
            if l2[1] < 0 or l2[1] >= n:
                raise ValueError
            if l2[0] >= l2[1]:
                raise ValueError
            if (l2[0], l2[1]) in F:
                raise ValueError
            F[(l2[0], l2[1])] = l1[1]
    return n, b, p, w, F


def criar_subprob(n, b, p, w):
    """Solve o subprob."""
    cpx_sub = cplex.Cplex()
    cpx_sub.objective.set_sense(cpx_sub.objective.sense.maximize)
    x = cpx_sub.variables.add(
        obj=[p[i] for i in range(n)],
        lb=[0] * n,
        ub=[1] * n,
        types=[cpx_sub.variables.type.binary] * n,
        names=["x_" + str(i) for i in range(n)],
    )
    # Mochila
    cpx_sub.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair([x[i] for i in range(n)], [float(w[i]) for i in range(n)])
        ],
        senses=["L"],
        rhs=[float(b)],
        names=["sac"],
    )

    # Tweak some CPLEX parameters so that CPLEX has a harder time to
    # solve the model and our cut separators can actually kick in.
    cpx_sub.parameters.mip.strategy.heuristicfreq.set(-1)
    cpx_sub.parameters.mip.cuts.mircut.set(-1)
    cpx_sub.parameters.mip.cuts.implied.set(-1)
    cpx_sub.parameters.mip.cuts.gomory.set(-1)
    cpx_sub.parameters.mip.cuts.flowcovers.set(-1)
    cpx_sub.parameters.mip.cuts.pathcut.set(-1)
    cpx_sub.parameters.mip.cuts.liftproj.set(-1)
    cpx_sub.parameters.mip.cuts.zerohalfcut.set(-1)
    cpx_sub.parameters.mip.cuts.cliques.set(-1)
    cpx_sub.parameters.mip.cuts.covers.set(-1)
    cpx_sub.parameters.threads.set(1)
    cpx_sub.parameters.clocktype.set(1)
    cpx_sub.parameters.timelimit.set(1800)

    cpx_sub.write("KPF_subprob.lp")
    return cpx_sub, x


def solve_subprob(cpx_sub, x, u, n, p, F):
    du = [0] * n
    k = 1
    for i, j in F:
        du[i] += u[k]
        du[j] += u[k]
        k += 1
    cpx_sub.objective.set_linear([(x[i], p[i] - du[i]) for i in range(n)])

    # cpx_sub.write("KPF_subprob.lp")

    cpx_sub.solve()

    # print("\t Subproblem:")
    # print('\t\t Solution status:                   %d' % cpx_sub.solution.get_status())
    # print('\t\t Nodes processed:                   %d' %
    # cpx_sub.solution.progress.get_num_nodes_processed())
    tol = cpx_sub.parameters.mip.tolerances.integrality.get()
    fobj = cpx_sub.solution.get_objective_value() - u[0]
    # print('\t\t Optimal value:                     %f' % fobj,"const: ",u[0])
    if fobj < 0.001:
        return fobj, set()
    values = cpx_sub.solution.get_values()
    col = set()
    for i in range(n):
        if values[x[i]] >= 1 - tol:
            col.add(i)
    return fobj, col


def BB_lado1(pilha, cpx_sub, cols, TF, ind, cpx):
    pilha.append([ind, 1])
    cpx_sub.variables.set_lower_bounds(ind, 1.0)
    for k in range(len(cols)):
        if ind in cols[k]:
            continue
        cpx.variables.set_upper_bounds(TF + k, 0.0)


def BB_lado0(pilha, cpx_sub, cols, TF, cpx):
    i = pilha[-1][0]
    print("Branch na variável x_" + str(i) + " lado zero - Profundidade:", len(pilha))
    pilha[-1][1] = 0
    cpx_sub.variables.set_lower_bounds(i, 0.0)
    cpx_sub.variables.set_upper_bounds(i, 0.0)
    for k in range(len(cols)):
        if i in cols[k]:
            cpx.variables.set_upper_bounds(TF + k, 0.0)
        else:
            for j in range(len(pilha) - 1):
                if (pilha[j][0] in cols[k]) == pilha[j][1]:
                    continue
                else:
                    break
            else:
                cpx.variables.set_upper_bounds(TF + k, 1.0)


def backtracking(pilha, cpx_sub, cols, TF, cpx):
    print("backtracking- Profundidade ini:", len(pilha), end=" ")
    while pilha != [] and pilha[-1][1] == 0:
        i, j = pilha.pop()
        cpx_sub.variables.set_upper_bounds(i, 1.0)
        for k in range(len(cols)):
            if i in cols[k]:
                for j in range(len(pilha)):
                    if (pilha[j][0] in cols[k]) == pilha[j][1]:
                        continue
                    else:
                        break
                else:
                    cpx.variables.set_upper_bounds(TF + k, 1.0)
    print("Produndidade Final:", len(pilha))
    if pilha != []:  # terminou o lado 1
        BB_lado0(pilha, cpx_sub, cols, TF, cpx)


def gecol(n, b, p, w, F):
    """Solve o KPF."""
    l = len(F)
    LB = 0
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    cols = [set()]

    v = cpx.variables.add(
        obj=[-F[t] for t in F],
        lb=[0] * l,
        ub=[1] * l,
        # types=[cpx.variables.type.continuous] * l,
        names=["v_" + str(i) + "_" + str(j) for i, j in F],
    )

    z = [
        cpx.variables.add(
            obj=[0],
            lb=[0],
            ub=[1],
            # types=[cpx.variables.type.continuous],
            names=["z_0"],
        )
    ]
    # comb
    comb = cpx.linear_constraints.add(
        lin_expr=[cplex.SparsePair([z[0][0]], [1.0])],
        senses=["E"],
        rhs=[1.0],
        names=["comb"],
    )

    # Linearizacao
    lin = cpx.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(["v_" + str(i) + "_" + str(j)], [-1.0]) for i, j in F
        ],
        senses=["L"] * l,
        rhs=[1.0] * l,
        names=["lin_" + str(i) + "_" + str(j) for i, j in F],
    )

    # Tweak some CPLEX parameters so that CPLEX has a harder time to
    # solve the model and our cut separators can actually kick in.
    # cpx.parameters.mip.strategy.heuristicfreq.set(-1)
    # cpx.parameters.mip.cuts.mircut.set(-1)
    # cpx.parameters.mip.cuts.implied.set(-1)
    # cpx.parameters.mip.cuts.gomory.set(-1)
    # cpx.parameters.mip.cuts.flowcovers.set(-1)
    # cpx.parameters.mip.cuts.pathcut.set(-1)
    # cpx.parameters.mip.cuts.liftproj.set(-1)
    # cpx.parameters.mip.cuts.zerohalfcut.set(-1)
    # cpx.parameters.mip.cuts.cliques.set(-1)
    # cpx.parameters.mip.cuts.covers.set(-1)
    cpx.parameters.threads.set(1)
    cpx.parameters.clocktype.set(1)
    cpx.parameters.timelimit.set(1800)
    cpx.set_results_stream(None)

    cpx_sub, x_sub = criar_subprob(n, b, p, w)
    cpx_sub.set_results_stream(None)

    pilha = []
    prt = False
    nos = -1
    while True:
        nos += 1
        r = 0
        while r < 100000:
            # cpx.write("KPF_gelcol.lp")
            # input("gerou LP")
            cpx.solve()
            print(
                "no: ", nos, " it: ", r + 1, " LP: ", cpx.solution.get_objective_value()
            )
            u = cpx.solution.get_dual_values()
            fobj, col = solve_subprob(cpx_sub, x_sub, u, n, p, F)
            # fobj = 0
            print("\t Subprob fobj:", fobj)
            if fobj < 0.001:
                break
            # print(col)
            cols += [col]
            v = 0
            for i in col:
                v += p[i]
            z += [
                cpx.variables.add(
                    obj=[v],
                    lb=[0],
                    ub=[1],
                    # types=[cpx.variables.type.continuous],
                    names=["z_" + str(len(z))],
                )
            ]
            cpx.linear_constraints.set_coefficients(
                [("comb", z[-1][0], 1.0)]
                + [
                    ("lin_" + str(i) + "_" + str(j), z[-1][0], (i in col) + (j in col))
                    for i, j in F
                ]
            )
            # print(col)
            r += 1

        # print('Solution status:                   %d' % cpx.solution.get_status())
        tol = cpx.parameters.mip.tolerances.integrality.get()
        # print('Optimal value:                     %f' %
        # cpx.solution.get_objective_value())
        values = cpx.solution.get_values()
        frac = False
        if prt:
            k = 0
            for i, j in F:
                if values[k] >= tol:
                    print("v_" + str(i) + "_" + str(j) + "= " + str(values[k]))
                k += 1
        x = [0] * n
        for i in range(len(F), len(values)):
            if values[i] >= tol:
                if prt:
                    print("z_" + str(i - len(F)) + "= " + str(values[i]))
                for iv in cols[i - len(F)]:
                    x[iv] += values[i]
                if values[i] <= 1 - tol:
                    frac = True
        if prt:
            print(x)

        if int(cpx.solution.get_objective_value()) < int(LB + 0.001) - 1:
            print(
                "Poda por limite - fobj: ",
                cpx.solution.get_objective_value(),
                " Best solution: ",
                LB,
            )
            # Pode por limite
            if pilha[-1][1] == 1:
                # Branch lado 0
                BB_lado0(pilha, cpx_sub, cols, len(F), cpx)
            else:
                # bracktracking
                backtracking(pilha, cpx_sub, cols, len(F), cpx)
        else:
            if frac:
                # Branch do lado 1
                tp = [(abs(x[i] - 0.5), i) for i in range(n)]
                val, i = min(tp)
                print(
                    "Branch na variável x_" + str(i) + " com valor ",
                    x[i],
                    " Profundidade: ",
                    len(pilha),
                )
                BB_lado1(pilha, cpx_sub, cols, len(F), i, cpx)
            else:
                LB = max(cpx.solution.get_objective_value(), LB)
                print(
                    "Poda por otimilidade - Solucao inteira: ",
                    cpx.solution.get_objective_value(),
                )
                print("Best solution: ", LB)
                if len(pilha) > 0:
                    if pilha[-1][1] == 1:
                        BB_lado0(pilha, cpx_sub, cols, len(F), cpx)
                    else:
                        backtracking(pilha, cpx_sub, cols, len(F), cpx)

        if pilha == []:
            break


def modelo(n, b, p, w, F):
    """Solve o KPF."""
    l = len(F)
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    x = cpx.variables.add(
        obj=[p[i] for i in range(n)],
        lb=[0] * n,
        ub=[1] * n,
        types=[cpx.variables.type.binary] * n,
        names=["x_" + str(i) for i in range(n)],
    )

    v = cpx.variables.add(
        obj=[-F[t] for t in F],
        lb=[0] * l,
        ub=[1] * l,
        types=[cpx.variables.type.continuous] * l,
        names=["v_" + str(i) + "_" + str(j) for i, j in F],
    )

    # Mochila
    cpx.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair([x[i] for i in range(n)], [float(w[i]) for i in range(n)])
        ],
        senses=["L"],
        rhs=[float(b)],
        names=["sac"],
    )

    # Linearizacao
    cpx.linear_constraints.add(
        lin_expr=[
            cplex.SparsePair(
                [x[i]] + [x[j]] + ["v_" + str(i) + "_" + str(j)], [1.0, 1.0, -1.0]
            )
            for i, j in F
        ],
        senses=["L"] * l,
        rhs=[1.0] * l,
        names=["lin_" + str(i) + "_" + str(j) for i, j in F],
    )
    # Tweak some CPLEX parameters so that CPLEX has a harder time to
    # solve the model and our cut separators can actually kick in.
    cpx.parameters.mip.strategy.heuristicfreq.set(-1)
    cpx.parameters.mip.strategy.presolvenode.set(-1)
    cpx.parameters.mip.cuts.mircut.set(-1)
    cpx.parameters.mip.cuts.implied.set(-1)
    cpx.parameters.mip.cuts.gomory.set(-1)
    cpx.parameters.mip.cuts.flowcovers.set(-1)
    cpx.parameters.mip.cuts.pathcut.set(-1)
    cpx.parameters.mip.cuts.liftproj.set(-1)
    cpx.parameters.mip.cuts.zerohalfcut.set(-1)
    cpx.parameters.mip.cuts.cliques.set(-1)
    cpx.parameters.mip.cuts.covers.set(-1)
    cpx.parameters.mip.limits.cutsfactor.set(0)
    cpx.parameters.mip.limits.eachcutlimit.set(0)
    cpx.parameters.mip.limits.cutpasses.set(-1)
    cpx.parameters.preprocessing.repeatpresolve.set(0)
    cpx.parameters.preprocessing.relax.set(0)
    cpx.parameters.preprocessing.boundstrength.set(0)
    cpx.parameters.preprocessing.symmetry.set(0)
    cpx.parameters.preprocessing.folding.set(0)
    cpx.parameters.preprocessing.aggregator.set(0)
    cpx.parameters.preprocessing.coeffreduce.set(0)
    cpx.parameters.preprocessing.dependency.set(0)
    cpx.parameters.preprocessing.dual.set(-1)
    cpx.parameters.preprocessing.presolve.set(0)
    cpx.parameters.preprocessing.numpass.set(0)
    cpx.parameters.threads.set(1)
    cpx.parameters.clocktype.set(1)
    cpx.parameters.timelimit.set(600)

    cpx.write("KPF.lp")

    cpx.solve()

    print("Solution status:                   %d" % cpx.solution.get_status())
    print(
        "Nodes processed:                   %d"
        % cpx.solution.progress.get_num_nodes_processed()
    )
    tol = cpx.parameters.mip.tolerances.integrality.get()
    print("Optimal value:                     %f" % cpx.solution.get_objective_value())
    values = cpx.solution.get_values()
    for i in range(n):
        if values[x[i]] >= 1 - tol:
            print("x_" + str(i) + "= " + str(values[x[i]]))


if len(sys.argv) != 3 or not (sys.argv[2] in "-mip", "-BP"):
    print("Uso: python KPF.py <arquivo_inst> -mip ou -BP")
    sys.exit(1)

nome = sys.argv[1]

# nome = "teste.dat"
n, b, p, w, F = ler_instancia(
    nome
)  # 5, [[0,1,2,3,4],[1,0,2,3,4],[2,2,0,1,2],[3,3,1,0,3],[4,4,2,3,0]]#
# print(n,b,p,w,F)
if sys.argv[2] == "-mip":
    modelo(n, b, p, w, F)
else:
    gecol(n, b, p, w, F)
