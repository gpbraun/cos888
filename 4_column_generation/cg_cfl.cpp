#include <ilcplex/ilocplex.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <cctype>

ILOSTLBEGIN

struct Data
{
    int m = 0;                          // facilities
    int n = 0;                          // customers
    std::vector<double> Q;              // capacity[i]
    std::vector<double> f;              // fixed cost[i]
    std::vector<double> d;              // demand[j]
    std::vector<std::vector<double>> c; // c[i][j] = cost to assign all of j to i
    std::string name;
};

static inline bool is_number(const std::string &s)
{
    char *end = nullptr;
    std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0';
}

Data parse_orlib_cap(const std::string &path, double cap_override = -1.0)
{
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("Cannot open instance file: " + path);
    Data D;
    D.name = path;

    // Read m n
    {
        std::string line;
        do
        {
            if (!std::getline(in, line))
                throw std::runtime_error("Unexpected EOF while reading m n");
        } while (line.find_first_not_of(" \t\r\n") == std::string::npos);

        std::istringstream iss(line);
        if (!(iss >> D.m >> D.n))
            throw std::runtime_error("Failed to read m and n");
    }

    D.Q.resize(D.m);
    D.f.resize(D.m);
    D.d.resize(D.n);
    D.c.assign(D.m, std::vector<double>(D.n));

    // Facilities: capacity, fixed cost
    for (int i = 0; i < D.m; ++i)
    {
        std::string cap_tok, fix_tok;
        if (!(in >> cap_tok >> fix_tok))
            throw std::runtime_error("Failed reading facility line " + std::to_string(i + 1));

        double cap_val = 0.0;
        if (is_number(cap_tok))
        {
            cap_val = std::stod(cap_tok);
        }
        else
        {
            // OR-Library files capa/capb/capc use literal "capacity"
            // We allow the user to pass --cap X to replace it on the fly
            if (cap_override <= 0.0)
                throw std::runtime_error(
                    "Found non-numeric capacity token '" + cap_tok +
                    "'. Provide --cap <value> for capa/capb/capc.");
            cap_val = cap_override;
        }
        double fix_val = 0.0;
        if (!is_number(fix_tok))
            throw std::runtime_error("Fixed cost is not numeric on facility line " + std::to_string(i + 1));
        fix_val = std::stod(fix_tok);

        D.Q[i] = cap_val;
        D.f[i] = fix_val;
    }

    // Customers: demand, then m costs for i=1..m
    for (int j = 0; j < D.n; ++j)
    {
        std::string dem_tok;
        if (!(in >> dem_tok))
            throw std::runtime_error("Failed reading demand for customer " + std::to_string(j + 1));
        if (!is_number(dem_tok))
            throw std::runtime_error("Demand is not numeric for customer " + std::to_string(j + 1));
        D.d[j] = std::stod(dem_tok);

        for (int i = 0; i < D.m; ++i)
        {
            std::string cost_tok;
            if (!(in >> cost_tok))
                throw std::runtime_error("Failed reading cost c(" + std::to_string(i + 1) + "," + std::to_string(j + 1) + ")");
            if (!is_number(cost_tok))
                throw std::runtime_error("Non-numeric cost at c(" + std::to_string(i + 1) + "," + std::to_string(j + 1) + ")");
            D.c[i][j] = std::stod(cost_tok);
        }
    }
    return D;
}

struct Options
{
    double cap_override = -1.0; // for capa/capb/capc
    double timelimit = -1.0;
    int threads = 0;
    bool print_assign = false;
};

Options parse_opts(int argc, char **argv)
{
    Options opt;
    for (int k = 2; k < argc; ++k)
    {
        std::string a = argv[k];
        auto need = [&](int &k)
        {
            if (k + 1 >= argc)
                throw std::runtime_error("Missing value after " + a);
            ++k;
            return std::string(argv[k]);
        };
        if (a == "--cap")
            opt.cap_override = std::stod(need(k));
        else if (a == "--timelimit")
            opt.timelimit = std::stod(need(k));
        else if (a == "--threads")
            opt.threads = std::stoi(need(k));
        else if (a == "--print")
            opt.print_assign = true;
        else
            throw std::runtime_error("Unknown option: " + a);
    }
    return opt;
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <instance_path> [--cap X] [--timelimit T] [--threads K] [--print]\n"
                                             "Notes:\n"
                                             "  - For OR-Library capa/capb/capc, pass --cap <value> (from Beasley 1988 Table 1).\n";
        return 1;
    }

    std::string path = argv[1];
    Options opt;
    try
    {
        opt = parse_opts(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Arg error: " << e.what() << "\n";
        return 1;
    }

    IloEnv env;
    try
    {
        Data D = parse_orlib_cap(path, opt.cap_override);
        std::cout << "Instance: " << D.name << "  m=" << D.m << "  n=" << D.n << "\n";

        IloModel model(env);

        // Decision variables
        IloBoolVarArray y(env, D.m); // open facility i
        for (int i = 0; i < D.m; ++i)
            y[i].setName(("y_" + std::to_string(i + 1)).c_str());

        IloArray<IloBoolVarArray> z(env, D.m); // assign j to i
        for (int i = 0; i < D.m; ++i)
        {
            z[i] = IloBoolVarArray(env, D.n);
            for (int j = 0; j < D.n; ++j)
                z[i][j].setName(("z_" + std::to_string(i + 1) + "_" + std::to_string(j + 1)).c_str());
        }

        // Objective: sum_i f_i y_i + sum_{i,j} c_ij z_ij
        IloExpr obj(env);
        for (int i = 0; i < D.m; ++i)
            obj += D.f[i] * y[i];
        for (int i = 0; i < D.m; ++i)
            for (int j = 0; j < D.n; ++j)
                obj += D.c[i][j] * z[i][j];
        model.add(IloMinimize(env, obj));
        obj.end();

        // Assignment: each customer exactly once
        for (int j = 0; j < D.n; ++j)
        {
            IloExpr sum(env);
            for (int i = 0; i < D.m; ++i)
                sum += z[i][j];
            model.add(sum == 1);
            sum.end();
        }

        // Capacity linking: sum_j d_j z_ij <= Q_i y_i
        for (int i = 0; i < D.m; ++i)
        {
            IloExpr lhs(env);
            for (int j = 0; j < D.n; ++j)
                lhs += D.d[j] * z[i][j];
            model.add(lhs <= D.Q[i] * y[i]);
            lhs.end();
        }

        // Optional strengthening: z_ij <= y_i  (redundant but helps LP)
        for (int i = 0; i < D.m; ++i)
            for (int j = 0; j < D.n; ++j)
                model.add(z[i][j] <= y[i]);

        // Solve
        IloCplex cplex(model);
        if (opt.timelimit > 0)
            cplex.setParam(IloCplex::Param::TimeLimit, opt.timelimit);
        if (opt.threads > 0)
            cplex.setParam(IloCplex::Param::Threads, opt.threads);
        cplex.setOut(env.getNullStream()); // comment out to see CPLEX log

        bool ok = cplex.solve();
        if (!ok)
        {
            std::cerr << "CPLEX status: " << cplex.getStatus() << "\n";
            throw std::runtime_error("No solution found.");
        }

        double zopt = cplex.getObjValue();
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Objective = " << zopt << "\n";
        std::cout << "Open facilities:\n";

        std::vector<int> open_idx;
        for (int i = 0; i < D.m; ++i)
        {
            if (cplex.getValue(y[i]) > 0.5)
            {
                open_idx.push_back(i);
            }
        }
        std::cout << "  count = " << open_idx.size() << " { ";
        for (size_t k = 0; k < open_idx.size(); ++k)
        {
            std::cout << (open_idx[k] + 1) << (k + 1 < open_idx.size() ? " " : "");
        }
        std::cout << "}\n";

        // Report capacity usage for open facilities
        for (int i : open_idx)
        {
            double used = 0.0;
            for (int j = 0; j < D.n; ++j)
                used += D.d[j] * cplex.getValue(z[i][j]);
            std::cout << "  i=" << (i + 1) << "  used=" << used << " / " << D.Q[i] << "\n";
        }

        if (opt.print_assign)
        {
            std::cout << "\nAssignments (customer -> facility):\n";
            int printed = 0;
            for (int j = 0; j < D.n; ++j)
            {
                int sel = -1;
                for (int i = 0; i < D.m; ++i)
                    if (cplex.getValue(z[i][j]) > 0.5)
                    {
                        sel = i;
                        break;
                    }
                std::cout << "  j=" << (j + 1) << " -> i=" << (sel + 1) << "\n";
                if (++printed % 200 == 0)
                    std::cout.flush();
            }
        }
    }
    catch (const IloException &e)
    {
        std::cerr << "CPLEX exception: " << e.getMessage() << "\n";
        env.end();
        return 2;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        env.end();
        return 3;
    }

    env.end();
    return 0;
}
