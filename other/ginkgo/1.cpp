// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <iomanip>
#include <limits>
#include <ginkgo/ginkgo.hpp>


// Some shortcuts
using ValueType = double;
//using RealValueType = gko::remove_complex<ValueType>;
using IndexType = int;
using mtx = gko::matrix::Csr<ValueType, IndexType>;
using vec = gko::matrix::Dense<ValueType>;
using real_vec = gko::matrix::Dense<ValueType>;
using mg = gko::solver::Multigrid;
using pgm = gko::multigrid::Pgm<ValueType, IndexType>;

double compute_residual(const std::shared_ptr<gko::Executor>& exec, const std::shared_ptr<mtx>& A, const std::shared_ptr<vec> &b, const std::shared_ptr<vec> &x)
{
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    auto b_clone = b->clone();
    A->apply(one, x, neg_one, b_clone);
    b_clone->compute_norm2(res);
    return res->clone(gko::OmpExecutor::create())->at(0);
}

struct ResidualNormLogger : public gko::log::Logger {
    ResidualNormLogger()
        : gko::log::Logger(gko::log::Logger::iteration_complete_mask)
    {}

    void on_iteration_complete(
        const gko::LinOp* solver,
        const gko::size_type& iteration,
        const gko::LinOp* r,
        const gko::LinOp* x,
        const gko::LinOp* residual_norm,
        const gko::LinOp* implicit_tau_sq	// the implicit residual norm squared (optional)
    ) const override
    {
        // if (residual_norm) {
        //     auto dense =
        //         gko::as<gko::matrix::Dense<double>>(residual_norm);
        //
        //     // residual_norm is a 1x1 Dense
        //     double value = dense->at(0, 0);
        //
        //     std::cout << "Iter " << iteration
        //               << " | residual norm = "
        //               << value << std::endl;
        // } else
        {
            auto e = gko::as<gko::matrix::Dense<double>>(r)->clone(gko::OmpExecutor::create());
            auto res = gko::initialize<real_vec>({0.0}, solver->get_executor());
            e->compute_norm2(res);
            // auto val = res->at(0);
            std::cout << "Iter " << iteration << "r " << res->clone(gko::OmpExecutor::create())->at(0) << std::endl;

            // if (std::isnan(val) || std::isinf(val) || val > 10000000.0 && iteration > 1000) {
            //     std::cerr << "Stopping solver at iteration "
            //               << iteration
            //               << " (invalid residual = " << r << ")\n";
            //     throw std::runtime_error("Solver diverged");
            // }
        }
    }
};



template <typename Solver>
void solve_system(const std::shared_ptr<gko::Executor>& exec, const std::shared_ptr<mtx>& A, const std::shared_ptr<vec> &b, const std::shared_ptr<vec> &x, std::shared_ptr<gko::LinOp> preconditioner, int max_iteration)
{

    // #####################################################
    // Criteria
    // #####################################################
    constexpr ValueType reduction_factor{1e-15};
    auto iteration_criteria = gko::stop::Iteration::build().with_max_iters(max_iteration);
    auto residual_criteria = gko::stop::ResidualNorm<ValueType>::build()
    .with_reduction_factor(reduction_factor);
    // .with_baseline(gko::stop::mode::absolute);


    // #####################################################
    // Solver
    // #####################################################


    std::shared_ptr<gko::LinOpFactory> solver_factory = Solver::build()
    .with_criteria(iteration_criteria, residual_criteria)
    .with_generated_preconditioner(preconditioner)
        .on(exec);


    // Generate preconditioned solver for a specific target system
    const auto solver = solver_factory->generate(A);
    // Solve system

    auto logger = gko::share(gko::log::Convergence<>::create());
    solver->add_logger(logger);
    //
    //
    // auto loggerr = std::make_shared<ResidualNormLogger>();
    // solver->add_logger(loggerr);


    exec->synchronize();
    auto time_start = std::chrono::steady_clock::now();
    solver->apply(b, x);
    exec->synchronize();
    auto time_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    std::cout << "[Iterations:" << logger->get_num_iterations() << "]\n";
    std::cout << "[Time:" << duration.count() << "ms" << "]\n";

}

void solve_system_multigrid(const std::shared_ptr<gko::Executor>& exec, const std::shared_ptr<mtx>& A, const std::shared_ptr<vec> &b, const std::shared_ptr<vec> &x, int max_iteration) {

    // auto ilu_preconditioner = gko::share(
    // gko::preconditioner::Ilu<gko::solver::LowerTrs<ValueType, IndexType>,
    //                          gko::solver::UpperTrs<ValueType, IndexType>,
    //                          false>::build()
    //     .on(exec));
    //
    auto jacobi_preconditioner = gko::share(
        gko::preconditioner::Jacobi<ValueType, IndexType>::build().on(exec));

    // Smoother
    auto smoother_gen = gko::share(
        gko::solver::build_smoother(jacobi_preconditioner, 1u, static_cast<ValueType>(1.0)));

    auto jacobi = gko::share(
    gko::preconditioner::Jacobi<ValueType, IndexType>::build().with_max_block_size(4)
        .on(exec));
    auto chebyshev_smoother = gko::share(
    gko::solver::Chebyshev<ValueType>::build()
        .with_preconditioner(jacobi)
        .with_criteria(gko::stop::Iteration::build().with_max_iters(1u).on(exec))
        .on(exec));




    // Coarser
    auto exact_tol_stop =
        gko::share(gko::stop::ResidualNorm<ValueType>::build()
                       .with_baseline(gko::stop::mode::rhs_norm)
                       .with_reduction_factor(1e-14)
                       .on(exec));
    auto coarsening_iter_stop =
     gko::share(gko::stop::Iteration::build().with_max_iters(50u).on(exec));
    auto coarsest_gen = gko::share(gko::solver::Gmres<ValueType>::build()
                                       .with_preconditioner(jacobi_preconditioner)
                                       .with_criteria(coarsening_iter_stop, exact_tol_stop)
                                       .on(exec));


    // Here we put the customized options together and create the multigrid factory.
    std::shared_ptr<gko::LinOpFactory> multigrid_gen;
    multigrid_gen =
        mg::build()
            .with_max_levels(15u) // stop coarsening if multigrid levels reaches max_multigrid_levels
            .with_min_coarse_rows(0u) // stop coarsening if coarse_rows < min_coarse_rows
            .with_pre_smoother(smoother_gen)
            .with_post_uses_pre(true)
            .with_mg_level(gko::share(pgm::build().with_deterministic(true).on(exec))) // Use Pgm as the MultigridLevel factory.
            .with_coarsest_solver(coarsest_gen)
            .with_default_initial_guess(gko::solver::initial_guess_mode::zero)
            .with_criteria(gko::stop::Iteration::build().with_max_iters(1u))
            .on(exec);


    // Create solver factory
    const gko::remove_complex<ValueType> tolerance = 1e-15;
    auto tol_stop = gko::share(gko::stop::ResidualNorm<ValueType>::build()
                               .with_baseline(gko::stop::mode::absolute)
                               .with_reduction_factor(tolerance)
                               .on(exec));

    auto iter_stop =
     gko::share(gko::stop::Iteration::build().with_max_iters(max_iteration).on(exec));
    auto solver_gen = gko::solver::Gmres<ValueType>::build()
                          .with_krylov_dim(30u)
                          .with_criteria(iter_stop, tol_stop)
                          .with_preconditioner(multigrid_gen)
                          .on(exec);



    // Generate the multigrid hierarchy using A
    auto mg_solver = solver_gen->generate(A);

    // ###############################
    //    SOLVE Ax = b
    // ###############################

    auto logger = gko::share(gko::log::Convergence<>::create());
    mg_solver->add_logger(logger);

    exec->synchronize();
    auto time_start = std::chrono::steady_clock::now();
    mg_solver->apply(b, x);
    exec->synchronize();
    auto time_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start);
    std::cout << "[Iterations:" << logger->get_num_iterations() << "]\n";
    std::cout << "[Time:" << duration.count() << "ms" << "]\n";
}

int main(int argc, char* argv[])
{
    std::cout << std::scientific << std::setprecision(15);
    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc != 7) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << " [solver name] [preconditioner name]" << " [data folder containing A,b,solution]" << " [max_iteration] [block_jacobi_size]" << std::endl;
        std::cerr << "executor: omp, cuda, hip, dpcpp, reference" << std::endl;
        std::cerr << "sover name: bicg ,bicgstab ,cgs ,cbgmres ,gmres ,gcr ,idr, cg, multigrid" << std::endl;
        std::cerr << "preconditioner name: isai_lower, isai_upper, isai_spd, isai_general, Ilu, ParIlu, ic, jacobi" << std::endl;
        std::exit(-1);
        //grep -oP '\[\K[^\]]+'
    }

    const auto executor_string = std::string(argv[1]);
    const auto solver_name = std::string(argv[2]);
    const auto preconditioner_name = std::string(argv[3]);
    const auto data_folder = std::string(argv[4]);
    const int max_iteration = std::atoi(argv[5]);
    const int block_jacobi_size = std::atoi(argv[6]);


    std::cout << "Using: [" << executor_string << "] as executor [" << solver_name << "] as solver";
    if (solver_name != "multigrid")
    {
        if (preconditioner_name == "jacobi")
        {
            std::cout << " [" << preconditioner_name << "(" << block_jacobi_size << ")] is jacobi";
        }
        else
        {
            std::cout << " [" << preconditioner_name << "]";
        }
    }
    std::cout << std::endl;
    // Figure out where to run the code
    std::map<std::string, std::function<std::shared_ptr<gko::Executor>()>>
        exec_map{
            {"omp", [] { return gko::OmpExecutor::create(); }},
            {"cuda",
             [] {
                 return gko::CudaExecutor::create(0,
                                                  gko::OmpExecutor::create());
             }},
            {"hip",
             [] {
                 return gko::HipExecutor::create(0, gko::OmpExecutor::create());
             }},
            {"dpcpp",
             [] {
                 return gko::DpcppExecutor::create(0,
                                                   gko::OmpExecutor::create());
             }},
            {"reference", [] { return gko::ReferenceExecutor::create(); }}};


    // executor where Ginkgo will perform the computation
    const auto exec = exec_map.at(executor_string)();  // throws if not valid

    // #####################################################
    // Build matrices
    // #####################################################
    auto A = gko::share(gko::read<mtx>(std::ifstream(data_folder + "/A.mtx"), exec));
    auto b = gko::share(gko::read<vec>(std::ifstream(data_folder + "/b.mtx"), exec));
    auto x = gko::share(b->clone());
    // auto x = gko::share(vec::create(exec, b->get_size()));
    x->fill(0.0);
    auto mumps_x = gko::share(gko::read<vec>(std::ifstream(data_folder + "/x.mtx"), exec));

    // gko::write(std::cout, A);



    // #####################################################
    // Preconditioners
    // #####################################################

    std::shared_ptr<gko::LinOp> preconditioner;
    if (preconditioner_name == "isai_lower")
    {
        auto isai_pre_factory = gko::preconditioner::Isai<gko::preconditioner::isai_type::lower, ValueType, IndexType>::build().on(exec);
        preconditioner = gko::share(isai_pre_factory->generate(A));
    } else if (preconditioner_name == "isai_upper")
    {
        auto isai_pre_factory = gko::preconditioner::Isai<gko::preconditioner::isai_type::upper, ValueType, IndexType>::build().on(exec);
        preconditioner = gko::share(isai_pre_factory->generate(A));
    } else if (preconditioner_name == "isai_spd")
    {
        auto isai_pre_factory = gko::preconditioner::Isai<gko::preconditioner::isai_type::spd, ValueType, IndexType>::build().on(exec);
        preconditioner = gko::share(isai_pre_factory->generate(A));
    } else if (preconditioner_name == "isai_general")
    {
        auto isai_pre_factory = gko::preconditioner::Isai<gko::preconditioner::isai_type::general, ValueType, IndexType>::build().on(exec);
        preconditioner = gko::share(isai_pre_factory->generate(A));
    } else if (preconditioner_name == "ic")
    {
        auto ic_pre_factory = gko::preconditioner::Ic<ValueType, IndexType>::build().on(exec);
        preconditioner = gko::share(ic_pre_factory->generate(A));
    } else if (preconditioner_name == "jacobi")
    {
        auto jacobi_pre_factory = gko::preconditioner::Jacobi<ValueType, IndexType>::build().with_max_block_size(block_jacobi_size).on(exec);
        preconditioner = gko::share(jacobi_pre_factory->generate(A));
    } else if (preconditioner_name == "Ilu")
    {
        auto ilu_pre_factory = gko::preconditioner::Ilu<ValueType>::build().on(exec);
        preconditioner = gko::share(ilu_pre_factory->generate(A));
    } else if (preconditioner_name == "ParIlu")
    {
        auto par_ilu_fact =
            gko::factorization::ParIlu<ValueType, IndexType>::build().on(exec);
        // Generate concrete factorization for input matrix
        auto par_ilu = gko::share(par_ilu_fact->generate(A));

        // Generate an ILU preconditioner factory by setting lower and upper
        // triangular solver - in this case the exact triangular solves
        auto ilu_pre_factory =
            gko::preconditioner::Ilu<gko::solver::LowerTrs<ValueType, IndexType>,
                                     gko::solver::UpperTrs<ValueType, IndexType>,
                                     false>::build()
                .on(exec);

        // Use incomplete factors to generate ILU preconditioner
        preconditioner = gko::share(ilu_pre_factory->generate(par_ilu));
    } else
    {
        std::cerr << "Unknown preconditioner type: " << preconditioner_name << std::endl;
        exit(-1);
    }

    // #####################################################
    // Solve system
    // #####################################################

    if (solver_name == "bicg")
        solve_system<gko::solver::Bicg<ValueType> >(exec, A, b, x, preconditioner, max_iteration);
    else if (solver_name == "bicgstab")
        solve_system<gko::solver::Bicgstab<ValueType> >(exec, A, b, x, preconditioner, max_iteration);
    else if (solver_name == "cgs")
        solve_system<gko::solver::Cgs<ValueType> >(exec, A, b, x, preconditioner, max_iteration);
    else if (solver_name == "cbgmres")
        solve_system<gko::solver::CbGmres<ValueType> >(exec, A, b, x, preconditioner, max_iteration);
    else if (solver_name == "gmres")
        solve_system<gko::solver::Gmres<ValueType> >(exec, A, b, x, preconditioner, max_iteration);
    else if (solver_name == "gcr")
        solve_system<gko::solver::Gcr<ValueType> >(exec, A, b, x, preconditioner, max_iteration);
    else if (solver_name == "idr")
        solve_system<gko::solver::Idr<ValueType> >(exec, A, b, x, preconditioner, max_iteration);
    else if (solver_name == "cg")
        solve_system<gko::solver::Cg<ValueType> >(exec, A, b, x, preconditioner, max_iteration);
    else if (solver_name == "multigrid")
        solve_system_multigrid(exec, A, b, x, max_iteration);
    else
    {
        std::cerr << "Unknown solver name: " << solver_name << std::endl;
        exit(-1);
    }


    // Calculate residual
    std::cout << "ginkgo Residual norm sqrt(r^T*r):[" << compute_residual(exec, A, b, x) << "]"<< std::endl;
    std::cout << "mumps Residual norm sqrt(r^T*r):" << compute_residual(exec, A, b, mumps_x) << std::endl;

    // Difference between mumps_sol and ginkgo_sol
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<real_vec>({0.0}, exec);
    // mumps_sol = mumps_sol - ginkgo_sol
    mumps_x->add_scaled(neg_one, x);
    // normal(mumps_sol)
    mumps_x->compute_norm2(res);
    std::cout << "mumps_sol vs ginkgo_sol: " << res->clone(gko::OmpExecutor::create())->at(0) << std::endl;
}
