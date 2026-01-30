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

double compute_residual(const std::shared_ptr<gko::Executor>& exec, const std::shared_ptr<mtx>& A, const std::shared_ptr<vec> b, const std::shared_ptr<vec> x)
{
    auto one = gko::initialize<vec>({1.0}, exec);
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<vec>({0.0}, exec);
    auto b_clone = b->clone();
    A->apply(one, x, neg_one, b_clone);
    b_clone->compute_norm2(res);
    return res->clone(gko::OmpExecutor::create())->at(0);
}

template <typename Solver>
void solve_system(const std::shared_ptr<gko::Executor>& exec, const std::shared_ptr<mtx>& A, const std::shared_ptr<vec> &b, const std::shared_ptr<vec> &x)
{
    // ------------------------------------------------------------------
    // 1.
    // Preconditioners (generate par_ilu, jacobi and ic preconditioners)
    // ------------------------------------------------------------------

    // #################################
    // ### Generate incomplete factors using ParILU
    // #################################
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
    auto ilu_preconditioner = gko::share(ilu_pre_factory->generate(par_ilu));

    // #################################
    // #### jacobi
    // #################################
    auto jacobi_pre_factory = gko::preconditioner::Jacobi<ValueType, IndexType>::build()
    .with_max_block_size(4)
    .on(exec);
    auto jacobi_preconditioner = gko::share(jacobi_pre_factory->generate(A));


    // #################################
    // #### Ic
    // #################################
    auto ic_pre_factory = gko::preconditioner::Ic<ValueType, IndexType>::build().on(exec);
    auto ic_preconditioner = gko::share(ic_pre_factory->generate(A));


    // ------------------------------------------------------------------
    // 2.
    // Create the solver factory
    // Specifying the preconditioner, iteration_criteria and residual_criteria
    // ------------------------------------------------------------------
    // Criteria
    constexpr ValueType reduction_factor{1e-15};
    auto iteration_criteria = gko::stop::Iteration::build().with_max_iters(10000u);
    auto residual_criteria = gko::stop::ResidualNorm<ValueType>::build().with_reduction_factor(reduction_factor);

    std::shared_ptr<gko::LinOpFactory> solver_factory = Solver::build()
    .with_criteria(iteration_criteria, residual_criteria)
    .with_generated_preconditioner(ilu_preconditioner)
    .on(exec);



    // ------------------------------------------------------------------
    // 3.
    // Generate the solver for a specific target system
    // ------------------------------------------------------------------
    const auto solver = solver_factory->generate(A);




    // ------------------------------------------------------------------
    // 4.
    // Solve the system
    // Calculate Time, Number of iterations
    // ------------------------------------------------------------------
    auto logger = gko::share(gko::log::Convergence<>::create());
    solver->add_logger(logger);

    exec->synchronize();
    auto time_start = std::chrono::steady_clock::now();
    solver->apply(b, x);
    exec->synchronize();
    auto time_end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    std::cout << "Iterations: " << logger->get_num_iterations() << "\n";
    std::cout << "Time: " << duration.count() << " microseconds" << "\n";

}

int main(int argc, char* argv[])
{
    std::cout << std::scientific << std::setprecision(15);
    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << " [solver name]" << " [data folder containing A.mtx,b.mtx,x_sol.mtx]" << std::endl;
        std::cerr << "executor: omp, cuda, hip, dpcpp, reference" << std::endl;
        std::cerr << "sover name: bicg ,bicgstab ,cgs ,cbgmres ,gmres ,gcr ,idr, cg" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = std::string(argv[1]);
    const auto solver_name = std::string(argv[2]);
    const auto data_folder = std::string(argv[3]);
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
    auto x = gko::share(vec::create(exec, b->get_size()));
    x->fill(0.0);
    // the solution provided using mumps
    auto mumps_x = gko::share(gko::read<vec>(std::ifstream(data_folder + "/x_sol.mtx"), exec));


    // #####################################################
    // Solve system
    // #####################################################
    if (solver_name == "bicg")
        solve_system<gko::solver::Bicg<ValueType> >(exec, A, b, x);
    else if (solver_name == "bicgstab")
        solve_system<gko::solver::Bicgstab<ValueType> >(exec, A, b, x);
    else if (solver_name == "cgs")
        solve_system<gko::solver::Cgs<ValueType> >(exec, A, b, x);
    else if (solver_name == "cbgmres")
        solve_system<gko::solver::CbGmres<ValueType> >(exec, A, b, x);
    else if (solver_name == "gmres")
        solve_system<gko::solver::Gmres<ValueType> >(exec, A, b, x);
    else if (solver_name == "gcr")
        solve_system<gko::solver::Gcr<ValueType> >(exec, A, b, x);
    else if (solver_name == "idr")
        solve_system<gko::solver::Idr<ValueType> >(exec, A, b, x);
    else if (solver_name == "cg")
        solve_system<gko::solver::Cg<ValueType> >(exec, A, b, x);
    else
    {
        std::cerr << "Unknown solver name: " << solver_name << std::endl;
        exit(-1);
    }


    // Calculate residual
    std::cout << "ginkgo Residual norm sqrt(r^T*r):" << compute_residual(exec, A, b, x) << std::endl;
    std::cout << "mumps Residual norm sqrt(r^T*r):" << compute_residual(exec, A, b, mumps_x) << std::endl;

    // Difference between mumps_sol and ginkgo_sol
    auto neg_one = gko::initialize<vec>({-1.0}, exec);
    auto res = gko::initialize<vec>({0.0}, exec);
    // mumps_sol = mumps_sol - ginkgo_sol
    mumps_x->add_scaled(neg_one, x);
    // normal(mumps_sol)
    mumps_x->compute_norm2(res);
    std::cout << "mumps_sol vs ginkgo_sol: " << res->clone(gko::OmpExecutor::create())->at(0) << std::endl;
}
