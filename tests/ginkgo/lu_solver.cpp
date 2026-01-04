#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <random>

#include <ginkgo/ginkgo.hpp>

template <typename MatrixPtrType>
void    save_matrix(const std::string &name, MatrixPtrType&& matrix)
{
    std::ofstream A_file(name);
    gko::write(A_file, matrix);
}

int main(int argc, char* argv[])
{
    // Some shortcuts
    using RealValueType = double;
    using IndexType = int;

    using dense_t = gko::matrix::Dense<RealValueType>;
    using csr_t = gko::matrix::Csr<RealValueType, IndexType>;
    using coo_t = gko::matrix::Coo<RealValueType, IndexType>;

    // Print version information
    std::cout << gko::version_info::get() << std::endl;

    if (argc == 2 && (std::string(argv[1]) == "--help")) {
        std::cerr << "Usage: " << argv[0] << " [executor]" << std::endl;
        std::exit(-1);
    }

    const auto executor_string = argc >= 2 ? argv[1] : "reference";
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
    const auto cpu_exec = exec_map.at("reference")();

    // Generate data
    constexpr IndexType N = 10;
    constexpr IndexType nnz = 10;
    std::default_random_engine rng(1234);
    std::uniform_int_distribution<IndexType> rand_row(0, N-1);
    std::uniform_int_distribution<IndexType> rand_col(0, N-1);
    std::uniform_real_distribution<RealValueType> rand_val(1.0, 100.0);

    auto A_cpu = coo_t::create(cpu_exec, gko::dim<2>{N, N}, nnz);
    for (IndexType i = 0; i < nnz; ++i)
    {

        A_cpu->get_row_idxs()[i] = i;
        A_cpu->get_col_idxs()[i] = i;
        A_cpu->get_values()[i] = rand_val(rng);

    }
    //Generate b and x
    auto b_cpu = dense_t::create(cpu_exec, gko::dim<2>(N, 1));
    auto x_cpu = dense_t::create(cpu_exec, gko::dim<2>(N, 1));
    for (IndexType i = 0; i < N; ++i)
    {
        b_cpu->at(i) = 1;
        x_cpu->at(i) = 0.0;
    }

    auto A_cpu_csr = gko::share(csr_t::create(exec));
    A_cpu->convert_to(A_cpu_csr);
    auto A_csr = gko::share(A_cpu_csr->clone(exec));
    auto b = b_cpu->clone(exec);
    auto x = x_cpu->clone(exec);

    // auto A_csr = gko::share(gko::read<csr_t>(std::ifstream("A.mtx"), exec));
    save_matrix("A.mtx", A_csr);
    save_matrix("b.mtx", b);
    save_matrix("x0.mtx", x);

    //ILU factory
    auto ilu_factory = gko::share(gko::preconditioner::Ilu<RealValueType>::build().on(exec));

    //Solver
    auto solver = gko::solver::Gmres<RealValueType>::build()
        .with_criteria(gko::stop::Iteration::build().with_max_iters(1000u),
            gko::stop::ResidualNorm<RealValueType>::build().with_reduction_factor(1e-7))
        .with_preconditioner(ilu_factory)
        .on(exec);

    std::chrono::time_point<std::chrono::steady_clock> t_tic, t_tac;

    exec->synchronize();
    t_tic = std::chrono::steady_clock::now();
    auto gmres = solver->generate(A_csr);
    exec->synchronize();
    t_tac = std::chrono::steady_clock::now();
    auto generate_time = std::chrono::duration_cast<std::chrono::microseconds>(t_tac - t_tic);
    std::cout << "Generate time: " << generate_time.count() << " microseconds" << std::endl;

    //Solve the system
    exec->synchronize();
    t_tic = std::chrono::steady_clock::now();
    gmres->apply(b, x);
    exec->synchronize();
    t_tac = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_tac - t_tic);


    // Print solution
    std::cout << ">> Solution (x): (" << duration.count() << " microseconds)" << std::endl;
    //gko::write(std::cout, x);

    // Print Residual
    // Compute r = A * x - b
    auto r = dense_t::create(exec, b->get_size());
    A_csr->apply(x, r); // r = Ax
    r->scale(gko::initialize<dense_t>({-1.0}, exec)); // r = -Ax
    r->add_scaled(gko::initialize<dense_t>({1.0}, exec), b); // r = b - Ax

    std::cout << ">> Ax-b (r):\n";
    // gko::write(std::cout, r);


    std::cout << ">> Residual Norm:\n";
    auto norm = gko::initialize<dense_t>({0.0}, exec);
    r->compute_norm2(norm);
    gko::write(std::cout, norm);
}