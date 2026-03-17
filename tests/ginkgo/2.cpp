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