#include <pybind11/pybind11.h>
#include "../app.hpp"

namespace py = pybind11;

PYBIND11_MODULE(manapyGUI, m, py::mod_gil_not_used())
{
    py::class_<App>(m, "App").def(py::init<>()).def("run", &App::run);
}
