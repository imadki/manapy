#include "bindings.hpp"
#include "application.hpp"

PYBIND11_MODULE(manapyGUI, m, py::mod_gil_not_used())
{
    py::class_<Application>(m, "Application").def(py::init<>()).def("run", &Application::run);
}
