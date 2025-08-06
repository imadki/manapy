#include "manapy_part.h"



# define PRINT_DEBUG false
void print_instant(const char *fmt, ...) {
#ifdef PRINT_DEBUG
    char buffer[1024];  // temp string buffer
    va_list args;
    va_start(args, fmt);
    vsnprintf(buffer, sizeof(buffer), fmt, args);  // format the string
    va_end(args);

    // Import sys module and write to sys.stdout
    PyObject *sys = PyImport_ImportModule("sys");
    if (!sys) return;

    PyObject *stdout = PyObject_GetAttrString(sys, "stdout");
    if (stdout) {
        const std::string str = "C\t[Rank 0]: " + std::string(buffer);
        PyObject *write_result = PyObject_CallMethod(stdout, "write", "s", str.c_str());
        Py_XDECREF(write_result);

        PyObject *flush_result = PyObject_CallMethod(stdout, "flush", NULL);
        Py_XDECREF(flush_result);

        Py_DECREF(stdout);
    }

    Py_DECREF(sys);
#endif
}