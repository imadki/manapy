#include <iomanip>


#include "manapy_part.h"

double	get_time(void)
{
    struct timeval	tv;

    gettimeofday(&tv, NULL);
    return (((tv.tv_sec * 1000000.0) + ((double)tv.tv_usec / 1.0)));
}



static std::string get_time_as_string(double time) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);

    if (time < 1000.0) {
        // less than 1 ms → keep in µs
        oss << time << " µs";
    } else if (time < 1e6) {
        // less than 1 second → convert to ms
        oss << (time / 1000.0) << " ms";
    } else {
        // otherwise → convert to seconds
        oss << (time / 1e6) << " s";
    }

    return oss.str();
}

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

void time_it(const std::string &msg) {
    static double begin = 0.0;
    static double start = 0.0;

    if (msg.empty()) {
        start = get_time();
        if (begin == 0.0) {
            begin = start;
        }
    } else {
        const double end = get_time();
        print_instant("%s: acc=%s delta=%s\n", msg.c_str(), get_time_as_string(end - begin).c_str(), get_time_as_string(end - start).c_str());
    }
}
