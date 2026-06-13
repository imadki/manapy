#include <iomanip>
#include <sys/time.h>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cctype>

#include "manapy_part.h"

double	get_time(void)
{
    struct timeval	tv;

    gettimeofday(&tv, NULL);
    return (((tv.tv_sec * 1000000.0) + ((double)tv.tv_usec / 1.0)));
}



static bool env_enabled(const char *value) {
    if (value == nullptr) {
        return false;
    }

    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return text == "1" || text == "true" || text == "yes" || text == "on" || text == "all" || text == "rank0";
}

bool manapy_debug_timing_enabled() {
    const char *value = std::getenv("MANAPY_DEBUG_TIMING");
    if (value == nullptr) {
        value = std::getenv("MANAPY_TIMING_DEBUG");
    }
    return env_enabled(value);
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

idx_t binary_search(const PyArray<idx_t, 1> &arr, const idx_t item) {
    const idx_t size = arr.last();
    idx_t left = 0;
    idx_t right = size - 1;
    while (left <= right) {
        const idx_t mid = (left + right) / 2;
        const idx_t mid_val = arr.get(mid);
        if (mid_val == item) {
            return mid;
        } else if (mid_val < item) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}

void intersect_arr(PyArray<idx_t, 2> *arr, PyArray<idx_t, 1> *indices, const idx_t size, std::vector<idx_t> &intersect_arr) {
    idx_t counter = 0;

    intersect_arr[0] = -1;
    intersect_arr[1] = -1;

    auto arr1 = arr->sub_array(indices->get(0));
    for (idx_t i = 0; i < arr1.last(); i++) {
        intersect_arr[counter] = arr1.get(i);
        for (idx_t j = 1; j < size; j++) {
            auto arr2 = arr->sub_array(indices->get(j));
            if (binary_search(arr2, arr1.get(i)) == -1){
                intersect_arr[counter] = -1;
                break;
            }
        }
        if (intersect_arr[counter] != -1)
            counter++;
        if (counter >= 2)
            break;
    }
}


std::array<idx_t, 3> get_max_info(const idx_t cell_type) {
    if (cell_type == CELL_TYPE::Triangle) {
        return {3, 2, 3};
    }
    else if (cell_type == CELL_TYPE::Quad) {
        return {4, 2, 4};
    }
    else if (cell_type == CELL_TYPE::Tetra) {
        return {4, 3, 4};
    }
    else if (cell_type == CELL_TYPE::Hexahedron) {
        return {6, 4, 8};
    }
    else if (cell_type == CELL_TYPE::Pyramid) {
        return {5, 4, 5};
    }
    return {0, 0, 0};
}


void print_instant(const char *fmt, ...) {
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
