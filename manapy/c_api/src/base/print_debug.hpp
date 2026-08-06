#pragma once

// Project-wide debug tracing, shared by every manapy_compute module (core,
// domain, boundary, solvers, partitioning).
//
// Consolidates what the original manapy c_api had as two separate helpers in
// its utils.cpp -- `print_instant` (formatted message) and `time_it`
// (accumulated / delta timing) -- under one `print_debug` facility:
//
//     print_instant("...")  ->  print_debug("...")
//     time_it("")           ->  print_debug_time_start()
//     time_it("phase")      ->  print_debug_time("phase")
//
// Header-only and free of any per-module state, so a translation unit only has
// to include it -- no CMake change, no library to link.
//
// Everything is off unless one of the environment variables below is set, and
// the MANAPY_PRINT_DEBUG* macros compile the call away entirely when
// MANAPY_DISABLE_PRINT_DEBUG is defined, so release builds pay nothing.
//
//   MANAPY_DEBUG          general switch (preferred)
//   MANAPY_DEBUG_TIMING   legacy name, still honoured
//   MANAPY_TIMING_DEBUG   legacy name, still honoured
//
// Accepted truthy values: 1, true, yes, on, all, rank0 (case-insensitive).
//
// GIL: output is routed through Python's sys.stdout (see print_debug), so
// callers must hold the GIL. Every kernel in this project is called from a
// binding with the GIL held, so that is the normal state; the fallback path
// covers the rest.

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

namespace nb = nanobind;

namespace print_debug_detail {

inline bool env_enabled(const char *value) {
  if (value == nullptr)
    return false;

  std::string text(value);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  return text == "1" || text == "true" || text == "yes" || text == "on" ||
         text == "all" || text == "rank0";
}

// Microseconds since an arbitrary epoch. steady_clock rather than
// gettimeofday: monotonic, so a clock adjustment mid-run cannot produce a
// negative delta.
inline double now_us() {
  using namespace std::chrono;
  return static_cast<double>(
             duration_cast<nanoseconds>(steady_clock::now().time_since_epoch())
                 .count()) /
         1000.0;
}

// `begin` is the epoch the accumulated total is measured from, `start` the
// most recent print_debug_time_start(). Function-local static rather than a
// namespace-scope variable so this header stays free of a .cpp; each extension
// module gets its own timer, which is what you want -- modules are timed
// independently of each other.
struct TimerState {
  double begin = 0.0;
  double start = 0.0;
};

inline TimerState &timer_state() {
  static TimerState state;
  return state;
}

inline std::string format_duration(double time_us) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3);

  if (time_us < 1000.0) {
    // less than 1 ms -> keep in microseconds
    oss << time_us << " us";
  } else if (time_us < 1e6) {
    // less than 1 second -> convert to ms
    oss << (time_us / 1000.0) << " ms";
  } else {
    // otherwise -> convert to seconds
    oss << (time_us / 1e6) << " s";
  }

  return oss.str();
}

} // namespace print_debug_detail

// Whether debug output is switched on. The environment is read once, on the
// first call -- flipping the variable mid-process has no effect, which keeps
// this cheap enough to sit in front of every trace point.
inline bool print_debug_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("MANAPY_DEBUG");
    if (value == nullptr)
      value = std::getenv("MANAPY_DEBUG_TIMING");
    if (value == nullptr)
      value = std::getenv("MANAPY_TIMING_DEBUG");
    return print_debug_detail::env_enabled(value);
  }();
  return enabled;
}

// printf-style debug line. Prefer the MANAPY_PRINT_DEBUG macro, which also
// skips evaluating the arguments when tracing is off.
inline void print_debug(const char *fmt, ...) {
  char buffer[1024];
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(buffer, sizeof(buffer), fmt, args);
  va_end(args);

  const std::string line = "C\t[Rank 0]: " + std::string(buffer);

  // Route through Python's sys.stdout, not C stdout, so tracing interleaves
  // correctly with the caller's own print() output -- including when stdout
  // has been redirected, as it is under Jupyter and under most MPI launchers.
  //
  // The "[Rank 0]" prefix is hardcoded, exactly as in the original c_api: this
  // layer has no notion of MPI rank. (The env var accepting "all"/"rank0"
  // anticipated per-rank filtering that was never implemented.)
  try {
    nb::object out = nb::module_::import_("sys").attr("stdout");
    out.attr("write")(line);
    out.attr("flush")();
  } catch (...) {
    // A debug printer must never turn a working computation into a failing
    // one, so any Python-side problem (no interpreter, stdout replaced by
    // something without write/flush, ...) falls back to C stdout.
    std::fputs(line.c_str(), stdout);
    std::fflush(stdout);
  }
}

// Start (or restart) the delta timer. The first call also fixes the epoch that
// the accumulated total is measured from.
inline void print_debug_time_start() {
  print_debug_detail::timer_state().start = print_debug_detail::now_us();
  if (print_debug_detail::timer_state().begin == 0.0)
    print_debug_detail::timer_state().begin =
        print_debug_detail::timer_state().start;
}

// Report time spent since the last print_debug_time_start() (delta) and since
// the very first one (acc).
inline void print_debug_time(const std::string &msg) {
  const double end = print_debug_detail::now_us();
  print_debug("%s: acc=%s delta=%s\n", msg.c_str(),
              print_debug_detail::format_duration(
                  end - print_debug_detail::timer_state().begin)
                  .c_str(),
              print_debug_detail::format_duration(
                  end - print_debug_detail::timer_state().start)
                  .c_str());
}

// Trace points. Use these rather than the functions directly: when tracing is
// off the arguments are never evaluated, and when MANAPY_DISABLE_PRINT_DEBUG
// is defined the whole statement disappears.
#if !defined(MANAPY_DISABLE_PRINT_DEBUG)
#define MANAPY_PRINT_DEBUG(fmt, ...)                                           \
  do {                                                                         \
    if (print_debug_enabled())                                                 \
      print_debug((fmt), ##__VA_ARGS__);                                       \
  } while (0)
#define MANAPY_PRINT_DEBUG_TIME_START()                                        \
  do {                                                                         \
    if (print_debug_enabled())                                                 \
      print_debug_time_start();                                                \
  } while (0)
#define MANAPY_PRINT_DEBUG_TIME(msg)                                           \
  do {                                                                         \
    if (print_debug_enabled())                                                 \
      print_debug_time((msg));                                                 \
  } while (0)
#else
#define MANAPY_PRINT_DEBUG(fmt, ...) ((void)0)
#define MANAPY_PRINT_DEBUG_TIME_START() ((void)0)
#define MANAPY_PRINT_DEBUG_TIME(msg) ((void)0)
#endif
