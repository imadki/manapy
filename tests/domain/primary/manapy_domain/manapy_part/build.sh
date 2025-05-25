#python -m pip install numpy           # ensure NumPy headers
gcc -O2 -shared -fPIC \
    $(python3 -m pybind11 --includes) \  # or `python3 -m numpy --cflags`
    -I/usr/include/metis -lmetis \
    manapy_domain.c \
    -o manapy_domain$(python3-config --extension-suffix)
