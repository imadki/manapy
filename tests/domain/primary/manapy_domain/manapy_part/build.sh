python -m pip install numpy           # ensure NumPy headers

# Need Cmake to build

# Install GKlib
git clone https://github.com/KarypisLab/GKlib.git
cd GKlib
make config cc=gcc prefix=~/local
make install
cd ..

# Install METIS
git clone https://github.com/KarypisLab/METIS.git
cd METIS
make config cc=gcc prefix=~/local
make install
cd ..

# Install manapy_domain lib
python3 -m pip install .

#gcc -O2 -shared -fPIC \
#    $(python3 -m pybind11 --includes) \  # or `python3 -m numpy --cflags`
#    -I/usr/include/metis -lmetis \
#    manapy_domain.c \
#    -o manapy_domain$(python3-config --extension-suffix)
