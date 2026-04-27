FROM quay.io/pypa/manylinux_2_28_x86_64

ENV PYTHON_ROOT=/opt/python/cp314-cp314 \
    PROJECT_ROOT=/workspace/manapy \
    PETSC_CONFIGURE_OPTIONS="--with-debugging=0 --download-fblaslapack --with-shared-libraries=1"

ENV PATH="${PYTHON_ROOT}/bin:/usr/lib64/openmpi/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH}" \
    PKG_CONFIG_PATH="/usr/lib64/openmpi/lib/pkgconfig:${PKG_CONFIG_PATH}"

#   find /usr -name "lib*mumps*.so" 2> /dev/null
#   find /usr -name "dmumps_c*.h" 2> /dev/null

ENV OMPI_MCA_mtl=^ofi \
    OMPI_MCA_btl=tcp,self \
    MUMPS_INC=/usr/include/openmpi-x86_64 \
    MUMPS_LIB=/usr/lib64/openmpi/lib \
    MUMPS_SOLVERS=dmumps,cmumps,zmumps,smumps

RUN dnf -y update && \
    dnf -y install \
        blas-devel \
        bzip2 \
        cmake \
        curl \
        findutils \
        gcc \
        gcc-c++ \
        gcc-gfortran \
        git \
        gmp-devel \
        lapack-devel \
        make && \
    dnf -y install \
        metis-devel \
        MUMPS-openmpi \
        MUMPS-openmpi-devel \
        openmpi \
        openmpi-devel \
        patch \
        suitesparse-devel \
        swig && \
    dnf -y install pkgconf-pkg-config \
        scalapack-openmpi \
        scalapack-openmpi-devel \
        tar \
        vim \
        wget \
        which \
        xz && \
        dnf clean all && \
        rm -rf /var/cache/dnf

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install cython mpi4py pythran pytest && \
    python -m pip install petsc petsc4py

RUN git clone https://github.com/imadki/mumps4py.git && \
    cd mumps4py && \
    python -m pip install .

RUN cd /tmp && \
    curl -LO https://www.mpfr.org/mpfr-4.2.1/mpfr-4.2.1.tar.gz && \
    tar -xzf mpfr-4.2.1.tar.gz && \
    cd mpfr-4.2.1 && \
    ./configure --prefix=/usr/local && \
    make -j 12 && \
    make install && \
    rm -rf /tmp/mpfr-4.2.1 /tmp/mpfr-4.2.1.tar.gz

RUN cd /tmp && \
    git clone --branch stable https://github.com/DrTimothyAldenDavis/SuiteSparse.git && \
    cd SuiteSparse && \
    make -j 12 && \
    make install && \
    python -m pip install scipy scikit-umfpack && \
    rm -rf /tmp/SuiteSparse

RUN useradd -m manapy
USER manapy
# COPY . ${PROJECT_ROOT}
WORKDIR ${PROJECT_ROOT}
VOLUME ["/workspace/manapy"]


CMD ["/bin/bash"]
