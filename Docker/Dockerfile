FROM amd64/python:3.8

RUN apt upgrade -y && apt update

RUN apt -y install vim -y && \
    apt -y install cython3 -y && \
    apt install openmpi-bin openmpi-common libopenmpi-dev -y && \
    apt install libopenmpi-dev -y && \
    apt-get install -y \
        wget \
        curl \
        build-essential \
        lsb-release \
        python3.8-dev && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install mpi4py && \
    python3 -m pip install cython && \
    python3 -m pip install pythran && \
    python3 -m pip install pytest

COPY manapy /manapy
COPY ./manapy/Docker/tests.sh /

RUN chmod +x /tests.sh

RUN echo "[mpi]\n\
mpicc = /usr/bin/mpicc\n\
mpicxx = /usr/bin/mpicxx" > "`python3 -c "import mpi4py; from os.path import join, dirname;print(dirname(mpi4py.__spec__.origin))"`/mpi.cfg"

WORKDIR /manapy

RUN python3 -m pip install .

CMD ["/bin/bash", "/tests.sh"]
