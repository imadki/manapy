**MANAPY** is a Python 3 Library for Finite Volume using unstructured grids.

## Requirements
-----

***Python3**:
```bash
sudo apt-get install python3 python3-dev
```

***pip3**:
```bash
sudo apt-get install python3-pip
```

## Installing the library
-----

***Standard mode**:
```bash
python3 -m pip install .
```
   
***Development mode**:
```bash
python3 -m pip install --user -e .
```

## Uninstall
-----
***Whichever the install mode**:
```bash
python3 -m pip uninstall manapy
```


## Running tests
-----
```bash
python3 -m pytest  manapy -m "not parallel"
```


To use Mumps solver

- Install mumps libraries
```sh
sudo apt install libmumps-ptscotch-dev
```

- Install mumps4py from https://github.com/imadki/mumps4py

To use petsc4py solver

https://petsc.org/release/install/