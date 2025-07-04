#!/bin/bash

set -e

result_file="tetra_test"



mkdirp -p meshes
for ((m=10; m<=300; m+=10)); do

  echo $m
  gmsh ../mesh/tetra_test_2.geo -3 -setnumber Nx $m -setnumber Ny $m -setnumber Nz $m  -o ./meshes/"$result_file""_$m.msh" > /dev/null

done


