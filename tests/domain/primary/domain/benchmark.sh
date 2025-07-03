#!/bin/bash

set -e

result_file="benchmark_results.csv"

n=2
echo -n '' > $result_file

while [ "$n" -le 4 ]; do

  for ((m=10; m<=20; m+=10)); do


    gmsh ../mesh/tetra_test_2.geo -3 -setnumber Nx $m -setnumber Ny $m -setnumber Nz $m  -o tetra_test.msh > /dev/null
    domain_time=$(python3 benchmark.py $n 0 | tail -n 1)
    alt_domain_time=$(python3 benchmark.py $n 1 | tail -n 1)
    results="$n $((m*m*m*6)) $domain_time $alt_domain_time"
    echo "$results"
    echo "$results" >> $result_file



  done

  n=$(( n * 2 ))
done
