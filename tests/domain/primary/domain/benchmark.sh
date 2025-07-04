#!/bin/bash

set -e

result_file="benchmark_results.csv"

n=2
echo -n '' > $result_file

for ((m=10; m<=300; m+=10)); do

  while [ "$n" -le 16384 ]; do

    mesh_name="./meshes/""tetra_test_$m.msh"
    domain_time=$(python3 benchmark.py $n 0 "$mesh_name"| tail -n 1)
    alt_domain_time=$(python3 benchmark.py $n 1 "$mesh_name" | tail -n 1)
    results="$n $((m*m*m*6)) $domain_time $alt_domain_time"
    echo "$results"
    echo "$results" >> $result_file



  done

  n=$(( n * 2 ))
done
