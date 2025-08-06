#!/bin/bash

set -e

mkdir -p bench_folder
result_file="bench_folder/benchmark_results"
mesh_folder="$1"


echo -n '' > $result_file

for ((m=10; m<=300; m+=30)); do





    number_of_cells=$((m*m*m*6))


    mesh_name="$mesh_folder""/tetra_test_$m.msh"
    domain_time=$(python3 benchmark_new_domain.py "$mesh_name")
    echo "$domain_time"
    echo "$domain_time" >> "${result_file}_${m}"






done
