#!/bin/bash

set -e

result_file="benchmark_results.csv"
mesh_folder="$1"


echo -n '' > $result_file

for ((m=10; m<=300; m+=10)); do





    number_of_cells=$((m*m*m*6))


    mesh_name="$mesh_folder""/tetra_test_$m.msh"
    domain_time=$(python3 benchmark_new_domain.py "$mesh_name" "$number_of_cells" | grep "=99>")
    echo "$domain_time"
    echo "$domain_time" >> $result_file






done
