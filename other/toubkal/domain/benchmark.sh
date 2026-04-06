#!/bin/bash

set -e

result_file="benchmark_results.csv"
mesh_folder="$1"


echo -n '' > $result_file

for ((m=10; m<=300; m+=10)); do

  n=2
  while [ "$n" -le 16384 ]; do



    number_of_cells=$((m*m*m*6))
    result=$(echo "$number_of_cells / $n < 5.0" | bc -l)
    if (( result == 1 )); then
      echo "Breaking loop: $m $n -> $number_of_cells / $n"
      break
    fi

    mesh_name="$mesh_folder""/tetra_test_$m.msh"
    domain_time=$(python3 benchmark.py $n 0 "$mesh_name"| tail -n 1)
    alt_domain_time=$(python3 benchmark.py $n 1 "$mesh_name" | tail -n 1)
    results="$n $number_of_cells $domain_time $alt_domain_time"
    echo "$results"
    echo "$results" >> $result_file


    n=$(( n * 2 ))

  done

done
