squeue -u $USER
#cat benchmark_results.csv | tail
#cat out.txt | tail
#cat gmsh_output.log | tail
#ls logs
cat logs/out_rank_0.log | tail -n 20
ls logs -lah | awk '$7 > 0' | grep "err_rank" | tail -n 20
#cat logs/out_0.log | tail -n 20
