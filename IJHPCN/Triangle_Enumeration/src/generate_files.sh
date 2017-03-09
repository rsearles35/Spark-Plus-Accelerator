#!/bin/bash

for graph_size in $(seq 1000 1000 10000); do
    for probability in $(seq 0.001 0.001 0.009) $(seq .01 .01 .05); do
        fn_prob=$(echo $probability | sed -r 's/0.([0-9]+)/\1/')
        filename="../../data/triangle_enum/N${graph_size}_P${fn_prob}.txt"
        ./random_graph $filename $graph_size $probability && bzip2 -zf $filename &
    done
done

wait
