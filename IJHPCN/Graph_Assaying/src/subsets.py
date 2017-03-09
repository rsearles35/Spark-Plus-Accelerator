#!/usr/bin/env python

import sys
import itertools
import numpy as np
import argparse
from functools import partial
import math
from collections import Counter

# Spark
from pyspark import SparkContext, SparkConf

# PyCUDA stuff
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context

# Number of partitions for Spark
NUM_PARTITIONS = 1024

# Flag to use gpu or not
USE_GPU = 0

def binom(n, k):        # Compute binomial coefficients.
    ntok = 1            # Code borrowed from: http://stackoverflow.com/a/3025547/4187033
    ktok = 1            
    for t in xrange(1, min(k, n - k) + 1):
        ntok *= n
        ktok *= t
        n -= 1
    return ntok / ktok


# This function turns a 0-1 matrix into a bit string reading across the rows first
# It reads in reverse order (i.e. the first bit read is the least significant bit).
# To illustrate:
#
#                        | 1 2 3 |
# the positions of A are | 4 5 6 | and the bit-string formed is (9 8 7 6 5 4 3 2 1) where the integers
#                        | 7 8 9 |
# 
# 9, 8, 7, ..., 1 are replaced by the bit (0 or 1) that appears in A in the corresponding position.

def examine_subsets_cuda(task, A, N, K, threads_per_block):
    # Unpack the task tuple
    subset_start, subset_end = task
    
    # Create CUDA context
    cuda.init()
    ctx = make_default_context()

    KFAC = math.factorial(K)

    # Keep track of the stride
    stride = subset_end - subset_start

    # Copy A to the GPU
    device_A = cuda.to_device(A.astype(np.int16))

    # Create results array
    results = np.zeros(stride, dtype=np.int32)

    # Copy results array
    device_results = cuda.to_device(results)

    # Number of CUDA blocks
    cuda_blocks = ((stride + threads_per_block - 1) / threads_per_block)
    
    # Compile CUDA kernel
    mod = SourceModule(open("/Users/rsearles/Documents/Repositories/cisc849-16s/project/src/Spike_Neural_Nets/subsets.cu", "r").read())
    kernel = mod.get_function("examine_subsets")
    
    # Run kernel
    kernel(np.int32(N), np.int32(K), np.int64(subset_start), np.int64(subset_end), np.int32(KFAC), device_A, device_results, block=(cuda_blocks, 1, 1), grid=(threads_per_block, 1))

    # Copy results back
    results = cuda.from_device_like(device_results, results)

    # Free GPU memory
    device_results.free()
    device_A.free()

    # Pop CUDA context (driver yells otherwise)
    ctx.pop()
    
    # Return the counts
    counts = Counter(results)
    return counts

def permute_and_score(permutations, K, Local_A):
    # Initialize to the value seen when the matrix is all 1's.
    smallest_seen = 2**(K*K)-1

    for perm in permutations:
        score = 0
        # GPU PARALLELIZE HERE
        for i in xrange(K):
            for j in xrange(K):
                if Local_A[perm[i], perm[j]] == 1:
                    score += 2**(i*K+j)

        smallest_seen = min(smallest_seen, score)

    return smallest_seen

def MapNeuralNet(subset, A, K):
    # From A, extract the local adjacency matrix:
    # This corresponds to selecting the rows and columns of A with indices in subset
    Local_A = np.zeros( (K, K), dtype=np.int32 )
    for i in xrange(K):
        for j in xrange(K):
            Local_A[i, j] = A.value[subset[i], subset[j]]

    # Create permutations
    permutations = np.asarray(list(itertools.permutations( xrange(K) )), dtype=np.int32)

    # Apply permutations to the names of the vertices in Local_A and permute the matrix accordingly looking for a "least" example.
    smallest_seen = permute_and_score(permutations, K, Local_A)

    return (smallest_seen, 1)

def cuda_map_partitions(tasks, A, N, K, threads_per_block):
    tasks = list(tasks)

    # Merge contiguous tasks
    new_task_list = [tasks[0]]
    for task in tasks[1:]:
        previous_start, previous_end = new_task_list[-1]
        if task[0] == previous_end:
            new_task_list[-1] = (previous_start, task[1])
        else:
            new_task_list.append(task)

    # Launch Kernel
    results = [examine_subsets_cuda(task, A, N, K, threads_per_block) for task in new_task_list]

    # Combine the counts
    counts = Counter()
    for res in results:
        counts += res

    # Return the results of this partition
    return [counts]

def main():
    # Read in adjacency matrix
    A = np.load(args.adjacency_matrix)
    
    N = A.shape[0]  # The number of vertices in the graph
    K = args.k_subsets  # The size of the subsets to examine
    
    # Create Spark context
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    if USE_GPU:
        # Calculate number of subsets and set number of thread blocks for CUDA
        number_of_subsets = binom(N,K)
        threads_per_block = args.threads_per_block

        # Generate starting indices for tuples
        stride = args.stride
        starting_indices = range(0, number_of_subsets, stride)
        ending_indices = [min(starting_index + stride, number_of_subsets) for starting_index in starting_indices]
        tasks = zip(starting_indices, ending_indices)

        # If they aren't evenly divisible, we're bad
        assert len(starting_indices) == len(tasks) and len(ending_indices) == len(tasks)

        # Create a Spark RDD of tasks
        tasks_RDD = sc.parallelize(tasks, NUM_PARTITIONS)
        
        # Run CUDA kernel
#        results = tasks_RDD.map(partial(examine_subsets_cuda, A=A, N=N, K=K, threads_per_block=threads_per_block)).collect()
        results = tasks_RDD.mapPartitions(partial(cuda_map_partitions, A=A, N=N, K=K, threads_per_block=threads_per_block)).collect()

        # Combine the counts
        counts = Counter()
        for res in results:
            counts += res

        # Print results
        for key, value in counts.items():
            print key, "\t", value
    else:
        # Create and broadcast the subsets
        subsets = list(itertools.combinations( xrange(N), K ))
        print "subsets: ", len(subsets)
        subsets_rdd = sc.parallelize(subsets, NUM_PARTITIONS)

        # Broadcast A to all nodes in the cluster
        broadcast_A = sc.broadcast(A)

        # Compute scores and count results
        sub_scores = subsets_rdd.map(partial(MapNeuralNet, A=broadcast_A, K=K)).countByKey()

        # Print the results
        for key, value in sub_scores.items():
            print key, "\t", value

if __name__ == "__main__":
    # Initialize argparse
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("adjacency_matrix", help="input matrix/spike neural net topology", type=str)
    parser.add_argument("-k", "--k-subsets", help="size of subsets to examine", type=int, default=2)
    parser.add_argument("-p", "--partitions", help="number of Spark partitions to use", type=int, default=1024)
    parser.add_argument("-g", "--gpu", help="If present, use the GPU for comparisons", action='store_true')
    parser.add_argument("-t", "--threads-per-block", help="number of threads per CUDA block", type=int, default=1024)
    parser.add_argument("-s", "--stride", help="minimum number of subsets per GPU task", type=int, default=128)

    # Parse the args
    args = parser.parse_args()

    if args.gpu:
        USE_GPU = 1

    NUM_PARTITIONS = args.partitions

    # Run
    main()
