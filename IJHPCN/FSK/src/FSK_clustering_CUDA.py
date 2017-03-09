#!/usr/bin/env python

'''
 Robert Searles
 Department of Computer and Information Sciences
 University of Delaware

 FSK_clustering.py Python implementation of the Fast Subtree
 Kernel. This version uses kmeans clustering to extract meaningful
 points from large amounts of data to build the similarity matrix.
'''

#################################################################
###    Includes
#################################################################

# Needed for system operations
import sys, os, shutil

# Parsing command line arguments
import argparse

# Configuration stuff
import json, socket

# Math functions
import math

# CSV reader
import csv

# Itertools for building graph comparison combinations
import itertools

# Numpy, because Numpy
import numpy as np

# PyCUDA for launching CUDA kernels
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context
from skcuda import misc
import pycuda.gpuarray as gpuarray

# Functools partial
from functools import partial

# Spark
from pyspark import SparkContext, SparkConf

# K-Means
from pyspark.mllib.clustering import KMeans

# Timing and tracebacks
import time, traceback

# Similarity Threshold
DELTA = 0.05

# Use Spark?
SPARK = True

# Which device are we running on
USE_GPU = 0

# CUDA Block size
CUDA_BLOCK_SIZE = 1024

# Number of partitions for spark
NUM_PARTITIONS = 1024

# File extension for encoded subtrees
FILE_SUFFIX = ".npy"

# Number of clusters for K-Means clustering
NUM_CLUSTERS = 10

# Number of closest points we want the extract from each centroid
CENTROID_POINTS = 1

# Minimum number of nodes in a graph in order for that graph to be considered
# Graphs with less nodes will be filtered out
NODE_FILTER = 5

#################################################################
###    Class/Function Declarations
#################################################################

def traceback_exception(e):
    '''
     @param e: exception that we want to trace

     Prints out the exception traceback.
    '''
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

def BuildFileList(file_list):
    '''
     @param file_list: text file containing the list of call graphs we would like to compare

     Builds a list of file names from the specified text file. We will use this list to iterate over and load the appropriate files during each pairwise comparison
    '''
    output = list()
    with open(file_list, 'r') as files:
        reader = csv.reader(files, delimiter=',')
        for line in reader:
             output.append(line[0])
    return output

def LoadGraph(file_hash, prefix):
    '''
     @param file_name: file name of the graph we would like to load
     @param prefix: file path prefix (this allows us to build an absolute path)

     Loads the specified graph into memory
    '''
    encoded_trees = np.load(prefix + os.sep + file_hash + FILE_SUFFIX)
    return np.array(encoded_trees, dtype=np.float32)

def FilterFileList(file_list, min_nodes, prefix):
    '''
     @param file_list: original list of files (contains all encoded graphs in dataset)
     @param min_nodes: minimum size of graph (number of nodes/subtrees)
     @param prefix: file path prefix (this allows us to build an absolute path)
    '''
    output = list()
    for graph in file_list:
        curr_graph = np.load(prefix + os.sep + graph + FILE_SUFFIX, mmap_mode='r')
        if curr_graph.shape[0] >= min_nodes:
            output.append(graph)
    return output

def BuildLookupDictionary(file_list, prefix):
    '''
     @param file_list: list of hashes we will be examining
     @param prefix: directory containing the graphs

     Builds a lookup dictionary by loading all the hashes contained in the file list into a dictionary. Entries are looked up according to their hash value
    '''
    lookup_dict = dict()
    for file_hash in file_list:
        lookup_dict[file_hash] = LoadGraph(file_hash, prefix)
    return lookup_dict

def CreateClusterVectors(hash_list, lookup_dict):
    '''
     @param hash_list: list of hashes we will be examining
     @param: lookup_dict: dictionary of graphs in dataset that we can look up by hash

     Takes a list of hashes, loads their corresponding graphs, and sums the feature vectors in those graphs.
     The result is a list of summed vectors
    '''
    cluster_vectors = list()
    for app in hash_list:
        # Load the Graph from the lookup dictionary
        feature_vectors = lookup_dict[app]

        # Create a new vector
        new_vector = np.zeros(len(feature_vectors[0]), dtype=np.float32)

        # Sum the vectors
        for fv in feature_vectors:
            new_vector = np.add(new_vector, fv)

        # Add the new vector with its corresponding hash to the list
        cluster_vectors.append(new_vector)
        
    return cluster_vectors

def VectorDistance(a, b):
    '''
     @param a: vector
     @param b: vector

     Measures the distance between two vectors by taking the average distance between each corresponding element of the vectors. In our use case, the order matters because we are examining feature vectors extracted from binary applications.
    '''
    distance_sum = 0.0
    
    for idx, element in enumerate(a):
        distance_sum += abs(float(element) - b[idx])

    return distance_sum/float(len(a))

def Combiner(a):
    '''
     @param a: value (or tuple) that we want to turn into a list for combining

     Combiner function for combineByKey. 
     This takes some 'a' (in this case a tuple), and turns it into a list
    '''
    return [a]

def MergeValue(data_list, new_entry):
    '''
     @param data_list: list we want to insert into
     @param new_entry: entry that is to be inserted into the list

     Merge value function for combineByKey
     This takes a list and a new entry (tuple) and inserts the entry into the list.
    '''
    # Insert the new value
    inserted = False
    for idx, data_entry in enumerate(data_list):
        if new_entry[1] <= data_entry[1]:
            data_list.insert(idx, new_entry)
            inserted = True
            break

    if len(data_list) < CENTROID_POINTS and inserted is False:
        data_list.append(new_entry)

    # Make sure we don't exceed number of centroid points
    if len(data_list) > CENTROID_POINTS:
        data_list.pop()

    return data_list

def MergeCombiners(a, b):
    '''
     @param a: list to merge
     @param b: list to merge

     Merge combiners function for combineByKey
     This takes two lists and merges them. Elements are inserted in sorted order, and the size is capped by a global variable.
    '''
    # Insert each element of b into our merged_list
    for new_entry in b:
        a = MergeValue(a, new_entry)
    
    return a

def ClusterGraphs(hash_list, clustering_vectors, sc):
    '''
     @param hash_list: list of hashes corresponding to the cluster vector data
     @param clustering_vectors: summed feature vectors for each of the hashes
     @param sc: spark context

     Performs our graph clustering. This will use kmeans to divide the dataset into a specified number of clusters, and then it will extract the specified number of nearest neighbors to each cluster's centroid. The result of this will be a new, condensed hash list that we can feed to the FSK algorithm
    '''
    # Create an RDD of hashes and their corresponding clustering vectors
    hashes = sc.parallelize(enumerate(hash_list), NUM_PARTITIONS)
    broadcast_clustering_vectors = sc.broadcast(clustering_vectors)
    hash_data = hashes.map(lambda (index,filename): (filename, broadcast_clustering_vectors.value[index]))

    # Use KMeans to train a clustering model
    clusters = KMeans.train(sc.parallelize(clustering_vectors), NUM_CLUSTERS, maxIterations=10, runs=10, initializationMode="random")

    # Broadcast the model
    broadcast_clusters = sc.broadcast(clusters)

    # Predict clusters for the data
    hash_clusters = hash_data.map(lambda (filename, vector): (broadcast_clusters.value.predict(vector), (filename, vector)))

    # Calculate the distances from the cluster centroids
    hash_distances = hash_clusters.map(lambda (cluster, (filename, vector)): (cluster, (filename, VectorDistance(vector, broadcast_clusters.value.centers[cluster]))))
    distances = hash_distances.collect()

    # Combine by key to select a certain number of closest points to each cluster centroid
    combined_clusters = hash_distances.combineByKey(Combiner, MergeValue, MergeCombiners)

    # Build and return condensed hash list
    condensed_hash_list = list()
    # For each cluster
    for cluster in combined_clusters.collect():
        # Iterate over the list of points
        for point in cluster[1]:
            # Grab the hash value (we can throw out distances: point[1])
            condensed_hash_list.append(point[0])
            
    return condensed_hash_list

def SubtreeSimilarity(fv_1, fv_2):
    '''
     @param fv_1: feature vector from one graph
     @param fv_2: feature vector from other graph

     Takes 2 feature vectors and returns a normalized distance value. The closer it is to 0, the more similar the feature vectors are. The closer it is to 1, the less similar they are.
    '''
    # Capture feature vector length
    fv_length = len(fv_1)

    # Sum feature distances and normalize
    distance = np.sum(abs(fv_1 - fv_2) / np.maximum(np.maximum(fv_1, fv_2), np.ones(fv_length, dtype=np.float32)))
    normalized_distance = distance / float(fv_length)

    return normalized_distance

def PairwiseGraphSimilarity_CPU(graph_A, graph_B, DELTA):
    '''
     @param graph_A: graph to compare
     @param graph_B: graph to compare
     @param DELTA: similarity threshold

     Compares 2 graphs. Each graph is a list of encoded subtrees represented as feature vectors. We map over one graph to search for the presence of its feature vectors in the other graph in parallel. This function returns a normalized similarity value (between 1 and 0...0 being identical, 1 being different) that we can add to our kernel matrix.
    '''
    # Keep track of total similarity score and number of comparisons for normalization
    global_similarity = 0

    # Number of total comparisons, feature vector length, and graph B size
    total_subtree_comparisons = len(graph_A) + len(graph_B)
    fv_length = len(graph_A[0])
    graph_b_size = len(graph_B)

    # Keep track of B to A similarities as we go
    BtoA_similarities = np.ones(len(graph_B), dtype=np.float32)

    # Compare graphs A and B
    for feature_vector_1 in graph_A:
        curr_min = 1.0
        for j, feature_vector_2 in enumerate(graph_B):
            # Calculate similarity of the 2 feature vectors
            curr_sim = SubtreeSimilarity(feature_vector_1, feature_vector_2)

            # Keep track of similarity from A to B
            curr_min = min(curr_min, curr_sim)

            # Keep track of similarity from B to A
            if curr_sim < BtoA_similarities[j]:
                BtoA_similarities[j] = curr_sim

        # Return and add to similarity if node from graph A is found in graph B
        if curr_min < DELTA:
            global_similarity += 1
        
    # Reduce BtoA_similarities
    for sim in BtoA_similarities:
        if sim < DELTA:
            global_similarity += 1

    # Normalize (between 1 and 0) and return. 1 is different, 0 is identical
    normalized_similarity = 1.0 - (float(global_similarity) / total_subtree_comparisons)
    return normalized_similarity
    
def min_GPU(a_gpu, axis, stream=None, keepdims=False):
    min_or_max = "min"
    alloc = cuda.mem_alloc

    n, m = a_gpu.shape if a_gpu.flags.c_contiguous else (a_gpu.shape[1], a_gpu.shape[0])
    col_kernel, row_kernel = misc._get_minmax_kernel(a_gpu.dtype, min_or_max)
    if (axis == 0 and a_gpu.flags.c_contiguous) or (axis == 1 and a_gpu.flags.f_contiguous):
        if keepdims:
            out_shape = (1, m) if axis == 0 else (m, 1)
        else:
            out_shape = (m,)
        target = gpuarray.empty(out_shape, dtype=a_gpu.dtype, allocator=alloc)
        idx = gpuarray.empty(out_shape, dtype=np.uint32, allocator=alloc)
        col_kernel(a_gpu, target, idx, np.uint32(m), np.uint32(n),
                   block=(32, 1, 1), grid=(m, 1, 1), stream=stream)
    else:
        if keepdims:
            out_shape = (1, n) if axis == 0 else (n, 1)
        else:
            out_shape = (n,)
        target = gpuarray.empty(out_shape, dtype=a_gpu, allocator=alloc)
        idx = gpuarray.empty(out_shape, dtype=np.uint32, allocator=alloc)
        row_kernel(a_gpu, target, idx, np.uint32(m), np.uint32(n),
                block=(32, 1, 1), grid=(n, 1, 1), stream=stream)
    return target

def PairwiseGraphSimilarity_GPU(graph_A, graph_B, DELTA):
    '''
     @param graph_A: graph to compare
     @param graph_B: graph to compare
     @param DELTA: similarity threshold

     Compares 2 graphs. Each graph is a list of encoded subtrees represented as feature vectors. We parallelize over one graph to search for the presence of its feature vectors in the other graph in parallel. This function returns a normalized similarity value (between 1 and 0...0 being identical, 1 being different) that we can add to our kernel matrix. This uses the FSK CUDA kernel in FSK.cu, and the pairwise similarity is accelerated using the GPU (examination of feature vectors is done in parallel).
    '''
    # Total number of comparisons, feature vector length, and size of other graph
    total_subtree_comparisons = len(graph_A) + len(graph_B)
    fv_length = len(graph_A[0])
    graph_a_size = len(graph_A)
    graph_b_size = len(graph_B)

    # Create CUDA context
    cuda.init()
    ctx = make_default_context()

    # Initialize distance matrix
    distance_matrix_device = misc.ones((graph_a_size, graph_b_size), dtype=np.float32)

    # Copy graphs to GPU
    device_graph_a = cuda.to_device(graph_A.astype(np.float32))
    device_graph_b = cuda.to_device(graph_B.astype(np.float32))

    # How many threads per block
    threads_per_block = int(math.ceil(float(graph_a_size) * float(graph_b_size) / float(CUDA_BLOCK_SIZE)))
    if threads_per_block == 0:
        threads_per_block = 1

    # print "\n\n\n\n\n", threads_per_block, "\n\n\n\n\n"

    # Compile CUDA kernel
    mod = SourceModule(open("./FSK.cu", "r").read())
    kernel = mod.get_function("fsk")

    # Run kernel
    kernel(distance_matrix_device, device_graph_a, device_graph_b, np.int32(fv_length), np.int32(graph_a_size), np.int32(graph_b_size), block=(CUDA_BLOCK_SIZE,1,1), grid=(threads_per_block, 1))

    # Reduce the matrix in both directions and gather the resulting min arrays from the GPU
    graph_a_mins = min_GPU(distance_matrix_device, axis=1).get()
    graph_b_mins = min_GPU(distance_matrix_device, axis=0).get()
    
    # Free GPU memory
    device_graph_a.free()
    device_graph_b.free()
    
    # Pop CUDA context (driver yells otherwise)
    ctx.pop()

    # Reduce and increment similarity
    global_similarity = 0.0

    # Reduce graph A
    for dist in graph_a_mins:
        if dist < DELTA:
            global_similarity += 1.0

    # Reduce graph B
    for dist in graph_b_mins:
        if dist < DELTA:
            global_similarity += 1.0

    # Normalize the calculated similarity value
    normalized_similarity = 1.0 - (float(global_similarity) / total_subtree_comparisons)
    return normalized_similarity

def MapGraphSimilarity(comp, hash_list, lookup_dict):
    '''
     @param comp: current comparison to be done (distributed by spark)
     @param hash_list: list of hashes used to create the matrix

     Mapping function for comparison. If it fails, we return 1.0 (completely different graphs). For example, this would happen in cases where we fail to load the graph.
    '''
    i = comp[0]
    j = comp[1]

    print "--> Comparing graphs", i, "and", j, "\r",
        
    try:
        # Look up graphs A and B in the dictionary
        graph_A = lookup_dict.value[hash_list.value[i]]
        graph_B = lookup_dict.value[hash_list.value[j]]

        # Compute pairwise similarity of graphs using either GPU or CPU implementation
        if USE_GPU:
            # Launch FSK kernel, but make sure the larger graph is passed in as graph A to exploit the most parallelism
            pairwise_sim = PairwiseGraphSimilarity_GPU(graph_A, graph_B, DELTA)
            # if len(graph_A) > len(graph_B):
            #     pairwise_sim = PairwiseGraphSimilarity_GPU(graph_A, graph_B, DELTA)
            # else:
            #     pairwise_sim = PairwiseGraphSimilarity_GPU(graph_B, graph_A, DELTA)
        else:
            pairwise_sim = PairwiseGraphSimilarity_CPU(graph_A, graph_B, DELTA)

    except RuntimeError:
        return 1.0

    # Return similarity value    
    return pairwise_sim

def ComputeGraphSimilarity_Spark(hash_list, lookup_dict, sc):
    '''
     @param hash_list: list of graphs we would like to compare (by filename/hash)

     This function creates our kernel matrix. We compute the pairwise similarity of each pair of graphs in our block, and we add that value to the kernel matrix, which ultimately is the result of this application. We distribute this pairwise computation using Apache Spark
    '''
    # 2D array to store the kernel matrix in
    num_hashes = len(hash_list)
    kernel_matrix = np.zeros((num_hashes, num_hashes), dtype=np.float32)

    # Create and broadcast the list of comparisons
    graph_comparisons = list(itertools.combinations(xrange(0, num_hashes), 2))
    comp_rdd = sc.parallelize(graph_comparisons, NUM_PARTITIONS)

    # Broadcast chunks of the comparison list out to workers
    broadcast_hash_list = sc.broadcast(hash_list)
    broadcast_lookup_dict = sc.broadcast(lookup_dict)

    # Compute pairwise similarity of graphs using either GPU or CPU implementation
    comp_sim = comp_rdd.map(partial(MapGraphSimilarity, hash_list=broadcast_hash_list, lookup_dict=broadcast_lookup_dict))
    collected_sim = comp_sim.collect()

    for k, comp in enumerate(graph_comparisons):
        i = comp[0]
        j = comp[1]

        # Update kernel matrix
        kernel_matrix[i][j] = collected_sim[k]
        kernel_matrix[j][i] = collected_sim[k]

    # Print resulting kernel matrix
    print "\n"
    for row in kernel_matrix:
        print " ".join(str("%.2f" % x) for x in row)
    return kernel_matrix

def WriteBlock(kernel_matrix, output_file):
    '''
     @param kernel_matrix: kernel matrix to write out
     @param output_file: file we want to write the matrix to

     Converts the kernel matrix to a numpy array and writes it out to a file
    '''

    np.save(output_file, kernel_matrix)

#################################################################
###    Script Execution
#################################################################

def main():
    # Create spark context
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    # Build the file list of graphs we will compare
    print "--> Building hash list"
    hash_list = BuildFileList(HASH_LIST)

    print "--> Filtering out graphs with less than " + str(NODE_FILTER) + " nodes"
    hash_list = FilterFileList(hash_list, NODE_FILTER, GRAPH_DIR)

    print "--> Loading graphs into memory"
    lookup_dict = BuildLookupDictionary(hash_list, GRAPH_DIR)

    print "--> Creating clustering vectors"
    clustering_vectors = CreateClusterVectors(hash_list, lookup_dict)

    print "--> Clustering graphs into " + str(NUM_CLUSTERS) + " clusters and extracting " + str(CENTROID_POINTS) + " points from each cluster centroid"
    condensed_hash_list = ClusterGraphs(hash_list, clustering_vectors, sc)

    print "--> Computing Graph Similarities"    
    # Compute the pairwise similarity of the graphs and return the resulting kernel matrix
    kernel_matrix = ComputeGraphSimilarity_Spark(condensed_hash_list, lookup_dict, sc)

    print "--> Writing matrix to disk as " + OUTPUT_MATRIX
    # Write matrix to numpy file
    WriteBlock(kernel_matrix, OUTPUT_MATRIX)

if __name__ == "__main__":
    # Used to parse options
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-g", "--gpu", help="If present, use the GPU for comparisons", action='store_true')
    parser.add_argument("-p", "--partitions", metavar="num_partitions", help="Number of partitions to split the comparisons up into when using Spark", type=int, default=1024)
    parser.add_argument("-k", "--kmeans", help="Number of clusters we would like to create", type=int, default=10)
    parser.add_argument("-c", "--centroids", help="Number of points closest to the cluster centroids we would like to extract", type=int, default=1)
    parser.add_argument("-f", "--filter", help="Filter out graphs with less nodes/subtrees than the specified filter value", type=int, default=5)

    # Mandatory arguments
    parser.add_argument("hash_list", help="List of hashes that we wish to examine similarities between", type=str)
    parser.add_argument("graph_dir", help="Directory containing the data on the local file system", type=str)

    # Parse arguments
    args = parser.parse_args()

    # Row and column hash lists that make up the block we are computing
    HASH_LIST = args.hash_list

    # Number of spark partitions
    NUM_PARTITIONS = args.partitions

    # Grab the row file and column file basenames
    HASH_NAME = HASH_LIST.split(os.sep)[-1].split(".")[0]

    # Local directory where the graphs are stored
    GRAPH_DIR = args.graph_dir

    # Output file (this is going to need to change to use unique IDs and map them to block configs in a json mapping file for the S3 bucket)
    OUTPUT_MATRIX = os.path.abspath("." + os.sep + HASH_NAME + FILE_SUFFIX)

    # Number of clusters and centroid points
    NUM_CLUSTERS = args.kmeans
    CENTROID_POINTS = args.centroids

    # Filter out small graphs
    NODE_FILTER = args.filter

    # Set flags for GPU and/or Spark use
    if args.gpu:
        USE_GPU = 1
    
    # execute
    main()
