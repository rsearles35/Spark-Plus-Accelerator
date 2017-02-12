#!/usr/bin/env python

'''
 Robert Searles
 Department of Computer and Information Sciences
 University of Delaware

 FSK.py
 Python implementation of the Fast Subtree Kernel.
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

# Pyopencl for running OpenCL kernels
import pyopencl as cl

# Functools partial
from functools import partial

# Spark
from pyspark import SparkContext, SparkConf

# Similarity Threshold
DELTA = 0.1

# Which device are we running on
USE_GPU = 0

# Are we using Spark
SPARK = False

# OpenCL device to use
OPENCL_DEVICE = 0

#################################################################
###    Class/Function Declarations
#################################################################

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

def LoadGraph(file_name, prefix):
    '''
     @param file_name: file name of the graph we would like to load
     @param prefix: file path prefix (this allows us to build an absolute path)

     Loads the specified graph into memory
    '''
    encoded_trees = list()
    with open(prefix + file_name, 'r') as graph:
        # Gather encoded feature vectors
        for line in graph:
            feature_vector = map(int, line.split())
            encoded_trees.append(feature_vector)

    # Return as numpy array
    return np.array(encoded_trees, dtype=np.float32)

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
    normalized_distance = distance / fv_length

    return normalized_distance

def CPUMap(feature_vector_1, graph_B, DELTA):
    '''
     @param feature_vector_1: feature vector (subtree) we would like to search for in graph_B
     @param graph_B: graph (list of encoded subtrees/feature vectors) that we would like to compare against
     @param DELTA: similarity threshold.

     Searches for a subtree (feature vector) in the given graph. This is a mapped function, so each instance of this function call takes a different feature vector from graph_A to search for (in parallel) in graph_B. We compare feature_vector_1 to every feature vector in graph_B. If any of them have a distance below our similarity threshold (DELTA), we consider it a match.
    '''
    curr_min = 1.0
    for feature_vector_2 in graph_B:
        # Calculate similarity of the 2 feature vectors
        curr_sim = SubtreeSimilarity(feature_vector_1, feature_vector_2)

        # Keep track of similarity from A to B
        curr_min = min(curr_min, curr_sim)

    # Return and add to similarity if node from graph A is found in graph B
    if curr_min < DELTA:
        return 1
    else:
        return 0

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
    
def PairwiseGraphSimilarity_GPU(graph_A, graph_B, DELTA, ctx, queue, mf, work_group_size):
    '''
     @param graph_A: graph to compare
     @param graph_B: graph to compare
     @param DELTA: similarity threshold
     @param ctx: OpenCL context
     @param queue: OpenCL command queue
     @param mf: OpenCL memory flags
     @param work_group_size: Size of our OpenCL work group (typically 256)

     Compares 2 graphs. Each graph is a list of encoded subtrees represented as feature vectors. We parallelize over one graph to search for the presence of its feature vectors in the other graph in parallel. This function returns a normalized similarity value (between 1 and 0...0 being identical, 1 being different) that we can add to our kernel matrix. This uses the FSK OpenCL kernel in FSK.cl, and the pairwise similarity is accelerated using the GPU (examination of feature vectors is done in parallel).
    '''

    # zeros array. This gives us the global size that we pass to the OpenCL kernel
    a_zeros = np.zeros(graph_A.shape[0], dtype=int)

    # Total number of comparisons, feature vector length, and size of other graph
    total_subtree_comparisons = len(graph_A) + len(graph_B)
    fv_length = len(graph_A[0])
    graph_b_size = len(graph_B)

    # Global similarity on device
    global_similarity = np.zeros(1, dtype=np.int32)
    global_similarity_device = cl.Buffer(ctx, mf.WRITE_ONLY | mf.COPY_HOST_PTR, hostbuf=global_similarity)

    # Copy graph A to the device
    graph_A_device = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=graph_A)

    # Copy graph B to the device
    graph_B_device = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=graph_B)

    # Copy BtoA_similarities to the device
    BtoA_similarities = np.ones(len(graph_B), dtype=np.float32)
    BtoA_similarities_device = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=BtoA_similarities)

    # Number of work groups for each comparison
    work_groups = math.ceil(float(len(graph_A)) / work_group_size)

    # Local and Global reduction arrays
    BtoA_temp = np.ones(len(graph_B), dtype=np.float32)
    BtoA_temp_device = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=BtoA_temp)

    BtoA_temp_global = np.ones(work_groups, dtype=np.float32)
    BtoA_temp_global_device = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=BtoA_temp_global)
    
    # Compile the FSK OpenCL kernel
    kernel_file = PROJECT_REPO + "src/FSK/FSK.cl"
    with open(kernel_file, 'r') as f:
        kernel_string = "".join(f.readlines())
    prg = cl.Program(ctx, kernel_string).build()

    # Run the OpenCL kernel
    prg.fsk(queue, a_zeros.shape, None, global_similarity_device, graph_A_device, graph_B_device, BtoA_similarities_device, BtoA_temp_device, BtoA_temp_global_device, np.int32(fv_length), np.int32(graph_b_size), np.float32(DELTA), np.int32(a_zeros.shape), np.int32(work_group_size), np.int32(work_groups))
    
    # Copy resulting similarity
    cl.enqueue_copy(queue, global_similarity, global_similarity_device)

    # Copy BtoA similarities
    cl.enqueue_copy(queue, BtoA_similarities, BtoA_similarities_device)

    # Free GPU memory
    graph_A_device.release()
    graph_B_device.release()
    global_similarity_device.release()
    BtoA_similarities_device.release()

    # Reduce B to A similarities vector (did we end up finding the nodes from graph B inside graph A?)
    for min_sim in BtoA_similarities:
        if min_sim < DELTA:
            global_similarity[0] += 1

    normalized_similarity = 1.0 - (float(global_similarity[0]) / total_subtree_comparisons)
    return normalized_similarity

def MapGraphSimilarity(comp, encoded_graph_list):
    # If we are using the GPU, do OpenCL stuff
    if USE_GPU:
        # Create OpenCL context, queue, and device list
        platform = cl.get_platforms()
        device_list = platform[0].get_devices()
        my_devices = [device_list[OPENCL_DEVICE]]
        ctx = cl.Context(devices = my_devices)
        queue = cl.CommandQueue(ctx)

        # Keep track of work group size (we will pass this to the kernel)
        work_group_size = my_devices[0].max_work_group_size
        
        # OpenCL memory flags object
        mf = cl.mem_flags

    i = comp[0]
    j = comp[1]

    # Filename prefix
    prefix = os.path.dirname(FILE_LIST) + os.sep

    print "--> Comparing graphs", i, "and", j, "\r",
        
    # Load graphs A and B into memory
    if SPARK:
        graph_A = LoadGraph(encoded_graph_list.value[i], prefix)
        graph_B = LoadGraph(encoded_graph_list.value[j], prefix)
    else:
        graph_A = LoadGraph(encoded_graph_list[i], prefix)
        graph_B = LoadGraph(encoded_graph_list[j], prefix)

    # Compute pairwise similarity of graphs using either GPU or CPU implementation
    if USE_GPU:
        pairwise_sim = PairwiseGraphSimilarity_GPU(graph_A, graph_B, DELTA, ctx, queue, mf, work_group_size)
        queue.finish()
    else:
        pairwise_sim = PairwiseGraphSimilarity_CPU(graph_A, graph_B, DELTA)

    return pairwise_sim

def ComputeGraphSimilarity(encoded_graph_list, num_graphs):
    '''
     @param encoded_graph_list: list of graphs we would like to compare (by filename)
     @param num_graphs: total number of graphs we will compare

     This function creates our kernel matrix. We compute the pairwise similarity of each pair of graphs in our dataset, and we add that value to the kernel matrix, which ultimately is the result of this application.
    '''
    # 2D array to store the kernel matrix in
    kernel_matrix = np.zeros((num_graphs, num_graphs), dtype=np.float32)

    graph_comparisons = itertools.combinations(xrange(0, num_graphs), 2)

    # Compute pairwise similarity of graphs using either GPU or CPU implementation
    for comp in graph_comparisons:
        i = comp[0]
        j = comp[1]

        pairwise_sim = MapGraphSimilarity(comp, encoded_graph_list)

        # Update kernel matrix
        kernel_matrix[i][j] = pairwise_sim
        kernel_matrix[j][i] = pairwise_sim

    # Print resulting kernel matrix
    for row in kernel_matrix:
        print " ".join(str("%.2f" % x) for x in row)
    return kernel_matrix

def ComputeGraphSimilarity_Spark(encoded_graph_list, num_graphs):
    '''
     @param encoded_graph_list: list of graphs we would like to compare (by filename)
     @param num_graphs: total number of graphs we will compare

     This function creates our kernel matrix. We compute the pairwise similarity of each pair of graphs in our dataset, and we add that value to the kernel matrix, which ultimately is the result of this application.
    '''
    # Create spark context
    conf = SparkConf()
    sc = SparkContext(conf=conf)

    # 2D array to store the kernel matrix in
    kernel_matrix = np.zeros((num_graphs, num_graphs), dtype=np.float32)

    graph_comparisons = list(itertools.combinations(xrange(0, num_graphs), 2))
    comp_rdd = sc.parallelize(graph_comparisons, 1024)

    broadcast_graph_list = sc.broadcast(encoded_graph_list)

    # Compute pairwise similarity of graphs using either GPU or CPU implementation
    comp_sim = comp_rdd.map(partial(MapGraphSimilarity, encoded_graph_list=broadcast_graph_list))
    collected_sim = comp_sim.collect()

    for k, comp in enumerate(graph_comparisons):
        i = comp[0]
        j = comp[1]

        # Update kernel matrix
        kernel_matrix[i][j] = collected_sim[k]
        kernel_matrix[j][i] = collected_sim[k]

    # Print resulting kernel matrix
    for row in kernel_matrix:
        print " ".join(str("%.2f" % x) for x in row)
    return kernel_matrix

#################################################################
###    Script Execution
#################################################################

def main():
    print "--> Building list of " + str(NUM_GRAPHS) + " graphs"

    # Build the file list of graphs we will compare
    encoded_graph_list = BuildFileList(FILE_LIST)

    print "--> Computing Graph Similarities"
    
    # Compute the pairwise similarity of the graphs and return the resulting kernel matrix
    if SPARK:
        ComputeGraphSimilarity_Spark(encoded_graph_list, NUM_GRAPHS)
    else:
        ComputeGraphSimilarity(encoded_graph_list, NUM_GRAPHS)

if __name__ == "__main__":
    # Used to parse options
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("json_config", help="Json file contatining the list of encoded graph files we will process, as well as the OpenCL device number for the machine we are running on.", type=str)
#    parser.add_argument("dataset_file_list", help="File containing the list of files in the dataset we would like to examine", type=str)
    parser.add_argument("-n", "--num_graphs", metavar="num_graphs", help="Number of graphs to process", type=int, default=10)
    parser.add_argument("-g", "--gpu", help="If present, use the GPU for comparisons", action='store_true')
    parser.add_argument("-s", "--spark", help="If present, use Spark.", action='store_true')

    # Parse arguments
    args = parser.parse_args()

    # Set globals
    JSON_CONFIG = args.json_config

    with open(JSON_CONFIG, 'r') as json_file:
        config = json.load(json_file)
        PROJECT_REPO, OPENCL_DEVICE = config[socket.gethostname()]

    FILE_LIST = PROJECT_REPO + "data/dataset.txt"
    NUM_GRAPHS = args.num_graphs
    if args.gpu:
        USE_GPU = 1
    if args.spark:
        SPARK = True
    
    # execute
    main()
