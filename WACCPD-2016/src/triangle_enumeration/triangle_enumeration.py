#!/usr/bin/env python2

#This code implements triangle enumeration by means of sparse matrix multiplication
#The code computes a_{i,i} for each i in A^3.

import os
from time import time
import argparse
from collections import defaultdict
import gzip
import bz2

from scipy.sparse import bsr_matrix
import numpy as np
import pycuda.driver as driver
import pycuda.tools as tools
import pycuda.gpuarray as gpuarray
from skcuda import cublas, linalg

def MakeVertexList(line):
    '''
    Input: Vi, Vj1, Vj2
    Output: (Vi, [Vj1, Vj2])
    '''
    vertices = line.split()
    row = int(vertices[0])
    neighbor_list = [int(i) for i in vertices[1:]]
    return SortMultiValue( (row, neighbor_list) )

def SortMultiValue(kv_pair):
    key, multivalue = kv_pair
    multivalue.sort()
    return (key, multivalue)

def MakeUpperEdgeList(G):
    '''
    edges of graph: Key = (Vi,Vj) with Vi < Vj, Value = NULL
    '''
    edge_list = []
    for vertex, neighbor_list in G:
        edge_list.extend([ ((vertex, int(i)), None) for i in neighbor_list ])
    return edge_list

def MakeUpperEdgeListSpark(kv_pair):
    vertex, neighbor_list = kv_pair
    return [ ((vertex, int(i)), None) for i in neighbor_list ]

def AddFirstDegree(edge_list_entry):
    '''
    Input: Key = Vi, MultiValue = (Vj,Vk,...)
    for each V in MultiValue:
       if Vi < V: Output: Key=(Vi,V),Value=(Di,0)
       else: Output: Key = (V, Vi), Value = (0, Di)
    '''
    key, multivalue = edge_list_entry
    degree = len(multivalue)
    output_list = []
    for other_key in multivalue:
        if key < other_key:
            output_list.append(( (key,other_key), (degree,0) ))
        else:
            output_list.append(( (other_key,key), (0,degree) ))
    return output_list

def AddSecondDegree(edge_list_entry):
    '''
    Input: Key = (Vi,Vj), MultiValue = ((Di,0),(0,Dj))
    Output: Key = (Vi,Vj), Value = (Di,Dj) with Vi < Vj
    '''
    key, multivalue = edge_list_entry
    value = (multivalue[0][0], multivalue[1][1]) if multivalue[0][1] == 0 else (multivalue[0][1], multivalue[1][0])
    return (key, value)

def Collate(kv_list):
    kv_dict = defaultdict(list)
    for key, value in kv_list:
        kv_dict[key].append(value)
    return kv_dict.iteritems()

def LowDegreeEmitEdges(kv_pair):
    '''
    Input: ((Vi, Vj), (Di, Dj))
    if Di < Dj: Output: Key = Vi, Value = Vj
    else if Dj < Di: Output: Key = Vj, Value = Vi
    else: Output: Key = Vi, Value = Vj
    '''
    key, value = kv_pair
    vi, vj = key
    di, dj = value
    output = None
    if di <= dj:
        output = (vi, vj)
    elif di > dj:
        output = (vj, vi)
    return output

def EmitAnglesOfVertex(kv_pair_list):
    '''
    Input: Key = Vi, MultiValue = (Vj,Vk,...)
    for each V1 in MultiValue:
        for each V2 beyond V1 in MultiValue:
        if V1 < V2: Output: Key = (V1, V2), Value = Vi
        else: Output: Key = (V2, V1), Value = Vi
    '''
    output_kv_pairs = []
    for kv_pair in kv_pair_list:
        input_key, multivalue = kv_pair
        for idx, v1 in enumerate(multivalue):
            for v2 in multivalue[idx+1:]:
                output_key = (v1, v2) if v1 < v2 else (v2, v1)
                output_kv_pairs.append( (output_key, input_key) )
    return output_kv_pairs

def MergeSubLists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]

def EmitTriangles(kv_pair_list):
    triangles = []
    for kv_pair in kv_pair_list:
        key, multivalue = kv_pair
        if None in multivalue:
            triangles.extend([(key[0], key[1], vertex) for vertex in multivalue if vertex is not None])
    return triangles

def EmitTrianglesSpark(kv_pair):
    key, multivalue = kv_pair
    triangles = []
    #print kv_pair
    if None in multivalue:
        for vertex in filter(lambda x: x is not None, multivalue):
            triangles.append( (key[0], key[1], vertex) )
    return triangles


def Combiner(a):
    return [a]

def MergeValue(a,b):
    a.append(b)
    return a

def MergeCombiners(a, b):
    a.extend(b)
    return a

def count_triangles_serial(adjacency_list):
    G = adjacency_list
    G_zero = MakeUpperEdgeList(G)
    #G_zero = reduce(lambda x,y: x+y, G_zero)

    G = MergeSubLists([AddFirstDegree(x) for x in G])
    #G = G.flatMap(AddFirstDegree)

    G = Collate(G)
    #G = G.combineByKey(Combiner, MergeValue, MergeCombiners)

    G = [AddSecondDegree(x) for x in G]
    #G = G.map(AddSecondDegree)

    G = [LowDegreeEmitEdges(x) for x in G]
    #G = G.map(LowDegreeEmitEdges)

    G = Collate(G)
    #G = G.combineByKey(Combiner, MergeValue, MergeCombiners)

    G = EmitAnglesOfVertex(G)
    #G = G.flatMap(EmitAnglesOfVertex)

    G = Collate(G + G_zero)
    #G = G.union(G_zero).combineByKey(Combiner, MergeValue, MergeCombiners)

    G = EmitTriangles(G)
    return len(G)

def count_triangles_sparse_scipy(adjacency_list):
    n = len(adjacency_list)
    transformed = [(1.0, row_idx, col_idx) for row_idx, neighbor_list in adjacency_list for col_idx in neighbor_list]
    data, row, col = zip(*transformed)
    A = bsr_matrix((data, (row, col)), shape=(n,n))
    return int(A.dot(A).dot(A).diagonal().sum() / 6.0)

def count_triangles_cublas(adjacency_list):
    driver.init()
    context = tools.make_default_context()
    h = cublas.cublasCreate()
    n = len(adjacency_list)
    A = np.zeros([n,n], dtype=np.float64)
    for row_idx, neighbor_list in adjacency_list:
        A[row_idx, neighbor_list] = 1.0
    a_gpu = gpuarray.to_gpu(A)
    b_gpu = gpuarray.empty(A.shape, A.dtype)
    c_gpu = gpuarray.empty(A.shape, A.dtype)
    one = np.float64(1.0)
    zero = np.float64(0.0)
    cublas.cublasDsymm(h, 'L', 'U', n, n, one, a_gpu.gpudata, n, a_gpu.gpudata, n, zero, b_gpu.gpudata, n)
    cublas.cublasDsymm(h, 'L', 'U', n, n, one, a_gpu.gpudata, n, b_gpu.gpudata, n, zero, c_gpu.gpudata, n)
    trace = linalg.trace(c_gpu, h)
    cublas.cublasDestroy(h)
    context.detach()
    return int(trace/6)

def filter_vertices(G):
    local_vertices = {vertex for vertex, neighbor_list in G}
    local_G = []
    global_angles = []
    for vertex_id, neighbor_list in G:
        local_neighbor_list = [vertex for vertex in neighbor_list if vertex in local_vertices]
        global_neighbor_list = [vertex for vertex in neighbor_list if vertex not in local_vertices]
        global_neighbor_list.sort()
        if len(local_neighbor_list) > 0:
            local_G.append((vertex_id, local_neighbor_list))
        if len(local_neighbor_list) < len(neighbor_list):
            angles = []
            valid_global_list = [global_id for global_id in global_neighbor_list if global_id > vertex_id]
            for idx, global_neighbor_id in enumerate(valid_global_list):
                for local_neighbor_id in local_neighbor_list:
                    if local_neighbor_id > vertex_id:
                        output_key = (global_neighbor_id, local_neighbor_id) if global_neighbor_id < local_neighbor_id else (local_neighbor_id, global_neighbor_id)
                        angles.append((output_key, vertex_id))
                for other_neighbor_id in valid_global_list[idx+1:]:
                    output_key = (global_neighbor_id, other_neighbor_id) if global_neighbor_id < other_neighbor_id else (other_neighbor_id, global_neighbor_id)
                    angles.append((output_key, vertex_id))
            global_angles.extend(angles)
    local_vertices = {vertex for vertex, neighbor_list in local_G}
    vertex_map = {old_id : new_id for old_id, new_id in zip(local_vertices, xrange(len(local_vertices)))}
    for idx, (vertex, neighbor_list) in enumerate(local_G):
        local_G[idx] = (vertex_map[vertex], neighbor_list)
        for idx, value in enumerate(neighbor_list):
            neighbor_list[idx] = vertex_map[value]
    return local_G, global_angles

def calculate_density(G):
    n = len(G)
    possible_edges = (n * (n-1))
    if possible_edges > 0:
        num_edges = sum([len(neighbor_list) for vertex, neighbor_list in G])
        return float(num_edges) / possible_edges
    else:
        return 0

def timed_run(G, args):
    pid = os.getpid()
    G = list(G)
    if args.spark:
        G_len = len(G)
        filter_start_time = time()
        G, global_angles = filter_vertices(G)
        filter_end_time = time()
        filtered_len = len(G)
        density = calculate_density(G)
        print "{} - Pre-filter len: {}, post-filter len: {}, density: {}".format(pid, G_len, filtered_len, density)
    else:
        filter_start_time = time()
        filter_end_time = filter_start_time
        global_angles = []
        filtered_len = len(G)
    if args.algorithm == 'cublas':
        func = count_triangles_cublas
    elif args.algorithm == 'serial':
        func = count_triangles_serial
    elif args.algorithm == 'sparse_scipy':
        func = count_triangles_sparse_scipy
    else:
        raise RuntimeError("Unknown algorithm")
    alg_start_time = time()
    if filtered_len > 0:
        num_tris = func(G)
    else:
        num_tris = 0
    alg_end_time = time()
    print "{} - filtering: {}, {}: {}".format(pid,
                                              filter_end_time - filter_start_time,
                                              args.algorithm,
                                              alg_end_time - alg_start_time)
    print pid, " - There are ", num_tris, " triangles in the graph."
    return num_tris, global_angles

def parse_input_file(args):
    if args.input_file[-3:] == ".gz":
        with gzip.open(args.input_file, 'r') as file:
            G = [MakeVertexList(line) for line in file]
    elif args.input_file[-4:] == ".bz2":
        with bz2.BZ2File(args.input_file, 'r') as file:
            G = [MakeVertexList(line) for line in file]
    else:
        with open(args.input_file, 'r') as file:
            G = [MakeVertexList(line) for line in file]
    return G

def spark(args):
    from pyspark import SparkContext, SparkConf
    conf = SparkConf().setAppName("Triangle Enumeration")
    sc = SparkContext(conf=conf)
    setup_start_time = time()
    if args.parallel_parse:
        graph_file = sc.textFile(args.input_file, minPartitions=args.num_partitions)
        G_spark = graph_file.map(MakeVertexList)
    else:
        G = parse_input_file(args)
        G_spark = sc.parallelize(G, numSlices=args.num_partitions)
    G_spark.count()
    setup_end_time = time()
    print "MainThread - setup time: {}".format(setup_end_time - setup_start_time)

    local_start_time = time()
    run_results = G_spark.mapPartitions(lambda x: [timed_run(x, args)])
    tris = run_results.map(lambda x: x[0])
    local_count = tris.reduce(lambda x, y: x+y)
    local_end_time = time()
    print "MainThread - local time: {}".format(local_end_time - local_start_time)

    global_start_time = time()
    global_angles = run_results.flatMap(lambda x: x[1])
    G_zero = G_spark.flatMap(MakeUpperEdgeListSpark)
    global_tris = global_angles.union(G_zero).combineByKey(Combiner, MergeValue, MergeCombiners)
    global_tris = global_tris.flatMap(EmitTrianglesSpark)
    global_count = global_tris.count()
    global_end_time = time()
    print "MainThread - global time: {}".format(global_end_time - global_start_time)
    return local_count + global_count

def nospark(args):
    G = parse_input_file(args)
    return timed_run(G, args)[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('algorithm', choices=['cublas', 'serial', 'sparse_scipy'])
    parser.add_argument('--num_partitions', help='number of partitions to use for the main RDD', type=int)
    parser.add_argument('--spark', action='store_true')
    parser.add_argument('--parallel_parse', action='store_true')
    #parser.add_argument('-z', dest='verbose', action='count', default=0, help="verbosity (pyspark eats up -v")
    args = parser.parse_args()

    if args.spark:
        print spark(args)
    else:
        print nospark(args)

if __name__ == "__main__":
    main()
