__global__ void fsk(float *distance_matrix, float *graph_a, float *graph_b, int fv_length, int graph_a_size, int graph_b_size)
{
  int gid = blockIdx.x*blockDim.x + threadIdx.x;

  // Make sure we don't overflow
  if (gid < graph_a_size * graph_b_size) {
    int idx_a = gid / graph_b_size;
    int idx_b = gid % graph_b_size;
    int fva_start = idx_a * fv_length;
    int fvb_start = idx_b * fv_length;

    // Compare to each feature vector in graph B
    float distance = 0.0;

    // Iterate through the feature vectors
    for (int k = 0, fv_a = fva_start; k < fv_length; k++, fv_a++) {
      int fv_b = fvb_start + k;
      
      // Compare feature vector and add to current similarity
      float fv_max = graph_b[fv_b];
      if(graph_a[fv_a] > graph_b[fv_b]) { fv_max = graph_a[fv_a]; }
      if (fv_max < 1.0) { fv_max = 1.0; }
      
      distance += abs(graph_a[fv_a] - graph_b[fv_b]) / fv_max;
    }

    // Normalize distance
    float curr_sim = distance / fv_length;

    // Write distance to matrix
    distance_matrix[gid] = curr_sim;
  }
}
