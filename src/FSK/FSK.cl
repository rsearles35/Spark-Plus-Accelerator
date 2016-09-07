__kernel void fsk(__global int *global_sim, __global const float *graph_a, __global const float *graph_b, __global float *BtoA_similarities, __global float *BtoA_temp, __global float *BtoA_temp_global, const int fv_length, const int graph_b_size, const float delta, const int num_threads, const int work_group_size, const int work_groups) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  if (gid < num_threads) { 
    float curr_min = 1.0;
    int fva_start = gid * fv_length;

    for (int j = 0; j < graph_b_size; j++) {
      float distance = 0.0;
      int fvb_start = j * fv_length;
      for (int k = 0, fv_a = fva_start; k < fv_length; k++, fv_a++) {
	int fv_b = fvb_start + k;
      
	// Compare feature vector and add to current similarity
	float fv_max = graph_b[fv_b];
	if(graph_a[fv_a] > graph_b[fv_b]) { fv_max = graph_a[fv_a]; }
	if (fv_max < 1.0) { fv_max = 1.0; }
      
	distance += fabs(graph_a[fv_a] - graph_b[fv_b]) / fv_max;
      }

      // Current similarity and current minimum values
      float curr_sim = distance / fv_length;
      if (curr_sim < curr_min) { curr_min = curr_sim; }

      BtoA_temp[gid] = curr_sim;

      // Synchronize threads
      barrier(CLK_GLOBAL_MEM_FENCE);

      // Local Reduction
      if (lid == 0) {
	float local_sim = 1.0;
	for (int local_i = 0; local_i < work_group_size; local_i++) {
	  if (gid + local_i < num_threads) {
	    if (BtoA_temp[gid + local_i] < local_sim) { local_sim = BtoA_temp[gid + local_i]; }
	  }
	}
	int global_index = (int)(gid / work_group_size);
	BtoA_temp_global[global_index] = local_sim;
      }

      // Synchronization
      barrier(CLK_GLOBAL_MEM_FENCE);

      // Global (work group) reduction
      if (gid == 0) {
	float synch_sim = 1.0;
	for (int i = 0; i < work_groups; i++) {
	  if (BtoA_temp_global[i] < synch_sim) { synch_sim = BtoA_temp_global[i]; }
	}
	BtoA_similarities[j] = synch_sim;
      }

      // Synchronization
      barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Add to similarity if node from graph A is found in graph B
    if (curr_min < delta) { atomic_add(global_sim, 1); }
  }
}
