#include<stdio.h>
#include<stdlib.h>

// This value is the largest subsets we must unrank.
#define LARGEST_SUBSET 5

// This is value is the largest number a graph could have.
// It is 2^(LARGEST_SUBSET*LARGEST_SUBSET)-1 (i.e. the LARGEST_SUBSET x LARGEST_SUBSET matrix of all 1's).
#define SMALLEST_GRAPH 33554431

// Device code to compute the binomial coefficient "n choose k."
__device__ int binom_d(int n, int k){
	if(k > n){
		return 0;
	}
	if(k == n){
		return 1;
	}
	int retVal = 1;
	for(int i=0; i<k; i++){
		retVal = retVal * (n-i) / (i+1);
	}
	return retVal;
}


__device__ void unrank_combination(int n, int k, int initial_value, int* Kset) {
	int cash_on_hand = initial_value;
	int digit;
	int cost_to_increment;
	Kset[0] = 0;	//Initialize the first element.
					//Each of the following elements will start off one bigger than the previous element.
	//Use the cash_on_hand value to "pay" for incrementing each digit.
	//Pay 1-unit for each combination that is "skipped" over.
	//E.g. To increment the 0 in 0, 1, 2, ..., k-1 to a 1 (and force the others to increment to 2, 3, ..., k)
	//it would cost binom(n-1, k-1) since we skipped over each combination of the form
	// 0 * * * ... * and there are binom(n-1, k-1) of those combinations
	for(digit=0; digit<k-1; digit++){
		//There are n-1-Kset[digit] elements left to choose from.
		//Those elements must be used to fill k-1-digit places.
		cost_to_increment = binom_d( n-1-Kset[digit], k-1-digit );
		while(cost_to_increment <= cash_on_hand){
			Kset[digit]++;
			cash_on_hand = cash_on_hand - cost_to_increment;
			cost_to_increment = binom_d( n-1-Kset[digit], k-1-digit );
		}
		Kset[digit+1] = Kset[digit]+1;	//Ititialize the next element of Kset making sure the elements
										//come in sorted order.
	}
	//Kset[k-1] has been initialized to Kset[k-2]+1 (last step).
	//Now, if there is anything left to pay, we simply increment Kset[k-1] by this amount.
	Kset[k-1] += cash_on_hand;
}

// The Myrvold/Rusky linear time algorithm for ranking/unranking permutations.
// See: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.216.9916&rep=rep1&type=pdf  [4 pages]
__device__ void unrank_permutation(int k, int initial_value, int* Perm){
	int r = initial_value;
	int tmp;
	int n, index;
	for(n=k; n>0; n--){
		index = r % n;
		//swap Perm[index1] with Perm[index2]
		tmp = Perm[n-1];
		Perm[n-1] = Perm[index];
		Perm[index] = tmp;
		r = r / (index + 1);
	}
}


// IMPORTANT: we must make sure that the number of subsets (N choose K) is a value that fits in a integer type (probably 4 bytes).
// Otherwise, "my_subset" will overflow and not be able to write to its location in the array.
// We don't allow bigger than 4 byte values of "my_subset" because we don't want to assume that we have more than 4 GB of memory on the card.
__global__ void examine_subsets(int n, int k, long offset, long MAX, int KFAC, short* A, int* Results){
	const long my_subset = threadIdx.x + blockIdx.x*blockDim.x + offset;		
	if( my_subset < MAX ){
		int i, j;
		int Kset[LARGEST_SUBSET];
		for(i=0; i<k; i++){
			Kset[i] = 0;
		}
		unrank_combination(n, k, my_subset, Kset);	//unrank should modify Kset to be the selection of k vertices to examine.
		short local_A[LARGEST_SUBSET*LARGEST_SUBSET];
		//short local_B[LARGEST_SUBSET*LARGEST_SUBSET];
		for(i=0; i<k; i++){
			for(j=0; j<k; j++){
				local_A[i*k + j] = A[ Kset[i]*n + Kset[j] ];	//A is n by n (in 1D form) but local_A is k by k (in 1D form)
			}
		}

		//Apply permutations to the vertices of local_A to try to create the "smallest" representative graph.
		int permutation_number;
		int Perm[LARGEST_SUBSET];
		int smallest_graph = SMALLEST_GRAPH;	//Default "maximum" graph.
		int this_graph;
		for(permutation_number=0; permutation_number < KFAC; permutation_number++){
			//Apply the Myrvold/Rusky Linear time unranking algorithm to determine the permutation.
			for(i=0; i<k; i++){
				Perm[i] = i;	//Initialize the permutaiton
			}
			unrank_permutation(k, permutation_number, Perm);
			this_graph = 0;
			for(i=0; i<k; i++){
				for(j=0; j<k; j++){
					//Applying Perm to local_A sends vertex i to Perm[i], vertex j to Perm[j].
					//local_B[ i*k + j] = local_A[ Perm[i]*k + Perm[j] ];
					if( local_A[Perm[i]*k + Perm[j]] == 1 ){
						this_graph = this_graph | (1 << (i*k + j));
					}
				}
			}
			if(this_graph < smallest_graph){
				smallest_graph = this_graph;
			}
		}
		Results[my_subset-offset] = smallest_graph;
	}
}




int binom_h(int n, int k){
	int retVal = 1;
	for(int i=0; i<k; i++){
		retVal = retVal*(n-i) / (i+1);
	}
	return retVal;
}


int main(int argc, char** argv){
	int N = atoi( argv[1] );		//Number of vertices in the graph (not necessarily required from command-line)
	int K = atoi( argv[2] );		//Size of the subsets to examine (upper bounded by LARGEST_SUBSET)
	int threads_per_block = atoi( argv[3] );

	short* h_A;
	short* d_A;
	int* h_Results;
	int* d_Results;

	int number_of_subsets = binom_h(N, K);
	int size_of_subsets = number_of_subsets*sizeof(int);
	int size_of_A = N*N*sizeof(short);

	h_A = (short *) malloc( size_of_A );
	h_Results = (int *) malloc( size_of_subsets );


	cudaSetDevice(1);
	cudaMalloc((void **) &d_A, size_of_A);
	cudaMalloc((void **) &d_Results, size_of_subsets);

	int i;
	int KFAC = 1;
	for(i=2; i<=K; i++){
		KFAC *= i;
	}

	//h_A is the adjacecy matrix (on the host)... here is just a dummy matrix.
	//
	// IN THESE LINES WE NEED TO MAKE SURE THAT THE ADJACENCY MATRIX IS THE ONE WE WANT TO EXAMINE
	//
	for(i=0; i<N*N; i++){
		h_A[i] = 0;
	}
	h_A[ 0*N+1 ] = 1;	//h_A[0][1] = 1
	h_A[ 0*N+2 ] = 1;	//h_A[0][2] = 1
	
	//Initialize the results matrix.
	//In h_Results[i] will be the integer representing the graph found by looking at subset i.
	for(i=0; i<N; i++){
		h_Results[i]=-1;
	}

	//Copy adjacency matrix and results array to the device
	cudaMemcpy(d_A, h_A, size_of_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_Results, h_Results, size_of_subsets, cudaMemcpyHostToDevice);

	examine_subsets<<<(number_of_subsets+threads_per_block-1)/threads_per_block, threads_per_block>>>(N, K, 0, number_of_subsets, KFAC, d_A, d_Results);
	printf("CUDA error: %s\n", cudaGetErrorString(cudaGetLastError()));
	cudaMemcpy(h_Results, d_Results, size_of_subsets, cudaMemcpyDeviceToHost);
	
	cudaDeviceSynchronize();

	for(i=0; i<number_of_subsets; i++){
		if(h_Results[i] != 0){
			printf("%d\t%d\n", i, h_Results[i]);
		}
	}
	
	free(h_A);
	free(h_Results);
	cudaFree(d_A);
	cudaFree(d_Results);

	return 0;
}
