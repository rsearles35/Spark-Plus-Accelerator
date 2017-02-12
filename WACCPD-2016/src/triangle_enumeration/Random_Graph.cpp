// Author: Travis Johnston

#include<iostream>
#include<fstream>
#include<time.h>
#include<stdlib.h>
#include<stdio.h>
#include<string.h>

using namespace::std;


int main(int argc, char** argv){

	char fileName [50];
	strncpy(fileName, argv[1], 50);	//Output file name
	const int N = atoi(argv[2]);
	const double p = atof(argv[3]);
	double coin_toss;

	int i, j;

	srand(time(NULL));

	char** Adj_Matrix;
	Adj_Matrix = (char**) malloc(N*sizeof(char*));
	for(i=0; i<N; i++){
		Adj_Matrix[i] = (char*) malloc(N*sizeof(char));
	}

	for(i=0; i<N; i++){
		Adj_Matrix[i][i] = 0;
		for(j=i+1; j<N; j++){
			coin_toss = rand()*1.0/RAND_MAX;
			if(coin_toss <= p){
				Adj_Matrix[i][j] = 1;
				Adj_Matrix[j][i] = 1;
			}else{
				Adj_Matrix[i][j] = 0;
				Adj_Matrix[j][i] = 0;
			}
		}
	}

	ofstream fout;
	fout.open(fileName);

	for(i=0; i<N; i++){
		fout << i;
		for(j=0; j<N; j++){
			if(Adj_Matrix[i][j] == 1){
				fout << " " << j;
			}
		}
		fout << "\n";
	}
	fout.close();

	return 0;
}
