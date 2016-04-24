#include <cstdio>
#include <cstdlib>
#include <math.h>
#include "utils.h"
// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define NUM_THREADS_PER_BLOCK 	5000	
#define NUM_BLOCKS 				1	
#define PRINT_TIME 				1
#define TOL						1e-6
#define TEST_CASE				1
#define X_DIM                   6
#define X_LENGTH                5000
#define ETA   					1.0
#define MAX_ITERS		10000
#define IMUL(a, b) __mul24(a, b)

__global__ void calculate_weights(data_t* X, char* Y, data_t* W, char* misclassified,int x_length, int x_dim, double eta){
	__shared__ data_t block_weights[NUM_THREADS_PER_BLOCK][X_DIM]; // 500 x 6
	int tx = threadIdx.x;

	int i,j;
	
		if(misclassified[tx] == 1){
			for(j= 0; j < x_dim;j++){
				block_weights[tx][j] = eta*X[tx*x_dim+j]*Y[tx];
			}	
		}
		else{
			for(j=0; j < x_dim; j++){
				block_weights[tx][j] = 0;
			}
		}
	__syncthreads();

	float sum;
	if(tx==399){
		for(j=0;j<x_dim;j++){
			sum = 0;
			for(i=0; i < NUM_THREADS_PER_BLOCK;i++){
				sum = sum+  block_weights[i][j];
			}
			W[j]+= sum;
		}
	}
}

__global__ void classify(data_t* X, char* Y, data_t* W, char* misclassified, char* not_classified, int* sum_missed,  int x_dim){
	__shared__  data_t score_shared[NUM_THREADS_PER_BLOCK];
	int tx = threadIdx.x;
	int j;
	score_shared[tx] =0;
	sum_missed[tx] = 0;
	not_classified[tx] = 0;
	for(j=0;j < x_dim; ++j){
		score_shared[tx] += X[tx*x_dim + j]*W[j];
	}
	__syncthreads();
	misclassified[tx] = score_shared[tx]*Y[tx] <= 0.0 ? 1:0;
	__syncthreads();
	if(misclassified[tx] == 1){
		sum_missed[tx] = 1;	
		not_classified[tx] = 1;
	}
	__syncthreads();
}


int main(int argc, char **argv){
	// GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	int test_case = TEST_CASE;
	int h_x_length = X_LENGTH;
	int h_x_dim = X_DIM;
	int line_counter = 0;
	int i;
	char not_classified = 1;
	int iters = 0;


	// global variables on GPU
	data_t* g_W;			
	data_t* g_X;
	float* g_score;
	char* g_Y;
	char* g_not_classified;
	char* g_misclassified;
	int* g_sum_missed;


	//global arrays on host
	float* h_W;
	float* h_X;
	char* h_Y;
	float* h_score;
	char* h_misclassified;	
	char* h_not_classified;
	int* h_sum_missed;
	int missed = 0;

	 // Select GPU
	CUDA_SAFE_CALL(cudaSetDevice(1));

	// Allocate GPU memory
	size_t allocSize_X =  h_x_dim * h_x_length * sizeof(data_t);
	size_t allocSize_Y = h_x_length * sizeof(char);
	size_t allocSize_W = h_x_dim * sizeof(data_t);
	size_t allocSize_Score = h_x_length * sizeof(float);
	size_t allocSize_sumMissed = sizeof(int)*h_x_length;

	CUDA_SAFE_CALL(cudaMalloc((void **)&g_W, allocSize_W))
	CUDA_SAFE_CALL(cudaMalloc((void **)&g_X, allocSize_X));
	CUDA_SAFE_CALL(cudaMalloc((void **)&g_Y, allocSize_Y));
	CUDA_SAFE_CALL(cudaMalloc((void **)&g_score, allocSize_Score));
	CUDA_SAFE_CALL(cudaMalloc((void **)&g_misclassified, allocSize_Y));	
	CUDA_SAFE_CALL(cudaMalloc((void **)&g_sum_missed, allocSize_sumMissed));
	CUDA_SAFE_CALL(cudaMalloc((void **)&g_not_classified, allocSize_Y));

	// Allocate arrays on host memory
	h_X                     = (data_t *) malloc(allocSize_X);
	h_Y                   	= (char *) malloc(allocSize_Y);
	h_W              	= (data_t *) malloc(allocSize_W);
	h_misclassified 	= (char *) malloc(allocSize_Y);
	h_score			= (float *) malloc(allocSize_Score);
	h_sum_missed		= (int *) malloc(allocSize_sumMissed);
	h_not_classified 	= (char *) malloc(allocSize_Y);

	for(i=0;i< h_x_length;i++){
		h_misclassified[i] = 1;
	}
	for(i=0; i < h_x_dim; i++){
		h_W[i] = 0;
	}

    FILE* stream = fopen("data.csv", "r");

    char line[1024];
    while (fgets(line, 1024, stream))
    {
        char* tmp = strdup(line);
        int idx = line_counter*h_x_dim;
        h_X[idx] = 1.0;
        h_X[idx + 1] = strtod(getfield(tmp, 1), NULL);
        tmp = strdup(line);
        h_X[idx + 2] = strtod(getfield(tmp, 2), NULL);
        h_X[idx + 3] = h_X[idx + 1]*h_X[idx + 2]; // xy
        h_X[idx + 4] = h_X[idx + 1]*h_X[idx + 1]; // x^2 
        h_X[idx + 5] = h_X[idx + 2]*h_X[idx + 2]; // y^2
        // NOTE strtok clobbers tmp
        free(tmp);
        line_counter++;
    }

    assign_labels(h_X, h_x_length, h_x_dim, test_case, h_Y);
/*
    for(i=0; i < h_x_length; ++i){ 
        switch(test_case) {
            case 1:
                h_Y[i] = (0.2*(h_X[i*h_x_dim + 0] - 0.5)) +
                    (.6-h_X[i*h_x_dim + 1]) > 0 ? 1 : -1;
                break;
            case 2:
                h_Y[i] = (h_X[i*h_x_dim + 0]-.5)*(h_X[i*h_x_dim + 0]-.5) +
                    (h_X[i*h_x_dim + 1]-.5)*(h_X[i*h_x_dim + 1]-.5) > 0.09 ? 1 : -1;
                break;
            case 3:
                h_Y[i] = 4*(h_X[i*h_x_dim + 0]-.5)*4*(h_X[i*h_x_dim + 0]-.5) +
                    (.2-h_X[i*h_x_dim + 1]) > 0 ? 1 : -1;
                break;
            default:
                h_Y[i] = 0;
        }
    }*/
 }


    // Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(g_X, h_X, allocSize_X, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(g_Y, h_Y, allocSize_Y, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(g_W, h_W, allocSize_W, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(g_misclassified, h_misclassified, allocSize_Y, cudaMemcpyHostToDevice));

#ifdef PRINT_TIME
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
#endif

while(not_classified && iters <= MAX_ITERS){
		
		// Increment iters
		iters++;
		printf("iters: %d \n", iters);
		// Set condition to zero (to avoid infinite while loop) and set it to one if there's an element that is misclassified
		not_classified = 0;
		// One block with 500 threads (one thread working on each row of data in X)
		calculate_weights<<<1,500>>>(g_X, g_Y,g_W,g_misclassified,500, 6, 1);
		CUDA_SAFE_CALL(cudaPeekAtLastError());
		// Copy weight vector to host
		CUDA_SAFE_CALL(cudaMemcpy(h_W, g_W, allocSize_W, cudaMemcpyDeviceToHost));

		// Check classification success		
		classify<<<1,500>>>(g_X, g_Y, g_W, g_misclassified, g_not_classified, g_sum_missed,6);
		CUDA_SAFE_CALL(cudaPeekAtLastError());

		// Copy arrays back to host
		CUDA_SAFE_CALL(cudaMemcpy(h_not_classified, g_not_classified,allocSize_Y, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(h_sum_missed, g_sum_missed,allocSize_sumMissed, cudaMemcpyDeviceToHost));
	//	CUDA_SAFE_CALL(cudaMemcpy(h_misclassified, g_misclassified,allocSize_Y, cudaMemcpyDeviceToHost));

		for(i=0;i<h_x_length;i++){
			not_classified += h_not_classified[i];		
		}
}

#ifdef PRINT_TIME
cudaEventRecord(stop,0);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&elapsed_gpu, start, stop);
printf("GPU execution time: %f (msec) \n", elapsed_gpu);
cudaEventDestroy(start);
#endif


	printf("\n");
	for(i=0;i < h_x_length; ++i){
		missed += h_sum_missed[i];
		printf("%d ", h_sum_missed[i]);	
	}
	printf("\nIterations: %d", iters);
	if(missed == 0){
		printf("Perfectly separated data\n");
	}
	else{
		printf("Finished MAX_ITERS and still %d misclassified \n", missed);
	}

		// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(g_X));
	CUDA_SAFE_CALL(cudaFree(g_Y));
	CUDA_SAFE_CALL(cudaFree(g_W));
	CUDA_SAFE_CALL(cudaFree(g_sum_missed));
	CUDA_SAFE_CALL(cudaFree(g_not_classified));
	CUDA_SAFE_CALL(cudaFree(g_score));
	CUDA_SAFE_CALL(cudaFree(g_misclassified));		   
	free(h_X);
	free(h_not_classified);
	free(h_sum_missed);
	free(h_Y);
	free(h_W);
	free(h_misclassified);
	free(h_score);
	return 0;
}


