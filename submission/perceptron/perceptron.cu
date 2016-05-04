#include <cstdio>
#include <cstdlib>
#include <math.h>
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

#define NUM_THREADS_PER_BLOCK 	500	
#define NUM_BLOCKS 		1	
#define PRINT_TIME		1
#define TEST_CASE		3
#define X_DIM                   6
#define X_LENGTH                500
#define START_ETA		0.1
#define ETA   			1.0
#define DELTA			.05
#define MAX_ITERS		10000
#define IMUL(a, b) __mul24(a, b)


const char* getfield(char* line, int num);

__global__ void calculate_weights(float* X, char* Y, float* W, char* misclassified,int x_length, int x_dim, double eta){
	__shared__ float block_weights[NUM_THREADS_PER_BLOCK][X_DIM]; // 500 x 6
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

__global__ void classify(float* X, char* Y, float* W, char* misclassified, char* not_classified, int* sum_missed,  int x_dim){
	__shared__  float score_shared[NUM_THREADS_PER_BLOCK];
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
	float eta = ETA;
	float start_eta = START_ETA;
	float delta = DELTA;
	int sum_missed_iters[19][2];
	// global variables on GPU
	float* g_W;			
	float* g_X;
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
	size_t allocSize_X =  h_x_dim * h_x_length * sizeof(float);
	size_t allocSize_Y = h_x_length * sizeof(char);
	size_t allocSize_W = h_x_dim * sizeof(float);
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
	h_X                     = (float *) malloc(allocSize_X);
	h_Y                   	= (char *) malloc(allocSize_Y);
	h_W              	= (float *) malloc(allocSize_W);
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
    while (fgets(line, 1024, stream)&&line_counter < 500)
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

//    assign_labels(h_X, h_x_length, h_x_dim, test_case, h_Y);

    for(i=0; i < h_x_length; ++i){ 
        switch(test_case) {
            case 1:
                h_Y[i] = (0.2*(h_X[i*h_x_dim + 1] - 0.5)) +
                    (.6-h_X[i*h_x_dim + 2]) > 0 ? 1 : -1;
                break;
            case 2:
                h_Y[i] = (h_X[i*h_x_dim + 1]-.5)*(h_X[i*h_x_dim + 1]-.5) +
                    (h_X[i*h_x_dim + 2]-.5)*(h_X[i*h_x_dim + 2]-.5) > 0.09 ? 1 : -1;
                break;
            case 3:
                h_Y[i] = 4*(h_X[i*h_x_dim + 1]-.5)*(h_X[i*h_x_dim + 1]-.5) +
                    (.2-h_X[i*h_x_dim + 2]) > 0 ? 1 : -1;
                break;
            default:
                h_Y[i] = 0;
        }
    }
int j;
 printf("X & Y : \n");
for(i = 0; i < h_x_length; i++){
	for(j= 0; j < h_x_dim; j++){
		printf("%f ", h_X[i*h_x_dim + j]);
	}
	printf("%f\n ", h_Y[i]);
}


    // Transfer the arrays to the GPU memory
	/*CUDA_SAFE_CALL(cudaMemcpy(g_X, h_X, allocSize_X, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(g_Y, h_Y, allocSize_Y, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(g_W, h_W, allocSize_W, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(g_misclassified, h_misclassified, allocSize_Y, cudaMemcpyHostToDevice));
*/
/*#ifdef PRINT_TIME
cudaEventCreate(&start);
cudaEventCreate(&stop);
cudaEventRecord(start,0);
#endif*/
float exec_times[19][2];
//int num_blocks = NUM_BLOCKS;
//int num_threads = NUM_THREADS_PER_BLOCK;
int k;
int index = 0;
float current_eta = start_eta;
for(k = 0; k < 19; k++){
	for(i=0;i< h_x_length;i++){
		h_misclassified[i] = 1;
	}
	for(i=0; i < h_x_dim; i++){
		h_W[i] = 0;
	}
	CUDA_SAFE_CALL(cudaMemcpy(g_X, h_X, allocSize_X, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(g_Y, h_Y, allocSize_Y, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(g_W, h_W, allocSize_W, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(g_misclassified, h_misclassified, allocSize_Y, cudaMemcpyHostToDevice));
	iters = 0;
	missed = 0;
	not_classified = 1;
	#ifdef PRINT_TIME
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	#endif
	int max_iters = MAX_ITERS;


while(not_classified && iters <= MAX_ITERS){
		// Increment iters
		iters++;
		// Set condition to zero (to avoid infinite while loop) and set it to one if there's an element that is misclassified
		not_classified = 0;
		// One block with 500 threads (one thread working on each row of data in X)
		calculate_weights<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(g_X, g_Y,g_W,g_misclassified,h_x_length, h_x_dim, current_eta);
		CUDA_SAFE_CALL(cudaPeekAtLastError());
		cudaThreadSynchronize();
		// Copy weight vector to host
		CUDA_SAFE_CALL(cudaMemcpy(h_W, g_W, allocSize_W, cudaMemcpyDeviceToHost));
		// Check classification success		
		classify<<<NUM_BLOCKS, NUM_THREADS_PER_BLOCK>>>(g_X, g_Y, g_W, g_misclassified, g_not_classified, g_sum_missed,h_x_dim);
		CUDA_SAFE_CALL(cudaPeekAtLastError());
		cudaThreadSynchronize();
		// Copy arrays back to host
		CUDA_SAFE_CALL(cudaMemcpy(h_not_classified, g_not_classified,allocSize_Y, cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(h_sum_missed, g_sum_missed,allocSize_sumMissed, cudaMemcpyDeviceToHost));
		for(i=0;i<h_x_length;i++){
			not_classified += h_not_classified[i];		
		}
}
	

	#ifdef PRINT_TIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	exec_times[index][1] = elapsed_gpu;
	exec_times[index][0] = current_eta;
	cudaEventDestroy(start);
	#endif
	printf("\n");
		for(i=0;i < h_x_length; ++i){
			missed += h_sum_missed[i];
		}
		printf("current_eta: %f, eta: %f, start_eta: %f, index: %d, delta: %f \n", current_eta, eta, start_eta, index, delta);
		if(missed == 0){
			printf("Perfectly separated data\n");
		}
		else{
			printf("Finished MAX_ITERS (%d iters) and still %d misclassified \n", iters, missed);
		}
	sum_missed_iters[index][0] = missed;
	sum_missed_iters[index][1] = iters;
	printf("iters out of loop: %d", iters);
	current_eta += delta;
	index++;
}
	printf("Iters		Exec time (ms)		Sum Missed: 		Iters: \n");
	for(i=0;i<19; i++){
		printf("%f\t\t%f\t\t%d\t\t%d\n", exec_times[i][0], exec_times[i][1], sum_missed_iters[i][0], sum_missed_iters[i][1]);
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

const char* getfield(char* line, int num) {
    const char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}
