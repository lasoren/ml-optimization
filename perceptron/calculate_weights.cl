__kernel void calculate_weights(__global float* X, __global char* Y, __global float* W, __global char* misclassified, int x_length, int x_dim){
	int worker_id = get_global_id(0);
	__local float block_weights[500][6];
	int i,j;
	if(misclassified[worker_id] == 1){
		for(j=0;j<x_dim; j++){
			block_weights[worker_id][j] = 1.0*X[worker_id*x_dim+j]*Y[worker_id];
		}
	}
	else{
		for(j=0;j<x_dim;j++){
			block_weights[worker_id][j] = 0;
		}
	}	

	barrier(CLK_LOCAL_MEM_FENCE);
	char counter = 0; //for debugging
	float sum;
	if(worker_id==399){
		for(j=0;j<x_dim;j++){
			sum = 0;
			for(i=0;i<x_length;i++){
				sum = sum+  block_weights[i][j];
				counter++;
			}
			W[j]+= sum;
		}
		misclassified[worker_id] = (char) (worker_id);
	}

}

__kernel void classify(__global float* X, __global char* Y, __global float* W, __global char* misclassified, __global char* not_classified, __global int* sum_missed, int x_dim, int x_length ){
	__local float score_shared[500];
	int worker_id = get_global_id(0);
	int j;
	score_shared[worker_id] =0;
	sum_missed[worker_id] =0;
	not_classified[worker_id] =0;
	for(j=0;j<x_dim;j++){
		score_shared[worker_id] += X[worker_id*x_dim +j]*W[j];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	misclassified[worker_id] = score_shared[worker_id]*Y[worker_id] <= 0.0 ? 1:0;
	barrier(CLK_LOCAL_MEM_FENCE);
	if(misclassified[worker_id] == 1){
		sum_missed[worker_id] = 1;
		not_classified[worker_id] = 1;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}
