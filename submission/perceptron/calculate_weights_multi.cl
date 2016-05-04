
__kernel void calculate_weights(__global float* X, __global char* Y, __global float* W, __global char* misclassified, int x_length, int x_dim, float eta){
	int worker_id = get_global_id(0);
	int local_id = get_local_id(0);
  	int group_id = get_group_id(0);
	__local float block_weights[300][6];
	int i,j;
	if(misclassified[worker_id] == 1){
		for(j=0;j<x_dim; j++){
			block_weights[local_id][j] = eta*X[worker_id*x_dim+j]*Y[worker_id];
		}
	}
	else{
		for(j=0;j<x_dim;j++){
			block_weights[local_id][j] = 0;
		}
	}	

	barrier(CLK_LOCAL_MEM_FENCE);
	float sum;
	if(local_id==1){
		for(j=0;j<x_dim;j++){
			sum = 0;
			for(i=0;i<300;i++){
				sum = sum + block_weights[i][j];
			}
			W[group_id*x_dim+j]= sum;
		}
	}

}


__kernel void classify(__global float* X, __global char* Y, __global float* W, 
	__global char* misclassified, __global int* not_classified, __global int* sum_missed, int x_dim, int x_length ){
	int local_id = get_local_id(0);
	__private float score;
	int worker_id = get_global_id(0);
	int j;
	score=0;
	sum_missed[worker_id] =0;
	not_classified[worker_id] =0;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(j=0;j<x_dim;j++){
		score += X[worker_id*x_dim +j]*W[j];
	}
	misclassified[worker_id] = score*Y[worker_id] <= 0.0 ? 1:0;
	if(misclassified[worker_id] == 1){
		sum_missed[worker_id] = 1;
		not_classified[worker_id] = 1;
	}
}
